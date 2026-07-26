# Document-Isolated Attention via Varlen — Plan & Justification

Companion doc. Goal: replace the crop-and-discard policy in `nanochat/dataloader.py` with proper document isolation in attention, and route it through `flash_attn_varlen_func` so we keep FA3 and avoid the per-step regression that an `attn_mask` fallback would cost.

## 1. TL;DR

Today, BOS-aligned documents are packed into rows with best-fit and the tail of each row is **cropped** (~35% of all tokens discarded). Attention inside the row is plain causal across **whatever tokens ended up packed**, regardless of document boundaries — the only "delimiter" is the BOS token, which the model must learn to interpret as a context-reset signal via the embedding.

This change makes document isolation **architectural** by using Flash Attention's varlen interface (`cu_seqlens`). Per-row compute stays the same. Per-step useful tokens roughly **1.5×** higher because we stop cropping. Expected val_bpb improvement driven by removal of two failure modes (cross-doc attention, semantic demand on BOS to be a learned reset).

Kill switch via `use_varlen_doc_attn` flag in `GPTConfig`. Default ON for new runs; flips to OFF with one CLI flag.

## 2. Background: how documents are currently handled

### 2.1 Storage

`nanochat/dataset.py` — pretraining data is parquet shards with a `text` column. **Each row is one document** (one web page / article from ClimbMix-400B). No explicit document IDs; document boundaries are implicit in the row structure.

### 2.2 Dataloader

`nanochat/dataloader.py` (`tokenizing_distributed_data_loader_with_state_bos_bestfit`) reads row-groups, tokenizes each document with a prepended BOS, and best-fit packs the resulting token lists into `(B, T+1)` rows:

1. From buffered docs, pick the **largest** doc that fits entirely in the remaining row capacity.
2. Place it whole.
3. Repeat until no doc fits.
4. Crop the **shortest** doc to fill the remaining capacity exactly.

`doc_buffer` is a list of full document token lists. Each row ends up as a flat `(T+1,)` vector. The model never sees the document-boundary information.

### 2.3 Attention

`nanochat/gpt.py:115` — single call site:

```python
y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
```

`q, k, v` are `(B, T, H, D)`. The only structured mask is `causal` plus a per-layer sliding window. **No document awareness at all.** A token near the end of a row can attend to any token earlier in the row, including tokens from previous documents packed into the same row.

### 2.4 The cost of the current scheme

- ~35% of tokens are crops (per the docstring at `nanochat/dataloader.py:5`). For d26 / T=2048, this is a structural data inefficiency: we paid for the tokenization, paid for the HtoD copy, computed attention over it, and contributed zero gradient.
- The model must learn that BOS = "new document, ignore prior context." This is theoretically possible (GPT-2 does it) but is a load-bearing demand on a single token embedding.
- The training distribution is mildly inconsistent: BOS sometimes appears in clean position-0 context, sometimes directly after a cropped tail (artifact of the packing).

## 3. The fix: varlen attention with `cu_seqlens`

### 3.1 What `cu_seqlens` does

`flash_attn_varlen_func` (FA2 and FA3 native) takes the same `q, k, v` but as a 1D sequence `(total_tokens, H, D)` plus a `cu_seqlens: int32[N_docs+1]` tensor. The kernel treats each "sequence" (each document) as **causally isolated**:

- No token in doc `i` sees any token in doc `j`.
- The BOS token at the head of each doc is now a true architectural reset, not a learned heuristic.

### 3.2 Why this approach over the alternatives

| Approach | Doc isolation | FA3 path? | Per-step cost | Useful tokens/step |
|---|---|---|---|---|
| **Current**: best-fit + crop | None (BOS-as-heuristic) | Yes (`flash_attn_func`) | baseline | ~65% of T·B |
| **Varlen** (`cu_seqlens`) | Architectural | Yes (`flash_attn_varlen_func`) | **0–3% slower kernel**, often flat | ~100% of T·B |
| SDPA + bool mask | Architectural | **No** — falls back to SDPA | **~1.5–2× slower per step** | ~65% (still padded) |
| Custom CUDA kernel | Architectural | N/A | N/A (weeks of work) | ~100% |

Two non-obvious points:

- **FA2/FA3 do not accept arbitrary boolean `attn_mask`.** The only structured masks they support are `causal` and `window_size`. Adding a per-pair document mask forces the SDPA fallback path, which loses FA3's kernel speed.
- **Varlen has no compute regression.** The attention FLOPs (`B · H · T²` worth of work) are unchanged — the kernel still processes every token in the row. The "savings" are in not throwing away 35% of the data, not in faster math.

### 3.3 What this is not

- This is **not** introducing a new sequence-level positional scheme. RoPE stays per-token within each document.
- This is **not** changing the loss. Targets are still just `inputs[:, 1:]`; BOS/regular tokens are all trained on equally.
- This is **not** changing the eval path (currently the same dataloader used for val; val gets the same treatment as train).

## 4. Expected impact

These are educated estimates, not measurements. Numbers should be treated as order-of-magnitude.

### 4.1 Data efficiency

- **Per useful token: same FLOPs.** Step compute is unchanged.
- **Per step: ~1.5× more useful tokens.** Today's ~35% crop means that of the `T·B` tokens the model sees, only ~65% contribute a gradient. With varlen padding (or doc padding at the end of rows), the model still sees ~`T·B` tokens per step, but ~100% of them contribute.
- Wall-clock-to-target drops by roughly the same factor (1.5×) all else equal, IF the bottleneck is data efficiency. The speedrun is currently FLOPs-bound on the H100s, so the win is **more likely to show up in val_bpb at a given wall-clock time, not in faster wall-clock to a given val_bpb**.

### 4.2 Quality signal

- **Cross-doc attention is removed.** Today, late tokens in a row attend to early tokens that came from a different document. In the best case this is noise; in the worst case it's actively misleading (the model learns that "groceries" is a plausible continuation of "quantum mechanics"). Removing it should improve sample efficiency slightly.
- **BOS is no longer load-bearing.** The model can still learn that BOS is a context-refresh signal (it's a useful inductive bias), but the model is no longer required to learn it under failure. Should reduce gradient noise on the BOS embedding.
- **Crop artifacts are removed.** The current scheme forces BOS to appear after a random crop tail ~35% of the time. With varlen, BOS is always position-0 of a doc, unconditionally.

### 4.3 What probably doesn't change

- `tok_per_sec` over a full step (negligible kernel overhead).
- `bf16_mfu` (FLOPs per token unchanged).
- `peak VRAM` (the per-token K/V footprint is the same; we add a `cu_seqlens` tensor of size `(B, max_docs_per_row + 1)` which is tens of KB).
- Optimizer state (no new parameters).

### 4.4 What might get worse

- **Bin-packing efficiency can drop slightly** if we no longer crop. The current packing is 100% utilization of the row; varlen with last-row padding is closer to ~85–95% depending on doc-length distribution. This is the **small** downside. Mitigation: pack smarter (best-fit-decreasing across rows instead of within-row) or accept it.
- **Sliding-window attention** (`window_size` parameter) currently applies uniformly across a row. With varlen, sliding window + `-1` left context is fine; with a small `window_size` it's a no-op for the last few tokens of a doc (by construction). Probably no behavior change in practice, but worth verifying.

## 5. Implementation plan

### 5.1 Config (`nanochat/gpt.py`)

Add to `GPTConfig` (around line 29, next to `window_pattern`):

```python
use_varlen_doc_attn: bool = True   # default ON for new runs
```

CLI flag in `scripts/base_train.py` (around the existing window-pattern args):

```python
parser.add_argument("--no-varlen-doc-attn", action="store_false",
                    dest="use_varlen_doc_attn", default=True,
                    help="Disable varlen document attention (fall back to crop)")
```

### 5.2 Dataloader (`nanochat/dataloader.py`)

The current loader writes two tensors: `cpu_inputs` and `cpu_targets`, both `(B, T)`. We add a third:

```python
doc_offsets = torch.zeros((B, max_docs_per_row + 1), dtype=torch.int32)
# doc_offsets[b, 0] = len(doc0 in row b)
# doc_offsets[b, 1] = len(doc0) + len(doc1)
# ...
# Pad the trailing slot with `T` so padding positions are out-of-bounds.
```

The best-fit packing loop is modified to:
- Track `max_docs_per_row` (pre-allocated, e.g. 32).
- On each packed doc, write the cumulative length into `doc_offsets[b, doc_idx]`.
- On the final crop (or on padding), write `row_capacity` into the next slot.

The dataloader's `yield` becomes `(inputs, targets, doc_offsets, state_dict)` instead of `(inputs, targets, state_dict)`.

Helper API for backward compat (the `tokenizing_distributed_data_loader_bos_bestfit` thin wrapper omits `doc_offsets` for callers that don't need it, mirroring how it already omits `state_dict`).

**DDP sharding**: unchanged. Each rank still reads its own row-groups (`rg_idx = ddp_rank`); documents are still sharded by row-group index. The `doc_offsets` tensor is per-rank, so no cross-rank coordination is needed.

**Resume state**: `doc_offsets` is not in the state_dict — it's deterministic from the iteration position. The state_dict only tracks `(pq_idx, rg_idx, epoch)`, same as today.

### 5.3 Model forward (`nanochat/gpt.py`)

Single change inside the attention block (around `gpt.py:115`):

```python
if kv_cache is None:
    if doc_offsets is not None:
        # Varlen training path
        from nanochat.flash_attention import _fa3, USE_FA3
        from nanochat.common import COMPUTE_DTYPE

        # Flatten (B, T, H, D) -> (total_tokens, H, D)
        # Build per-row cu_seqlens from doc_offsets + drop the trailing padding slot
        cu_seqlens = build_cu_seqlens(doc_offsets, T)  # (B, max_docs_per_row,)

        if USE_FA3:
            y = _fa3.flash_attn_varlen_func(
                q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                causal=True, window_size=window_size,
            )
        else:
            y = sdpa_varlen_attention(q, k, v, cu_seqlens, window_size)
        # Reshape (total_tokens, H, D) -> (B, T, H, D)
    else:
        # Existing path (inference, or fallback when flag is off)
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
else:
    # Inference path — unchanged
    y = flash_attn.flash_attn_with_kvcache(...)
```

Helper `build_cu_seqlens(doc_offsets, T)`:
- Input: `doc_offsets` of shape `(B, max_docs_per_row + 1)`, padded with `T`.
- Output: a 1D `int32` tensor of shape `(B * max_docs_per_row,)` for the varlen kernel, OR a per-row padded `(B, max_docs_per_row)` tensor if the kernel accepts batched lengths.
- varlen convention is to flatten: each row's lengths are concatenated with zero prepended. So `cu_seqlens = flatten([0, doc_offsets[b, 0], doc_offsets[b, 1], ..., 0, ...])` minus trailing duplicates. Verify with FA3's exact signature.

The `max_seqlen` parameter is `doc_offsets[:, -2].max()` (the longest doc across all rows). For homogeneous docs (the typical case in ClimbMix), this is roughly `T / 5` or so.

### 5.4 SDPA fallback (`nanochat/flash_attention.py`)

The existing `_sdpa_attention` gets a varlen variant. Two options:

- **Easy**: `torch.nn.functional.scaled_dot_product_attention` does not directly accept `cu_seqlens`. So we'd need to construct a per-row boolean mask per sample (still expensive) — defeating the purpose.
- **Right**: use `torch.nn.attention.flexible_attention` (PyTorch 2.5+) or roll a small varlen SDPA wrapper. For H100s that have FA3, the SDPA path is the slow path; can punt for v1 by asserting varlen requires FA3.

Pragmatic choice: **v1 asserts FA3 is available when varlen is on**. The nanochat d26 speedrun is always on H100s, so FA3 is always there. Smaller-dev users (CPU/MPS/older CUDA) keep the crop path by defaulting the flag to ON but documenting the dependency.

### 5.5 Async / metrics / wrappers

- `nanochat/dataloader_async.py`: no change. The async wrapper just shuttles whatever the generator yields.
- `nanochat/dataloader_metrics.py`: one addition — emit `data_crop_pct` (current `(cropped_tokens / total_tokens) * 100`) for the crop path. With varlen this becomes 0. Useful to keep the metric for the rollback path.
- `scripts/base_train.py` line 333: pass `use_varlen_doc_attn=args.use_varlen_doc_attn` into the dataloader factory, propagate into `doc_offsets`.

### 5.6 What stays the same

- Tokenizer BOS handling
- `window_pattern` (sliding window)
- `value_embeds`
- `resid_lambdas`, `x0_lambdas`
- Optimizer
- DDP sharding
- Eval path (same loader, same flag plumbing)
- `kv_cache` inference path

### 5.7 What might break

- **`torch.compile`**: the new `if doc_offsets is not None` branch is a dynamic control-flow edge. Compile will recompile. Put the branch outside the hot loop if possible, or use `torch.compile(dynamic=False)` override.
- **Memory peaks during backward**: varlen backward is a single fused kernel call that retains `q, k, v, o, do, dq, dk, dv` — same as today. No new autograd graph nodes introduced by the masking.
- **Doc-aware packed rows + sliding window**: when `window_size < T`, the per-doc window constraint is naturally tighter (a token cannot look further than its window even if it would have been able to across a doc boundary). This is correct, but check that the existing tests don't assume cross-doc attention ranges.
- **`chat_sft.py`**: uses its own `sft_data_generator_bos_bestfit` (already pads, doesn't crop). It does NOT use `dataloader.py`. Independent — but worth porting to varlen in a follow-up PR for consistency.

## 6. Measurement plan

### 6.1 Primary signal

`val_bpb` at fixed wall-clock time, d26 speedrun. Compare against the current best `0.71800` baseline (LEADERBOARD row 6). Expected Δ: `-0.005` to `-0.015`. Anything worse than `-0.002` is "no result" and we should investigate.

### 6.2 Secondary signals

- **Token utilization**: log `useful_tokens / total_tokens` per step. Should be 1.0 with varlen, ~0.65 with crop. Sanity check.
- **BOS gradient norm**: track the gradient norm of the BOS embedding via TensorBoard. With varlen, this should drop (the model doesn't need to fight cross-doc attention). A rise would suggest that BOS is being asked to do something even more load-bearing than before.
- **CORE**: only meaningful after a full speedrun. Use the same eval pipeline.

### 6.3 Per-step performance

- `tok_per_sec`: should be unchanged within ±3%.
- `bf16_mfu`: unchanged within ±1%.
- `train/data_*` metrics from `nanochat/dataloader_metrics.py`: `producer_total_ms` and `producer_max_ms` should be ~same as crop path (the varlen write is just an extra `int32` tensor copy).
- `peak VRAM`: should be unchanged or slightly lower (we hold `doc_offsets` instead of the cropped-tokens buffer).

### 6.4 Ablations to run

1. **Varlen OFF, crop ON**: confirm we can reproduce the baseline bpb. (Validates the kill switch.)
2. **Varlen ON, no doc_offsets (force-flat)**: every row treated as one doc. Should be equivalent to the current crop path. (Validates the cu_seqlens implementation.)
3. **Varlen ON, random doc_offsets**: garbage doc boundaries. Should regress bpb. (Validates that the doc isolation itself is the source of the win, not some artifact of the implementation.)

## 7. Kill switch and rollback

### 7.1 Single flag

`GPTConfig.use_varlen_doc_attn: bool = True` (default ON). One CLI flag flips it. When OFF, the dataloader falls back to the current crop path and the model uses `flash_attn_func` (not varlen). Zero new code paths executed.

### 7.2 Rollback

The diff touches:
- `nanochat/gpt.py` (config + attention branch + `build_cu_seqlens` helper)
- `nanochat/dataloader.py` (yield `doc_offsets`, pack with doc boundaries)
- `nanochat/flash_attention.py` (varlen exports)
- `scripts/base_train.py` (CLI flag + threading)

No public API changes outside the dataloader's `yield` signature, which is already internal. Revert is two clicks.

## 8. Open questions

- **Should padding tokens contribute a loss contribution, or be masked like in SFT?** Today's code trains on every emitted token. With varlen, the "padding" tokens (positions after the last doc ends in a row) are computed by attention but contribute nothing to the model output. They're equivalent to the current crops. Masking them with `ignore_index=-1` is cleaner but doesn't change the gradient (the model output is masked first).
- **Should we also reduce `T` to compensate for the lower (~85%) packing efficiency?** Probably not for v1; wait for the measurement. If the win is clean, accept the small packing loss.
- **Best-fit decreasing across rows** instead of within-row: would tighten packing to ~95% utilization. Probably overkill for v1.
- **Porting to `chat_sft.py`**: SFT already pads, so the change is trivial there (~0.5 day). Worth bundling in a follow-up PR.

## 9. Out of scope (v1)

- `cu_doc_lens` (cross-doc boundaries inside a single sequence) — research-grade, not in upstream FA3.
- Custom CUDA kernels for varlen sliding-window.
- Replacing the parquet row-group shuffle with a true streaming dataloader.
- Inference-time changes (the path is per-token KV cache, no doc concept needed).

## 10. Estimated effort

- **Implementation**: 2–3 days for someone fluent in the codebase. Heaviest lift is the dataloader refactor around `doc_offsets`; the model-side change is ~50 lines; the SDPA fallback is a punt.
- **Validation**: 1 day run + analyze on d12 first, then d20, then d26.
- **Total**: ~1 week calendar time from start to "ready to merge or kill."

## 11. Why this is a reasonable PR

- It replaces a known inefficiency (crop) with a strictly better mechanism (varlen) **using FA's native API**, not a custom kernel.
- It has a single, clean kill switch.
- It does not change FLOPs, memory, optimizer state, or any other architectural parameter.
- It removes a load-bearing demand on the BOS embedding that the current design accidentally places.
- The pattern is well-precedented: every modern large-scale pretraining setup (Megatron, llama.cpp, HF transformers' `varlen` flavor) uses cu_seqlens for exactly this reason.

The only reason to **not** do this is if the speedrun is currently FLOPs-bound rather than data-bound, in which case the win is smaller than projected. The measurement plan §6 is designed to detect this within one d12 run.
