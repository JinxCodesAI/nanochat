# Block AttnRes — Implementation Plan

Companion to `att_res.md` (Kimi paper notes). Goal: a measured, killable experiment.

## 1. What Block AttnRes is (one-paragraph version)

Replace the fixed `h_l = h_{l-1} + f(h_{l-1})` with softmax attention over depth: each layer attends over N block-level summaries of prior layers (plus the embedding). With 1 learned pseudo-query per layer and an RMSNorm on keys, total added compute is a `(1×d) × (d×(N+1))` matmul per call (~free). Storage: N `(B,T,d)` tensors. See `att_res.md` for full motivation; this doc is the implementation plan.

## 2. How it relates to existing code

| Existing | Interaction with Block AttnRes |
|---|---|
| `Block.forward` (gpt.py:152) — `x = x + attn(norm(x)); x = x + mlp(norm(x))` | AttnRes lives between `norm` and the residual add. The `x + ...` is *replaced* by `attn_res_weighted(sources) + ...`. |
| `resid_lambdas` / `x0_lambdas` (gpt.py:182) — applied at GPT.forward level, pre-block | Order: AttnRes first (replaces the `x + f(x)` accumulation), then existing lambdas. The lambdas become interpretable as gating the AttnRes output rather than gating a fixed residual add. |
| `value_embeds` — alternating layers, added to V inside attention (gpt.py:95) | Unaffected. VE modifies V inside attention; AttnRes modifies the residual stream. They compose. |
| `smear` / `backout` (gpt.py:184) | Smear happens before the trunk loop (one place in code); backout reads `x_backout` cached mid-loop. AttnRes block summaries are a *third* side-channel of intermediate state. Reasonable to leave smear/backout unchanged for v0. |
| `setup_optimizer` (gpt.py:402) — AdamW groups for scalars (`resid_params`, `x0_params`, `smear_params`) | Pseudo-queries `w_l` are 1D-ish — they fit naturally next to `resid_params`. Same LR × 0.01 as resid (per paper: small scalar-like params). |
| `_compute_window_sizes` (gpt.py:316) — sliding window | Independent. AttnRes attends over depth, not sequence. |
| `kv_cache` inference path (engine.py not checked, but pattern clear) | Block summaries for inference need a per-call argument list. Like `kv_cache`, has to be threaded through. v0 should test training only. |

**Key compatibility point:** nanochat uses vanilla autograd (no activation checkpointing, no pipeline parallelism). So Full AttnRes's "impossible due to O(Ld) memory" caveat doesn't apply — the L layer outputs are already in HBM. We *could* do Full AttnRes for free memory-wise. We use Block AttnRes anyway because it has ~half the HBM bandwidth cost (N=8 reads vs L=26 reads per layer). See `att_res.md` discussion §"Where Full AttnRes would be the right call".

## 3. Implementation

### 3.1 New config (`nanochat/gpt.py:29`)

Add to `GPTConfig`:
```python
use_block_attn_res: bool = False           # off-by-default, kill switch
block_attn_res_n_blocks: int = 8           # paper default
block_attn_res_rmsnorm_eps: float = 1e-6   # default
```

`CLI flag` (`scripts/base_train.py:50`-ish): `--block-attn-res` (bool) and `--block-attn-res-n-blocks` (int). Default OFF — `--block-attn-res` toggles on.

### 3.2 New submodules (`nanochat/gpt.py`)

Three small modules, total ~50-80 lines:

```python
class RMSNormNoBias(nn.Module):
    """RMSNorm used for AttnRes keys. No learnable scale (paper uses plain RMSNorm)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class BlockAttnRes(nn.Module):
    """One AttnRes call site (pre-attn or pre-MLP). Holds its own pseudo-query + RMSNorm."""
    def __init__(self, n_embd):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_embd))   # zero-init per paper §5
        self.key_norm = RMSNormNoBias(n_embd)
    def forward(self, sources):
        """
        sources: list of (B, T, d) tensors — block summaries + current partial.
        Length = N+1 (N completed blocks + embedding + 1 partial) or fewer at the first block.
        Returns: (B, T, d) aggregated hidden state.
        """
        stacked = torch.stack(sources, dim=0)                  # (K, B, T, d)
        keys = self.key_norm(stacked)                          # (K, B, T, d)
        logits = (keys * self.w.view(1, 1, 1, -1)).sum(-1)     # (K, B, T)
        weights = logits.softmax(dim=0)                        # softmax over sources
        return torch.einsum('kbt,kbtd->btd', weights, stacked)
```

Note: `keys * w` is implemented as einsum against a 1D `w` so the matmul fuses into a single `(1,d) × (K,B,T,d)` reduction along `d`. Tiny.

### 3.3 Hook into `Block.forward` and `GPT.forward`

**Do block management in `GPT.forward`**, not in `Block.forward`. Block management state (list of completed block summaries, current partial sum) is trunk-level.

Add to `GPT.__init__` (when `config.use_block_attn_res`):
```python
# 2 AttnRes call sites per layer (pre-attn, pre-MLP) for the first layer of each block;
# subsequent intra-block layers use just the current partial_block.
# Simplest: 2 call sites per layer, both reading same sources list.
self.attn_res_pre_attn = nn.ModuleList([BlockAttnRes(config.n_embd) for _ in range(n_layer)])
self.attn_res_pre_mlp  = nn.ModuleList([BlockAttnRes(config.n_embd) for _ in range(n_layer)])
```

Add 1D-params group in `setup_optimizer`: put `attn_res_pre_attn.*.w` and `attn_res_pre_mlp.*.w` into the same AdamW group as `resid_params` (LR × 0.01, beta1=0.8, weight_decay=0.05). Or a separate group if we want to tune.

**Modify `GPT.forward` loop** (currently gpt.py:495-503). Sketch — the existing `x = block(...)` line becomes:

```python
if self.config.use_block_attn_res:
    block_size = self.config.n_layer // self.config.block_attn_res_n_blocks
    # `block_summaries` holds one (B,T,d) tensor per completed block
    # `partial_block` is the running intra-block sum (starts as x after embed+norm+smear)
    block_summaries = []
    partial_block = x.clone()
    sources = [x0]   # b_0 = embedding, always a source

for i, block in enumerate(self.transformer.h):
    if self.config.use_block_attn_res:
        # Replace x with AttnRes-weighted aggregate before the block
        cur_sources = sources + [partial_block]
        x_for_block = self.attn_res_pre_attn[i](cur_sources)   # pre-attn
        # Run only the attention path
        ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
        attn_out = block.attn_only(x_for_block, ve, cos_sin, self.window_sizes[i], kv_cache)
        # Update partial_block with attn contribution
        partial_block = partial_block + attn_out
        # Compute MLP input from pre-MLP AttnRes
        x_for_mlp = self.attn_res_pre_mlp[i](cur_sources + [partial_block])
        mlp_out = block.mlp_only(x_for_mlp)
        # Update partial_block with mlp contribution
        partial_block = partial_block + mlp_out
        # x remains the running residual — set it to partial_block (or to its prev value + AttnRes output)
        x = partial_block
        # Apply existing resid/x0 lambdas on the new x
        x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        # Block boundary check
        if (i + 1) % block_size == 0:
            block_summaries.append(partial_block.detach() if not training else partial_block)  # detach debatable
            partial_block = x.clone()
    else:
        # Existing path, unchanged
        x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
        x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        if i == backout_layer:
            x_backout = x
```

**Note**: this sketch splits `Block.forward` into `attn_only` and `mlp_only` paths. Alternative (less invasive): keep `Block.forward` as-is, accept some redundancy. Decide based on what compiles cleanly under `torch.compile`.

**Memory model**: `block_summaries` is a Python list of N tensors, each `(B, T, d)`. PyTorch autograd handles this naturally — they're inputs to `block_attn_res`, so they're retained for backward via reference counting. No manual `.detach()` is needed (and would break gradient flow).

**Inference concern**: out of scope for v0. Note in docstring that the current sketch is training-only — `engine.py`'s inference loop would need updating to thread `block_summaries` through generation.

### 3.4 What stays the same

- All matmul shapes — no architectural FLOPs change.
- Optimizer state for matrix params — unchanged.
- Dataloader, masking, FA3, sliding window — all orthogonal.
- LR schedule, weight decay scaling — unchanged.
- FP8 conversion path — should still work (the new `BlockAttnRes` modules have only `nn.Parameter` and small ops, FP8 conversion doesn't touch them).

### 3.5 What might break

- **`torch.compile` recompiles** when the loop body conditional branches on `use_block_attn_res`. Use `torch._dynamo.allow_in_graph` or a separate compiled function for the two paths.
- **`init_weights` on meta device**: the new `BlockAttnRes.w` is 1D. Must initialize actual data in `init_weights`, not in `__init__` (existing footgun documented at gpt.py:158).
- **Block boundary alignment**: if `n_layer % n_blocks != 0`, the last block is partial. Code already handles this — just append the partial as the last summary at end of loop.
- **Backward memory**: storing N block summaries means the autograd graph retains those tensors. For d26, N=8 × (16, 2048, 1664) bf16 = ~870 MB extra retention. Probably fine, but profile.
- **Gradient through the softmax + weighted sum**: standard differentiable, but multiply with the optimizer's per-shape Muon grouping — `attn_res_pre_attn.*.w` and `attn_res_pre_mlp.*.w` are 1D params, naturally go in AdamW group.

## 4. How to measure

Three signals, in order of sensitivity:

### 4.1 `val_bpb` vs wall-clock time (primary metric, per LOG.md conventions)
- Plot training curve vs step and vs `total_training_time`.
- A meaningful improvement is `val_bpb` lower at the same wall-clock (capability win).
- For Block AttnRes, expect Δ val_bpb ~0.005-0.015 at d12-d26 (per paper scaling).
- Watch for the "early-training plateau": AttnRes needs ~100-500 steps for the pseudo-queries to learn meaningful weights (paper Table 4 ablation shows it converges from uniform).

### 4.2 `tok_per_sec` and `train/mfu`
- Block AttnRes should *not* significantly reduce tok/s. The `block_attn_res` call is one tiny matmul + one weighted sum; total per-layer FLOPS are unchanged.
- Realistic targets: tok/s unchanged within ±3%, MFU unchanged within ±1%.
- If tok/s drops >5%: the einsum isn't fusing properly under `torch.compile` — debug.
- If tok/s *increases* slightly: the AttnRes pre-projection might help by reducing effective `||x||` growth that confuses RMSNorm downstream (the paper observes this).

### 4.3 `CORE` metric (final, noisy)
- Only meaningful on full d24/d26 runs after training completes.
- Per Run 4 in `LEADERBOARD.md`, CORE has spread ~0.016 even at fixed seed. Need 3+ runs to disambiguate.

### 4.4 Memory: `peak VRAM` per rank
- Expected: +870 MB block summaries per rank at d26, batch=16 (8 blocks × 109 MB each).
- Watch for OOM at batch=32 — the block summaries might tip it over the cliff.
- If so, consider halving `block_attn_res_n_blocks` (N=4).

## 5. Kill switch (cleanly implemented)

Two layers of kill switch:

### 5.1 Config flag (zero runtime cost when off)
```python
# GPTConfig
use_block_attn_res: bool = False

# CLI
parser.add_argument("--block-attn-res", action="store_true", default=False,
                    help="Enable Block AttnRes (pre-attn + pre-MLP softmax over depth)")
parser.add_argument("--block-attn-res-n-blocks", type=int, default=8,
                    help="Number of blocks for Block AttnRes (only used if --block-attn-res)")
```

When `use_block_attn_res=False`, the `if/else` in `GPT.forward` chooses the *existing* code path. Zero overhead, binary identical to baseline (verify with a quick `git diff` of the compiled graph).

### 5.2 Pseudo-query zero-init safety net
Even if the flag *is* on, the `w_l` parameters start at zero, so initial attention weights are uniform (each source gets weight `1/(N+1)`). At init, Block AttnRes ≈ equal-weight average of sources ≈ equal-weight average of layer outputs (within a block) + embedding — which approximates the standard residual stream at depth-1 with some scaling.

This means: **even if AttnRes hurts quality, training should be stable**. Compare to x0_lambdas / resid_lambdas which the existing code is already used to.

## 6. Suggested experimental sequence

1. **d12 + Block AttnRes, N=4 blocks (size 3 layers)**: ~5 min run. Check val_bpb matches or beats baseline within first 1k steps. Token/s unchanged. If no regression, proceed.

2. **d12 + Block AttnRes, N=6 blocks (size 2 layers)**: same time budget. Compare N=4 vs N=6 to see if the paper's "S=2-8 are roughly equivalent" claim holds at d12 scale.

3. **d20 + Block AttnRes, N=8 blocks**: ~20 min run. This is the closest analog to d26. Should give the clearest signal.

4. **d26 + Block AttnRes, N=8 blocks**: full speedrun-time run. Compare against Run 6 baseline (1.65h, val_bpb 0.71800, CORE 0.2626).

5. **If positive at d26**: tune `block_attn_res_n_blocks ∈ {6, 8, 13}` and re-check.

## 7. Open questions for the implementer

- **Should `partial_block` be `detach()`-ed across micro-batch boundaries in DDP?** My instinct is no (let autograd do its thing), but DDP hook weirdness with retained tensors across micro-batches could surprise. Test.
- **Should `x0_lambdas` and `resid_lambdas` be removed when AttnRes is on?** They were added to fix the same problem AttnRes fixes. Probably should be left in (initialize to identity ~1.0) for the first experiment, then ablate later.
- **Interaction with FP8**: the new `block_attn_res` matmul is `(1×d) × (d×(N+1))` which is tiny. FP8 should not need to convert it. Verify `Float8Linear`'s module filter excludes these.
- **Optimizer placement**: pseudo-queries are 1D, ~26 of them per call site, ~52 total. Either group with `resid_params` (AdamW × 0.01 LR) or carve a new group. Start with grouped; ablate later.

## 8. Out of scope (for v1)

- Full AttnRes (overkill given HBM bandwidth)
- Inference-time support (engine.py updates)
- Cache-based pipeline parallel (paper §4.1, not relevant for nanochat's DDP setup)
- Two-phase inference strategy (paper §4.2, same reason)

## 9. Rollback

The flag is off by default. The PR diff touches `gpt.py` (config + new modules + forward loop) and `base_train.py` (CLI flags). All gates via `if self.config.use_block_attn_res:`. If the experiment fails, revert the PR — no other code depends on the new path.
