"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW

# Our custom Flash Attention module that automatically uses FA3 when compatible and SDPA fallback otherwise
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # Document isolation: when True, training uses flash_attn_varlen_func with
    # document boundaries from the dataloader. When False, falls back to plain
    # causal attention with the crop-and-discard packing policy.
    use_varlen_doc_attn: bool = True
    # Chunk size for the chunked cross-entropy loss. The lm_head output (B, T, vocab) in fp32
    # is O(B*T*vocab) bytes and dominates peak HBM at training scale (e.g. 4 GB at B=16,
    # T=2048, vocab=32768, fp32). Chunking along T avoids materializing the full buffer.
    # Set to T (or larger) to fall back to the one-shot path.
    loss_chunk_size: int = 512
    # Block Attention Residuals (AttnRes): replace uniform residual accumulation with
    # learned softmax attention over depth. Off by default (kill switch).
    # When enabled, layers are partitioned into N blocks; cross-block attention uses
    # block-level summaries, intra-block uses standard uniform accumulation.
    use_block_attn_res: bool = False
    block_attn_res_n_blocks: int = 8


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class BlockAttnRes(nn.Module):
    """One AttnRes call site (pre-attn or pre-MLP). Holds a learned pseudo-query w
    and an RMSNorm for the key representations. The forward pass computes softmax
    attention over a list of source tensors (block summaries + partial block + embedding)
    and returns the weighted aggregate.

    The pseudo-query w is zero-initialized, so at init all sources get uniform weight
    1/K and the output approximates a standard equal-weight residual average.
    """
    def __init__(self, n_embd, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_embd))
        self.eps = eps

    def forward(self, sources):
        """
        Args:
            sources: list of (B, T, d) tensors to attend over.
        Returns:
            (B, T, d) softmax-weighted aggregate of sources.
        """
        stacked = torch.stack(sources, dim=0)                          # (K, B, T, d)
        keys = stacked * torch.rsqrt(stacked.pow(2).mean(-1, keepdim=True) + self.eps)  # RMSNorm
        logits = (keys * self.w.view(1, 1, 1, -1)).sum(-1)             # (K, B, T)
        weights = logits.softmax(dim=0).to(dtype=stacked.dtype)         # softmax over sources, match dtype
        return torch.einsum('kbt,kbtd->btd', weights, stacked)


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    # note: this rotates by -theta, the transpose of the textbook convention. Functionally
    # equivalent (only the relative q/k rotation matters), kept for checkpoint compatibility.
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def _flatten_doc_offsets(doc_offsets, B, T):
    """
    Convert the per-row (B, max_docs+1) doc-boundary tensor into a fixed-shape
    1D cu_seqlens tensor for FA3's varlen interface.

    The output is always padded to the maximum possible size
    (B * max_docs_per_row + 1) so that torch.compile(dynamic=False) sees a
    static tensor shape across batches.  Trailing entries past the real docs
    are filled with B*T, which FA3 treats as zero-length sequences (no-ops).

    The model should use max_seqlen = T (the row length) as a safe, constant
    upper bound for FA3's workspace allocation.  This avoids torch.compile
    scalar guards on a per-batch value that varies with doc-length distribution.

    Dataloader layout: doc_offsets[b, 0] = 0 (start of doc 0), and
    doc_offsets[b, k] = end position of doc k-1 in row b (for k >= 1). Unused
    trailing slots are filled with row_capacity = T+1.

    This is fully vectorized — no Python loops, no .item() syncs — so it runs
    at GPU speed and can be fused with torch.compile if placed inside the model.

    Returns:
        cu_seqlens: 1D int32 tensor of fixed length (B*max_docs_per_row + 1).
    """
    assert doc_offsets.dim() == 2 and doc_offsets.size(0) == B, (
        f"doc_offsets must be (B, max_docs+1), got {tuple(doc_offsets.shape)}"
    )
    D = doc_offsets.size(1) - 1  # max_docs_per_row
    max_cu_len = B * D + 1       # fixed output length
    device = doc_offsets.device

    # Row offset: flat global position = doc_offsets[b,k] + b*T
    row_offset = torch.arange(B, dtype=torch.int32, device=device) * T  # (B,)
    flat = doc_offsets.to(torch.int32) + row_offset.unsqueeze(1)        # (B, D+1)

    # Select real doc ends: columns 1..D that are < T+1 (the start marker
    # at column 0 is always 0 and handled implicitly by the row boundaries).
    cols = torch.arange(D + 1, device=device)
    is_doc_end = (doc_offsets < (T + 1)) & (cols >= 1).unsqueeze(0)  # (B, D+1)
    doc_ends = flat[is_doc_end]  # flat global positions of all inter-document boundaries

    # Row boundaries [T, 2T, ..., B*T]: these guarantee each row is represented
    # as a contiguous segment even when its docs don't fully pack to T.
    # Doc ends that land exactly on a row boundary (doc fills to exactly T)
    # produce duplicates — sort + dedup below removes them.
    row_ends = (torch.arange(1, B + 1, dtype=torch.int32, device=device)) * T  # (B,)

    # All candidate boundary points in one flat tensor
    all_ends = torch.cat([doc_ends, row_ends])  # (~B*D + B)
    all_ends_sorted, _ = all_ends.sort()

    # Remove consecutive duplicates (row boundaries overlapping with doc ends
    # or with each other in degenerate cases).
    if all_ends_sorted.numel() >= 2:
        keep = torch.ones(all_ends_sorted.numel(), dtype=torch.bool, device=device)
        keep[1:] = (all_ends_sorted[1:] != all_ends_sorted[:-1])
        boundaries = all_ends_sorted[keep]
    else:
        boundaries = all_ends_sorted

    # Assemble: [0, ...boundaries..., B*T]
    B_T = torch.tensor([B * T], dtype=torch.int32, device=device)
    cu = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),
        boundaries,
        B_T,
    ])

    # Final dedup (handles B*T overlapping with the last row_end entry)
    if cu.numel() >= 2:
        keep = torch.ones(cu.numel(), dtype=torch.bool, device=device)
        keep[1:] = (cu[1:] != cu[:-1])
        cu = cu[keep]

    # Pad to fixed length: trailing B*T entries are zero-length segments to FA3
    if cu.numel() < max_cu_len:
        pad = torch.full((max_cu_len - cu.numel(),), B * T, dtype=torch.int32, device=device)
        cu = torch.cat([cu, pad])

    return cu.contiguous()

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache, cu_seqlens=None, max_seqlen=0):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 or SDPA fallback)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None and cu_seqlens is not None:
            # Document-isolated training via varlen attention: the cu_seqlens
            # tensor was built once in GPT.forward (outside torch.compile) from
            # the doc_offsets buffer. Row-major flattening matches the dataloader
            # layout.
            q_flat = q.reshape(-1, self.n_head, self.head_dim)
            k_flat = k.reshape(-1, self.n_kv_head, self.head_dim)
            v_flat = v.reshape(-1, self.n_kv_head, self.head_dim)
            y_flat = flash_attn.flash_attn_varlen_func(
                q_flat, k_flat, v_flat,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                causal=True, window_size=window_size,
            )
            y = y_flat.reshape(B, T, self.n_head, self.head_dim)
        elif kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, cu_seqlens=None, max_seqlen=0):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        x = x + self.mlp(norm(x))
        return x

    def attn_forward(self, x, ve, cos_sin, window_size, kv_cache, cu_seqlens=None, max_seqlen=0):
        """Norm + attention sub-block, without the residual add. Used by AttnRes path."""
        return self.attn(norm(x), ve, cos_sin, window_size, kv_cache, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

    def mlp_forward(self, x):
        """Norm + MLP sub-block, without the residual add. Used by AttnRes path."""
        return self.mlp(norm(x))


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Block AttnRes: learned softmax attention over depth (off by default)
        if config.use_block_attn_res:
            self.attn_res_pre_attn = nn.ModuleList([BlockAttnRes(config.n_embd) for _ in range(config.n_layer)])
            self.attn_res_pre_mlp = nn.ModuleList([BlockAttnRes(config.n_embd) for _ in range(config.n_layer)])
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Block AttnRes pseudo-queries: zero-init → uniform attention at start
        if self.config.use_block_attn_res:
            for m in self.attn_res_pre_attn:
                torch.nn.init.zeros_(m.w)
            for m in self.attn_res_pre_mlp:
                torch.nn.init.zeros_(m.w)

        # Smear/backout scalars and smear gate must be explicitly initialized 
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * self.num_matmul_params() + attn_flops
        return num_flops_per_token

    def num_matmul_params(self):
        """
        The number of parameters that participate in matmuls with the token stream,
        i.e. contribute 2 FLOPs/param to the forward pass. Counted structurally: every
        matmul in this model goes through the Linear class, while non-matmul params
        (embeddings = lookups, per-layer scalars) are nn.Embedding or raw Parameters.
        """
        matmul_params = sum(m.weight.numel() for m in self.modules() if isinstance(m, Linear))
        return matmul_params

    def estimate_decode_flops(self, context_len):
        """
        Forward FLOPs to decode one token at a given context length during inference:
        2 FLOPs per matmul param, plus attention over min(context, window) per layer.
        """
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = sum(4 * h * q * min(context_len, window) for window, _ in self.window_sizes)
        decode_flops = 2 * self.num_matmul_params() + attn_flops
        return decode_flops

    def estimate_prefill_flops(self, num_tokens):
        """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window)."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = 0
        for window, _ in self.window_sizes:
            w = min(window, num_tokens)
            attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w # ramp up to w, then flat
            attn_flops += 4 * h * q * attended_tokens
        prefill_flops = 2 * self.num_matmul_params() * num_tokens + attn_flops
        return prefill_flops

    def kv_bytes_per_token(self):
        """Bytes to *store* one token of KV cache during inference, per row (all layers)."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize # the KV cache is kept in the compute dtype
        return self.config.n_layer * 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes

    def kv_read_bytes(self, context_len):
        """Bytes of KV cache *read* by one decode step at a given context length, per row.
        Sliding window layers only attend to (and read) the last `window` tokens."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        total = 0
        for window, _ in self.window_sizes:
            total += 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes * min(context_len, window)
        return total

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        # Count AttnRes pseudo-queries if enabled (1D params, naturally scalars)
        if self.config.use_block_attn_res:
            scalars += sum(m.w.numel() for m in self.attn_res_pre_attn)
            scalars += sum(m.w.numel() for m in self.attn_res_pre_mlp)
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        # Block AttnRes pseudo-queries: 1D scalar-like params, group with resid (LR×0.01)
        if self.config.use_block_attn_res:
            attn_res_w_params = [m.w for m in self.attn_res_pre_attn] + [m.w for m in self.attn_res_pre_mlp]
            resid_params = resid_params + attn_res_w_params
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', cu_seqlens=None, max_seqlen=0):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None
        if self.config.use_block_attn_res:
            # Block AttnRes: replace uniform residual accumulation with learned softmax
            # attention over depth. Layers are partitioned into N blocks; completed blocks
            # are saved as summaries. Each layer uses attn_res_pre_attn/attn_res_pre_mlp to
            # compute a weighted aggregate of (block_summaries + embedding + partial_block)
            # before feeding into the sub-layer.
            n_blocks = self.config.block_attn_res_n_blocks
            block_size = max(1, n_layer // n_blocks)
            block_summaries = []   # one (B,T,d) tensor per completed block
            partial_block = x  # running intra-block sum (no clone: tensors are never mutated in-place)
            for i, block in enumerate(self.transformer.h):
                # Pre-attn: AttnRes over completed blocks + embedding + partial
                sources = block_summaries + [x0, partial_block]
                x_attn = self.attn_res_pre_attn[i](sources)
                ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
                attn_out = block.attn_forward(x_attn, ve, cos_sin, self.window_sizes[i], kv_cache, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
                partial_block = partial_block + attn_out
                # Pre-MLP: AttnRes (with updated partial_block after attn)
                sources = block_summaries + [x0, partial_block]
                x_mlp = self.attn_res_pre_mlp[i](sources)
                mlp_out = block.mlp_forward(x_mlp)
                partial_block = partial_block + mlp_out
                # Apply lambdas as post-hoc scaling on the layer output
                x = self.resid_lambdas[i] * partial_block + self.x0_lambdas[i] * x0
                partial_block = x
                # Backout cache (same as standard path)
                if i == backout_layer:
                    x_backout = x
                # Block boundary: save completed block summary, continue with current x
                if (i + 1) % block_size == 0 and i < n_layer - 1:
                    block_summaries.append(x)  # no clone: x is reassigned next iter, tensor is immutable
        else:
            # Standard path: uniform residual accumulation (unchanged)
            for i, block in enumerate(self.transformer.h):
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
                x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
                if i == backout_layer:
                    x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head and compute loss in chunks along T to avoid materializing the
        # full (B, T, vocab) fp32 logits buffer in HBM. Peak buffer size drops from
        # B*T*vocab*4 bytes to B*chunk_size*vocab*4 bytes (e.g. 4 GB -> 1 GB at B=16, vocab=32K).
        # Mathematically identical to F.cross_entropy(reduction=loss_reduction, ignore_index=-1).
        if targets is not None:
            return self._chunked_loss(x, targets, loss_reduction=loss_reduction)

        # Inference / sampling path: return softcapped logits in COMPUTE_DTYPE.
        # The softcap is applied in fp32 for numerical stability (matches training),
        # then cast back to COMPUTE_DTYPE so callers see bf16 logits. Note that
        # `logits.to(torch.float32)` does materialize a full (B, T, vocab) fp32 buffer
        # in HBM (eager mode does not elide this cast). This is fine here because
        # generate() only consumes the last token's logits; the fp32 buffer is transient.
        softcap = 15
        logits = self.lm_head(x)  # (B, T, padded_vocab_size)
        logits = logits[..., :self.config.vocab_size]
        logits = (softcap * torch.tanh(logits.to(torch.float32) / softcap)).to(logits.dtype)
        return logits

    def _chunked_loss(self, x, targets, loss_reduction='mean'):
        """
        Compute the cross-entropy loss in chunks along the sequence dimension.

        Why: at training scale, the lm_head output (B, T, vocab) in fp32 is the single
        largest activation buffer (e.g. 4 GB at B=16, T=2048, vocab=32768, fp32).
        Chunking along T reduces the peak buffer to (B, chunk, vocab) per iteration.

        Math: identical to F.cross_entropy(logits, targets, ignore_index=-1, reduction=...).
        - 'mean': average NLL over non-ignored tokens (matches PyTorch's reduction='mean' with ignore_index).
        - 'none': per-token NLL tensor of shape (B, T) (ignored positions are 0).

        Args:
            x: hidden states, shape (B, T, n_embd), in COMPUTE_DTYPE
            targets: target token ids, shape (B, T), dtype long; use -1 for ignored positions
            loss_reduction: 'mean' or 'none' (also accepts 'sum' for completeness)
        """
        B, T, _ = x.shape
        V = self.config.vocab_size
        softcap = 15

        # If T is small enough to fit in one chunk, skip the chunking overhead and use
        # the one-shot path. This also makes the common inference / short-seq case fast.
        chunk_size = self.config.loss_chunk_size
        if chunk_size <= 0 or chunk_size >= T:
            logits = self.lm_head(x)[..., :V]            # (B, T, vocab)
            logits = logits.float()                      # fp32 for stable softcap + CE
            logits = softcap * torch.tanh(logits / softcap)
            return F.cross_entropy(
                logits.view(-1, V), targets.view(-1),
                ignore_index=-1, reduction=loss_reduction,
            )

        # Chunked path: keep at most (B, chunk, vocab) fp32 logits live at once.
        if loss_reduction == 'mean':
            # Accumulate sum-of-NLL and count-of-valid-tokens across chunks, divide at the end.
            # This matches F.cross_entropy(reduction='mean', ignore_index=-1) exactly:
            # the denominator excludes ignored positions.
            total_sum = x.new_zeros((), dtype=torch.float32)
            total_count = x.new_zeros((), dtype=torch.int64)
            for t0 in range(0, T, chunk_size):
                t1 = min(t0 + chunk_size, T)
                chunk_logits = self.lm_head(x[:, t0:t1])[..., :V].float()
                chunk_logits = softcap * torch.tanh(chunk_logits / softcap)
                chunk_targets = targets[:, t0:t1]
                total_sum = total_sum + F.cross_entropy(
                    chunk_logits.reshape(-1, V), chunk_targets.reshape(-1),
                    ignore_index=-1, reduction='sum',
                )
                # count of non-ignored targets in this chunk
                total_count = total_count + (chunk_targets >= 0).sum()
            # clamp(min=1) avoids 0/0 when all targets in a micro-batch are ignored
            return total_sum / total_count.clamp(min=1)

        # 'none' (or 'sum'): write per-token losses into a pre-allocated (B, T) buffer.
        # For 'sum' we still allocate the full buffer and reduce at the end; this is rare.
        per_token = torch.empty(B, T, dtype=torch.float32, device=x.device)
        for t0 in range(0, T, chunk_size):
            t1 = min(t0 + chunk_size, T)
            chunk_logits = self.lm_head(x[:, t0:t1])[..., :V].float()
            chunk_logits = softcap * torch.tanh(chunk_logits / softcap)
            chunk_targets = targets[:, t0:t1]
            per_token[:, t0:t1] = F.cross_entropy(
                chunk_logits.reshape(-1, V), chunk_targets.reshape(-1),
                ignore_index=-1, reduction='none',
            ).view(B, t1 - t0)
        if loss_reduction == 'sum':
            return per_token.sum()
        return per_token

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
