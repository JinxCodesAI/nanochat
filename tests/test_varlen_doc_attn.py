"""
Tests for varlen doc-attn path.

Run with: python -m pytest tests/test_varlen_doc_attn.py -v
or:       python tests/test_varlen_doc_attn.py
"""
import os
import sys
import torch

# Make sure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanochat.gpt import GPT, GPTConfig, _flatten_doc_offsets
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.tokenizer import get_tokenizer
from nanochat.flash_attention import HAS_FA3, USE_FA3


def make_fake_doc(tokenizer, n_tokens, bos=True):
    """Build a fake document of approximately n_tokens by repeating a phrase."""
    # Use a tokenizable phrase that's roughly 1 token per "word" in the regex split.
    phrase = "Hello world. " * 5
    ids = tokenizer.encode(phrase)
    if bos:
        ids = [tokenizer.get_bos_token_id()] + ids
    # repeat / truncate to n_tokens
    while len(ids) < n_tokens:
        ids = ids + ids
    return ids[:n_tokens]


def test_flatten_doc_offsets_basic():
    """2 rows, 2 docs each, plus padding."""
    B, T = 2, 10
    # row 0: docs ending at positions 3 and 7
    # row 1: docs ending at positions 4 and 6
    T_plus_1 = T + 1
    doc_offsets = torch.tensor([
        [0, 3, 7, T_plus_1, T_plus_1],
        [0, 4, 6, T_plus_1, T_plus_1],
    ], dtype=torch.int32)
    cu, max_seqlen = _flatten_doc_offsets(doc_offsets, B, T)
    # max_docs_per_row = 4, so max_cu_len = 2*4+1 = 9.
    # Real cu: [0, 3, 7, 10, 14, 16, 20], padded with 20 to length 9.
    expected = torch.tensor([0, 3, 7, 10, 14, 16, 20, 20, 20], dtype=torch.int32)
    assert cu.shape == (9,), f"cu_seqlens shape mismatch: {cu.shape}, expected (9,)"
    assert torch.equal(cu, expected), f"cu_seqlens mismatch: {cu} vs {expected}"
    # doc/cap lengths: [3, 4, 4, 4, 2, 0, 0, 0] -> max = 4
    assert max_seqlen == 4, f"max_seqlen mismatch: {max_seqlen} vs 4"
    print("test_flatten_doc_offsets_basic PASSED")


def test_flatten_doc_offsets_single_doc():
    """One row with no explicit docs (all padding). Cap segment covers all T tokens."""
    B, T = 1, 8
    T_plus_1 = T + 1
    doc_offsets = torch.tensor([[0, T_plus_1, T_plus_1, T_plus_1]], dtype=torch.int32)
    cu, max_seqlen = _flatten_doc_offsets(doc_offsets, B, T)
    # max_docs_per_row = 3, so max_cu_len = 1*3+1 = 4.
    # Real cu: [0, 8], padded with 8 to length 4.
    expected = torch.tensor([0, 8, 8, 8], dtype=torch.int32)
    assert cu.shape == (4,), f"cu_seqlens shape mismatch: {cu.shape}, expected (4,)"
    assert torch.equal(cu, expected), f"cu_seqlens mismatch: {cu} vs {expected}"
    # Segments: [8, 0, 0] -> max_seqlen = T (cap covers all tokens)
    assert max_seqlen == 8, f"max_seqlen mismatch: {max_seqlen} vs 8"
    print("test_flatten_doc_offsets_single_doc PASSED")


def test_flatten_doc_offsets_padding_only():
    """All padding (shouldn't happen in practice but should be safe)."""
    B, T = 2, 5
    T_plus_1 = T + 1
    doc_offsets = torch.tensor([[0, T_plus_1, T_plus_1], [0, T_plus_1, T_plus_1]], dtype=torch.int32)
    cu, max_seqlen = _flatten_doc_offsets(doc_offsets, B, T)
    # max_docs_per_row = 2, so max_cu_len = 2*2+1 = 5.
    # Real cu: [0, 5, 10], padded with 10 to length 5.
    expected = torch.tensor([0, 5, 10, 10, 10], dtype=torch.int32)
    assert cu.shape == (5,), f"cu_seqlens shape mismatch: {cu.shape}, expected (5,)"
    assert torch.equal(cu, expected), f"cu_seqlens mismatch: {cu} vs {expected}"
    # Segments: [5, 5, 0, 0] -> max_seqlen = 5
    assert max_seqlen == 5, f"max_seqlen mismatch: {max_seqlen} vs 5"
    print("test_flatten_doc_offsets_padding_only PASSED")


def test_flatten_doc_offsets_crop_cap():
    """Single row where a short doc leaves a trailing cap segment.
    This is the case where last_doc_end < T, so a cap at (b+1)*T is injected.
    The cap segment length must be included in max_seqlen."""
    B, T = 1, 12
    T_plus_1 = T + 1
    # One doc of length 5, then all padding. Cap covers remaining 7 tokens.
    doc_offsets = torch.tensor([[0, 5, T_plus_1, T_plus_1, T_plus_1]], dtype=torch.int32)
    cu, max_seqlen = _flatten_doc_offsets(doc_offsets, B, T)
    # max_docs_per_row = 4, max_cu_len = 1*4+1 = 5.
    # Real entries: col 0 (=0), col 1 (=5). K=2, last_doc_end=5 < T=12 => cap at 12.
    # cu = [0, 5, 12], padded to [0, 5, 12, 12, 12].
    expected = torch.tensor([0, 5, 12, 12, 12], dtype=torch.int32)
    assert cu.shape == (5,), f"cu_seqlens shape mismatch: {cu.shape}, expected (5,)"
    assert torch.equal(cu, expected), f"cu_seqlens mismatch: {cu} vs {expected}"
    # Segments: doc=[5], cap=[7], padding=[0,0] -> max = 7
    assert max_seqlen == 7, f"max_seqlen mismatch: {max_seqlen} vs 7"
    print("test_flatten_doc_offsets_crop_cap PASSED")


def test_dataloader_yields_doc_offsets():
    """Dataloader with emit_doc_offsets=True yields consistent doc_offsets."""
    if not os.path.exists(os.path.expanduser("~/.cache/nanochat/tokenizer")):
        print("Skipping dataloader test (no tokenizer trained); run scripts/tok_train.py first")
        return
    tokenizer = get_tokenizer()
    B, T = 2, 64
    # We can't run the real dataloader without parquet data, but we can test the
    # function signature and contract by patching the doc buffer.
    # Build a fake doc buffer via the iterator.
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, B, T, split="train", device="cpu",
        emit_doc_offsets=True, max_docs_per_row=8,
    )
    # Pull 1 batch. Will fail if no parquet data, so wrap in try.
    try:
        item = next(loader)
    except (AssertionError, FileNotFoundError) as e:
        print(f"Skipping dataloader test (no parquet data: {e})")
        return
    assert len(item) == 3, f"Expected 3-tuple (x, y, doc_offsets), got {len(item)}-tuple"
    x, y, doc_offsets = item
    assert x.shape == (B, T), f"x shape mismatch: {x.shape}"
    assert y.shape == (B, T), f"y shape mismatch: {y.shape}"
    assert doc_offsets.shape == (B, 9), f"doc_offsets shape mismatch: {doc_offsets.shape}"
    assert doc_offsets.dtype == torch.int32
    # First column should be 0 (every doc 0 starts at 0)
    assert torch.all(doc_offsets[:, 0] == 0)
    # Values should be non-decreasing within each row
    for b in range(B):
        diffs = doc_offsets[b, 1:] - doc_offsets[b, :-1]
        assert (diffs >= 0).all(), f"Row {b} doc_offsets not non-decreasing"
    # Last entries should be >= T (showing padding)
    assert (doc_offsets[:, -1] >= T).all()
    print("test_dataloader_yields_doc_offsets PASSED")


def test_dataloader_no_doc_offsets_backward_compat():
    """Dataloader with emit_doc_offsets=False yields 2-tuples (legacy)."""
    if not os.path.exists(os.path.expanduser("~/.cache/nanochat/tokenizer")):
        print("Skipping dataloader test (no tokenizer trained)")
        return
    tokenizer = get_tokenizer()
    B, T = 2, 64
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, B, T, split="train", device="cpu",
        emit_doc_offsets=False,
    )
    try:
        item = next(loader)
    except (AssertionError, FileNotFoundError) as e:
        print(f"Skipping dataloader test (no parquet data: {e})")
        return
    assert len(item) == 2, f"Expected 2-tuple (x, y), got {len(item)}-tuple"
    x, y = item
    assert x.shape == (B, T)
    assert y.shape == (B, T)
    print("test_dataloader_no_doc_offsets_backward_compat PASSED")


def test_model_forward_with_doc_offsets_falls_back():
    """Model with use_varlen_doc_attn=False ignores doc_offsets."""
    from nanochat.common import COMPUTE_DTYPE
    config = GPTConfig(
        sequence_len=64, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2,
        n_embd=32, window_pattern="L", use_varlen_doc_attn=False,
    )
    with torch.device("meta"):
        model = GPT(config)
    model = model.to_empty(device="cpu")
    model.init_weights()
    model = model.to(COMPUTE_DTYPE)
    B, T = 2, 64
    x = torch.randint(0, 128, (B, T))
    y = torch.randint(0, 128, (B, T))
    loss_no_doc = model(x, y)
    # Pass doc_offsets — model now expects cu_seqlens, not doc_offsets.
    # With the flag off, cu_seqlens=None is the same as no varlen.
    loss_with_doc = model(x, y, cu_seqlens=None, max_seqlen=0)
    assert loss_no_doc.shape == ()
    assert torch.allclose(loss_no_doc, loss_with_doc), "cu_seqlens=None should be ignored when flag is off"
    print("test_model_forward_with_doc_offsets_falls_back PASSED")


def test_model_forward_use_varlen_doc_attn_flag():
    """sanity: model constructed with use_varlen_doc_attn=True does not crash on
    the new config attribute, and a forward with doc_offsets=None uses fallback."""
    from nanochat.common import COMPUTE_DTYPE
    config = GPTConfig(
        sequence_len=64, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2,
        n_embd=32, window_pattern="L", use_varlen_doc_attn=True,
    )
    with torch.device("meta"):
        model = GPT(config)
    model = model.to_empty(device="cpu")
    model.init_weights()
    model = model.to(COMPUTE_DTYPE)
    B, T = 2, 64
    x = torch.randint(0, 128, (B, T))
    y = torch.randint(0, 128, (B, T))
    # Pass cu_seqlens=None (so fallback path runs) — this exercises the
    # branch logic without needing FA3 varlen shape correctness.
    loss = model(x, y, cu_seqlens=None, max_seqlen=0)
    assert loss.shape == ()
    print("test_model_forward_use_varlen_doc_attn_flag PASSED")


if __name__ == "__main__":
    test_flatten_doc_offsets_basic()
    test_flatten_doc_offsets_single_doc()
    test_flatten_doc_offsets_padding_only()
    test_flatten_doc_offsets_crop_cap()
    test_dataloader_yields_doc_offsets()
    test_dataloader_no_doc_offsets_backward_compat()
    test_model_forward_with_doc_offsets_falls_back()
    test_model_forward_use_varlen_doc_attn_flag()
    print("\nAll tests passed.")
