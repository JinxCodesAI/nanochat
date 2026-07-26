#!/bin/bash
# Real d10 training on Strix Halo iGPU (~5-6 hours).
# Run from project root:  bash runs/rund10_strix.sh
#
# Note: NO `set -e`. A failure in any single step (e.g. one benchmark in
# CORE eval tripping an assertion, BPB OOM, sample crash) must NOT prevent
# the rest of the run from completing. Each Python entry point has its own
# internal try/except so per-benchmark failures are logged but don't
# bubble up; this shell wrapper does the same at the script level.
cd "$(dirname "$0")/.."

# ---- Hardware-tuned environment ----
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export TORCHDYNAMO_DISABLE=1
export NANOCHAT_DTYPE=bfloat16
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p "$NANOCHAT_BASE_DIR"
source .venv/bin/activate

WANDB_RUN="${WANDB_RUN:-d20_strix_halo}"

# ---- 1) Download pretraining data (40 shards ~ 10 GB compressed) ----
python -m nanochat.dataset -n 40

# ---- 2) Train the BPE tokenizer (~30-60 s) ----
python -m scripts.tok_train --max-chars=2000000000
python -m scripts.tok_eval

# ---- 3) Pretrain d10 ----
# At B=8, T=2048 the model uses ~14 GB VRAM (measured). Raise --device-batch-size
# if you want faster steps; 16 will land you ~28 GB.
#
# Optional: Markov-style forgivable-mistake loss discount.
#   --markov-discount=1.0   (default — feature disabled, full loss everywhere)
#   --markov-discount=0.3   (scale loss down to 30% when target is in source's
#                            bigram top-100. Requires ~/.cache/nanochat/next_token_table.pt
#                            — build with: python -m scripts.build_next_token_table --num-shards 40)
# Try: --markov-discount=0.3 for a strong regularizer, or 0.5 for a gentle one.
MARKOV_DISCOUNT="${MARKOV_DISCOUNT:-1.0}"   # set in env to override (e.g. MARKOV_DISCOUNT=0.3)
# Track per-step exit codes so a failing step doesn't poison the rest of
# the script — we just log and keep going.
overall_status=0
run_step() {
    local label="$1"; shift
    echo
    echo "============================================================"
    echo "STEP: $label"
    echo "============================================================"
    if ! "$@"; then
        echo "[rund10_strix.sh] STEP FAILED: $label  (continuing with next step)"
        overall_status=1
    fi
}

run_step "base_train (d10_loop3_strix)" \
    python -m scripts.base_train \
        --depth=10 \
        --head-dim=128 \
        --window-pattern=L \
        --max-seq-len=2048 \
        --device-batch-size=16 \
        --total-batch-size=131072 \
        --eval-every=200 \
        --eval-tokens=1048576 \
        --core-metric-every=-1 \
        --sample-every=20 \
        --save-every=600 \
        --num-iterations=2000 \
        --no-varlen-doc-attn \
        --model-tag=d10_strix_2_varlen

# ---- 4) Quick base eval ----
run_step "base_eval" \
    python -m scripts.base_eval --model-tag=d10_strix_2_varlen --device-batch-size=4 --split-tokens=524288

# ---- 5) SFT (~30-60 min) ----
run_step "chat_sft" \
    python -m scripts.chat_sft \
        --device-batch-size=4 \
        --max-seq-len=2048 \
        --eval-every=200 \
        --eval-tokens=1048576 \
        --num-iterations=800 \
        --run="${WANDB_RUN}_sft_varlen"

if [ "$overall_status" -ne 0 ]; then
    echo
    echo "============================================================"
    echo "rund10_strix.sh: one or more steps reported a non-zero exit."
    echo "See the log above for which step failed and why."
    echo "============================================================"
    exit 0   # exit 0 so callers (and the runner) still see a "completed run"
fi

echo
echo "============================================================"
echo "TRAINING + SFT DONE. To chat with the model:"
echo "  source .venv/bin/activate"
echo "  python -m scripts.chat_cli -g d10_strix -p \"Why is the sky blue?\""
echo "============================================================"
