#!/usr/bin/env bash
# Stage-1 of the T2S baseline: pretrain the LA-VAE (mix 24/48/96) for one source.
# Series are global min-max normalized to [0,1] per length; the per-length stats
# are stored in the checkpoint and reused by the diffusion stage.
set -euo pipefail
NODE_NAME="${SLURMD_NODENAME:-$(hostname -s)}"
echo "Running on node: $NODE_NAME"

# Resolve to the project root (parent of this script's t2ts_jit/ dir).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------- config ----------------
SOURCE="${SOURCE:-ETTm1}"
CACHE_DIR="./hf_cache"
NORMALIZE="${NORMALIZE:-global}"
MAX_ROWS_PER_LENGTH="${MAX_ROWS_PER_LENGTH:-}"   # empty = full
BATCH_SIZE="${BATCH_SIZE:-512}"
EPOCHS="${EPOCHS:-2000}"
LR="${LR:-1e-3}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
# Split params MUST match the diffusion stage (run_train_cond_v7_t2s.sh) so the VAE
# holds out exactly the diffusion val rows: val_ratio=0.1, val_split_seed=0.
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_SPLIT_SEED="${VAL_SPLIT_SEED:-0}"

EMBEDDING_DIM=64   # yields the (B,64,30) latent the T2S DiT requires
SAVE_DIR="${SAVE_DIR:-./t2ts_jit/vae_ckpts_t2s}"
SAVE_NAME="${SAVE_NAME:-vae_${SOURCE}_mix_ep${EPOCHS}}"

WANDB_PROJECT="${WANDB_PROJECT:-t2s-baseline-vae}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${SAVE_NAME}}"

# ---------------- HF / wandb caches (avoid /home quota) ----------------
export HF_HOME="${HF_HOME:-$PWD/.hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$CACHE_DIR"
export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PWD/.wandb_cache}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

echo "SOURCE=$SOURCE EPOCHS=$EPOCHS SAVE_DIR=$SAVE_DIR/$SAVE_NAME"

EXTRA=()
[[ -n "$MAX_ROWS_PER_LENGTH" ]] && EXTRA+=(--max-rows-per-length "$MAX_ROWS_PER_LENGTH")

python t2ts_jit/pretrain_vae_t2s.py \
  --source "$SOURCE" \
  --cache-dir "$CACHE_DIR" \
  --normalize "$NORMALIZE" \
  --mix-train \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --seed "$SEED" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --val-ratio "$VAL_RATIO" \
  --val-split-seed "$VAL_SPLIT_SEED" \
  --embedding-dim "$EMBEDDING_DIM" \
  --save-dir "$SAVE_DIR" \
  --save-name "$SAVE_NAME" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-mode "$WANDB_MODE" \
  --wandb-run-name "$WANDB_RUN_NAME" \
  --wandb-tags t2s vae "$SOURCE" \
  "${EXTRA[@]}"
