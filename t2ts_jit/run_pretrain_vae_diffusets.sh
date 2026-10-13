#!/usr/bin/env bash
# Stage-1 of the DiffuSETS baseline: pretrain the SD-style conv VAE (mix 24/48/96)
# for one source. Series are global min-max normalized to [0,1] per length; the
# per-length stats are stored in the checkpoint and reused by the diffusion stage.
set -euo pipefail
NODE_NAME="${SLURMD_NODENAME:-$(hostname -s)}"
echo "Running on node: $NODE_NAME"

# Resolve to the project root (parent of this script's t2ts_jit/ dir).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------- config ----------------
SOURCE="${SOURCE:-ETTm1}"
LENGTH="${LENGTH:-24}"          # per-length VAE (ConTSG trains one model per length, no mix)
CACHE_DIR="./hf_cache"
NORMALIZE="${NORMALIZE:-global}"
MAX_ROWS_PER_LENGTH="${MAX_ROWS_PER_LENGTH:-}"   # empty = full
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-1000}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
# Constant KL weight, no annealing — keep the baseline VAE simple.
KLD_WEIGHT="${KLD_WEIGHT:-0.01}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
# Split params MUST match the diffusion stage (run_train_cond_v7_diffusets.sh) so
# the VAE holds out exactly the diffusion val rows: val_ratio=0.1, val_split_seed=0.
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_SPLIT_SEED="${VAL_SPLIT_SEED:-0}"

# DiffuSETS VAE geometry. downsample-factor=2 -> latent length L/2 (default the
# baseline is tuned for); 4/8 are also valid but shorten the latent.
DOWNSAMPLE_FACTOR="${DOWNSAMPLE_FACTOR:-2}"
LATENT_CHANNELS="${LATENT_CHANNELS:-4}"
BASE_CHANNELS="${BASE_CHANNELS:-128}"

SAVE_DIR="${SAVE_DIR:-./t2ts_jit/vae_ckpts_diffusets}"
SAVE_NAME="${SAVE_NAME:-vae_${SOURCE}_L${LENGTH}_ds${DOWNSAMPLE_FACTOR}_ep${EPOCHS}}"

WANDB_PROJECT="${WANDB_PROJECT:-diffusets-baseline-vae}"
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

echo "SOURCE=$SOURCE LENGTH=$LENGTH EPOCHS=$EPOCHS DOWNSAMPLE_FACTOR=$DOWNSAMPLE_FACTOR SAVE_DIR=$SAVE_DIR/$SAVE_NAME"

EXTRA=()
[[ -n "$MAX_ROWS_PER_LENGTH" ]] && EXTRA+=(--max-rows-per-length "$MAX_ROWS_PER_LENGTH")

python t2ts_jit/pretrain_vae_diffusets.py \
  --source "$SOURCE" \
  --cache-dir "$CACHE_DIR" \
  --normalize "$NORMALIZE" \
  --no-mix-train \
  --single-length "$LENGTH" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --grad-clip "$GRAD_CLIP" \
  --kld-weight "$KLD_WEIGHT" \
  --seed "$SEED" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --val-ratio "$VAL_RATIO" \
  --val-split-seed "$VAL_SPLIT_SEED" \
  --downsample-factor "$DOWNSAMPLE_FACTOR" \
  --latent-channels "$LATENT_CHANNELS" \
  --base-channels "$BASE_CHANNELS" \
  --save-dir "$SAVE_DIR" \
  --save-name "$SAVE_NAME" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-mode "$WANDB_MODE" \
  --wandb-run-name "$WANDB_RUN_NAME" \
  --wandb-tags diffusets vae "$SOURCE" \
  "${EXTRA[@]}"
