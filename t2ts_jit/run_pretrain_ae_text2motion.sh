#!/usr/bin/env bash
# Stage-1 of the Text2Motion baseline: pretrain the movement conv autoencoder for
# ONE (source, length). Series are global min-max normalized to [0,1]; the stats
# are stored in the checkpoint and reused by the conditional stage so the frozen
# encoder always sees in-distribution inputs.
set -euo pipefail
NODE_NAME="${SLURMD_NODENAME:-$(hostname -s)}"
echo "Running on node: $NODE_NAME"

# Resolve to the project root (parent of this script's t2ts_jit/ dir).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------- config ----------------
SOURCE="${SOURCE:-ETTm1}"
LENGTH="${LENGTH:-24}"
CACHE_DIR="./hf_cache"
NORMALIZE="${NORMALIZE:-global}"
MAX_ROWS="${MAX_ROWS:-}"   # empty = full (matches the conditional stage)
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-270}"
LR="${LR:-1e-4}"
# Match the Text2Motion Decomp baseline optimizer: plain Adam (lr=1e-4) with no
# weight decay and no gradient clipping (DecompTrainerV3 uses optim.Adam and
# leaves clip_norm commented out). batch_size=128, max_epoch=270.
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
GRAD_CLIP="${GRAD_CLIP:-0.0}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
# Split params MUST match the conditional stage (run_train_cond_v7_text2motion.sh)
# so the AE holds out exactly the Stage-2 val rows: val_ratio=0.1, val_split_seed=0.
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_SPLIT_SEED="${VAL_SPLIT_SEED:-0}"

DIM_MOVEMENT_LATENT="${DIM_MOVEMENT_LATENT:-512}"
# Latent regularizers (Text2Motion DecompTrainerV3 defaults: 0.001 each).
LAMBDA_SPARSITY="${LAMBDA_SPARSITY:-0.001}"
LAMBDA_SMOOTH="${LAMBDA_SMOOTH:-0.001}"
SAVE_DIR="${SAVE_DIR:-./t2ts_jit/ae_ckpts_text2motion}"
SAVE_NAME="${SAVE_NAME:-ae_${SOURCE}_L${LENGTH}_ep${EPOCHS}}"

WANDB_PROJECT="${WANDB_PROJECT:-text2motion-baseline-ae}"
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

echo "SOURCE=$SOURCE LENGTH=$LENGTH EPOCHS=$EPOCHS SAVE_DIR=$SAVE_DIR/$SAVE_NAME"

EXTRA=()
[[ -n "$MAX_ROWS" ]] && EXTRA+=(--max-rows "$MAX_ROWS")

python t2ts_jit/pretrain_ae_text2motion.py \
  --source "$SOURCE" \
  --length "$LENGTH" \
  --cache-dir "$CACHE_DIR" \
  --normalize "$NORMALIZE" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --grad-clip "$GRAD_CLIP" \
  --seed "$SEED" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --val-ratio "$VAL_RATIO" \
  --val-split-seed "$VAL_SPLIT_SEED" \
  --dim-movement-latent "$DIM_MOVEMENT_LATENT" \
  --lambda-sparsity "$LAMBDA_SPARSITY" \
  --lambda-smooth "$LAMBDA_SMOOTH" \
  --save-dir "$SAVE_DIR" \
  --save-name "$SAVE_NAME" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-mode "$WANDB_MODE" \
  --wandb-run-name "$WANDB_RUN_NAME" \
  --wandb-tags text2motion ae "$SOURCE" "L${LENGTH}" \
  "${EXTRA[@]}"
