#!/usr/bin/env bash
# Train the independent evaluation aligner (eval_aligner.py) on one
# (source, length) split of TSFragment-600K.
#
# This scorer exists to break the evaluation circularity: its text encoder is a
# BERT-family model (no Gemma lineage, no Qwen lineage) and its time-series
# encoder is a CNN trained from scratch (no Chronos-2 lineage). VAL_RATIO and
# VAL_SPLIT_SEED MUST match the Stage-2 diffusion runs, otherwise the scorer
# would be trained on the very windows generation is evaluated on.
set -euo pipefail

NODE_NAME="${SLURMD_NODENAME:-$(hostname -s)}"
echo "Running on node: $NODE_NAME"

# Resolve the working directory from this script's location.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------- dataset ----------------
SOURCE="${SOURCE:-ETTh1s}"
LENGTH="${LENGTH:-48}"
MIN_LENGTH="${MIN_LENGTH:-64}"
MAX_ROWS="${MAX_ROWS:-none}"
CACHE_DIR="${CACHE_DIR:-./hf_cache}"

# ---------------- caches (avoid quota-limited $HOME) ----------------
export HF_HOME="${HF_HOME:-$PWD/.hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$PWD/.wandb_data}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PWD/.wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$PWD/.wandb_config}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$CACHE_DIR" \
         "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

# ---------------- text encoder ----------------
# Must not be Gemma-family (the method's encoder) nor Qwen-family (the LLM that
# wrote the captions). bge-large-en-v1.5: BertModel, 24 layers, 335M params.
TEXT_MODEL_ID="${TEXT_MODEL_ID:-BAAI/bge-large-en-v1.5}"
# bf16 for the frozen base: ~2.6x faster than fp32 on A6000 (measured), and the
# pooled output is cast to fp32 before the projector, so logits and the InfoNCE
# loss still run in fp32. Matches what aligner_v7 does on its text side.
TEXT_DTYPE="${TEXT_DTYPE:-bfloat16}"
TEXT_POOLING="${TEXT_POOLING:-cls}"       # cls for BGE, mean for e5 / sentence-transformers
TEXT_MAX_LENGTH="${TEXT_MAX_LENGTH:-512}"
TEXT_LORA_R="${TEXT_LORA_R:-16}"
TEXT_LORA_ALPHA="${TEXT_LORA_ALPHA:-32}"
TEXT_LORA_DROPOUT="${TEXT_LORA_DROPOUT:-0.05}"
TEXT_LORA_TARGET_MODULES=(${TEXT_LORA_TARGET_MODULES:-query key value})

# ---------------- time-series encoder (from scratch) ----------------
CNN_HIDDEN_CHANNELS="${CNN_HIDDEN_CHANNELS:-256}"
CNN_KERNEL_SIZE="${CNN_KERNEL_SIZE:-3}"
CNN_DILATIONS=(${CNN_DILATIONS:-1 2 4 8})
CNN_DROPOUT="${CNN_DROPOUT:-0.1}"

# ---------------- shared space ----------------
# Keep proj_dim well below the number of gen-eval samples: FID estimates a
# proj_dim covariance and JFTSD a 2*proj_dim one.
PROJ_DIM="${PROJ_DIM:-128}"
INIT_TEMPERATURE="${INIT_TEMPERATURE:-0.07}"
MIN_TEMPERATURE="${MIN_TEMPERATURE:-0.03}"
HF_TOKEN_ENV="${HF_TOKEN_ENV:-HF_TOKEN}"

# ---------------- training ----------------
BATCH_SIZE="${BATCH_SIZE:-128}"           # per rank
EPOCHS="${EPOCHS:-30}"
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

# ---------------- validation (MUST match the diffusion runs) ----------------
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_SPLIT_SEED="${VAL_SPLIT_SEED:-0}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-1}"
SELECT_BY="${SELECT_BY:-loss}"            # loss | r@1  (both held-out, generator-free)

LOG_EVERY="${LOG_EVERY:-20}"
WANDB_PROJECT="${WANDB_PROJECT:-text-ts-eval-aligner}"
WANDB_MODE="${WANDB_MODE:-online}"

RESULTS_DIR="./results_eval_aligner/${SOURCE}_L${LENGTH}"
mkdir -p "$RESULTS_DIR"
SAVE="${RESULTS_DIR}/eval_aligner_ckpt.pt"
SAVE_BEST="${RESULTS_DIR}/eval_aligner_best.pt"

LAUNCHER=(torchrun --standalone --nproc_per_node="$NPROC_PER_NODE")
if [[ "$NPROC_PER_NODE" -eq 1 ]]; then
  LAUNCHER=(python)
fi
export PYTHONUNBUFFERED=1

echo "------------------------------------------------------------"
echo "DATASET=${SOURCE}/L${LENGTH}  TEXT=${TEXT_MODEL_ID}  PROJ_DIM=${PROJ_DIM}"
echo "SPLIT: val_ratio=${VAL_RATIO} val_split_seed=${VAL_SPLIT_SEED} (must match Stage-2)"
echo "Results directory: $RESULTS_DIR"

"${LAUNCHER[@]}" train_eval_aligner.py \
  --source "$SOURCE" \
  --length "$LENGTH" \
  --min-length "$MIN_LENGTH" \
  --max-rows "$MAX_ROWS" \
  --cache-dir "$CACHE_DIR" \
  --text-model-id "$TEXT_MODEL_ID" \
  --text-dtype "$TEXT_DTYPE" \
  --text-pooling "$TEXT_POOLING" \
  --text-max-length "$TEXT_MAX_LENGTH" \
  --text-lora-r "$TEXT_LORA_R" \
  --text-lora-alpha "$TEXT_LORA_ALPHA" \
  --text-lora-dropout "$TEXT_LORA_DROPOUT" \
  --text-lora-target-modules "${TEXT_LORA_TARGET_MODULES[@]}" \
  --cnn-hidden-channels "$CNN_HIDDEN_CHANNELS" \
  --cnn-kernel-size "$CNN_KERNEL_SIZE" \
  --cnn-dilations "${CNN_DILATIONS[@]}" \
  --cnn-dropout "$CNN_DROPOUT" \
  --proj-dim "$PROJ_DIM" \
  --init-temperature "$INIT_TEMPERATURE" \
  --min-temperature "$MIN_TEMPERATURE" \
  --hf-token-env "$HF_TOKEN_ENV" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --warmup-steps "$WARMUP_STEPS" \
  --grad-clip "$GRAD_CLIP" \
  --num-workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --val-ratio "$VAL_RATIO" \
  --val-split-seed "$VAL_SPLIT_SEED" \
  --eval-every-epochs "$EVAL_EVERY_EPOCHS" \
  --select-by "$SELECT_BY" \
  --log-every "$LOG_EVERY" \
  --save "$SAVE" \
  --save-best "$SAVE_BEST" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-mode "$WANDB_MODE" \
  --wandb-tags eval-aligner "$SOURCE" "L${LENGTH}" \
  --wandb-run-name "eval_aligner_${SOURCE}_L${LENGTH}"
