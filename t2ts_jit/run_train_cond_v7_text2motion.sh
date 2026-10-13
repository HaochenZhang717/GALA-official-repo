#!/usr/bin/env bash
# Stage-2 of the Text2Motion baseline: train the text-conditioned autoregressive
# CVAE generator on top of the frozen per-(source,length) movement AE, with
# v7-aligned ConTSG generation eval. Single GPU.
set -euo pipefail
NODE_NAME="${SLURMD_NODENAME:-$(hostname -s)}"
echo "Running on node: $NODE_NAME"

# Resolve to the project root (parent of this script's t2ts_jit/ dir).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------- dataset ----------------
SOURCE="${SOURCE:-ETTm1}"
LENGTH="${LENGTH:-24}"
MAX_ROWS="${MAX_ROWS:-}"        # empty = full (matches v7)
CACHE_DIR="./hf_cache"
NORMALIZE="global"

# ---------------- pretrained movement AE (per source+length) ----------------
AE_EPOCHS="${AE_EPOCHS:-270}"   # matches Decomp baseline max_epoch (AE ckpt path)
AE_ROOT="${AE_ROOT:-./t2ts_jit/ae_ckpts_text2motion}"
AE_RUN_NAME="${AE_RUN_NAME:-ae_${SOURCE}_L${LENGTH}_ep${AE_EPOCHS}}"
PRETRAINED_AE_PATH="${PRETRAINED_AE_PATH:-${AE_ROOT}/${AE_RUN_NAME}/best_model.pth}"

# ---------------- metrics aligner (v7) ----------------
METRICS_ALIGNER_ROOT="${METRICS_ALIGNER_ROOT:-./feature_alignment/results_aligner_v7}"
METRICS_ALIGNER_CKPT="${METRICS_ALIGNER_CKPT:-${METRICS_ALIGNER_ROOT}/${SOURCE}_L${LENGTH}/aligner_v7_best.pt}"
LLM_ID="google/embeddinggemma-300m"
HF_TOKEN_ENV="HF_TOKEN"

# ---------------- CVAE generator / losses ----------------
LAMBDA_REC_TS="${LAMBDA_REC_TS:-1.0}"
LAMBDA_REC_MOV="${LAMBDA_REC_MOV:-1.0}"
LAMBDA_KLD="${LAMBDA_KLD:-0.005}"
TEACHER_FORCING_RATIO="${TEACHER_FORCING_RATIO:-0.4}"
EMA_DECAY1=0.9999
EMA_DECAY2=0.999

# ---------------- training ----------------
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-2000}"
LR="${LR:-2e-4}"
WEIGHT_DECAY=0.0
# Text2Motion Comp baseline (CompTrainerV6) clips grads to 0.5; no weight decay.
GRAD_CLIP=0.5
NUM_WORKERS=8
SEED=0
DEVICE="${DEVICE:-cuda}"

# ---------------- validation / gen-eval ----------------
VAL_RATIO=0.1
VAL_SPLIT_SEED=0
VAL_SEED=12345
EVAL_EVERY_STEPS=0
ENABLE_GEN_EVAL="${ENABLE_GEN_EVAL:-1}"
GEN_EVAL_EVERY_EPOCHS="${GEN_EVAL_EVERY_EPOCHS:-100}"
GEN_EVAL_MAX_BATCHES=0
METRICS_K=5
METRICS_MDD_BINS=32
METRICS_MAX_SAMPLES=0     # <=0 = full val set
METRICS_SEED=123
ALIGNER_MIN_LENGTH=64

# ---------------- I/O ----------------
LOG_EVERY=10
RUN_TAG="${RUN_TAG:-kld${LAMBDA_KLD}_tf${TEACHER_FORCING_RATIO}_bs${BATCH_SIZE}_lr${LR}}"
SAVE_DIR="${SAVE_DIR:-./t2ts_jit/results_text2motion_baseline_${SOURCE}_L${LENGTH}_${RUN_TAG}}"
SAVE="${SAVE_DIR}/gen_ckpt.pt"
SAVE_BEST="${SAVE_DIR}/gen_best.pt"
SAVE_EVERY_EPOCHS=100

# ---------------- wandb ----------------
WANDB_PROJECT="${WANDB_PROJECT:-text-ts-formal-experiment}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-text2motion_baseline_${SOURCE}_L${LENGTH}_${RUN_TAG}}"
WANDB_MODE="${WANDB_MODE:-online}"

# ---------------- HF / wandb caches ----------------
export HF_HOME="${HF_HOME:-$PWD/.hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$CACHE_DIR"
export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PWD/.wandb_cache}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

echo "DATASET=tsfragment:${SOURCE}:${LENGTH}"
echo "PRETRAINED_AE_PATH=$PRETRAINED_AE_PATH"
echo "METRICS_ALIGNER_CKPT=$METRICS_ALIGNER_CKPT"

if [[ ! -f "$PRETRAINED_AE_PATH" ]]; then
  echo "ERROR: movement AE checkpoint not found: $PRETRAINED_AE_PATH"
  echo "Run t2ts_jit/run_pretrain_ae_text2motion.sh (SOURCE=$SOURCE LENGTH=$LENGTH) first."
  exit 1
fi

# embeddinggemma (used only by the metric aligner) is gated.
if [[ "$LLM_ID" == "google/embeddinggemma-300m" ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: $LLM_ID is gated. Set $HF_TOKEN_ENV before running."
    exit 1
  fi
fi

EXTRA=()
[[ -n "$MAX_ROWS" ]] && EXTRA+=(--max-rows "$MAX_ROWS")
[[ "$ENABLE_GEN_EVAL" -eq 1 ]] && EXTRA+=(--enable-gen-eval)

python t2ts_jit/train_cond_v7_text2motion.py \
  --source "$SOURCE" \
  --length "$LENGTH" \
  --cache-dir "$CACHE_DIR" \
  --normalize "$NORMALIZE" \
  --pretrained-ae-path "$PRETRAINED_AE_PATH" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --grad-clip "$GRAD_CLIP" \
  --num-workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --val-ratio "$VAL_RATIO" \
  --val-split-seed "$VAL_SPLIT_SEED" \
  --val-seed "$VAL_SEED" \
  --eval-every-steps "$EVAL_EVERY_STEPS" \
  --ema-decay1 "$EMA_DECAY1" \
  --ema-decay2 "$EMA_DECAY2" \
  --lambda-rec-ts "$LAMBDA_REC_TS" \
  --lambda-rec-mov "$LAMBDA_REC_MOV" \
  --lambda-kld "$LAMBDA_KLD" \
  --teacher-forcing-ratio "$TEACHER_FORCING_RATIO" \
  --gen-eval-every-epochs "$GEN_EVAL_EVERY_EPOCHS" \
  --gen-eval-max-batches "$GEN_EVAL_MAX_BATCHES" \
  --metrics-aligner-ckpt "$METRICS_ALIGNER_CKPT" \
  --metrics-k "$METRICS_K" \
  --metrics-mdd-bins "$METRICS_MDD_BINS" \
  --metrics-max-samples "$METRICS_MAX_SAMPLES" \
  --metrics-seed "$METRICS_SEED" \
  --aligner-min-length "$ALIGNER_MIN_LENGTH" \
  --hf-token-env "$HF_TOKEN_ENV" \
  --log-every "$LOG_EVERY" \
  --save "$SAVE" \
  --save-best "$SAVE_BEST" \
  --save-every-epochs "$SAVE_EVERY_EPOCHS" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-run-name "$WANDB_RUN_NAME" \
  --wandb-mode "$WANDB_MODE" \
  --wandb-tags text2motion cond baseline "$SOURCE" "L${LENGTH}" \
  "${EXTRA[@]}"
