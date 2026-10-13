#!/usr/bin/env bash
# Train conditional diffusion model using the VerbalTS architecture
# (--model-type=verbalts) inside train_cond_v7.py, with Aligner V7 as the
# text encoder. Single GPU.
#
# This mirrors run_train_cond_v6_verbalts_arch.sh but swaps the text encoder /
# metric embedder path to aligner-v7 so we can compare v6/v7 under the same
# VerbalTS generator architecture.
set -euo pipefail

NODE_NAME="${SLURMD_NODENAME:-$(hostname -s)}"
echo "Running on node: $NODE_NAME"

# Resolve the project root from this script's location.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------- dataset ----------------
DATASET_BACKEND="tsfragment"
SOURCE="${SOURCE:-ETTm1}"
LENGTH="${LENGTH:-24}"
MIN_LENGTH=64
MAX_ROWS=none
CACHE_DIR="./hf_cache"
NORMALIZE="global"

# ---------------- text encoder ----------------
TEXT_ENCODER="aligner-v7"
ALIGNER_CKPT="${ALIGNER_CKPT:-./feature_alignment/results_aligner_v7/${SOURCE}_L${LENGTH}/aligner_v7_best.pt}"
LLM_ID="google/embeddinggemma-300m"
LLM_DTYPE="bfloat16"
HF_TOKEN_ENV="HF_TOKEN"
TEXT_CACHE_BATCH_SIZE=256
TEXT_COND_NORM="layernorm"
TEXT_ADAPTER_HIDDEN_DIM=512
TEXT_ADAPTER_DROPOUT=0.0
COND_DROP_PROB=0.0

# ---------------- denoiser (VerbalTS architecture) ----------------
MODEL_TYPE="verbalts"
IN_CHANNELS=1
VERBALTS_CHANNELS=64
VERBALTS_MULTIPATCH_NUM=3
VERBALTS_BASE_PATCH=2
VERBALTS_L_PATCH_LEN=3
VERBALTS_LAYERS=3
VERBALTS_NHEADS=8
VERBALTS_DIFF_EMBED_DIM=128
VERBALTS_VAR_EMB=16
VERBALTS_TIME_EMB=112
VERBALTS_ATTENTION_MASK_TYPE="parallel"
VERBALTS_CONDITION_TYPE="adaLN"
VERBALTS_NUM_STEPS=50
VERBALTS_BETA_START=1e-4
VERBALTS_BETA_END=0.5
VERBALTS_SCHEDULE="quad"
VERBALTS_TEXT_PROJECTOR_STAGES=3
VERBALTS_TEXTEMB_HIDDEN_DIM=512
VERBALTS_TEXT_EMB_DIM=64

# ---------------- diffusion ----------------
NOISE_SCALE=1.0
EMA_DECAY1=0.999
EMA_DECAY2=0.9999

# ---------------- training ----------------
BATCH_SIZE="${BATCH_SIZE:-512}"
EPOCHS=1000
LR="${LR:-1e-3}"
WEIGHT_DECAY=0.0
GRAD_CLIP=1.0
NUM_WORKERS="${NUM_WORKERS:-16}"
SEED="${SEED:-0}"
DEVICE="cuda"

# ---------------- validation ----------------
VAL_RATIO=0.1
VAL_SPLIT_SEED=0
VAL_SEED=12345
EVAL_EVERY_STEPS=0

# ---------------- generation eval ----------------
ENABLE_GEN_EVAL=1
GEN_EVAL_EVERY_EPOCHS=100
GEN_EVAL_MAX_BATCHES=0
METRICS_ALIGNER_CKPT="${METRICS_ALIGNER_CKPT:-$ALIGNER_CKPT}"
METRICS_K=5
METRICS_MDD_BINS=32
METRICS_MAX_SAMPLES=0
METRICS_SEED="${METRICS_SEED:-123}"

# ---------------- I/O ----------------
LOG_EVERY=10
SAVE_DIR="./t2ts_jit/results_jit_cond_v7_verbalts_arch_${SOURCE}_L${LENGTH}_seed${SEED}"
SAVE="${SAVE_DIR}/denoiser_ckpt.pt"
SAVE_BEST="${SAVE_DIR}/denoiser_best.pt"
SAVE_EVERY_EPOCHS=100

# ---------------- wandb ----------------
WANDB_PROJECT="text-ts-formal-experiment"
WANDB_ENTITY=""
WANDB_RUN_NAME="aligner_v7_verbalts_arch_${SOURCE}_L${LENGTH}_seed${SEED}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_TAGS=(cond v7 verbalts_arch aligner_v7 "$SOURCE" "L${LENGTH}" "seed${SEED}")
WANDB_NOTES=""

# ---------------- Hugging Face cache ----------------
export HF_HOME="${HF_HOME:-$PWD/.hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$CACHE_DIR"
echo "HF_HOME=$HF_HOME"
echo "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"

export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$PWD/.wandb_data}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PWD/.wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$PWD/.wandb_config}"
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
echo "WANDB_DIR=$WANDB_DIR"
echo "WANDB_DATA_DIR=$WANDB_DATA_DIR"
echo "WANDB_CACHE_DIR=$WANDB_CACHE_DIR"
echo "DATASET=tsfragment:${SOURCE}:${LENGTH}"
echo "SEED=$SEED"
echo "ALIGNER_CKPT=$ALIGNER_CKPT"
echo "METRICS_ALIGNER_CKPT=$METRICS_ALIGNER_CKPT"

EXTRA=()
[[ -n "$WANDB_ENTITY"   ]] && EXTRA+=(--wandb-entity "$WANDB_ENTITY")
[[ -n "$WANDB_RUN_NAME" ]] && EXTRA+=(--wandb-run-name "$WANDB_RUN_NAME")
[[ -n "$WANDB_NOTES"    ]] && EXTRA+=(--wandb-notes "$WANDB_NOTES")
[[ -n "$VAL_SPLIT_SEED" ]] && EXTRA+=(--val-split-seed "$VAL_SPLIT_SEED")
[[ "$ENABLE_GEN_EVAL" -eq 1 ]] && EXTRA+=(--enable-gen-eval)
# ---- eval-only hooks (re-score saved checkpoints with a swapped metric aligner) ----
[[ -n "${METRICS_ALIGNER_KIND:-}" ]] && EXTRA+=(--metrics-aligner-kind "$METRICS_ALIGNER_KIND")
[[ -n "${EVAL_ONLY_CKPT_DIR:-}"   ]] && EXTRA+=(--eval-only-ckpt-dir "$EVAL_ONLY_CKPT_DIR")
[[ -n "${EVAL_ONLY_OUT_CSV:-}"    ]] && EXTRA+=(--eval-only-out-csv "$EVAL_ONLY_OUT_CSV")

if [[ "$LLM_ID" == "google/embeddinggemma-300m" ]]; then
  TOKEN_VALUE="${HF_TOKEN:-}"
  if [[ -z "$TOKEN_VALUE" ]]; then
    echo "ERROR: $LLM_ID is gated. Set $HF_TOKEN_ENV (default: HF_TOKEN) before running."
    echo "Example:"
    echo "  export HF_TOKEN=hf_xxx"
    exit 1
  fi
fi

python t2ts_jit/train_cond_v7.py \
  --dataset-backend "$DATASET_BACKEND" \
  --source "$SOURCE" \
  --length "$LENGTH" \
  --min-length "$MIN_LENGTH" \
  --max-rows "$MAX_ROWS" \
  --cache-dir "$CACHE_DIR" \
  --normalize "$NORMALIZE" \
  --text-encoder "$TEXT_ENCODER" \
  --aligner-ckpt "$ALIGNER_CKPT" \
  --llm-id "$LLM_ID" \
  --llm-dtype "$LLM_DTYPE" \
  --hf-token-env "$HF_TOKEN_ENV" \
  --text-cache-batch-size "$TEXT_CACHE_BATCH_SIZE" \
  --text-cond-norm "$TEXT_COND_NORM" \
  --text-adapter-hidden-dim "$TEXT_ADAPTER_HIDDEN_DIM" \
  --text-adapter-dropout "$TEXT_ADAPTER_DROPOUT" \
  --cond-drop-prob "$COND_DROP_PROB" \
  --model-type "$MODEL_TYPE" \
  --in-channels "$IN_CHANNELS" \
  --verbalts-channels "$VERBALTS_CHANNELS" \
  --verbalts-multipatch-num "$VERBALTS_MULTIPATCH_NUM" \
  --verbalts-base-patch "$VERBALTS_BASE_PATCH" \
  --verbalts-L-patch-len "$VERBALTS_L_PATCH_LEN" \
  --verbalts-layers "$VERBALTS_LAYERS" \
  --verbalts-nheads "$VERBALTS_NHEADS" \
  --verbalts-diff-embed-dim "$VERBALTS_DIFF_EMBED_DIM" \
  --verbalts-var-emb "$VERBALTS_VAR_EMB" \
  --verbalts-time-emb "$VERBALTS_TIME_EMB" \
  --verbalts-attention-mask-type "$VERBALTS_ATTENTION_MASK_TYPE" \
  --verbalts-condition-type "$VERBALTS_CONDITION_TYPE" \
  --verbalts-num-steps "$VERBALTS_NUM_STEPS" \
  --verbalts-beta-start "$VERBALTS_BETA_START" \
  --verbalts-beta-end "$VERBALTS_BETA_END" \
  --verbalts-schedule "$VERBALTS_SCHEDULE" \
  --verbalts-text-projector-stages "$VERBALTS_TEXT_PROJECTOR_STAGES" \
  --verbalts-textemb-hidden-dim "$VERBALTS_TEXTEMB_HIDDEN_DIM" \
  --verbalts-text-emb-dim "$VERBALTS_TEXT_EMB_DIM" \
  --noise-scale "$NOISE_SCALE" \
  --ema-decay1 "$EMA_DECAY1" \
  --ema-decay2 "$EMA_DECAY2" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --grad-clip "$GRAD_CLIP" \
  --num-workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --val-ratio "$VAL_RATIO" \
  --val-seed "$VAL_SEED" \
  --eval-every-steps "$EVAL_EVERY_STEPS" \
  --gen-eval-every-epochs "$GEN_EVAL_EVERY_EPOCHS" \
  --gen-eval-max-batches "$GEN_EVAL_MAX_BATCHES" \
  --metrics-aligner-ckpt "$METRICS_ALIGNER_CKPT" \
  --metrics-k "$METRICS_K" \
  --metrics-mdd-bins "$METRICS_MDD_BINS" \
  --metrics-max-samples "$METRICS_MAX_SAMPLES" \
  --metrics-seed "$METRICS_SEED" \
  --log-every "$LOG_EVERY" \
  --save "$SAVE" \
  --save-best "$SAVE_BEST" \
  --save-every-epochs "$SAVE_EVERY_EPOCHS" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-mode "$WANDB_MODE" \
  --wandb-tags "${WANDB_TAGS[@]}" \
  "${EXTRA[@]}"
