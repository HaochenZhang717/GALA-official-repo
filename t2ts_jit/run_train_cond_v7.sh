#!/usr/bin/env bash
# Train conditional diffusion model with Aligner V7 text encoder (single GPU).
# Supported TSFragment sources for this launcher: ETTm1, ETTh1s, electricity, traffic.
set -euo pipefail
NODE_NAME="${SLURMD_NODENAME:-$(hostname -s)}"
echo "Running on node: $NODE_NAME"

# Resolve to the project root (parent of this script's t2ts_jit/ dir).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------- dataset ----------------
DATASET_BACKEND="${DATASET_BACKEND:-tsfragment}"
SOURCE="${SOURCE:-ETTm1}"
LENGTH="${LENGTH:-24}"
MIN_LENGTH=64
MAX_ROWS=none
CACHE_DIR="./hf_cache"
NORMALIZE="global"          # per-sample | global | none

# ---------------- text encoder ----------------
TEXT_ENCODER="aligner-v7"
ALIGNER_DIFFUSION_LENGTH="${ALIGNER_DIFFUSION_LENGTH:-64}"
ALIGNER_MIN_LENGTH="${ALIGNER_MIN_LENGTH:-$MIN_LENGTH}"
ALIGNER_DW0_EXPERIMENT_NAME="${ALIGNER_DW0_EXPERIMENT_NAME:-diffusion_weight0_ablation}"
ALIGNER_DW0_ROOT="${ALIGNER_DW0_ROOT:-./feature_alignment/results_aligner_v7_diffusion_weight0_ablation}"
ALIGNER_DW0_RUN_TAG="${ALIGNER_DW0_RUN_TAG:-${ALIGNER_DW0_EXPERIMENT_NAME}_M${ALIGNER_MIN_LENGTH}_dl${ALIGNER_DIFFUSION_LENGTH}_dw0.0}"
METRICS_ALIGNER_ROOT="${METRICS_ALIGNER_ROOT:-./feature_alignment_for_metric/results_aligner_v7}"
ALIGNER_CKPT="${ALIGNER_CKPT:-${METRICS_ALIGNER_ROOT}/${SOURCE}_L${LENGTH}/aligner_v7_best.pt}"
LLM_ID="google/embeddinggemma-300m"
LLM_DTYPE="bfloat16"
HF_TOKEN_ENV="HF_TOKEN"
TEXT_CACHE_BATCH_SIZE=256
TEXT_COND_NORM="layernorm"   # layernorm | none
TEXT_ADAPTER_HIDDEN_DIM=512
TEXT_ADAPTER_DROPOUT=0.0
COND_DROP_PROB=0.0   # per-sample prob of replacing text emb with learned null at train time

# ---------------- denoiser ----------------
MODEL_TYPE="jit2d"
NUM_REGISTER_TOKENS=0
# Default jit2d geometry is length-aware; still overridable via env.
# PATCH_SIZE=2; defaults above satisfy the coverage constraint for native TSFragment lengths.
IMG_SIZE="${IMG_SIZE:-6}"
DELAY="${DELAY:-4}"   # column stride; equals IMG_SIZE -> non-overlapping columns
PATCH_SIZE=2
IN_CHANNELS=1
HIDDEN_CHANNELS=128
DEPTH=4
NUM_HEADS=4
ATTN_DROPOUT=0.1
PROJ_DROPOUT=0.1
BOTTLENECK_DIM=128

# ---------------- diffusion ----------------
P_MEAN="${P_MEAN:-0.0}"
P_STD="${P_STD:-1.0}"
T_EPS=1e-5
NOISE_SCALE=1.0
EMA_DECAY1=0.9999
EMA_DECAY2=0.999
SAMPLING_METHOD="heun"
NUM_SAMPLING_STEPS=50
INTERVAL_MIN=0.0
INTERVAL_MAX=1.0

# ---------------- training ----------------
BATCH_SIZE="${BATCH_SIZE:-512}"
EPOCHS=2000
LR="${LR:-1e-4}"
WEIGHT_DECAY=0.0
GRAD_CLIP=1.0
NUM_WORKERS=16
SEED=0
DEVICE="${DEVICE:-cuda}"

# ---------------- validation ----------------
VAL_RATIO=0.1
VAL_SPLIT_SEED=0
VAL_SEED=12345
EVAL_EVERY_STEPS=0

# generation metrics every N epochs (set <=0 to disable)
ENABLE_GEN_EVAL=1
GEN_EVAL_EVERY_EPOCHS=100
GEN_EVAL_MAX_BATCHES=0
METRICS_ALIGNER_CKPT="${METRICS_ALIGNER_CKPT:-${METRICS_ALIGNER_ROOT}/${SOURCE}_L${LENGTH}/aligner_v7_best.pt}"
METRICS_K=5
METRICS_MDD_BINS=32
METRICS_MAX_SAMPLES=0          # <=0 means full val set
METRICS_SEED=123

# ---------------- I/O ----------------
LOG_EVERY=10
RUN_TAG="ps${PATCH_SIZE}_pm${P_MEAN}_ps${P_STD}_img${IMG_SIZE}_d${DELAY}_bs${BATCH_SIZE}_lr${LR}_v7enc"
SAVE_DIR="./t2ts_jit/results_jit_cond_v7_${SOURCE}_L${LENGTH}_${RUN_TAG}"
SAVE="${SAVE_DIR}/denoiser_ckpt.pt"
SAVE_BEST="${SAVE_DIR}/denoiser_best.pt"
SAVE_EVERY_EPOCHS=100

# ---------------- wandb ----------------
WANDB_PROJECT="text-ts-formal-experiment"
WANDB_ENTITY=""
WANDB_RUN_NAME="${WANDB_RUN_NAME:-aligner_v7_${SOURCE}_L${LENGTH}_${RUN_TAG}}"
WANDB_MODE="online"          # online | offline | disabled
WANDB_TAGS=(cond v7 "$SOURCE" "L${LENGTH}")
WANDB_NOTES=""

# ---------------- Hugging Face cache ----------------
export HF_HOME="${HF_HOME:-$PWD/.hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$CACHE_DIR"
echo "HF_HOME=$HF_HOME"
echo "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"

# Avoid /home quota issues for wandb runtime/artifacts cache.
export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$PWD/.wandb_data}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PWD/.wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$PWD/.wandb_config}"
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
echo "WANDB_DIR=$WANDB_DIR"
echo "WANDB_DATA_DIR=$WANDB_DATA_DIR"
echo "WANDB_CACHE_DIR=$WANDB_CACHE_DIR"
echo "DATASET=tsfragment:${SOURCE}:${LENGTH}"
echo "ALIGNER_CKPT=$ALIGNER_CKPT"
echo "METRICS_ALIGNER_CKPT=$METRICS_ALIGNER_CKPT"
echo "IMG_SIZE=$IMG_SIZE DELAY=$DELAY"

EXTRA=()
[[ -n "$WANDB_ENTITY"   ]] && EXTRA+=(--wandb-entity "$WANDB_ENTITY")
[[ -n "$WANDB_RUN_NAME" ]] && EXTRA+=(--wandb-run-name "$WANDB_RUN_NAME")
[[ -n "$WANDB_NOTES"    ]] && EXTRA+=(--wandb-notes "$WANDB_NOTES")
[[ -n "$VAL_SPLIT_SEED" ]] && EXTRA+=(--val-split-seed "$VAL_SPLIT_SEED")
[[ "$ENABLE_GEN_EVAL" -eq 1 ]] && EXTRA+=(--enable-gen-eval)

# Gated model guard
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
  --img-size "$IMG_SIZE" \
  --delay "$DELAY" \
  --patch-size "$PATCH_SIZE" \
  --in-channels "$IN_CHANNELS" \
  --hidden-channels "$HIDDEN_CHANNELS" \
  --depth "$DEPTH" \
  --num-heads "$NUM_HEADS" \
  --attn-dropout "$ATTN_DROPOUT" \
  --proj-dropout "$PROJ_DROPOUT" \
  --bottleneck-dim "$BOTTLENECK_DIM" \
  --num-register-tokens "$NUM_REGISTER_TOKENS" \
  --p-mean "$P_MEAN" \
  --p-std "$P_STD" \
  --t-eps "$T_EPS" \
  --noise-scale "$NOISE_SCALE" \
  --ema-decay1 "$EMA_DECAY1" \
  --ema-decay2 "$EMA_DECAY2" \
  --sampling-method "$SAMPLING_METHOD" \
  --num-sampling-steps "$NUM_SAMPLING_STEPS" \
  --interval-min "$INTERVAL_MIN" \
  --interval-max "$INTERVAL_MAX" \
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
