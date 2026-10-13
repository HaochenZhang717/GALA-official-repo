#!/usr/bin/env bash
# Stage-2 of the DiffuSETS baseline: train the latent DDPM U-Net, conditioned on
# the dataset's native OpenAI TextEmbedding, with v7-aligned ConTSG generation eval.
# Single GPU. Supported sources: ETTm1, ETTh1s, electricity, traffic.
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

# ---------------- pretrained DiffuSETS VAE ----------------
VAE_EPOCHS="${VAE_EPOCHS:-1000}"
DOWNSAMPLE_FACTOR="${DOWNSAMPLE_FACTOR:-2}"
VAE_ROOT="${VAE_ROOT:-./t2ts_jit/vae_ckpts_diffusets}"
VAE_RUN_NAME="${VAE_RUN_NAME:-vae_${SOURCE}_L${LENGTH}_ds${DOWNSAMPLE_FACTOR}_ep${VAE_EPOCHS}}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-${VAE_ROOT}/${VAE_RUN_NAME}/best_model.pth}"

# ---------------- metrics aligner (v7) ----------------
METRICS_ALIGNER_ROOT="${METRICS_ALIGNER_ROOT:-./feature_alignment/results_aligner_v7}"
METRICS_ALIGNER_CKPT="${METRICS_ALIGNER_CKPT:-${METRICS_ALIGNER_ROOT}/${SOURCE}_L${LENGTH}/aligner_v7_best.pt}"
LLM_ID="google/embeddinggemma-300m"
HF_TOKEN_ENV="HF_TOKEN"

# ---------------- diffusion / sampling ----------------
DIFFUSION_STEPS="${DIFFUSION_STEPS:-50}"   # DDPM steps (full reverse loop); ConTSG default
NOISE_SCHEDULE="${NOISE_SCHEDULE:-quad}"
BETA_START="${BETA_START:-0.0001}"
BETA_END="${BETA_END:-0.5}"
UNET_NUM_LEVELS="${UNET_NUM_LEVELS:-3}"
UNET_KERNEL_SIZE="${UNET_KERNEL_SIZE:-3}"
UNET_HEADS="${UNET_HEADS:-8}"
SAMPLER="${SAMPLER:-ddim}"
CFG_SCALE="${CFG_SCALE:-1.0}"               # 1.0 = no guidance (faithful DiffuSETS)
COND_DROP_PROB="${COND_DROP_PROB:-0.0}"     # raise (e.g. 0.1) together with CFG_SCALE>1
EMA_DECAY1=0.9999
EMA_DECAY2=0.999

# ---------------- training ----------------
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-500}"
LR="${LR:-1e-3}"
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0
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
RUN_TAG="${RUN_TAG:-ds${DOWNSAMPLE_FACTOR}_T${DIFFUSION_STEPS}_${SAMPLER}_cfg${CFG_SCALE}_bs${BATCH_SIZE}_lr${LR}}"
SAVE_DIR="${SAVE_DIR:-./t2ts_jit/results_diffusets_baseline_${SOURCE}_L${LENGTH}_${RUN_TAG}}"
SAVE="${SAVE_DIR}/denoiser_ckpt.pt"
SAVE_BEST="${SAVE_DIR}/denoiser_best.pt"
SAVE_EVERY_EPOCHS=100

# ---------------- wandb ----------------
WANDB_PROJECT="${WANDB_PROJECT:-text-ts-formal-experiment}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-diffusets_baseline_${SOURCE}_L${LENGTH}_${RUN_TAG}}"
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
echo "PRETRAINED_MODEL_PATH=$PRETRAINED_MODEL_PATH"
echo "METRICS_ALIGNER_CKPT=$METRICS_ALIGNER_CKPT"
echo "DIFFUSION_STEPS=$DIFFUSION_STEPS SAMPLER=$SAMPLER CFG_SCALE=$CFG_SCALE"

if [[ ! -f "$PRETRAINED_MODEL_PATH" ]]; then
  echo "ERROR: VAE checkpoint not found: $PRETRAINED_MODEL_PATH"
  echo "Run t2ts_jit/run_pretrain_vae_diffusets.sh (SOURCE=$SOURCE) first."
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

python t2ts_jit/train_cond_v7_diffusets.py \
  --source "$SOURCE" \
  --length "$LENGTH" \
  --cache-dir "$CACHE_DIR" \
  --normalize "$NORMALIZE" \
  --pretrained-model-path "$PRETRAINED_MODEL_PATH" \
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
  --diffusion-steps "$DIFFUSION_STEPS" \
  --noise-schedule "$NOISE_SCHEDULE" \
  --beta-start "$BETA_START" \
  --beta-end "$BETA_END" \
  --unet-num-levels "$UNET_NUM_LEVELS" \
  --unet-kernel-size "$UNET_KERNEL_SIZE" \
  --unet-heads "$UNET_HEADS" \
  --sampler "$SAMPLER" \
  --cfg-scale "$CFG_SCALE" \
  --cond-drop-prob "$COND_DROP_PROB" \
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
  --wandb-tags diffusets cond baseline "$SOURCE" "L${LENGTH}" \
  "${EXTRA[@]}"
