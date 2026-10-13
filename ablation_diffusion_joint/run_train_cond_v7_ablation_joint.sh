#!/usr/bin/env bash
# Stage-2 ablation: single-stage JOINT training. Stage 1 and Stage 2 collapse
# into ONE run: the aligner (contrastive loss, same TextToTimeSeriesAlignerV7
# architecture as Stage 1) is trained TOGETHER with a SINGLE generator built at
# the SAME size/architecture as the real Stage-2 generator, in one optimizer /
# backward pass -- no frozen aligner, no separate preceding stage, no smaller
# stand-in network.
#
# Everything else (generator architecture, diffusion hyperparameters, training
# schedule, train/val split, gen-eval protocol) matches the other stage-2
# ablations byte-for-byte, so FID/CTTP/JFTSD stay directly comparable to the
# main pipeline and ablation_diffusion_noalign/.
#
# NOTE on GPU memory: this ablation holds Gemma-300M+LoRA, Chronos-2+LoRA, the
# generator, AND a second frozen reference aligner (for FID scoring) resident
# on one GPU simultaneously -- meaningfully heavier than either single-stage
# script it's compared against. BATCH_SIZE defaults to 128 (not the main
# pipeline's 512); raise it only after confirming it fits your GPU.
set -euo pipefail
NODE_NAME="${SLURMD_NODENAME:-$(hostname -s)}"
echo "Running on node: $NODE_NAME"

# Resolve to the project root
# Resolve the project root from this script's location.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------- dataset ----------------
SOURCE="${SOURCE:-ETTh1s}"
LENGTH="${LENGTH:-48}"
MIN_LENGTH=64
MAX_ROWS=none
CACHE_DIR="./hf_cache"

# ---------------- aligner (encoder E, jointly trained) ----------------
LLM_ID="google/embeddinggemma-300m"
CHRONOS_MODEL_ID="amazon/chronos-2"
PROJ_DIM=512
LLM_DTYPE="bfloat16"
HF_TOKEN_ENV="HF_TOKEN"
INIT_TEMPERATURE=0.07
MIN_TEMPERATURE=0.03
TUNE_SIDE="both"
TEXT_LORA_R=8
TEXT_LORA_ALPHA=16
TEXT_LORA_DROPOUT=0.05
TS_LORA_R=8
TS_LORA_ALPHA=16
TS_LORA_DROPOUT=0.05
ALIGNER_LR="${ALIGNER_LR:-1e-4}"

# ---------------- text conditioning (generator side) ----------------------
TEXT_COND_NORM="layernorm"
TEXT_ADAPTER_HIDDEN_DIM=512
TEXT_ADAPTER_DROPOUT=0.0
COND_DROP_PROB=0.0

# ---------------- generator (Stage-2-sized, jit2d) -------------------------
IMG_SIZE="${IMG_SIZE:-8}"
DELAY="${DELAY:-6}"
PATCH_SIZE=2
IN_CHANNELS=1
HIDDEN_CHANNELS=128
DEPTH=4
NUM_HEADS=4
ATTN_DROPOUT=0.1
PROJ_DROPOUT=0.1
BOTTLENECK_DIM=128

# ---------------- diffusion (flow-matching) --------------------------------
P_MEAN="${P_MEAN:-0.5}"
P_STD="${P_STD:-1.2}"
T_EPS=1e-5
NOISE_SCALE=1.0
EMA_DECAY1=0.9999
EMA_DECAY2=0.999
SAMPLING_METHOD="heun"
NUM_SAMPLING_STEPS=50
INTERVAL_MIN=0.0
INTERVAL_MAX=1.0

# Metric embedder is FIXED to the main "both" aligner for comparability with
# the main pipeline and every other ablation.
METRICS_ALIGNER_ROOT="${METRICS_ALIGNER_ROOT:-./feature_alignment/results_aligner_v7}"
METRICS_ALIGNER_CKPT="${METRICS_ALIGNER_CKPT:-${METRICS_ALIGNER_ROOT}/${SOURCE}_L${LENGTH}/aligner_v7_best.pt}"

# ---------------- training ----------------
BATCH_SIZE="${BATCH_SIZE:-128}"   # lower than the main pipeline's 512 -- see memory note above
EPOCHS=2000
LR="${LR:-1e-4}"
WEIGHT_DECAY=0.0
GRAD_CLIP=1.0
NUM_WORKERS="${NUM_WORKERS:-16}"
SEED="${SEED:-0}"
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
METRICS_K=5
METRICS_MDD_BINS=32
METRICS_MAX_SAMPLES=0
METRICS_SEED="${METRICS_SEED:-123}"

# ---------------- I/O ----------------
LOG_EVERY=10
RUN_TAG="ps${PATCH_SIZE}_pm${P_MEAN}_ps${P_STD}_img${IMG_SIZE}_d${DELAY}_bs${BATCH_SIZE}_lr${LR}_joint_seed${SEED}"
SAVE_DIR="./ablation_diffusion_joint/results_cond_v7_joint_${SOURCE}_L${LENGTH}_${RUN_TAG}"
SAVE="${SAVE_DIR}/denoiser_ckpt.pt"
SAVE_BEST="${SAVE_DIR}/denoiser_best.pt"
SAVE_EVERY_EPOCHS=100

# ---------------- wandb ----------------
WANDB_PROJECT="${WANDB_PROJECT:-text-ts-ablation-joint}"
WANDB_ENTITY=""
WANDB_RUN_NAME="${WANDB_RUN_NAME:-cond_joint_${SOURCE}_L${LENGTH}_${RUN_TAG}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_TAGS=(cond v7 joint "$SOURCE" "L${LENGTH}" "seed${SEED}")
WANDB_NOTES=""

# ---------------- Hugging Face cache ----------------
export HF_HOME="${HF_HOME:-$PWD/.hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$CACHE_DIR"
echo "HF_HOME=$HF_HOME"

# Avoid /home quota issues for wandb runtime/artifacts cache.
export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$PWD/.wandb_data}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PWD/.wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$PWD/.wandb_config}"
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

echo "DATASET=tsfragment:${SOURCE}:${LENGTH}"
echo "METRICS_ALIGNER_CKPT(fixed reference)=$METRICS_ALIGNER_CKPT"
echo "IMG_SIZE=$IMG_SIZE DELAY=$DELAY P_MEAN=$P_MEAN P_STD=$P_STD BATCH_SIZE=$BATCH_SIZE"
echo "SEED=$SEED  VAL_SPLIT_SEED=$VAL_SPLIT_SEED (split fixed across seeds)"
echo "SAVE_DIR=$SAVE_DIR"

if [[ ! -f "$METRICS_ALIGNER_CKPT" ]]; then
  echo "ERROR: reference metric aligner not found: $METRICS_ALIGNER_CKPT"
  echo "       This should be the main 'both' aligner used by your formal experiment."
  exit 1
fi

EXTRA=()
[[ -n "$WANDB_ENTITY"   ]] && EXTRA+=(--wandb-entity "$WANDB_ENTITY")
[[ -n "$WANDB_RUN_NAME" ]] && EXTRA+=(--wandb-run-name "$WANDB_RUN_NAME")
[[ -n "$WANDB_NOTES"    ]] && EXTRA+=(--wandb-notes "$WANDB_NOTES")
[[ -n "$VAL_SPLIT_SEED" ]] && EXTRA+=(--val-split-seed "$VAL_SPLIT_SEED")
[[ "$ENABLE_GEN_EVAL" -eq 1 ]] && EXTRA+=(--enable-gen-eval)
[[ -n "${METRICS_ALIGNER_KIND:-}" ]] && EXTRA+=(--metrics-aligner-kind "$METRICS_ALIGNER_KIND")
[[ -n "${EVAL_ONLY_CKPT_DIR:-}"   ]] && EXTRA+=(--eval-only-ckpt-dir "$EVAL_ONLY_CKPT_DIR")
[[ -n "${EVAL_ONLY_OUT_CSV:-}"    ]] && EXTRA+=(--eval-only-out-csv "$EVAL_ONLY_OUT_CSV")

# Gated model guard
if [[ "$LLM_ID" == "google/embeddinggemma-300m" ]]; then
  TOKEN_VALUE="${HF_TOKEN:-}"
  if [[ -z "$TOKEN_VALUE" ]]; then
    echo "ERROR: $LLM_ID is gated. Set $HF_TOKEN_ENV (default: HF_TOKEN) before running."
    exit 1
  fi
fi

python ablation_diffusion_joint/train_cond_v7_ablation_joint.py \
  --source "$SOURCE" \
  --length "$LENGTH" \
  --min-length "$MIN_LENGTH" \
  --max-rows "$MAX_ROWS" \
  --cache-dir "$CACHE_DIR" \
  --llm-id "$LLM_ID" \
  --chronos-model-id "$CHRONOS_MODEL_ID" \
  --proj-dim "$PROJ_DIM" \
  --llm-dtype "$LLM_DTYPE" \
  --hf-token-env "$HF_TOKEN_ENV" \
  --init-temperature "$INIT_TEMPERATURE" \
  --min-temperature "$MIN_TEMPERATURE" \
  --tune-side "$TUNE_SIDE" \
  --text-lora-r "$TEXT_LORA_R" \
  --text-lora-alpha "$TEXT_LORA_ALPHA" \
  --text-lora-dropout "$TEXT_LORA_DROPOUT" \
  --ts-lora-r "$TS_LORA_R" \
  --ts-lora-alpha "$TS_LORA_ALPHA" \
  --ts-lora-dropout "$TS_LORA_DROPOUT" \
  --aligner-lr "$ALIGNER_LR" \
  --text-cond-norm "$TEXT_COND_NORM" \
  --text-adapter-hidden-dim "$TEXT_ADAPTER_HIDDEN_DIM" \
  --text-adapter-dropout "$TEXT_ADAPTER_DROPOUT" \
  --cond-drop-prob "$COND_DROP_PROB" \
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
