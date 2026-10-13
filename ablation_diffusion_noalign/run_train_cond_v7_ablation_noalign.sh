#!/usr/bin/env bash
# Stage-2 ablation: NO stage-1 feature alignment. Train the conditional diffusion
# model while the conditioning text encoder is trained JOINTLY (end-to-end) with
# the diffusion loss, then score generation metrics with a FIXED reference
# aligner (the main "both" aligner) so FID/adherence stay comparable to the main
# pipeline and the other stage-2 ablations.
#
# The conditioning text encoder is selected by TEXT_VARIANT:
#   frozen -> frozen embeddinggemma-300m backbone + a TRAINABLE projector
#             (the denoiser's TextConditioner adapter). Maps to
#             --text-encoder frozen-gemma; embeddings are cached, projector learns.
#   lora   -> LoRA-adapted embeddinggemma-300m (base frozen) + trainable projector,
#             trained on-the-fly with gradients. Maps to --text-encoder
#             tunable-gemma; train_cond_v7_ablation_noalign.py swaps the upstream
#             FULL-finetune TunableGemmaTextEncoder for the LoRA variant.
#
# NOTE: unlike the other stage-2 ablations there is NO --aligner-ckpt here -- the
# whole point is that no stage-1 aligner exists. Only the FIXED metric embedder
# (--metrics-aligner-ckpt, the main "both" aligner) is loaded, for FID scoring.
#
# Everything else (denoiser, diffusion, training schedule, split) is identical to
# the other stage-2 ablations. gen-eval metrics are teed to
# <save_dir>/test_metrics.txt by train_cond_v7_ablation_noalign.py.
set -euo pipefail
NODE_NAME="${SLURMD_NODENAME:-$(hostname -s)}"
echo "Running on node: $NODE_NAME"

# Resolve to the project root
# Resolve the project root from this script's location.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------- dataset ----------------
DATASET_BACKEND="${DATASET_BACKEND:-tsfragment}"
SOURCE="${SOURCE:-ETTm1}"
LENGTH="${LENGTH:-24}"
MIN_LENGTH=64
MAX_ROWS=none
CACHE_DIR="./hf_cache"
NORMALIZE="global"          # per-sample | global | none

# ---------------- text encoder (trained JOINTLY -- no stage-1 alignment) ------
# frozen -> frozen Gemma + trainable projector ; lora -> LoRA Gemma + projector
TEXT_VARIANT="${TEXT_VARIANT:-lora}"
LLM_ID="google/embeddinggemma-300m"
LLM_DTYPE="bfloat16"
HF_TOKEN_ENV="HF_TOKEN"
TEXT_CACHE_BATCH_SIZE=256
TEXT_COND_NORM="layernorm"   # layernorm | none
TEXT_ADAPTER_HIDDEN_DIM=512
TEXT_ADAPTER_DROPOUT=0.0
COND_DROP_PROB=0.0   # per-sample prob of replacing text emb with learned null at train time

# Projector output dim for the LoRA/tunable encoder (mirrors the main aligner's
# proj_dim so the conditioning dim matches the formal pipeline).
TUNABLE_PROJ_DIM="${TUNABLE_PROJ_DIM:-512}"
# Separate LR for the jointly-trained Gemma encoder (LoRA adapters + projector).
# Only used in the lora arm (the optimizer special-cases tunable-gemma).
TUNABLE_GEMMA_LR="${TUNABLE_GEMMA_LR:-1e-4}"

# LoRA hyper-params for the lora arm. Read by train_cond_v7_ablation_noalign.py
# from the environment (defaults match the stage-1 V7 aligner text-side LoRA).
export LORA_R="${LORA_R:-8}"
export LORA_ALPHA="${LORA_ALPHA:-16}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"
# Gradient checkpointing on the Gemma backbone (lora arm only). On by default to
# keep bs=512 within a single 94GB GPU; set LORA_GRAD_CKPT=0 to disable.
export LORA_GRAD_CKPT="${LORA_GRAD_CKPT:-1}"

case "$TEXT_VARIANT" in
  frozen) TEXT_ENCODER="frozen-gemma" ;;
  lora)   TEXT_ENCODER="tunable-gemma" ;;   # swapped to LoRA by the driver
  *) echo "ERROR: TEXT_VARIANT must be 'frozen' or 'lora', got '$TEXT_VARIANT'"; exit 1 ;;
esac

# Metric embedder is FIXED to the main "both" aligner for cross-variant
# comparability (results_aligner_v7/<SOURCE>_L<LENGTH>/aligner_v7_best.pt).
METRICS_ALIGNER_ROOT="${METRICS_ALIGNER_ROOT:-./feature_alignment/results_aligner_v7}"
METRICS_ALIGNER_CKPT="${METRICS_ALIGNER_CKPT:-${METRICS_ALIGNER_ROOT}/${SOURCE}_L${LENGTH}/aligner_v7_best.pt}"

# ---------------- denoiser ----------------
MODEL_TYPE="jit2d"
NUM_REGISTER_TOKENS=0
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
NUM_WORKERS="${NUM_WORKERS:-16}"
SEED="${SEED:-0}"   # training seed (model init / dropout / shuffle); vary for error bars
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
METRICS_MAX_SAMPLES=0          # <=0 means full val set
METRICS_SEED="${METRICS_SEED:-123}"

# ---------------- I/O ----------------
LOG_EVERY=10
RUN_TAG="ps${PATCH_SIZE}_pm${P_MEAN}_ps${P_STD}_img${IMG_SIZE}_d${DELAY}_bs${BATCH_SIZE}_lr${LR}_noalign_${TEXT_VARIANT}_seed${SEED}"
SAVE_DIR="./ablation_diffusion_noalign/results_cond_v7_noalign_${SOURCE}_L${LENGTH}_${RUN_TAG}"
SAVE="${SAVE_DIR}/denoiser_ckpt.pt"
SAVE_BEST="${SAVE_DIR}/denoiser_best.pt"
SAVE_EVERY_EPOCHS=100

# ---------------- wandb ----------------
WANDB_PROJECT="${WANDB_PROJECT:-text-ts-ablation-noalign}"
WANDB_ENTITY=""
WANDB_RUN_NAME="${WANDB_RUN_NAME:-cond_noalign_${TEXT_VARIANT}_${SOURCE}_L${LENGTH}_${RUN_TAG}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_TAGS=(cond v7 noalign "var${TEXT_VARIANT}" "$SOURCE" "L${LENGTH}" "seed${SEED}")
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
echo "TEXT_VARIANT=$TEXT_VARIANT -> TEXT_ENCODER=$TEXT_ENCODER (trained jointly, NO stage-1 alignment)"
if [[ "$TEXT_VARIANT" == "lora" ]]; then
  echo "LoRA: r=$LORA_R alpha=$LORA_ALPHA dropout=$LORA_DROPOUT targets=$LORA_TARGET_MODULES proj_dim=$TUNABLE_PROJ_DIM gemma_lr=$TUNABLE_GEMMA_LR"
fi
echo "METRICS_ALIGNER_CKPT(fixed reference)=$METRICS_ALIGNER_CKPT"
echo "IMG_SIZE=$IMG_SIZE DELAY=$DELAY P_MEAN=$P_MEAN P_STD=$P_STD"
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
# Separate LR for the jointly-trained Gemma encoder (only used by the lora arm).
[[ "$TEXT_VARIANT" == "lora" ]] && EXTRA+=(--tunable-gemma-lr "$TUNABLE_GEMMA_LR")

# Gated model guard
if [[ "$LLM_ID" == "google/embeddinggemma-300m" ]]; then
  TOKEN_VALUE="${HF_TOKEN:-}"
  if [[ -z "$TOKEN_VALUE" ]]; then
    echo "ERROR: $LLM_ID is gated. Set $HF_TOKEN_ENV (default: HF_TOKEN) before running."
    exit 1
  fi
fi

python ablation_diffusion_noalign/train_cond_v7_ablation_noalign.py \
  --dataset-backend "$DATASET_BACKEND" \
  --source "$SOURCE" \
  --length "$LENGTH" \
  --min-length "$MIN_LENGTH" \
  --max-rows "$MAX_ROWS" \
  --cache-dir "$CACHE_DIR" \
  --normalize "$NORMALIZE" \
  --text-encoder "$TEXT_ENCODER" \
  --tunable-proj-dim "$TUNABLE_PROJ_DIM" \
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
