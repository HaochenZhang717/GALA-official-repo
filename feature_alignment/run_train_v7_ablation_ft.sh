#!/usr/bin/env bash
# Stage-1 fine-tuning-METHOD ablation launcher for TextToTimeSeriesAlignerV7AblationFT.
#
# Varies HOW both encoders are fine-tuned (tune_side is pinned to 'both'):
#   FT_METHOD=lora -> LoRA adapters on both encoders        (current main method)
#   FT_METHOD=head -> frozen backbones + appended Transformer head per side
#   FT_METHOD=full -> full fine-tuning of both backbones (+ gradient checkpointing)
#
# Crossed with the diffusion arm via DIFFUSION_WEIGHT (0.0 = no diffusion loss).
# Supported TSFragment sources: ETTm1, ETTh1s, electricity, traffic.
set -euo pipefail

NODE_NAME="${SLURMD_NODENAME:-$(hostname -s)}"
echo "Running on node: $NODE_NAME"

# Resolve the working directory from this script's location.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------- dataset ----------------
DATASET_BACKEND="${DATASET_BACKEND:-tsfragment}"  # tsfragment | verbalts
SOURCE="${SOURCE:-ETTm1}"
LENGTH="${LENGTH:-24}"
VERBALTS_NAME="${VERBALTS_NAME:-synthetic_u}"
VERBALTS_SPLIT="${VERBALTS_SPLIT:-train}"
VERBALTS_VAL_SPLIT="${VERBALTS_VAL_SPLIT:-valid}"
# Only used when DATASET_BACKEND=verbalts; TSFragment auto-downloads.
VERBALTS_ROOT="${VERBALTS_ROOT:-./data/VerbalTSDatasets}"
MIN_LENGTH="${MIN_LENGTH:-64}"
MAX_ROWS="${MAX_ROWS:-none}"
CACHE_DIR="${CACHE_DIR:-./hf_cache}"

# ---------------- Hugging Face cache ----------------
export HF_HOME="${HF_HOME:-$PWD/.hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$CACHE_DIR"
echo "HF_HOME=$HF_HOME"
echo "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"

# ---------------- Weights & Biases cache/data ----------------
export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$PWD/.wandb_data}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PWD/.wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$PWD/.wandb_config}"
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
echo "WANDB_DIR=$WANDB_DIR"
echo "WANDB_DATA_DIR=$WANDB_DATA_DIR"
echo "WANDB_CACHE_DIR=$WANDB_CACHE_DIR"
if [[ "$DATASET_BACKEND" == "tsfragment" ]]; then
  echo "DATASET=tsfragment:${SOURCE}:${LENGTH} MIN_LENGTH=$MIN_LENGTH"
else
  echo "DATASET=verbalts:${VERBALTS_NAME}:${VERBALTS_SPLIT} (val=${VERBALTS_VAL_SPLIT})"
  echo "VERBALTS_ROOT=$VERBALTS_ROOT"
fi

# ---------------- aligner model ----------------
LLM_ID="${LLM_ID:-google/embeddinggemma-300m}"
CHRONOS_MODEL_ID="${CHRONOS_MODEL_ID:-amazon/chronos-2}"
PROJ_DIM="${PROJ_DIM:-512}"
LLM_DTYPE="${LLM_DTYPE:-bfloat16}"
HF_TOKEN_ENV="${HF_TOKEN_ENV:-HF_TOKEN}"
INIT_TEMPERATURE="${INIT_TEMPERATURE:-0.07}"
MIN_TEMPERATURE="${MIN_TEMPERATURE:-0.07}"
# This ablation runs only under tune_side='both'.
TUNE_SIDE="${TUNE_SIDE:-both}"
TEXT_LORA_R="${TEXT_LORA_R:-64}"
TEXT_LORA_ALPHA="${TEXT_LORA_ALPHA:-16}"
TEXT_LORA_DROPOUT="${TEXT_LORA_DROPOUT:-0.05}"
TEXT_LORA_TARGET_MODULES=(${TEXT_LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj})
TS_LORA_R="${TS_LORA_R:-64}"
TS_LORA_ALPHA="${TS_LORA_ALPHA:-16}"
TS_LORA_DROPOUT="${TS_LORA_DROPOUT:-0.05}"
TS_LORA_TARGET_MODULES=(${TS_LORA_TARGET_MODULES:-q k v o})

# ---------------- fine-tuning method (ablation dimension) ----------------
FT_METHOD="${FT_METHOD:-lora}"          # lora | head | full
HEAD_DEPTH="${HEAD_DEPTH:-1}"
HEAD_NUM_HEADS="${HEAD_NUM_HEADS:-12}"
HEAD_FFN_DIM="${HEAD_FFN_DIM:-2560}"    # depth1 + ffn2560 ~= 12.6M (matches LoRA r=64)
HEAD_DROPOUT="${HEAD_DROPOUT:-0.0}"
# Gradient checkpointing: default ON for full FT, OFF otherwise (overridable).
if [[ -z "${GRAD_CHECKPOINTING:-}" ]]; then
  if [[ "$FT_METHOD" == "full" ]]; then GRAD_CHECKPOINTING=1; else GRAD_CHECKPOINTING=0; fi
fi

# ---------------- diffusion branch (lightweight) ----------------
DIFFUSION_WEIGHT="${DIFFUSION_WEIGHT:-1.0}"
DIFFUSION_LENGTH="${DIFFUSION_LENGTH:-64}"
IMG_SIZE="${IMG_SIZE:-10}"
PATCH_SIZE="${PATCH_SIZE:-2}"
IN_CHANNELS="${IN_CHANNELS:-1}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-64}"
DEPTH="${DEPTH:-2}"
NUM_HEADS="${NUM_HEADS:-4}"
ATTN_DROPOUT="${ATTN_DROPOUT:-0.0}"
PROJ_DROPOUT="${PROJ_DROPOUT:-0.0}"
BOTTLENECK_DIM="${BOTTLENECK_DIM:-64}"
P_MEAN="${P_MEAN:-0.5}"
P_STD="${P_STD:-1.2}"
T_EPS="${T_EPS:-1e-5}"
NOISE_SCALE="${NOISE_SCALE:-1.0}"
EMA_DECAY1="${EMA_DECAY1:-0.9999}"
EMA_DECAY2="${EMA_DECAY2:-0.999}"
SAMPLING_METHOD="${SAMPLING_METHOD:-heun}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-50}"
INTERVAL_MIN="${INTERVAL_MIN:-0.0}"
INTERVAL_MAX="${INTERVAL_MAX:-1.0}"

# ---------------- training ----------------
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-100}"
# Per-method default LR: full FT is more sensitive, so default lower.
if [[ -z "${LR:-}" ]]; then
  if [[ "$FT_METHOD" == "full" ]]; then LR=1e-5; else LR=2e-5; fi
fi
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"

# ---------------- DDP / multi-GPU ----------------
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

# ---------------- validation ----------------
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_SPLIT_SEED="${VAL_SPLIT_SEED:-0}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-200}"

# ---------------- I/O ----------------
LOG_EVERY="${LOG_EVERY:-10}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-ftmethod_ablation}"
RESULTS_ROOT="${RESULTS_ROOT:-./results_aligner_v7_${EXPERIMENT_NAME}}"

# ---------------- wandb ----------------
WANDB_PROJECT="text-ts-aligner-v7"
WANDB_ENTITY=""
WANDB_RUN_NAME=""
WANDB_MODE="${WANDB_MODE:-online}"      # online | offline | disabled
WANDB_TAGS=(v7 "$DATASET_BACKEND" ftmethod "$FT_METHOD")
WANDB_NOTES=""

EXTRA=()
[[ -n "$WANDB_ENTITY"   ]] && EXTRA+=(--wandb-entity   "$WANDB_ENTITY")
[[ -n "$WANDB_RUN_NAME" ]] && EXTRA+=(--wandb-run-name "$WANDB_RUN_NAME")
[[ -n "$WANDB_NOTES"    ]] && EXTRA+=(--wandb-notes    "$WANDB_NOTES")
[[ -n "$VAL_SPLIT_SEED" ]] && EXTRA+=(--val-split-seed "$VAL_SPLIT_SEED")
# store_true flag: only append when enabled.
[[ "$GRAD_CHECKPOINTING" == "1" ]] && EXTRA+=(--grad-checkpointing)

# Gated model guard
if [[ "$LLM_ID" == "google/embeddinggemma-300m" ]]; then
  TOKEN_VALUE="${HF_TOKEN:-}"
  if [[ -z "$TOKEN_VALUE" ]]; then
    echo "ERROR: $LLM_ID is gated. Set $HF_TOKEN_ENV (default: HF_TOKEN) before running."
    echo "Example:"
    echo "  export HF_TOKEN=hf_xxx"
    echo "  bash feature_alignment/run_train_v7_ablation_ft.sh"
    exit 1
  fi
fi

LAUNCHER=(torchrun --standalone --nproc_per_node="$NPROC_PER_NODE")
if [[ "$NPROC_PER_NODE" -eq 1 ]]; then
  LAUNCHER=(python)
fi
export PYTHONUNBUFFERED=1

echo "DEVICE=$DEVICE"
DEFAULT_RUN_TAG="${EXPERIMENT_NAME}_M${MIN_LENGTH}_dl${DIFFUSION_LENGTH}_dw${DIFFUSION_WEIGHT}_ft${FT_METHOD}_seed${SEED}"
RUN_TAG="${RUN_TAG:-$DEFAULT_RUN_TAG}"
if [[ "$DATASET_BACKEND" == "tsfragment" ]]; then
  RESULTS_DIR="${RESULTS_ROOT}/${SOURCE}_L${LENGTH}_${RUN_TAG}"
  RUN_NAME_SUFFIX="${SOURCE}_L${LENGTH}_${RUN_TAG}"
  WANDB_EXTRA_TAGS=("$SOURCE" "L${LENGTH}" "M${MIN_LENGTH}" "ft${FT_METHOD}" "dw${DIFFUSION_WEIGHT}" "seed${SEED}")
else
  RESULTS_DIR="${RESULTS_ROOT}/${VERBALTS_NAME}_${VERBALTS_SPLIT}_${RUN_TAG}"
  RUN_NAME_SUFFIX="${VERBALTS_NAME}_${VERBALTS_SPLIT}_${RUN_TAG}"
  WANDB_EXTRA_TAGS=("$VERBALTS_NAME" "$VERBALTS_SPLIT" "M${MIN_LENGTH}" "ft${FT_METHOD}" "dw${DIFFUSION_WEIGHT}" "seed${SEED}")
fi
mkdir -p "$RESULTS_DIR"
SAVE="${RESULTS_DIR}/aligner_v7_ckpt.pt"
SAVE_BEST="${RESULTS_DIR}/aligner_v7_best.pt"
echo "------------------------------------------------------------"
echo "Running dataset backend=$DATASET_BACKEND"
echo "MIN_LENGTH=$MIN_LENGTH"
echo "FT_METHOD=$FT_METHOD (head_depth=$HEAD_DEPTH head_num_heads=$HEAD_NUM_HEADS head_ffn_dim=$HEAD_FFN_DIM grad_ckpt=$GRAD_CHECKPOINTING)"
echo "DIFFUSION_LENGTH=$DIFFUSION_LENGTH"
echo "DIFFUSION_WEIGHT=$DIFFUSION_WEIGHT"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "TUNE_SIDE=$TUNE_SIDE  LR=$LR  SEED=$SEED"
echo "RUN_TAG=$RUN_TAG"
echo "Results directory: $RESULTS_DIR"
echo "Launcher: ${LAUNCHER[*]} train_v7_ablation_ft.py"

CUR_EXTRA=("${EXTRA[@]}")
if [[ -z "$WANDB_RUN_NAME" ]]; then
  CUR_EXTRA+=(--wandb-run-name "v7_${RUN_NAME_SUFFIX}")
fi

"${LAUNCHER[@]}" train_v7_ablation_ft.py \
  --dataset-backend "$DATASET_BACKEND" \
  --source "$SOURCE" \
  --length "$LENGTH" \
  --verbalts-name "$VERBALTS_NAME" \
  --verbalts-split "$VERBALTS_SPLIT" \
  --verbalts-val-split "$VERBALTS_VAL_SPLIT" \
  --verbalts-root "$VERBALTS_ROOT" \
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
  --ft-method "$FT_METHOD" \
  --head-depth "$HEAD_DEPTH" \
  --head-num-heads "$HEAD_NUM_HEADS" \
  --head-ffn-dim "$HEAD_FFN_DIM" \
  --head-dropout "$HEAD_DROPOUT" \
  --text-lora-r "$TEXT_LORA_R" \
  --text-lora-alpha "$TEXT_LORA_ALPHA" \
  --text-lora-dropout "$TEXT_LORA_DROPOUT" \
  --text-lora-target-modules "${TEXT_LORA_TARGET_MODULES[@]}" \
  --ts-lora-r "$TS_LORA_R" \
  --ts-lora-alpha "$TS_LORA_ALPHA" \
  --ts-lora-dropout "$TS_LORA_DROPOUT" \
  --ts-lora-target-modules "${TS_LORA_TARGET_MODULES[@]}" \
  --diffusion-weight "$DIFFUSION_WEIGHT" \
  --diffusion-length "$DIFFUSION_LENGTH" \
  --img-size "$IMG_SIZE" \
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
  --eval-every-steps "$EVAL_EVERY_STEPS" \
  --log-every "$LOG_EVERY" \
  --save "$SAVE" \
  --save-best "$SAVE_BEST" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-mode "$WANDB_MODE" \
  --wandb-tags "${WANDB_TAGS[@]}" "${WANDB_EXTRA_TAGS[@]}" \
  "${CUR_EXTRA[@]}"
