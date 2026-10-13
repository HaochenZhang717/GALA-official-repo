#!/usr/bin/env bash
# Train the M3-JEPA baseline on one (source, length) split of TSFragment-600K
# (or one VerbalTS split). Mirrors run_train_v7.sh but uses the MMoE/JEPA model
# (jepa_align) and has no diffusion branch.
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

# ---------------- Weights & Biases cache/data ----------------
export WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$PWD/.wandb_data}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$PWD/.wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$PWD/.wandb_config}"
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

if [[ "$DATASET_BACKEND" == "tsfragment" ]]; then
  echo "DATASET=tsfragment:${SOURCE}:${LENGTH} MIN_LENGTH=$MIN_LENGTH"
else
  echo "DATASET=verbalts:${VERBALTS_NAME}:${VERBALTS_SPLIT} (val=${VERBALTS_VAL_SPLIT})"
  echo "VERBALTS_ROOT=$VERBALTS_ROOT"
fi

# ---------------- encoders ----------------
LLM_ID="${LLM_ID:-google/embeddinggemma-300m}"
CHRONOS_MODEL_ID="${CHRONOS_MODEL_ID:-amazon/chronos-2}"
LLM_DTYPE="${LLM_DTYPE:-bfloat16}"       # float32 | float16 | bfloat16
HF_TOKEN_ENV="${HF_TOKEN_ENV:-HF_TOKEN}"
# which encoder side to LoRA-adapt: both | text | ts
TUNE_SIDE="${TUNE_SIDE:-both}"
TEXT_LORA_R="${TEXT_LORA_R:-64}"
TEXT_LORA_ALPHA="${TEXT_LORA_ALPHA:-16}"
TEXT_LORA_DROPOUT="${TEXT_LORA_DROPOUT:-0.05}"
TEXT_LORA_TARGET_MODULES=(${TEXT_LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj})
TS_LORA_R="${TS_LORA_R:-64}"
TS_LORA_ALPHA="${TS_LORA_ALPHA:-16}"
TS_LORA_DROPOUT="${TS_LORA_DROPOUT:-0.05}"
TS_LORA_TARGET_MODULES=(${TS_LORA_TARGET_MODULES:-q k v o})

# ---------------- M3-JEPA predictor ----------------
NUM_EXPERTS="${NUM_EXPERTS:-12}"
K_EXPERT="${K_EXPERT:-4}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"
DROPOUT_RATE="${DROPOUT_RATE:-0.1}"
TEMPERATURE="${TEMPERATURE:-0.03}"
L2_WEIGHT="${L2_WEIGHT:-1.0}"
CL_WEIGHT="${CL_WEIGHT:-1.0}"
TASK_WEIGHT_TEXT2TS="${TASK_WEIGHT_TEXT2TS:-0.5}"
TASK_WEIGHT_TS2TEXT="${TASK_WEIGHT_TS2TEXT:-0.5}"
TASK_SCHEDULE="${TASK_SCHEDULE:-alternate}"   # alternate | joint

# ---------------- training ----------------
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-2e-5}"
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

# ---------------- I/O ----------------
LOG_EVERY="${LOG_EVERY:-10}"

# ---------------- wandb ----------------
WANDB_PROJECT="${WANDB_PROJECT:-text-ts-m3jepa}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_MODE="${WANDB_MODE:-online}"      # online | offline | disabled
WANDB_TAGS=(m3jepa "$DATASET_BACKEND")
WANDB_NOTES="${WANDB_NOTES:-}"

EXTRA=()
[[ -n "$WANDB_ENTITY"   ]] && EXTRA+=(--wandb-entity   "$WANDB_ENTITY")
[[ -n "$WANDB_RUN_NAME" ]] && EXTRA+=(--wandb-run-name "$WANDB_RUN_NAME")
[[ -n "$WANDB_NOTES"    ]] && EXTRA+=(--wandb-notes    "$WANDB_NOTES")
[[ -n "$VAL_SPLIT_SEED" ]] && EXTRA+=(--val-split-seed "$VAL_SPLIT_SEED")

# Gated model guard
if [[ "$LLM_ID" == "google/embeddinggemma-300m" ]]; then
  TOKEN_VALUE="${HF_TOKEN:-}"
  if [[ -z "$TOKEN_VALUE" ]]; then
    echo "ERROR: $LLM_ID is gated. Set $HF_TOKEN_ENV (default: HF_TOKEN) before running."
    exit 1
  fi
fi

LAUNCHER=(torchrun --standalone --nproc_per_node="$NPROC_PER_NODE")
if [[ "$NPROC_PER_NODE" -eq 1 ]]; then
  LAUNCHER=(python)
fi
export PYTHONUNBUFFERED=1

echo "DEVICE=$DEVICE"
if [[ "$DATASET_BACKEND" == "tsfragment" ]]; then
  RESULTS_DIR="./results_jepa/${SOURCE}_L${LENGTH}"
  RUN_NAME_SUFFIX="${SOURCE}_L${LENGTH}"
  WANDB_EXTRA_TAGS=("$SOURCE" "L${LENGTH}")
else
  RESULTS_DIR="./results_jepa/${VERBALTS_NAME}_${VERBALTS_SPLIT}"
  RUN_NAME_SUFFIX="${VERBALTS_NAME}_${VERBALTS_SPLIT}"
  WANDB_EXTRA_TAGS=("$VERBALTS_NAME" "$VERBALTS_SPLIT")
fi
mkdir -p "$RESULTS_DIR"
SAVE="${RESULTS_DIR}/m3jepa_ckpt.pt"
SAVE_BEST="${RESULTS_DIR}/m3jepa_best.pt"
echo "------------------------------------------------------------"
echo "Running dataset backend=$DATASET_BACKEND"
echo "MIN_LENGTH=$MIN_LENGTH  TASK_SCHEDULE=$TASK_SCHEDULE"
echo "NUM_EXPERTS=$NUM_EXPERTS  K_EXPERT=$K_EXPERT  HIDDEN_SIZE=$HIDDEN_SIZE"
echo "Results directory: $RESULTS_DIR"
echo "Launcher: ${LAUNCHER[*]} train_jepa.py"

CUR_EXTRA=("${EXTRA[@]}")
if [[ -z "$WANDB_RUN_NAME" ]]; then
  CUR_EXTRA+=(--wandb-run-name "m3jepa_${RUN_NAME_SUFFIX}")
fi

"${LAUNCHER[@]}" train_jepa.py \
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
  --llm-dtype "$LLM_DTYPE" \
  --hf-token-env "$HF_TOKEN_ENV" \
  --tune-side "$TUNE_SIDE" \
  --text-lora-r "$TEXT_LORA_R" \
  --text-lora-alpha "$TEXT_LORA_ALPHA" \
  --text-lora-dropout "$TEXT_LORA_DROPOUT" \
  --text-lora-target-modules "${TEXT_LORA_TARGET_MODULES[@]}" \
  --ts-lora-r "$TS_LORA_R" \
  --ts-lora-alpha "$TS_LORA_ALPHA" \
  --ts-lora-dropout "$TS_LORA_DROPOUT" \
  --ts-lora-target-modules "${TS_LORA_TARGET_MODULES[@]}" \
  --num-experts "$NUM_EXPERTS" \
  --k-expert "$K_EXPERT" \
  --hidden-size "$HIDDEN_SIZE" \
  --dropout-rate "$DROPOUT_RATE" \
  --temperature "$TEMPERATURE" \
  --l2-weight "$L2_WEIGHT" \
  --cl-weight "$CL_WEIGHT" \
  --task-weight-text2ts "$TASK_WEIGHT_TEXT2TS" \
  --task-weight-ts2text "$TASK_WEIGHT_TS2TEXT" \
  --task-schedule "$TASK_SCHEDULE" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --grad-clip "$GRAD_CLIP" \
  --num-workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --val-ratio "$VAL_RATIO" \
  --log-every "$LOG_EVERY" \
  --save "$SAVE" \
  --save-best "$SAVE_BEST" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-mode "$WANDB_MODE" \
  --wandb-tags "${WANDB_TAGS[@]}" "${WANDB_EXTRA_TAGS[@]}" \
  "${CUR_EXTRA[@]}"
