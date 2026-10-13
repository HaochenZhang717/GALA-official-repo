#!/usr/bin/env bash
# Stage-1 launcher for the STRUCTURE feature-alignment baseline.
#
# Architecture (faithful STRUCTURE, Gröger et al. NeurIPS 2025):
#   * both encoders (Gemma + Chronos-2) FULLY FROZEN
#   * a Transformer head + linear projector per side = the trainable alignment layer
#   * objective = diagonal CLIP (fixed temperature) + lambda * STRUCTURE regularizer
#   * NO diffusion term (that is the V7-side regularizer being compared against)
#
# The produced checkpoint carries ft_method="head" in its saved args, so it is
# loaded verbatim by the existing stage-2 builder
# ablation_diffusion_df_n_arch/train_cond_v7_ablation_ft.py.
#
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
# Transformer-head capacity (parameter-matched to LoRA r=64 ~= 12.6M, like the
# ft-method=head ablation arm). Both encoders are frozen by construction.
HEAD_DEPTH="${HEAD_DEPTH:-1}"
HEAD_NUM_HEADS="${HEAD_NUM_HEADS:-12}"
HEAD_FFN_DIM="${HEAD_FFN_DIM:-2560}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.0}"

# ---------------- objective (CLIP + STRUCTURE) ----------------
TEMPERATURE="${TEMPERATURE:-0.05}"                  # fixed CLIP temperature (STRUCTURE recipe)
STRUCTURE_LAMBDA="${STRUCTURE_LAMBDA:-10.0}"
STRUCTURE_LEVELS="${STRUCTURE_LEVELS:-1}"
STRUCTURE_WARMUP_STEPS="${STRUCTURE_WARMUP_STEPS:-1000}"
STRUCTURE_TEMPERATURE="${STRUCTURE_TEMPERATURE:-0.05}"   # STRUCTURE ties this to the CLIP temperature
STRUCTURE_WEIGHTING="${STRUCTURE_WEIGHTING:-none}"       # none | inverse
STRUCTURE_MARGIN="${STRUCTURE_MARGIN:-0.0}"
STRUCTURE_CENTERING="${STRUCTURE_CENTERING:-mean}"       # none | mean | standard
STRUCTURE_DISTANCE="${STRUCTURE_DISTANCE:-cosine}"       # cosine | rbf | spearman
STRUCTURE_CENTER_FIRST="${STRUCTURE_CENTER_FIRST:-false}"

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
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-200}"

# ---------------- I/O ----------------
LOG_EVERY="${LOG_EVERY:-10}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-structure_baseline}"
RESULTS_ROOT="${RESULTS_ROOT:-./results_aligner_${EXPERIMENT_NAME}}"

# ---------------- wandb ----------------
WANDB_PROJECT="text-ts-aligner-v7"
WANDB_ENTITY=""
WANDB_RUN_NAME=""
WANDB_MODE="${WANDB_MODE:-online}"      # online | offline | disabled
WANDB_TAGS=(structure "$DATASET_BACKEND" head)
WANDB_NOTES=""

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
    echo "Example:"
    echo "  export HF_TOKEN=hf_xxx"
    echo "  bash feature_alignment/run_train_structure.sh"
    exit 1
  fi
fi

LAUNCHER=(torchrun --standalone --nproc_per_node="$NPROC_PER_NODE")
if [[ "$NPROC_PER_NODE" -eq 1 ]]; then
  LAUNCHER=(python)
fi
export PYTHONUNBUFFERED=1

echo "DEVICE=$DEVICE"
DEFAULT_RUN_TAG="${EXPERIMENT_NAME}_M${MIN_LENGTH}_lvl${STRUCTURE_LEVELS}_lam${STRUCTURE_LAMBDA}_seed${SEED}"
RUN_TAG="${RUN_TAG:-$DEFAULT_RUN_TAG}"
if [[ "$DATASET_BACKEND" == "tsfragment" ]]; then
  RESULTS_DIR="${RESULTS_ROOT}/${SOURCE}_L${LENGTH}_${RUN_TAG}"
  RUN_NAME_SUFFIX="${SOURCE}_L${LENGTH}_${RUN_TAG}"
  WANDB_EXTRA_TAGS=("$SOURCE" "L${LENGTH}" "M${MIN_LENGTH}" "lvl${STRUCTURE_LEVELS}" "lam${STRUCTURE_LAMBDA}" "seed${SEED}")
else
  RESULTS_DIR="${RESULTS_ROOT}/${VERBALTS_NAME}_${VERBALTS_SPLIT}_${RUN_TAG}"
  RUN_NAME_SUFFIX="${VERBALTS_NAME}_${VERBALTS_SPLIT}_${RUN_TAG}"
  WANDB_EXTRA_TAGS=("$VERBALTS_NAME" "$VERBALTS_SPLIT" "M${MIN_LENGTH}" "lvl${STRUCTURE_LEVELS}" "lam${STRUCTURE_LAMBDA}" "seed${SEED}")
fi
mkdir -p "$RESULTS_DIR"
SAVE="${RESULTS_DIR}/aligner_structure_ckpt.pt"
SAVE_BEST="${RESULTS_DIR}/aligner_structure_best.pt"
echo "------------------------------------------------------------"
echo "Running dataset backend=$DATASET_BACKEND"
echo "MIN_LENGTH=$MIN_LENGTH"
echo "HEAD (depth=$HEAD_DEPTH num_heads=$HEAD_NUM_HEADS ffn_dim=$HEAD_FFN_DIM)"
echo "STRUCTURE (lambda=$STRUCTURE_LAMBDA levels=$STRUCTURE_LEVELS warmup=$STRUCTURE_WARMUP_STEPS temp=$STRUCTURE_TEMPERATURE)"
echo "CLIP temperature=$TEMPERATURE"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "LR=$LR  SEED=$SEED"
echo "RUN_TAG=$RUN_TAG"
echo "Results directory: $RESULTS_DIR"
echo "Launcher: ${LAUNCHER[*]} train_structure.py"

CUR_EXTRA=("${EXTRA[@]}")
if [[ -z "$WANDB_RUN_NAME" ]]; then
  CUR_EXTRA+=(--wandb-run-name "structure_${RUN_NAME_SUFFIX}")
fi

"${LAUNCHER[@]}" train_structure.py \
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
  --ft-method head \
  --head-depth "$HEAD_DEPTH" \
  --head-num-heads "$HEAD_NUM_HEADS" \
  --head-ffn-dim "$HEAD_FFN_DIM" \
  --head-dropout "$HEAD_DROPOUT" \
  --temperature "$TEMPERATURE" \
  --structure-lambda "$STRUCTURE_LAMBDA" \
  --structure-levels "$STRUCTURE_LEVELS" \
  --structure-warmup-steps "$STRUCTURE_WARMUP_STEPS" \
  --structure-temperature "$STRUCTURE_TEMPERATURE" \
  --structure-weighting "$STRUCTURE_WEIGHTING" \
  --structure-margin "$STRUCTURE_MARGIN" \
  --structure-centering "$STRUCTURE_CENTERING" \
  --structure-distance "$STRUCTURE_DISTANCE" \
  --structure-center-first "$STRUCTURE_CENTER_FIRST" \
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
