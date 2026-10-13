#!/usr/bin/env bash
# Train the BRIDGE baseline (--model-type=bridge) inside train_cond_v7.py, with
# Aligner V7 as the text encoder. Single GPU.
#
# BRIDGE (ICML 2025) is a data-space DDPM: a DomainUnifiedPrototyper extracts
# domain prototypes from an example series (during training the target series
# itself; during generation a cached batch of real training series) which are
# injected into a 1D UNet via cross-attention, with the pooled text embedding
# fused in by a gated ConditioningMLP. CFG is BRIDGE's own (drops the prototype
# context to a learnable null embedding).
#
# This mirrors run_train_cond_v7_verbalts_arch.sh but swaps the VerbalTS denoiser
# for the Bridge denoiser. Loss is noise-MSE (NOT comparable to jit v-MSE).
set -euo pipefail

NODE_NAME="${SLURMD_NODENAME:-$(hostname -s)}"
echo "Running on node: $NODE_NAME"

# Resolve to the project root (parent of this script's t2ts_jit/ dir).
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
# BRIDGE conditions on the ORIGINAL LongCLIP text embedding (zer0int LongCLIP-GmP-
# ViT-L-14, the same family ConTSG's CTTP is built on), pooled CLS -> (B, D).
# This is the conditioning encoder only; gen-eval metrics still use the Aligner V7
# embedder (--metrics-aligner-ckpt) so BRIDGE stays comparable to the other v7
# baselines.
TEXT_ENCODER="frozen-longclip"
LONGCLIP_MODEL_ID="zer0int/LongCLIP-GmP-ViT-L-14"
LONGCLIP_OUTPUT_TYPE="cls"      # pooled (B, D); 'all' (3D) is not allowed for bridge
LLM_DTYPE="bfloat16"            # dtype for the frozen-longclip encoder
HF_TOKEN_ENV="HF_TOKEN"
TEXT_CACHE_BATCH_SIZE=256
TEXT_COND_NORM="layernorm"
TEXT_ADAPTER_HIDDEN_DIM=512
TEXT_ADAPTER_DROPOUT=0.0
# BRIDGE has its own CFG (--bridge-cond-drop-prob); keep v6's TextConditioner
# condition-drop OFF so the two CFG mechanisms don't stack.
COND_DROP_PROB=0.0
# Aligner V7 ckpt used ONLY as the gen-eval metric embedder (not for conditioning).
ALIGNER_CKPT="${ALIGNER_CKPT:-./feature_alignment/results_aligner_v7/${SOURCE}_L${LENGTH}/aligner_v7_best.pt}"

# ---------------- denoiser (BRIDGE architecture) ----------------
MODEL_TYPE="bridge"
IN_CHANNELS=1
BRIDGE_LATENT_DIM=32
BRIDGE_NUM_LATENTS=16
BRIDGE_LATENT_UNIT=1
BRIDGE_PROTOTYPE_DIM=0          # 0 -> fall back to BRIDGE_LATENT_DIM
BRIDGE_MODEL_CHANNELS=32
BRIDGE_NUM_RES_BLOCKS=2
BRIDGE_ATTENTION_RESOLUTIONS=(1 2 4)
# seq_len must be divisible by 2**(len(channel_mult)-1). For LENGTH=24 that is
# 8 (24%8==0). Shorten this for shorter series.
BRIDGE_CHANNEL_MULT=(1 2 4 4)
BRIDGE_DROPOUT=0.0
BRIDGE_NUM_HEADS=8
BRIDGE_NUM_HEAD_CHANNELS=-1
BRIDGE_TRANSFORMER_DEPTH=1
BRIDGE_FUSION_TYPE="gated_add"
BRIDGE_COND_DROP_PROB=0.5       # train-time prototype-context drop for CFG
BRIDGE_CFG_SCALE=1.0            # sampling guidance scale (ConTSG default 1.0; try 3-9)
# Diffusion schedule aligned with ConTSG bridge_*.yaml (== repo VerbalTS settings):
# 50 quad steps, beta 1e-4 -> 0.5.
BRIDGE_DIFFUSION_STEPS=50
BRIDGE_BETA_START=1e-4
BRIDGE_BETA_END=0.5
BRIDGE_SCHEDULE="quad"
BRIDGE_SAMPLER="ddim"
BRIDGE_DDIM_STEPS=0            # 0 -> use all BRIDGE_DIFFUSION_STEPS (ConTSG samples all steps)
BRIDGE_ETA=0.0
# BRIDGE_CLIP_DENOISED left OFF (z-score data exceeds +/-1); pass --bridge-clip-denoised to enable.

# ---------------- diffusion (shared) ----------------
EMA_DECAY1=0.999
EMA_DECAY2=0.9999

# ---------------- training ----------------
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS=1000
LR="${LR:-1e-3}"
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0
NUM_WORKERS=16
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
METRICS_SEED=123

# ---------------- I/O ----------------
LOG_EVERY=10
SAVE_DIR="./t2ts_jit/results_jit_cond_v7_bridge_${SOURCE}_L${LENGTH}_seed${SEED}"
SAVE="${SAVE_DIR}/denoiser_ckpt.pt"
SAVE_BEST="${SAVE_DIR}/denoiser_best.pt"
SAVE_EVERY_EPOCHS=100

# ---------------- wandb ----------------
WANDB_PROJECT="text-ts-formal-experiment"
WANDB_ENTITY=""
WANDB_RUN_NAME="longclip_bridge_${SOURCE}_L${LENGTH}_seed${SEED}"
WANDB_MODE="online"
WANDB_TAGS=(cond v7 bridge longclip "$SOURCE" "L${LENGTH}" "seed${SEED}")
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

# LongCLIP itself is public, but the gen-eval metric aligner (V7 ckpt) internally
# loads gemma which IS gated — require HF_TOKEN whenever the metric aligner is used.
if [[ "$ENABLE_GEN_EVAL" -eq 1 && -n "${METRICS_ALIGNER_CKPT:-}" ]]; then
  TOKEN_VALUE="${HF_TOKEN:-}"
  if [[ -z "$TOKEN_VALUE" ]]; then
    echo "ERROR: metrics aligner ($METRICS_ALIGNER_CKPT) uses gated gemma."
    echo "Set $HF_TOKEN_ENV (default: HF_TOKEN) before running."
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
  --longclip-model-id "$LONGCLIP_MODEL_ID" \
  --longclip-output-type "$LONGCLIP_OUTPUT_TYPE" \
  --llm-dtype "$LLM_DTYPE" \
  --hf-token-env "$HF_TOKEN_ENV" \
  --text-cache-batch-size "$TEXT_CACHE_BATCH_SIZE" \
  --text-cond-norm "$TEXT_COND_NORM" \
  --text-adapter-hidden-dim "$TEXT_ADAPTER_HIDDEN_DIM" \
  --text-adapter-dropout "$TEXT_ADAPTER_DROPOUT" \
  --cond-drop-prob "$COND_DROP_PROB" \
  --model-type "$MODEL_TYPE" \
  --in-channels "$IN_CHANNELS" \
  --bridge-latent-dim "$BRIDGE_LATENT_DIM" \
  --bridge-num-latents "$BRIDGE_NUM_LATENTS" \
  --bridge-latent-unit "$BRIDGE_LATENT_UNIT" \
  --bridge-prototype-dim "$BRIDGE_PROTOTYPE_DIM" \
  --bridge-model-channels "$BRIDGE_MODEL_CHANNELS" \
  --bridge-num-res-blocks "$BRIDGE_NUM_RES_BLOCKS" \
  --bridge-attention-resolutions "${BRIDGE_ATTENTION_RESOLUTIONS[@]}" \
  --bridge-channel-mult "${BRIDGE_CHANNEL_MULT[@]}" \
  --bridge-dropout "$BRIDGE_DROPOUT" \
  --bridge-num-heads "$BRIDGE_NUM_HEADS" \
  --bridge-num-head-channels "$BRIDGE_NUM_HEAD_CHANNELS" \
  --bridge-transformer-depth "$BRIDGE_TRANSFORMER_DEPTH" \
  --bridge-fusion-type "$BRIDGE_FUSION_TYPE" \
  --bridge-cond-drop-prob "$BRIDGE_COND_DROP_PROB" \
  --bridge-cfg-scale "$BRIDGE_CFG_SCALE" \
  --bridge-diffusion-steps "$BRIDGE_DIFFUSION_STEPS" \
  --bridge-beta-start "$BRIDGE_BETA_START" \
  --bridge-beta-end "$BRIDGE_BETA_END" \
  --bridge-schedule "$BRIDGE_SCHEDULE" \
  --bridge-sampler "$BRIDGE_SAMPLER" \
  --bridge-ddim-steps "$BRIDGE_DDIM_STEPS" \
  --bridge-eta "$BRIDGE_ETA" \
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
