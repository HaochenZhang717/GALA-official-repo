#!/usr/bin/env bash
# Train conditional diffusion model using the VerbalTS *architecture*
# (--model-type=verbalts) inside train_cond_v6.py, with frozen LongCLIP text
# encoder returning unpooled token sequences. Single GPU.
#
# Here we train the VerbalTS multi-patch diffusion architecture on TSFragment,
# so that it is directly comparable with the other train_cond_v6.py runs.
#
# ==========================================================================
# ALIGNMENT WITH THE OFFICIAL VerbalTS RELEASE ("repro" below)
# ==========================================================================
# All architectural / training hyperparameters that can be set via CLI are
# aligned with the effective config of the official VerbalTS training
# script. The text path now also includes the
# TrainableTextProjector-equivalent pre-projector (768 -> 512 -> 64) so
# TextProjectorMVarMScaleMStep operates at d=64 like repro. The
# TextConditioner is disabled (norm=none, adapter_hidden_dim=0) so no
# extra LayerNorm/MLP sits between cached CLIP and the pre-projector.
#
# One intentional divergence remains:
#
# (A) DATASET: repro trains on VerbalTS-synthetic_u (DATASET_BACKEND=verbalts);
#     we train on TSFragment/ETTm1 so this run is apples-to-apples with the
#     other train_cond_v6.py runs (frozen_gemma, aligner_v6) on the same
#     source. Switch DATASET_BACKEND="verbalts" + supply --verbalts-* args
#     below if you want to also replicate the repro's data side.
#
# Annotations on each variable use:
#   [match repro]            value identical to effective repro config
#   [diff repro=<X>]         intentionally diverges; X = repro's effective value
#   [N/A in repro]           variable has no counterpart in the baseline
# ==========================================================================
set -euo pipefail

NODE_NAME="${SLURMD_NODENAME:-$(hostname -s)}"
echo "Running on node: $NODE_NAME"

# Resolve to the project root (parent of this script's t2ts_jit/ dir).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------- dataset ----------------
DATASET_BACKEND="tsfragment"        # tsfragment | verbalts                            [diff repro=verbalts (see header A)]
# SOURCE is swept via env var; defaults to ETTm1 for single-run use.
SOURCE="${SOURCE:-ETTm1}"           # TSFragment sub-source (ETTm1 | traffic | electricity | ...)
LENGTH=24                           # target sequence length (pre-upsample)
MIN_LENGTH=64                       # drop samples shorter than this                   [match repro]
MAX_ROWS=none                       # cap on rows; "none" = all                        [match repro]
CACHE_DIR="./hf_cache"              # HuggingFace datasets cache
NORMALIZE="global"                  # per-sample | global | none                       [match repro]

# ---------------- text encoder (frozen LongCLIP, UNPOOLED tokens) ----------------
TEXT_ENCODER="frozen-longclip"
LONGCLIP_MODEL_ID="zer0int/LongCLIP-GmP-ViT-L-14"       # [match repro]
LONGCLIP_OUTPUT_TYPE="all"          # "all" -> per-token (B, T=248, D=768)             [match repro]
LLM_DTYPE="bfloat16"                # CLIP load dtype to save VRAM
HF_TOKEN_ENV="HF_TOKEN"             # env var name for HF token (gated models)
TEXT_CACHE_BATCH_SIZE=256           # cache-time batch size                            [match repro]
# TextConditioner is disabled so the text path matches repro exactly:
#   CLIP-cached -> VerbalTSDenoiser(pre_proj -> projector -> net)
# (repro has no LayerNorm/adapter between cached CLIP and its TrainableTextProjector.)
TEXT_COND_NORM="none"               # layernorm | none -- "none" => TextConditioner.norm = Identity   [match repro (no LN)]
TEXT_ADAPTER_HIDDEN_DIM=0           # 0 disables the adapter MLP entirely                              [match repro (no adapter)]
TEXT_ADAPTER_DROPOUT=0.0            # adapter dropout (no-op when adapter is disabled)
# Pre-projector inside VerbalTSDenoiser (= repro's TrainableTextProjector):
# Linear(768 -> VERBALTS_TEXTEMB_HIDDEN_DIM) -> LayerNorm -> LeakyReLU -> Linear(-> VERBALTS_TEXT_EMB_DIM)
VERBALTS_TEXTEMB_HIDDEN_DIM=512     # pre-projector hidden dim                         [match repro]
VERBALTS_TEXT_EMB_DIM=64            # pre-projector output dim = TextProjectorMVarMScaleMStep dim_in  [match repro]

# ---------------- denoiser (VerbalTS architecture) ----------------
MODEL_TYPE="verbalts"               # routes build_denoiser() to VerbalTSDenoiser
IN_CHANNELS=1                       # n_var (variates). TSFragment is univariate       [match repro hardcoded n_var=1]
VERBALTS_CHANNELS=64                # VerbalTS internal d_model                        [match repro]
VERBALTS_MULTIPATCH_NUM=3           # # of parallel patch scales K                     [match repro shell override]
VERBALTS_BASE_PATCH=3               # smallest patch length                            [match repro shell override]
VERBALTS_L_PATCH_LEN=2              # patch length grows as base*L^i -> [3, 6, 12]     [match repro shell override]
VERBALTS_LAYERS=3                   # # of ResidualBlocks                              [match repro]
VERBALTS_NHEADS=8                   # attention heads in time/feature self-attn        [match repro]
VERBALTS_DIFF_EMBED_DIM=128         # DiffusionEmbedding output dim                    [match repro]
VERBALTS_VAR_EMB=16                 # SideEncoder_Var per-variate emb dim              [match repro hardcoded]
VERBALTS_TIME_EMB=112               # SideEncoder_Var time-position sin/cos dim        [match repro hardcoded]
VERBALTS_ATTENTION_MASK_TYPE="parallel"  # full=cross-scale attn; parallel=block-diag  [match repro]
VERBALTS_CONDITION_TYPE="adaLN"     # adaLN | add | cross_attention                    [match repro]
VERBALTS_NUM_STEPS=50               # discrete DDPM steps                              [match repro]
VERBALTS_BETA_START=1e-4            # DDPM beta schedule start                         [match repro]
VERBALTS_BETA_END=0.5               # DDPM beta schedule end                           [match repro]
VERBALTS_SCHEDULE="quad"            # quad | linear                                    [match repro]
VERBALTS_TEXT_PROJECTOR_STAGES=3    # # of step-bins for TextProjectorMVarMScaleMStep  [match repro num_stages=3]

# ---------------- diffusion (only --noise-scale / --ema-* are read for verbalts) ----------------
# For --model-type=verbalts the jit-only flags (p_mean/p_std/t_eps/interval_*/
# sampling_method/num_sampling_steps) are silently ignored. Sampling is
# hardcoded to deterministic DDIM that walks all VERBALTS_NUM_STEPS in reverse.
NOISE_SCALE=1.0                     # initial x_T scale at generate(); leave at 1.0 for DDPM-consistent prior   [N/A in repro]
# EMA semantics aligned with repro convention (fast first, slow second):
EMA_DECAY1=0.999                    # fast EMA                                         [match repro]
EMA_DECAY2=0.9999                   # slow EMA                                         [match repro]

# ---------------- training ----------------
BATCH_SIZE=512                      # per-rank batch size                              [match repro shell override]
EPOCHS=1000                         #                                                  [match repro shell override]
LR=1e-3                             #                                                  [match repro shell override]
WEIGHT_DECAY=0.0                    #                                                  [match repro]
GRAD_CLIP=1.0                       # max grad norm (<=0 disables)                     [match repro]
NUM_WORKERS=16                      # DataLoader worker processes                      [match repro shell]
SEED=0                              #                                                  [match repro]
DEVICE="cuda"                       # overridden under torchrun

# ---------------- validation ----------------
VAL_RATIO=0.1                       # train/val split ratio                            [match repro]
VAL_SPLIT_SEED=0                    # RNG seed for the train/val split (separate from --seed)
VAL_SEED=12345                      # RNG seed inside evaluate(); makes the stochastic diffusion val loss comparable across epochs  [match repro]
EVAL_EVERY_STEPS=0                  # 0 = run val only at epoch end                    [match repro]

# generation metrics every N epochs (set <=0 to disable)
ENABLE_GEN_EVAL=1                   # whether to sample + compute ConTSG metrics       [match repro]
GEN_EVAL_EVERY_EPOCHS=100            # frequency of generation-quality eval             [match repro shell override]
GEN_EVAL_MAX_BATCHES=0              # cap on val batches for generation eval (0=all)   [match repro]
# Metrics aligner: reuse the V6 aligner trained on the same dataset+length so
# generation metrics are comparable to the frozen-gemma / aligner-v6 method runs.
# Default path is per-source; override METRICS_ALIGNER_CKPT in the env to force a specific ckpt.
METRICS_ALIGNER_CKPT="${METRICS_ALIGNER_CKPT:-./feature_alignment/results_aligner_v6/${SOURCE}_L24/aligner_v6_best.pt}"
METRICS_K=5                         # top-K for retrieval-style metrics                [match repro]
METRICS_MDD_BINS=32                 # # of histogram bins for MDD                       [match repro]
METRICS_MAX_SAMPLES=0               # cap on samples per metric (0 = full val)         [match repro shell override]
METRICS_SEED=123                    # RNG seed for metric sampling                     [match repro]

# ---------------- I/O ----------------
LOG_EVERY=10                        # print one stdout line every N steps              [match repro]
SAVE_DIR="./t2ts_jit/results_jit_cond_v6_verbalts_arch_${SOURCE}_L24"
SAVE="${SAVE_DIR}/denoiser_ckpt.pt"
SAVE_BEST="${SAVE_DIR}/denoiser_best.pt"
SAVE_EVERY_EPOCHS=100               # periodic checkpoint frequency                    [match repro]

# ---------------- wandb ----------------
WANDB_PROJECT="text-ts-ablation"    # shared project with frozen-gemma / aligner-v6 runs
WANDB_ENTITY=""                     # team/user; empty = default
WANDB_RUN_NAME="verbalts_arch_${SOURCE}_L24"
WANDB_MODE="online"                 # online | offline | disabled
WANDB_TAGS=(cond v6 verbalts_arch frozen_longclip "$SOURCE" L24)
WANDB_NOTES=""

# ---------------- Hugging Face cache ----------------
# Redirect HF / wandb caches to project-local dirs (avoid $HOME quota on cluster).
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

# Optional CLI flags appended only when their backing var is non-empty / enabled.
EXTRA=()
[[ -n "$WANDB_ENTITY"   ]] && EXTRA+=(--wandb-entity "$WANDB_ENTITY")
[[ -n "$WANDB_RUN_NAME" ]] && EXTRA+=(--wandb-run-name "$WANDB_RUN_NAME")
[[ -n "$WANDB_NOTES"    ]] && EXTRA+=(--wandb-notes "$WANDB_NOTES")
[[ -n "$VAL_SPLIT_SEED" ]] && EXTRA+=(--val-split-seed "$VAL_SPLIT_SEED")
[[ "$ENABLE_GEN_EVAL" -eq 1 ]] && EXTRA+=(--enable-gen-eval)

python t2ts_jit/train_cond_v6.py \
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
