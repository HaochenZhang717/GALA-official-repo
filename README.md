# GALA: Generation-Aware Cross-Modal Alignment for Text-to-Time-Series Synthesis

Anonymized code for review. GALA generates a time-series fragment from a
free-form natural-language caption in two decoupled stages:

- **Stage 1 — Cross-modal alignment** (`feature_alignment/`). A frozen
  Embedding-Gemma-300M text encoder and a frozen Chronos-2 series encoder are
  LoRA-adapted and contrastively aligned into a shared 512-d space, with an
  auxiliary flow-matching generation loss that keeps the caption embedding
  sufficient to synthesize its fragment.
- **Stage 2 — Conditional generation** (`t2ts_jit/`). The aligned text encoder
  is frozen and used as the sole condition of a JiT/DiT flow-matching generator
  that operates on a delay-embedded image of the series.

Everything reported in the paper is covered here: the method, the five
text-conditioned baselines, the two alternative alignment objectives, all
ablations, the independent evaluation model, and the benchmark-diversity
analysis in the supplement.

## Repository layout

```
feature_alignment/            Stage 1
  aligner_v7.py               dual-encoder aligner (LoRA text + Chronos-2 towers)
  train_v7.py                 Stage-1 training entry point
  dataset.py                  TSFragment-600K / VerbalTS data loading
  run_train_v7.sh             launcher for Stage 1
  # --- Stage-1 ablations ---
  aligner_v7_ablation_ft.py   FT-Full / FT-Head / LoRA variants of the aligner
  train_v7_ablation_ft.py     trainer for the fine-tuning-method sweep
  run_train_v7_ablation_ft.sh   fine-tuning method x auxiliary-loss weight
  run_train_v7_ablation.sh      adapted-tower sweep (text / series / both)
  # --- alternative Stage-1 objectives compared in the paper ---
  jepa_align/, train_jepa.py, run_train_jepa.sh              M3-JEPA
  structure_align/, train_structure.py, run_train_structure.sh  STRUCTURE
  # --- independent evaluation model ---
  eval_aligner.py             EvalAligner: BGE-large text tower + from-scratch
                              CNN series tower, plus the metric embedder that
                              train_cond_v6.py loads
  train_eval_aligner.py       EvalAligner training entry point
  run_train_eval_aligner.sh   launcher

t2ts_jit/                     Stage 2, metrics, and baselines
  train_cond_v7.py            Stage-2 training entry point
  train_cond_v6.py            shared Stage-2 trainer (v7 wraps/patches it)
  contsg_v4_metrics.py        FID / CTTP / JFTSD metric implementation
  run_train_cond_v7.sh        launcher for Stage 2
  run_train_cond_v7_persample.sh   normalization ablation (Chronos-2 instance norm)
  cond_jit_basics/            denoiser, JiT/DiT model, delay embedding, and the
                              vendored baseline architectures
  pick_best_fid.py, pick_best_fid_ablation.py     checkpoint selection
  aggregate_repeat.py, aggregate_repeat_ablation.py
                              multi-seed aggregation into the reported numbers
  # --- baselines, see "Baselines" below ---

ablation_diffusion/           adapted-tower ablation (Tune-Text/TS/Both)
ablation_diffusion_df_n_arch/ fine-tuning method x auxiliary-loss weight
ablation_diffusion_joint/     single-stage joint training
ablation_diffusion_noalign/   no Stage-1 alignment (frozen / LoRA text encoder)
compare_jepa/, compare_structure/
                              Stage 2 trained on the M3-JEPA / STRUCTURE aligners

analysis/                     benchmark-diversity study in the supplement
  analyze_shape_diversity.py  PCA spectrum / effective rank / KMeans per source

baseline_codes/T2S/           vendored subset of the official T2S release (Apache-2.0)
```

Each ablation directory resolves its siblings relative to its own location, so
please keep the top-level layout as shipped.

## Setup

```bash
pip install -r requirements.txt
# Embedding-Gemma-300M is gated on the Hugging Face Hub:
export HF_TOKEN=hf_xxx
```

TSFragment-600K is downloaded automatically from the Hub
(`WinfredGe/TSFragment-600K`) on first use; no manual data placement is needed
for the default `tsfragment` backend. The optional VerbalTS backend reads from
`VERBALTS_ROOT` (defaults to `./data/VerbalTSDatasets`).

BGE-large-en-v1.5, used by the evaluation model, is **not** gated and needs no
token.

## Reproducing the paper

The three stages must run in this order, because Stage 2 consumes the Stage-1
aligner checkpoint and the metrics consume the EvalAligner checkpoint.

```bash
# 0. Evaluation model: one per (source, length). Scores every number in the
#    paper and shares no weights with either GALA tower.
SOURCE=ETTh1s LENGTH=48 bash feature_alignment/run_train_eval_aligner.sh

# 1. Stage 1: alignment.
#    -> feature_alignment/results_aligner_v7/<SOURCE>_L<LENGTH>/aligner_v7_best.pt
SOURCE=ETTh1s LENGTH=48 bash feature_alignment/run_train_v7.sh

# 2. Stage 2: conditional generation on the frozen Stage-1 aligner.
SOURCE=ETTh1s LENGTH=48 bash t2ts_jit/run_train_cond_v7.sh
```

All hyperparameters are environment variables at the top of each `run_*.sh`;
their defaults match the settings reported in the paper.

### Evaluation protocol

Generation metrics (FID, CTTP, JFTSD) are computed during Stage-2 training by
`t2ts_jit/contsg_v4_metrics.py`, in the embedding space of the EvalAligner
(`--metrics-aligner-kind eval`). To turn raw runs into the reported numbers:

```bash
python t2ts_jit/pick_best_fid.py      # lowest-FID checkpoint x {online,ema1,ema2}
python t2ts_jit/aggregate_repeat.py   # mean +- std over the 3 sampling seeds
```

### Ablations

| Paper item | Directory / launcher |
| --- | --- |
| No Stage-1 alignment (frozen / LoRA) | `ablation_diffusion_noalign/run_train_cond_v7_ablation_noalign.sh` |
| Single-stage joint training | `ablation_diffusion_joint/run_train_cond_v7_ablation_joint.sh` |
| Fine-tuning method x auxiliary-loss weight | `feature_alignment/run_train_v7_ablation_ft.sh` then `ablation_diffusion_df_n_arch/run_train_cond_v7_ablation_ft.sh` |
| Adapted tower (text / series / both) | `feature_alignment/run_train_v7_ablation.sh` then `ablation_diffusion/run_train_cond_v7_ablation.sh` |
| Series normalization (global z-score vs per-sample) | `t2ts_jit/run_train_cond_v7_persample.sh` |
| Alternative alignment objectives | `feature_alignment/run_train_{jepa,structure}.sh` then `compare_{jepa,structure}/run_train_cond_*.sh` |

Each Stage-1 ablation produces an aligner checkpoint that the matching Stage-2
launcher then freezes and trains on, so the generator is identical across
variants and only the conditioning representation changes.

### Benchmark-diversity analysis

```bash
python analysis/analyze_shape_diversity.py   # per-source diversity metrics
```

## Baselines

All five text-conditioned baselines (VerbalTS, T2S, DiffuSETS, BRIDGE,
Text2Motion) are run on exactly the same data splits, normalization and
evaluation protocol as GALA, so the numbers in the paper are directly
comparable. Each baseline keeps its own text encoder and its own objective;
only the data/eval harness is shared.

| Baseline | Launcher | Notes |
| --- | --- | --- |
| VerbalTS | `run_train_cond_v7_verbalts_arch.sh` | multi-patch VerbalTS denoiser + discrete DDPM, frozen LongCLIP tokens |
| BRIDGE | `run_train_cond_v7_bridge.sh` | data-space DDPM, prototype cross-attention + gated text fusion |
| T2S | `run_pretrain_vae_t2s.sh` -> `run_train_cond_v7_t2s.sh` | two-stage: LA-VAE, then latent rectified-flow DiT |
| DiffuSETS | `run_pretrain_vae_diffusets.sh` -> `run_train_cond_v7_diffusets.sh` | two-stage: VAE, then latent DDPM U-Net |
| Text2Motion | `run_pretrain_ae_text2motion.sh` -> `run_train_cond_v7_text2motion.sh` | two-stage: movement AE, then text-conditioned CVAE generator |

The VerbalTS, BRIDGE, DiffuSETS and Text2Motion architectures are vendored as
dependency-free single-file copies under `t2ts_jit/cond_jit_basics/`
(`verbalts.py`, `bridge_denoiser.py` + `bridge_modules/`, `diffusets_modules.py`,
`text2motion.py`).

The T2S baseline imports its backbone modules from the official T2S release; the
minimal subset it needs (`model.pretrained.vqvae`, `model.backbone.rectified_flow`,
`model.denoiser.transformer` and their dependency `model.pretrained.core`) is
vendored verbatim under `baseline_codes/T2S/` together with its Apache-2.0
license, so no external checkout is required.

See `THIRD_PARTY.md` for the full provenance and license of every borrowed
component.

## Notes

- Plotting and report-formatting scripts are omitted; the code here is what is
  needed to run the experiments and to reproduce the reported numbers.
- Trained checkpoints are **not** included; run the EvalAligner and Stage 1
  first to produce the checkpoints that Stage 2 loads (`ALIGNER_CKPT` /
  `METRICS_ALIGNER_CKPT`).
- The launchers run on a single GPU (`python`) or multi-GPU via `torchrun`
  (`NPROC_PER_NODE`); cluster-specific SLURM wrappers are omitted, and every
  script resolves its working directory from its own location.
- Weights & Biases logging is optional and off by default (`WANDB_ENTITY` is
  empty); set `WANDB_MODE=disabled` to be sure.
