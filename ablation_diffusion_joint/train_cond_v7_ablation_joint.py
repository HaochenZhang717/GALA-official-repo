"""
Stage-2 driver for the "single-stage JOINT training" ablation.

Question this ablation answers
-------------------------------
The main pipeline trains a Stage-1 text<->series aligner (contrastive InfoNCE),
FREEZES it, then trains the real Stage-2 generator on top of the frozen
embeddings (t2ts_jit/train_cond_v7.py). Table 2 only compares that
frozen-aligner pipeline against NO alignment at all (ablation_diffusion_noalign/):
it shows alignment helps, but never tests whether alignment needs to be a
SEPARATE, PRECEDING stage. Eq. 2's decomposition (R(E) + generation term) is
an algebraic identity that holds for ANY (E, theta) -- it does not, by
itself, privilege sequential optimization over joint optimization.

This script is the missing baseline: Stage 1 and Stage 2 collapse into ONE
run. The SAME TextToTimeSeriesAlignerV7 encoder (contrastive loss, LoRA on
both the Gemma text side and the Chronos-2 series side) is trained TOGETHER,
in one optimizer / one backward pass, with a SINGLE diffusion generator built
at the SAME size/architecture as the real Stage-2 generator (not a smaller
stand-in) -- there is no separate small "auxiliary" network. Gradients from
the generation loss flow directly into the aligner's LoRA adapters, something
the frozen two-stage pipeline never allows.

Fair-comparison contract (mirrors ablation_diffusion_noalign/ exactly)
-----------------------------------------------------------------------
Same dataset backend/split (TSFragmentDataset, global normalize, val_ratio
0.1, val_split_seed 0), same generator architecture/hyperparameters (jit2d
denoiser via t2ts_jit/train_cond_v6.py's build_denoiser / TextConditioner /
DenoiserWithTextConditioner, imported unmodified), same training schedule
(epochs / batch size / lr / grad clip), same gen-eval protocol
(ConTSGMetricsAccumulator every N epochs) against the SAME FIXED "both"
aligner checkpoint used as the metric embedder everywhere else in this repo
(feature_alignment/results_aligner_v7/{source}_L{length}/aligner_v7_best.pt),
so FID / CTTP / JFTSD stay directly comparable to the main pipeline and the
no-align ablation.

The aligner side (LoRA hyperparameters, contrastive loss) mirrors
feature_alignment/train_v7.py's aligner exactly, so the *architecture* being
ablated is identical to Stage 1's -- only the SCHEDULE (joint vs. sequential)
and the generator (Stage-2-sized, not a small auxiliary model) differ.

(For a standalone read-out of the ORIGINAL two-stage pipeline's own small
Stage-1 auxiliary denoiser g_xi -- a separate question from this ablation --
see ablation_diffusion_joint/eval_gxi_standalone.py, which needs no
retraining since g_xi's weights are already stored in every Stage-1
checkpoint.)

gen-eval metrics are teed to <save_dir>/test_metrics.txt, one JSON line per
eval, matching every other ablation driver in this repo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Dataset, Subset, random_split

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_T2TS_DIR = _REPO_ROOT / "t2ts_jit"
_FA_DIR = _REPO_ROOT / "feature_alignment"
sys.path.insert(0, str(_T2TS_DIR))
sys.path.insert(0, str(_FA_DIR))

import train_cond_v6 as _v6  # noqa: E402 -- build_denoiser/TextConditioner/DenoiserWithTextConditioner reuse
from train_cond_v7 import AlignerV7MetricEmbedder  # noqa: E402 -- fixed metric embedder
from aligner_v7 import TextToTimeSeriesAlignerV7  # noqa: E402
from train_v7 import batch_series_hash  # noqa: E402
from dataset import TSFragmentDataset, SOURCES, LENGTHS, collate  # noqa: E402
from contsg_v4_metrics import ConTSGMetricsAccumulator, MetricConfig, seed_all  # noqa: E402

DTYPES = _v6.DTYPES


# --------------------------------------------------------------------------- #
# gen-eval TXT tee (identical behavior to the other ablation drivers)
# --------------------------------------------------------------------------- #
def _resolve_txt_path(save_arg: Path | None) -> Path:
    if save_arg is not None:
        return Path(save_arg).resolve().parent / "test_metrics.txt"
    return _HERE / "test_metrics.txt"


def _make_tee(txt_path: Path, orig_log):
    """`orig_log` must be `wandb.log` captured AFTER `wandb.init()` returns --
    `wandb.init()` rebinds the module-level `wandb.log` to the real run logger,
    so capturing it before `init()` would tee into a no-op stub forever."""

    def _append_record(rec: dict) -> None:
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(txt_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def tee_log(data=None, *args, **kwargs):
        try:
            if isinstance(data, dict) and any(k.startswith("gen_eval/") for k in data):
                step = kwargs.get("step")
                if step is None and args:
                    step = args[0]
                rec: dict = {"kind": "gen_eval", "step": step}
                for k, v in data.items():
                    if k == "gen_eval/epoch":
                        rec["epoch"] = v
                    elif k.startswith("gen_eval/"):
                        rec[k[len("gen_eval/"):]] = v
                _append_record(rec)
        except Exception as e:  # never let TXT logging crash training
            print(f"[ablation-txt] WARN failed to write gen-eval metrics: {e}")
        return orig_log(data, *args, **kwargs)

    return tee_log, _append_record


# --------------------------------------------------------------------------- #
# dataset: one item -> (caption, raw series, main-denoiser-normalized series)
# --------------------------------------------------------------------------- #
class JointTSDataset(Dataset):
    """Wraps a native-length TSFragmentDataset to also expose the [-1, 1]
    globally-rescaled series used by the generator, without needing a second
    synchronized DataLoader.

    `global_min`/`global_max` MUST be computed over the whole base dataset
    (train+val) exactly like train_cond_v6.NormalizedTSDataset(mode="global"),
    so this ablation stays numerically identical to the main pipeline / the
    no-align ablation.
    """

    def __init__(self, base: TSFragmentDataset, global_min: float, global_max: float):
        self.base = base
        self.global_min = float(global_min)
        self.global_max = float(global_max)
        self._span = max(self.global_max - self.global_min, 1e-6)

    @property
    def upsampled_length(self) -> int:
        return self.base.upsampled_length

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        caption, series = self.base[idx]  # series: (1, L) raw scale
        norm = 2.0 * (series - self.global_min) / self._span - 1.0
        return caption, series, norm


def collate_joint(batch):
    captions = [b[0] for b in batch]
    raw = torch.stack([b[1] for b in batch], dim=0)
    norm = torch.stack([b[2] for b in batch], dim=0)
    return captions, raw, norm


def series_to_xts(series: torch.Tensor) -> torch.Tensor:
    return series.permute(0, 2, 1).contiguous()


def _denormalize_minmax(x: torch.Tensor, mn: float, mx: float) -> torch.Tensor:
    return ((x + 1.0) * 0.5) * (mx - mn) + mn


def _compute_global_minmax(base: TSFragmentDataset) -> tuple[float, float]:
    series = base.series.float()
    return float(series.min().item()), float(series.max().item())


def _compute_aligner_ts_stats(base: TSFragmentDataset, train_indices, min_length: int) -> tuple[float, float]:
    """Mirrors feature_alignment/train_v7.py::_compute_series_stats's mean/std
    half, on a train-only subset upsampled to `min_length` (matching Stage 1's
    own TSFragmentDataset(min_length=...) baked-in upsampling)."""
    idx = torch.as_tensor(list(train_indices), dtype=torch.long)
    series = base.series.index_select(0, idx).float()
    if series.size(-1) < min_length:
        series = F.interpolate(series.unsqueeze(1), size=min_length, mode="linear", align_corners=False).squeeze(1)
    mean = float(series.mean().item())
    std = max(float(series.std(unbiased=False).item()), 1e-5)
    return mean, std


def upsample_for_aligner(series_bvl: torch.Tensor, min_length: int) -> torch.Tensor:
    if min_length <= 0 or series_bvl.size(-1) >= min_length:
        return series_bvl
    return F.interpolate(series_bvl, size=min_length, mode="linear", align_corners=False)


# --------------------------------------------------------------------------- #
# argparse
# --------------------------------------------------------------------------- #
def int_or_none(s: str) -> int | None:
    if s is None or str(s).strip().lower() in ("", "none", "null"):
        return None
    return int(s)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("dataset")
    g.add_argument("--dataset-backend", default="tsfragment", choices=["tsfragment"])
    g.add_argument("--source", default="ETTh1", choices=SOURCES)
    g.add_argument("--length", type=int, default=24, choices=LENGTHS)
    g.add_argument("--min-length", type=int, default=64,
                   help="On-the-fly upsample target fed to the aligner's ts encoder "
                        "(and used to compute aligner GlobalZScoreNorm train stats).")
    g.add_argument("--max-rows", type=int_or_none, default=None)
    g.add_argument("--cache-dir", type=Path, default=Path("./hf_cache"))

    g = p.add_argument_group("aligner (encoder E, jointly trained)")
    g.add_argument("--llm-id", default="google/embeddinggemma-300m")
    g.add_argument("--chronos-model-id", default="amazon/chronos-2")
    g.add_argument("--proj-dim", type=int, default=512)
    g.add_argument("--llm-dtype", default="bfloat16", choices=list(DTYPES))
    g.add_argument("--hf-token-env", default="HF_TOKEN")
    g.add_argument("--init-temperature", type=float, default=0.07)
    g.add_argument("--min-temperature", type=float, default=0.03)
    g.add_argument("--tune-side", default="both", choices=["both", "text", "ts"])
    g.add_argument("--text-lora-r", type=int, default=8)
    g.add_argument("--text-lora-alpha", type=int, default=16)
    g.add_argument("--text-lora-dropout", type=float, default=0.05)
    g.add_argument("--text-lora-target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    g.add_argument("--ts-lora-r", type=int, default=8)
    g.add_argument("--ts-lora-alpha", type=int, default=16)
    g.add_argument("--ts-lora-dropout", type=float, default=0.05)
    g.add_argument("--ts-lora-target-modules", nargs="+", default=["q", "k", "v", "o"])
    g.add_argument("--aligner-lr", type=float, default=None,
                   help="Optional separate LR for the aligner (LoRA + projectors + logit_scale). "
                        "Defaults to --lr.")

    g = p.add_argument_group("text conditioning (generator side)")
    g.add_argument("--text-cond-norm", default="layernorm", choices=["none", "layernorm"])
    g.add_argument("--text-adapter-hidden-dim", type=int, default=512)
    g.add_argument("--text-adapter-dropout", type=float, default=0.0)
    g.add_argument("--cond-drop-prob", type=float, default=0.0)

    g = p.add_argument_group("generator (Stage-2-sized, jit2d)")
    g.add_argument("--model-type", default="jit2d", choices=["jit2d"])
    g.add_argument("--img-size", type=int, default=8)
    g.add_argument("--delay", type=int, default=None)
    g.add_argument("--patch-size", type=int, default=2)
    g.add_argument("--in-channels", type=int, default=1)
    g.add_argument("--hidden-channels", type=int, default=128)
    g.add_argument("--depth", type=int, default=4)
    g.add_argument("--num-heads", type=int, default=4)
    g.add_argument("--attn-dropout", type=float, default=0.1)
    g.add_argument("--proj-dropout", type=float, default=0.1)
    g.add_argument("--bottleneck-dim", type=int, default=128)
    g.add_argument("--num-register-tokens", type=int, default=0)

    g = p.add_argument_group("diffusion (flow-matching)")
    g.add_argument("--p-mean", type=float, default=0.0)
    g.add_argument("--p-std", type=float, default=1.0)
    g.add_argument("--t-eps", type=float, default=1e-5)
    g.add_argument("--noise-scale", type=float, default=1.0)
    g.add_argument("--ema-decay1", type=float, default=0.9999)
    g.add_argument("--ema-decay2", type=float, default=0.999)
    g.add_argument("--sampling-method", default="heun", choices=["euler", "heun"])
    g.add_argument("--num-sampling-steps", type=int, default=50)
    g.add_argument("--interval-min", type=float, default=0.0)
    g.add_argument("--interval-max", type=float, default=1.0)

    g = p.add_argument_group("training")
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--num-workers", type=int, default=0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="cuda")

    g = p.add_argument_group("validation")
    g.add_argument("--val-ratio", type=float, default=0.1)
    g.add_argument("--val-split-seed", type=int, default=None)
    g.add_argument("--val-seed", type=int, default=12345)
    g.add_argument("--eval-every-steps", type=int, default=0)
    g.add_argument("--enable-gen-eval", action="store_true")
    g.add_argument("--gen-eval-every-epochs", type=int, default=100)
    g.add_argument("--gen-eval-max-batches", type=int, default=0)
    g.add_argument("--metrics-aligner-ckpt", type=Path, default=None,
                   help="FIXED reference aligner (the main 'both' aligner) used only for FID/CTTP/JFTSD scoring.")
    g.add_argument("--metrics-k", type=int, default=5)
    g.add_argument("--metrics-mdd-bins", type=int, default=32)
    g.add_argument("--metrics-max-samples", type=int, default=0)
    g.add_argument("--metrics-seed", type=int, default=123)
    g.add_argument("--metrics-aligner-kind", choices=("v7", "eval"), default="v7",
                   help="'eval' loads EvalAlignerMetricEmbedder (bge-large) for scoring.")
    g.add_argument("--eval-only-ckpt-dir", type=str, default=None,
                   help="EVAL-ONLY: skip training; score every *.pt here (online/ema1/ema2) -> CSV.")
    g.add_argument("--eval-only-out-csv", type=str, default=None)

    g = p.add_argument_group("io")
    g.add_argument("--log-every", type=int, default=10)
    g.add_argument("--save", type=Path, default=Path("./denoiser_ckpt.pt"))
    g.add_argument("--save-best", type=Path, default=Path("./denoiser_best.pt"))
    g.add_argument("--save-every-epochs", type=int, default=100)

    g = p.add_argument_group("wandb")
    g.add_argument("--wandb-project", default="text-ts-ablation-joint")
    g.add_argument("--wandb-entity", default=None)
    g.add_argument("--wandb-run-name", default=None)
    g.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    g.add_argument("--wandb-tags", nargs="+", default=[])
    g.add_argument("--wandb-notes", default=None)

    return p.parse_args()


# --------------------------------------------------------------------------- #
# checkpointing
# --------------------------------------------------------------------------- #
def save_ckpt(path: Path, aligner, main_model, args: argparse.Namespace,
              seq_len: int, normalization: dict, split_info: dict,
              extra: dict | None = None) -> dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    aligner_state = {n: p.detach().cpu() for n, p in aligner.state_dict().items()}
    main_state = {n: p.detach().cpu() for n, p in main_model.state_dict().items()}
    payload = {
        "aligner_state": aligner_state,
        "main_denoiser_state": main_state,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "seq_len": seq_len,
        "normalization": normalization,
        "split_info": split_info,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return {"aligner": len(aligner_state), "main": len(main_state)}


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(aligner, main_model, loader, device: torch.device, min_length: int, seed: int) -> dict[str, float]:
    was_training_align = aligner.training
    was_training_main = main_model.training
    aligner.eval()
    main_model.eval()

    modes = ("online", "ema1", "ema2")
    sums = {f"loss_main/{m}": 0.0 for m in modes}
    sums.update({"loss_contrastive": 0.0, "mean_cos": 0.0, "acc_t2s": 0.0, "acc_s2t": 0.0})
    n_seen = 0
    seed_all(seed)
    for captions, series_raw, series_norm in loader:
        series_raw = series_raw.to(device)
        series_norm = series_norm.to(device)
        b = series_raw.size(0)

        series_hash = batch_series_hash(series_raw).to(device)
        align_out = aligner(
            captions, upsample_for_aligner(series_raw, min_length),
            series_hash=series_hash, gather=False,
        )
        text_emb = align_out["text_emb"]
        loss_ctr = align_out["loss"]
        acc_t2s = align_out["acc_t2s"]
        acc_s2t = align_out["acc_s2t"]
        sim_diag_mean = align_out["cosine"].mean()

        x_ts_main = series_to_xts(series_norm)
        for m in modes:
            if m == "online":
                loss_main = main_model(x_ts_main, text_emb)
            else:
                loss_main = main_model.forward_ema(x_ts_main, text_emb, which=m)
            sums[f"loss_main/{m}"] += loss_main.item() * b

        sums["loss_contrastive"] += loss_ctr.item() * b
        sums["mean_cos"] += sim_diag_mean.item() * b
        sums["acc_t2s"] += acc_t2s.item() * b
        sums["acc_s2t"] += acc_s2t.item() * b
        n_seen += b

    if was_training_align:
        aligner.train()
    if was_training_main:
        main_model.train()

    n = max(n_seen, 1)
    out = {k: v / n for k, v in sums.items()}
    out["loss_total/ema2"] = out["loss_contrastive"] + out["loss_main/ema2"]
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    txt_path = _resolve_txt_path(args.save)
    print(f"[ablation-txt] gen-eval metrics -> {txt_path}")

    wandb_config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    run = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name,
        mode=args.wandb_mode, tags=args.wandb_tags, notes=args.wandb_notes, config=wandb_config,
    )
    tee_log, append_record = _make_tee(txt_path, wandb.log)
    wandb.log = tee_log
    append_record({
        "kind": "meta",
        "ablation": "joint_single_stage",
        "source": args.source,
        "length": args.length,
        "seed": args.seed,
        "metrics_aligner_ckpt": str(args.metrics_aligner_ckpt) if args.metrics_aligner_ckpt else None,
        "save": str(args.save),
    })

    hf_token = os.environ.get(args.hf_token_env) or None

    # ---------- dataset ----------
    base_ds = TSFragmentDataset(
        source=args.source, length=args.length, min_length=0,
        max_rows=args.max_rows, cache_dir=args.cache_dir,
    )
    seq_len = base_ds.upsampled_length
    global_min, global_max = _compute_global_minmax(base_ds)
    ds = JointTSDataset(base_ds, global_min, global_max)
    print(f"normalize: global(whole dataset)  min={global_min:.4f}  max={global_max:.4f}")

    if args.delay is None:
        args.delay = args.img_size
    if args.delay <= 0:
        raise ValueError(f"--delay must be positive, got {args.delay}")
    coverage = args.delay * (args.img_size - 1) + args.img_size
    if coverage < seq_len:
        raise ValueError(
            f"delay*(img_size-1)+img_size ({coverage}) < seq_len ({seq_len}); "
            f"increase --img-size, or increase --delay (currently {args.delay})."
        )

    n_total = len(ds)
    n_val = min(int(round(n_total * args.val_ratio)) if args.val_ratio > 0 else 0, n_total - 1)
    n_train = n_total - n_val
    split_seed = args.seed if args.val_split_seed is None else args.val_split_seed
    if n_val > 0:
        train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(split_seed))
        train_indices = list(train_ds.indices)
        val_indices = list(val_ds.indices)
    else:
        train_ds, val_ds = ds, None
        train_indices = list(range(n_total))
        val_indices = []

    print(f"dataset: tsfragment / {args.source} L={args.length}  ->  seq_len={seq_len}")
    print(f"size : {n_total} total ({n_train} train / {n_val} val)")

    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_joint, num_workers=args.num_workers, drop_last=True,
    )
    val_loader = None
    metric_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_joint, num_workers=args.num_workers, drop_last=False,
        )
        metric_loader = DataLoader(
            Subset(base_ds, val_indices), batch_size=args.batch_size, shuffle=False,
            collate_fn=collate, num_workers=args.num_workers, drop_last=False,
        )

    # ---------- aligner normalization stats (mirrors feature_alignment/train_v7.py) ----------
    ts_mean, ts_std = _compute_aligner_ts_stats(base_ds, train_indices, args.min_length)
    print(
        f"aligner GlobalZScoreNorm stats (train split, upsampled to {args.min_length}): "
        f"mean={ts_mean:.6f} std={ts_std:.6f}"
    )

    # ---------- aligner (encoder E, jointly trained) ----------
    aligner = TextToTimeSeriesAlignerV7(
        llm_id=args.llm_id, chronos_model_id=args.chronos_model_id, proj_dim=args.proj_dim,
        llm_dtype=DTYPES[args.llm_dtype], device=device,
        init_temperature=args.init_temperature, min_temperature=args.min_temperature,
        hf_token=hf_token, tune_side=args.tune_side,
        text_lora_r=args.text_lora_r, text_lora_alpha=args.text_lora_alpha,
        text_lora_dropout=args.text_lora_dropout,
        text_lora_target_modules=tuple(args.text_lora_target_modules),
        ts_lora_r=args.ts_lora_r, ts_lora_alpha=args.ts_lora_alpha, ts_lora_dropout=args.ts_lora_dropout,
        ts_lora_target_modules=tuple(args.ts_lora_target_modules),
        ts_global_mean=ts_mean, ts_global_std=ts_std,
    )
    text_embed_dim = aligner.text_proj.out_features

    # ---------- generator -- unmodified train_cond_v6 builders, same size as Stage 2 ----------
    base_denoiser = _v6.build_denoiser(args, text_embed_dim=text_embed_dim, seq_len=seq_len, device=device)
    text_conditioner = _v6.TextConditioner(
        in_dim=text_embed_dim, norm_mode=args.text_cond_norm,
        adapter_hidden_dim=args.text_adapter_hidden_dim, adapter_dropout=args.text_adapter_dropout,
    ).to(device)
    for p in text_conditioner.parameters():
        p.requires_grad_(False)
    if args.cond_drop_prob > 0.0:
        text_conditioner.null_embed.requires_grad_(True)
    main_model = _v6.DenoiserWithTextConditioner(base_denoiser, text_conditioner).to(device)

    n_align = sum(p.numel() for p in aligner.parameters() if p.requires_grad)
    n_main = sum(p.numel() for p in main_model.parameters() if p.requires_grad)
    print(f"trainable params: aligner={n_align:,}  generator={n_main:,}")

    # ---------- optimizer (single optimizer over ALL trainable params -- the actual "joint" part) ----------
    aligner_lr = args.lr if args.aligner_lr is None else float(args.aligner_lr)
    param_groups = [
        {"params": [p for p in aligner.parameters() if p.requires_grad], "lr": aligner_lr},
        {"params": [p for p in main_model.parameters() if p.requires_grad], "lr": args.lr},
    ]
    opt = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    all_trainable = [p for group in param_groups for p in group["params"]]
    print(f"optimizer lr: aligner={aligner_lr:.2e}  generator={args.lr:.2e}")

    # ---------- metrics embedder (FIXED reference aligner, for FID/CTTP/JFTSD only) ----------
    metric_embedder = None
    if args.enable_gen_eval:
        if args.metrics_aligner_ckpt is not None:
            print(f"[metrics] loading fixed reference aligner from {args.metrics_aligner_ckpt}")
            if args.metrics_aligner_kind == "eval":
                from eval_aligner import EvalAlignerMetricEmbedder  # feature_alignment/ on sys.path via _v6
                metric_embedder = EvalAlignerMetricEmbedder(
                    ckpt_path=args.metrics_aligner_ckpt, device=device, hf_token=hf_token)
            else:
                metric_embedder = AlignerV7MetricEmbedder(
                    ckpt_path=args.metrics_aligner_ckpt, device=device, hf_token=hf_token,
                    ts_global_mean=ts_mean, ts_global_std=ts_std,
                )
        else:
            print("[metrics] --metrics-aligner-ckpt not set; generation metrics will be skipped.")

    normalization = {"scheme": "minmax_-1_1", "mode": "global", "global_min": global_min, "global_max": global_max}
    split_info = {
        "split_seed": split_seed, "val_ratio": args.val_ratio, "n_total": n_total,
        "n_train": n_train, "n_val": n_val, "train_indices": train_indices, "val_indices": val_indices,
        "source": args.source, "length": args.length, "min_length": args.min_length,
    }

    best_val_loss = float("inf")

    def run_validation(step_now: int, epoch_now: int) -> None:
        nonlocal best_val_loss
        if val_loader is None:
            return
        metrics = evaluate(aligner, main_model, val_loader, device, min_length=args.min_length, seed=args.val_seed)
        wandb.log({f"val/{k}": v for k, v in metrics.items()} | {"val/epoch": epoch_now}, step=step_now)
        print(
            f"  [val] epoch {epoch_now} step {step_now:>5} "
            f"total={metrics['loss_total/ema2']:.4f} main_ema2={metrics['loss_main/ema2']:.4f} "
            f"ctr={metrics['loss_contrastive']:.4f}"
        )
        if metrics["loss_total/ema2"] < best_val_loss:
            best_val_loss = metrics["loss_total/ema2"]
            n_tensors = save_ckpt(
                args.save_best, aligner, main_model, args, seq_len, normalization, split_info,
                extra={"val_loss_total_ema2": best_val_loss, "step": step_now, "epoch": epoch_now},
            )
            wandb.log({"val/best_loss_total_ema2": best_val_loss}, step=step_now)
            print(f"        -> new best val/loss_total(ema2) {best_val_loss:.4f}  saved {n_tensors} -> {args.save_best}")

    def run_generation_metrics(step_now: int, epoch_now: int) -> None:
        if not args.enable_gen_eval or args.gen_eval_every_epochs <= 0:
            return
        if (epoch_now + 1) % args.gen_eval_every_epochs != 0:
            return
        if metric_loader is None or metric_embedder is None:
            return

        seed_all(args.metrics_seed + epoch_now)
        was_align, was_main = aligner.training, main_model.training
        aligner.eval()
        main_model.eval()

        modes = ("online", "ema1", "ema2")
        mode_metrics = {}
        for m in modes:
            acc = ConTSGMetricsAccumulator(
                embedder=metric_embedder,
                cfg=MetricConfig(
                    k=args.metrics_k, mdd_bins=args.metrics_mdd_bins,
                    max_samples=args.metrics_max_samples, seed=args.metrics_seed + epoch_now,
                ),
            )
            with torch.no_grad():
                for bi, (captions, series) in enumerate(metric_loader):
                    if args.gen_eval_max_batches > 0 and bi >= args.gen_eval_max_batches:
                        break
                    real_bvl = series.to(device).float()
                    text_emb = aligner.encode_text(captions)
                    if m == "online":
                        gen_blv = main_model.generate(text_emb).float()
                    else:
                        gen_blv = main_model.generate_ema(text_emb, which=m).float()
                    gen_bvl = _denormalize_minmax(gen_blv.permute(0, 2, 1).contiguous(), global_min, global_max)
                    real_for_align = upsample_for_aligner(real_bvl, args.min_length)
                    gen_for_align = upsample_for_aligner(gen_bvl, args.min_length)
                    acc.update(captions=captions, real_series_bvl=real_for_align, gen_series_bvl=gen_for_align)
            mode_metrics[m] = acc.compute()

        payload = {}
        for m in modes:
            payload.update({f"gen_eval/{m}/{k}": v for k, v in mode_metrics[m].items()})
        payload["gen_eval/epoch"] = epoch_now
        wandb.log(payload, step=step_now)
        print(
            f"  [gen-eval] epoch {epoch_now}  "
            f"online_fid={mode_metrics['online']['fidelity/fid']:.4f}  "
            f"ema2_fid={mode_metrics['ema2']['fidelity/fid']:.4f}"
        )

        if was_align:
            aligner.train()
        if was_main:
            main_model.train()

    # ---------- EVAL-ONLY: score saved checkpoints, no training ----------
    if args.eval_only_ckpt_dir is not None:
        import glob as _glob
        import csv as _csv
        assert metric_loader is not None and metric_embedder is not None, (
            "eval-only requires --enable-gen-eval and --metrics-aligner-ckpt")
        ckpts = sorted(_glob.glob(os.path.join(args.eval_only_ckpt_dir, "*.pt")))
        out_csv = args.eval_only_out_csv or os.path.join(
            args.eval_only_ckpt_dir, "eval_evalaligner_metrics.csv")
        modes = ("online", "ema1", "ema2")
        fields = ["ckpt", "epoch", "mode", "fid", "cttp", "jftsd"]
        rows: list[dict] = []
        aligner.eval(); main_model.eval()
        print(f"[eval-only] scoring {len(ckpts)} ckpts x {len(modes)} modes "
              f"(aligner={args.metrics_aligner_ckpt}) -> {out_csv}", flush=True)
        for cp in ckpts:
            payload = torch.load(cp, map_location=device, weights_only=False)
            mstate = payload.get("main_denoiser_state", payload.get("denoiser_state"))
            if mstate is None:
                print(f"[eval-only] SKIP {os.path.basename(cp)} (no main_denoiser_state)")
                continue
            if "aligner_state" in payload:
                am, au = aligner.load_state_dict(payload["aligner_state"], strict=False)
                if au:
                    print(f"[eval-only] WARN {os.path.basename(cp)}: aligner {len(au)} unexpected keys")
            mm, mu = main_model.load_state_dict(mstate, strict=False)
            if mm or mu:
                print(f"[eval-only] WARN {os.path.basename(cp)}: main {len(mm)} missing / {len(mu)} unexpected keys")
            epoch_tag = payload.get("epoch", "")
            for m in modes:
                seed_all(args.metrics_seed)
                acc = ConTSGMetricsAccumulator(
                    embedder=metric_embedder,
                    cfg=MetricConfig(k=args.metrics_k, mdd_bins=args.metrics_mdd_bins,
                                     max_samples=args.metrics_max_samples, seed=args.metrics_seed))
                with torch.no_grad():
                    for captions, series in metric_loader:
                        real_bvl = series.to(device).float()
                        text_emb = aligner.encode_text(captions)
                        if m == "online":
                            gen_blv = main_model.generate(text_emb).float()
                        else:
                            gen_blv = main_model.generate_ema(text_emb, which=m).float()
                        gen_bvl = _denormalize_minmax(gen_blv.permute(0, 2, 1).contiguous(), global_min, global_max)
                        acc.update(captions=captions,
                                   real_series_bvl=upsample_for_aligner(real_bvl, args.min_length),
                                   gen_series_bvl=upsample_for_aligner(gen_bvl, args.min_length))
                met = acc.compute()
                rows.append({"ckpt": os.path.basename(cp), "epoch": epoch_tag, "mode": m,
                             "fid": met["fidelity/fid"], "cttp": met["adherence/cttp"],
                             "jftsd": met["adherence/jftsd"]})
                print(f"[eval-only] {os.path.basename(cp):30s} {m:6s} "
                      f"fid={met['fidelity/fid']:.4f} cttp={met['adherence/cttp']:.4f} "
                      f"jftsd={met['adherence/jftsd']:.4f}", flush=True)
            with open(out_csv, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        print(f"[eval-only] DONE wrote {len(rows)} rows -> {out_csv}", flush=True)
        return

    # ---------- train ----------
    aligner.train()
    main_model.train()
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for captions, series_raw, series_norm in loader:
            series_raw = series_raw.to(device)
            series_norm = series_norm.to(device)

            series_hash = batch_series_hash(series_raw).to(device)
            align_out = aligner(
                captions, upsample_for_aligner(series_raw, args.min_length),
                series_hash=series_hash, gather=False,
            )
            text_emb = align_out["text_emb"]
            loss_ctr = align_out["loss"]

            drop_mask = None
            if args.cond_drop_prob > 0.0:
                drop_mask = torch.rand(text_emb.size(0), device=device) < args.cond_drop_prob
            x_ts_main = series_to_xts(series_norm)
            loss_main = main_model(x_ts_main, text_emb, drop_mask=drop_mask)

            total = loss_ctr + loss_main

            opt.zero_grad()
            total.backward()
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable, args.grad_clip)
            else:
                grad_norm = torch.tensor(float("nan"))
            opt.step()
            main_model.update_ema()
            step += 1

            acc_t2s = align_out["acc_t2s"].item()
            acc_s2t = align_out["acc_s2t"].item()

            wandb.log({
                "train/loss_total": total.item(),
                "train/loss_contrastive": loss_ctr.item(),
                "train/loss_main": loss_main.item(),
                "train/acc_t2s": acc_t2s,
                "train/acc_s2t": acc_s2t,
                "train/mean_cos": align_out["cosine"].mean().item(),
                "train/grad_norm": float(grad_norm),
                "train/lr_aligner": opt.param_groups[0]["lr"],
                "train/lr_main": opt.param_groups[1]["lr"],
                "train/epoch": epoch,
            }, step=step)
            if step % args.log_every == 0:
                dt = time.time() - t0
                print(
                    f"epoch {epoch}  step {step:>5}  total={total.item():.4f}  "
                    f"ctr={loss_ctr.item():.4f}  main={loss_main.item():.4f}  "
                    f"acc_t2s={acc_t2s:.3f}  acc_s2t={acc_s2t:.3f}  gn={float(grad_norm):.3f}  ({dt:.1f}s)"
                )

            if args.eval_every_steps > 0 and step % args.eval_every_steps == 0:
                run_validation(step, epoch)

        run_validation(step, epoch)
        run_generation_metrics(step, epoch)

        if args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0:
            ckpt_path = args.save.with_name(f"{args.save.stem}_epoch{epoch + 1}{args.save.suffix}")
            n_tensors = save_ckpt(
                ckpt_path, aligner, main_model, args, seq_len, normalization, split_info,
                extra={"step": step, "epoch": epoch},
            )
            print(f"  [ckpt] epoch {epoch + 1}  saved {n_tensors} -> {ckpt_path}")

    n_tensors = save_ckpt(
        args.save, aligner, main_model, args, seq_len, normalization, split_info,
        extra={"final_step": step, "final_epoch": args.epochs - 1},
    )
    print(f"saved final state ({n_tensors}) -> {args.save.resolve()}")
    if val_loader is not None:
        print(f"best val/loss_total(ema2) = {best_val_loss:.4f}  -> {args.save_best.resolve()}")

    if run is not None and args.wandb_mode != "disabled":
        if args.save.exists():
            art = wandb.Artifact(name=f"joint-last-{run.id}", type="model")
            art.add_file(str(args.save))
            run.log_artifact(art)
        if args.save_best.exists():
            art = wandb.Artifact(name=f"joint-best-{run.id}", type="model")
            art.add_file(str(args.save_best))
            run.log_artifact(art)
    wandb.finish()


if __name__ == "__main__":
    main()
