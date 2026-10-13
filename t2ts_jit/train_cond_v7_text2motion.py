"""Train the Text2Motion baseline (frozen movement AE + text-conditioned
autoregressive CVAE generator) with the SAME generation eval as ``train_cond_v7.py``.

This is the conditional / Stage-2 trainer for the Text2Motion baseline. It is the
Text2Motion analogue of ``train_cond_v7_t2s.py`` and is kept deliberately
line-for-line comparable with it (same data rows, same val split, same
EMA + AlignerV7 ConTSG generation metrics) so the two baselines are directly
comparable. The only substantive swaps are the model and its train/gen ops:

  * **Frozen movement AE.** A per-(source,length) MovementConvEncoder/Decoder
    pretrained by ``pretrain_ae_text2motion.py``; its stored ``(min,max)`` defines
    the [0,1] scaling the encoder sees (in-distribution parity).
  * **Text2Motion generator.** Trained with the CVAE recon + KL objective
    (posterior path, teacher forcing). No diffusion / flow; generation is the
    prior-only autoregressive rollout decoded back to a series.
  * **Native condition.** Conditioned on the dataset's precomputed 128-d OpenAI
    ``TextEmbedding`` column (same as the T2S baseline). Captions are kept only for
    the generation metrics.
  * **v7 generation metrics.** ``ConTSGMetricsAccumulator`` + ``AlignerV7MetricEmbedder``
    on raw-scale series, upsampled to ``--aligner-min-length`` — identical to
    ``train_cond_v7_t2s.py``.

Example:
    python t2ts_jit/train_cond_v7_text2motion.py --source ETTm1 --length 24 \
        --pretrained-ae-path t2ts_jit/ae_ckpts_text2motion/ae_ETTm1_L24_ep2000/best_model.pth \
        --metrics-aligner-ckpt feature_alignment/results_aligner_v7/ETTm1_L24/aligner_v7_best.pt \
        --enable-gen-eval
"""

from __future__ import annotations

import argparse
import ast
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset, Subset, random_split

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_FA_DIR = _REPO_ROOT / "feature_alignment"
_JIT_DIR = _REPO_ROOT / "t2ts_jit"
for _d in (_FA_DIR, _JIT_DIR):
    sys.path.insert(0, str(_d))

from cond_jit_basics.text2motion import Text2Motion, Text2MotionConfig  # noqa: E402
from contsg_v4_metrics import ConTSGMetricsAccumulator, MetricConfig, seed_all  # noqa: E402

REPO_ID = "WinfredGe/TSFragment-600K"
SOURCES = ("ETTh1", "ETTh1s", "ETTm1", "airquality", "electricity", "exchangerate", "traffic")
LENGTHS = (24, 48, 96)

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def _parse_floats(s) -> np.ndarray:
    if isinstance(s, (list, tuple, np.ndarray)):
        return np.asarray(s, dtype=np.float32)
    s = str(s).strip()
    if "," in s:
        return np.asarray(ast.literal_eval(s), dtype=np.float32)
    return np.asarray([float(x) for x in s.replace("[", "").replace("]", "").split()], dtype=np.float32)


# ====================================================================== data ===
class TSFragmentWithEmbedding(Dataset):
    """One (source, length) split of TSFragment-600K with the OpenAI embedding.

    Row selection mirrors v7 / T2S exactly (``dropna`` on ``["Text","OT"]`` only),
    so a seeded ``random_split`` yields the same val subset. ``__getitem__`` returns
    ``(caption, emb_128, normed_series, raw_series)`` with ``normed_series`` in [0,1]
    (fed to the frozen AE) and ``raw_series`` at original scale (for the aligner).
    """

    def __init__(self, source: str, length: int, max_rows: int | None, cache_dir: Path,
                 global_min: float | None, global_max: float | None, normalize: str = "global"):
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}, expected one of {SOURCES}")
        if length not in LENGTHS:
            raise ValueError(f"unknown length {length!r}, expected one of {LENGTHS}")
        if normalize not in ("none", "global"):
            raise ValueError("normalize must be one of: none, global")

        fname = f"embedding_cleaned_{source}_{length}.csv"
        local = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=fname, cache_dir=str(cache_dir))
        df = pd.read_csv(local, usecols=["Text", "TextEmbedding", "OT"], nrows=max_rows)
        df = df.dropna(subset=["Text", "OT"]).reset_index(drop=True)
        if df["TextEmbedding"].isna().any():
            n_bad = int(df["TextEmbedding"].isna().sum())
            raise ValueError(
                f"{n_bad} rows have null TextEmbedding but valid Text/OT in {fname}; "
                "this would break v7 split alignment. Inspect the CSV before proceeding."
            )

        self.captions: list[str] = df["Text"].astype(str).tolist()
        self.series = torch.from_numpy(np.stack([_parse_floats(s) for s in df["OT"]])).float()  # (N, L) raw
        self.embeddings = torch.from_numpy(np.stack([_parse_floats(s) for s in df["TextEmbedding"]])).float()
        self.embedding_dim = int(self.embeddings.shape[1])
        self.source = source
        self.length = length
        self.normalize = normalize

        if normalize == "global":
            self.global_min = float(global_min) if global_min is not None else float(self.series.min().item())
            self.global_max = float(global_max) if global_max is not None else float(self.series.max().item())
        else:
            self.global_min = None
            self.global_max = None

    def __len__(self) -> int:
        return len(self.captions)

    def _normed(self, raw: torch.Tensor) -> torch.Tensor:
        if self.normalize == "none":
            return raw
        span = max(self.global_max - self.global_min, 1e-6)
        return (raw - self.global_min) / span

    def denormalize(self, normed_bvl: torch.Tensor) -> torch.Tensor:
        """Map [0,1]-space series (B,V,L) back to raw scale for the aligner."""
        if self.normalize == "none":
            return normed_bvl
        span = max(self.global_max - self.global_min, 1e-6)
        return normed_bvl * span + self.global_min

    def __getitem__(self, idx: int):
        raw = self.series[idx]
        return self.captions[idx], self.embeddings[idx], self._normed(raw), raw


def collate_train(batch):
    captions = [x[0] for x in batch]
    emb = torch.stack([x[1] for x in batch], dim=0)
    normed = torch.stack([x[2] for x in batch], dim=0)
    return captions, emb, normed


def collate_metric(batch):
    captions = [x[0] for x in batch]
    emb = torch.stack([x[1] for x in batch], dim=0)
    raw = torch.stack([x[3] for x in batch], dim=0).unsqueeze(1)  # (B, 1, L) raw
    return captions, emb, raw


# ====================================================================== ema ====
class EMAHelper:
    def __init__(self, module: nn.Module, decay: float):
        self.module = module
        self.decay = float(decay)
        self.shadow = [p.detach().clone() for p in self.module.parameters()]

    @torch.no_grad()
    def update(self):
        for s, p in zip(self.shadow, self.module.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self):
        self.backup = [p.detach().clone() for p in self.module.parameters()]
        for p, s in zip(self.module.parameters(), self.shadow):
            p.data.copy_(s.data)

    @torch.no_grad()
    def swap_out(self):
        for p, b in zip(self.module.parameters(), self.backup):
            p.data.copy_(b.data)
        del self.backup

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": [t.detach().cpu() for t in self.shadow]}


# ============================================================ aligner-v7 metric =
def _load_v7_ckpt(ckpt_path: Path) -> tuple[dict, dict]:
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    saved_args = payload["args"]
    ckpt_state = payload.get("aligner_trainable_state", payload.get("trainable_state"))
    if ckpt_state is None:
        raise KeyError("checkpoint missing aligner_trainable_state/trainable_state")
    return saved_args, ckpt_state


def _build_v7_aligner(saved_args: dict, ckpt_state: dict, device: torch.device,
                      hf_token: str | None, ts_global_mean: float, ts_global_std: float):
    from aligner_v7 import TextToTimeSeriesAlignerV7

    llm_dtype = DTYPES.get(saved_args.get("llm_dtype", "bfloat16"), torch.bfloat16)
    model = TextToTimeSeriesAlignerV7(
        llm_id=saved_args["llm_id"],
        chronos_model_id=saved_args["chronos_model_id"],
        proj_dim=int(saved_args["proj_dim"]),
        llm_dtype=llm_dtype,
        device=device,
        init_temperature=float(saved_args["init_temperature"]),
        min_temperature=float(saved_args["min_temperature"]),
        hf_token=hf_token,
        text_lora_r=int(saved_args.get("text_lora_r", 8)),
        text_lora_alpha=int(saved_args.get("text_lora_alpha", 16)),
        text_lora_dropout=float(saved_args.get("text_lora_dropout", 0.05)),
        text_lora_target_modules=tuple(
            saved_args.get("text_lora_target_modules", ("q_proj", "k_proj", "v_proj", "o_proj"))
        ),
        ts_lora_r=int(saved_args.get("ts_lora_r", 8)),
        ts_lora_alpha=int(saved_args.get("ts_lora_alpha", 16)),
        ts_lora_dropout=float(saved_args.get("ts_lora_dropout", 0.05)),
        ts_lora_target_modules=tuple(saved_args.get("ts_lora_target_modules", ("q", "k", "v", "o"))),
        ts_global_mean=float(ts_global_mean),
        ts_global_std=float(ts_global_std),
    )
    model_state = model.state_dict()
    filtered = {k: v for k, v in ckpt_state.items()
                if (k in model_state and tuple(model_state[k].shape) == tuple(v.shape))}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return model, filtered, missing, unexpected


class AlignerV7MetricEmbedder:
    """Full TextToTimeSeriesAlignerV7 used as the metric embedding backend (same as v7)."""

    def __init__(self, ckpt_path: Path, device: torch.device, ts_global_mean: float,
                 ts_global_std: float, hf_token: str | None = None):
        saved_args, ckpt_state = _load_v7_ckpt(Path(ckpt_path))
        self.model, filtered, missing, unexpected = _build_v7_aligner(
            saved_args, ckpt_state, device=device, hf_token=hf_token,
            ts_global_mean=ts_global_mean, ts_global_std=ts_global_std,
        )
        print(f"[AlignerV7MetricEmbedder] matched={len(filtered)} missing={len(missing)} "
              f"unexpected={len(unexpected)} stats(mean={ts_global_mean:.6f}, std={ts_global_std:.6f})")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = torch.device(device)

    @torch.no_grad()
    def encode_text(self, captions):
        return self.model.encode_text(list(captions))

    @torch.no_grad()
    def encode_series(self, series_bvl: torch.Tensor):
        return self.model.encode_time_series(series_bvl)


@torch.no_grad()
def evaluate_aligner_retrieval(aligner_model: nn.Module, loader, device: torch.device,
                               min_length: int, max_batches: int = 0):
    total = correct_t2s = correct_s2t = 0
    cosine_sum = 0.0
    for bi, (captions, _emb, series) in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        series = series.to(device)
        if series.size(-1) < min_length:
            series = F.interpolate(series, size=min_length, mode="linear", align_corners=False)
        out = aligner_model(captions, series, gather=False)
        b = len(captions)
        correct_t2s += int((out["acc_t2s"] * b).item())
        correct_s2t += int((out["acc_s2t"] * b).item())
        cosine_sum += float(out["cosine"].mean().item()) * b
        total += b
    if total == 0:
        return {"acc_t2s": 0.0, "acc_s2t": 0.0, "mean_cosine": 0.0}
    return {"acc_t2s": correct_t2s / total, "acc_s2t": correct_s2t / total, "mean_cosine": cosine_sum / total}


def compute_series_stats(series: torch.Tensor) -> tuple[float, float]:
    series = series.float()
    mean = float(series.mean().item())
    std = max(float(series.std(unbiased=False).item()), 1e-5)
    return mean, std


# ===================================================================== ckpt ====
def save_ckpt(path: Path, args, model: nn.Module, optimizer, ema1: EMAHelper, ema2: EMAHelper,
              cfg: Text2MotionConfig, split_info: dict, normalization: dict, extra: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_cfg": cfg.to_dict(),
        "optimizer_state": optimizer.state_dict(),
        "ema1_state": ema1.state_dict(),
        "ema2_state": ema2.state_dict(),
        "split_info": split_info,
        "normalization": normalization,
        "text_encoder_kind": "t2s-openai-textembedding",
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source", type=str, default="ETTm1", choices=SOURCES)
    p.add_argument("--length", type=int, default=24, choices=LENGTHS)
    p.add_argument("--max-rows", type=int, default=None, help="None = full (matches v7).")
    p.add_argument("--cache-dir", type=Path, default=Path("./hf_cache"))
    p.add_argument("--normalize", type=str, default="global", choices=["none", "global"])

    p.add_argument("--pretrained-ae-path", type=Path, required=True,
                   help="movement AE best_model.pth from pretrain_ae_text2motion.py (same source+length).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--val-split-seed", type=int, default=0)
    p.add_argument("--val-seed", type=int, default=12345)
    p.add_argument("--eval-every-steps", type=int, default=0)

    p.add_argument("--ema-decay1", type=float, default=0.9999)
    p.add_argument("--ema-decay2", type=float, default=0.999)

    # Text2Motion generator hyper-parameters / losses
    p.add_argument("--text-latent-dim", type=int, default=512)
    p.add_argument("--dim-att-vec", type=int, default=512)
    p.add_argument("--dim-z", type=int, default=128)
    p.add_argument("--dim-pri-hidden", type=int, default=1024)
    p.add_argument("--dim-pos-hidden", type=int, default=1024)
    p.add_argument("--dim-dec-hidden", type=int, default=1024)
    p.add_argument("--n-layers-pri", type=int, default=1)
    p.add_argument("--n-layers-pos", type=int, default=1)
    p.add_argument("--n-layers-dec", type=int, default=1)
    p.add_argument("--lambda-rec-ts", type=float, default=1.0)
    p.add_argument("--lambda-rec-mov", type=float, default=1.0)
    p.add_argument("--lambda-kld", type=float, default=0.005)
    p.add_argument("--teacher-forcing-ratio", type=float, default=0.4)
    p.add_argument("--no-detach-movement-latents", dest="detach_movement_latents",
                   action="store_false", default=True)

    # generation metrics (v7-aligned ConTSG suite)
    p.add_argument("--enable-gen-eval", action="store_true", default=False)
    p.add_argument("--gen-eval-every-epochs", type=int, default=100)
    p.add_argument("--gen-eval-max-batches", type=int, default=0)
    p.add_argument("--metrics-aligner-ckpt", "--metrics-aligner-v7-ckpt",
                   dest="metrics_aligner_ckpt", type=Path, default=None)
    p.add_argument("--metrics-k", type=int, default=5)
    p.add_argument("--metrics-mdd-bins", type=int, default=32)
    p.add_argument("--metrics-max-samples", type=int, default=0, help="<=0 = full val set.")
    p.add_argument("--metrics-seed", type=int, default=123)
    p.add_argument("--aligner-min-length", type=int, default=64)
    p.add_argument("--check-aligner-max-batches", type=int, default=0)
    p.add_argument("--hf-token-env", type=str, default="HF_TOKEN")

    p.add_argument("--save", type=Path, default=Path("./t2ts_jit/results_text2motion_baseline/gen_ckpt.pt"))
    p.add_argument("--save-best", type=Path, default=Path("./t2ts_jit/results_text2motion_baseline/gen_best.pt"))
    p.add_argument("--save-every-epochs", type=int, default=100)
    p.add_argument("--log-every", type=int, default=10)

    p.add_argument("--wandb-project", default="text-ts-formal-experiment")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb-tags", nargs="+", default=[])
    p.add_argument("--wandb-notes", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    hf_token = os.environ.get(args.hf_token_env) or None

    wandb.init(
        project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name,
        mode=args.wandb_mode, tags=args.wandb_tags, notes=args.wandb_notes,
        config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    )

    # ---------- frozen movement AE (load first; its stats define [0,1]) ----------
    ae_obj = torch.load(str(args.pretrained_ae_path), map_location="cpu", weights_only=False)
    if not (isinstance(ae_obj, dict) and "state_dict" in ae_obj and "cfg" in ae_obj):
        raise ValueError("pretrained AE checkpoint must contain 'state_dict' and 'cfg' "
                         "(produced by pretrain_ae_text2motion.py).")
    ae_cfg = Text2MotionConfig.from_dict(ae_obj["cfg"])
    norm_stats = ae_obj.get("norm_stats")
    if isinstance(norm_stats, dict) and norm_stats.get("global_min") is not None:
        g_min, g_max = float(norm_stats["global_min"]), float(norm_stats["global_max"])
        print(f"[norm] using AE-stored stats for L={args.length}: min={g_min:.6f} max={g_max:.6f}")
    else:
        g_min = g_max = None
        print("[norm] AE checkpoint has no stats; computing min/max from this split.")

    # ---------- dataset (v7-aligned rows + split) ----------
    full_ds = TSFragmentWithEmbedding(
        source=args.source, length=args.length, max_rows=args.max_rows, cache_dir=args.cache_dir,
        global_min=g_min, global_max=g_max, normalize=args.normalize,
    )
    print(f"[data] {args.source} L{args.length} N={len(full_ds)} text_emb_dim={full_ds.embedding_dim}")
    if full_ds.embedding_dim != 128:
        raise ValueError(f"unexpected TextEmbedding dim {full_ds.embedding_dim}; expected 128 (OpenAI).")

    n_total = len(full_ds)
    n_val = int(round(n_total * args.val_ratio)) if args.val_ratio > 0 else 0
    n_val = min(n_val, max(n_total - 1, 0))
    n_train = n_total - n_val
    split_seed = args.seed if args.val_split_seed is None else args.val_split_seed
    if n_val > 0:
        train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                        generator=torch.Generator().manual_seed(split_seed))
        train_indices = list(train_ds.indices)
        val_indices = list(val_ds.indices)
    else:
        train_ds, val_ds = full_ds, None
        train_indices = list(range(n_total))
        val_indices = []

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, collate_fn=collate_train)
    val_loader = metric_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, drop_last=False, collate_fn=collate_train)
        metric_loader = DataLoader(Subset(full_ds, val_indices), batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, drop_last=False, collate_fn=collate_metric)

    # ---------- Text2Motion model (frozen AE + trainable generator) ----------
    cfg = Text2MotionConfig(
        n_var=1,
        unit_length=ae_cfg.unit_length,
        dim_movement_latent=ae_cfg.dim_movement_latent,
        dim_movement_enc_hidden=ae_cfg.dim_movement_enc_hidden,
        dim_movement_dec_hidden=ae_cfg.dim_movement_dec_hidden,
        text_input_dim=full_ds.embedding_dim,
        text_latent_dim=args.text_latent_dim,
        dim_att_vec=args.dim_att_vec,
        dim_z=args.dim_z,
        n_layers_pri=args.n_layers_pri,
        n_layers_pos=args.n_layers_pos,
        n_layers_dec=args.n_layers_dec,
        dim_pri_hidden=args.dim_pri_hidden,
        dim_pos_hidden=args.dim_pos_hidden,
        dim_dec_hidden=args.dim_dec_hidden,
        lambda_rec_ts=args.lambda_rec_ts,
        lambda_rec_mov=args.lambda_rec_mov,
        lambda_kld=args.lambda_kld,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        detach_movement_latents=args.detach_movement_latents,
    )
    model = Text2Motion(cfg).to(device)
    model.load_movement_ae(ae_obj["state_dict"], strict=True)
    model.set_stage("finetune")  # freeze AE, train generator

    # Sanity: confirm the frozen AE round-trips the expected latent geometry.
    with torch.no_grad():
        probe_x = full_ds[0][2].unsqueeze(0).unsqueeze(-1).to(device).float()  # (1, L, 1) normalized
        probe_z = model.mov_enc(probe_x)
    print(f"[latent] movement shape = {tuple(probe_z.shape)} "
          f"(expect (1, {args.length // cfg.unit_length}, {cfg.dim_movement_latent}))")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Plain Adam to match the Text2Motion Comp baseline (CompTrainerV6 uses
    # optim.Adam, no weight decay); weight_decay defaults to 0.
    opt = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    ema1 = EMAHelper(model, decay=args.ema_decay1)
    ema2 = EMAHelper(model, decay=args.ema_decay2)
    print(f"[model] trainable params = {sum(p.numel() for p in trainable_params)}")

    split_info = {
        "split_seed": split_seed, "val_ratio": args.val_ratio,
        "n_total": n_total, "n_train": n_train, "n_val": n_val,
        "train_indices": train_indices, "val_indices": val_indices,
        "source": args.source, "length": args.length, "max_rows": args.max_rows,
    }
    normalization = {
        "scheme": "minmax_0_1", "mode": args.normalize,
        "global_min": full_ds.global_min, "global_max": full_ds.global_max,
    }

    # ---------- aligner-v7 metric embedder + stats (raw train split) ----------
    metric_embedder = None
    if args.enable_gen_eval and args.metrics_aligner_ckpt is not None:
        ts_mean, ts_std = compute_series_stats(full_ds.series.index_select(0, torch.as_tensor(train_indices)))
        print(f"[aligner-stats] mean={ts_mean:.6f} std={ts_std:.6f} (raw train split)")
        metric_embedder = AlignerV7MetricEmbedder(
            args.metrics_aligner_ckpt, device=device,
            ts_global_mean=ts_mean, ts_global_std=ts_std, hf_token=hf_token,
        )
        if metric_loader is not None:
            retr = evaluate_aligner_retrieval(metric_embedder.model, metric_loader, device,
                                              min_length=args.aligner_min_length,
                                              max_batches=args.check_aligner_max_batches)
            print(f"[sanity] aligner mean_cos={retr['mean_cosine']:.4f} "
                  f"acc_t2s={retr['acc_t2s']:.4f} acc_s2t={retr['acc_s2t']:.4f}")
            wandb.log({"sanity/aligner_mean_cosine": retr["mean_cosine"],
                       "sanity/aligner_acc_t2s": retr["acc_t2s"],
                       "sanity/aligner_acc_s2t": retr["acc_s2t"]}, step=0)
    elif args.enable_gen_eval:
        print("[gen-eval] enabled but --metrics-aligner-ckpt not set; metrics will be skipped.")

    # ---------- eval helpers ----------
    @torch.no_grad()
    def eval_loss(mode: str, seed: int):
        if val_loader is None:
            return {"loss": float("nan")}
        model.eval()
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        if mode == "ema1":
            ema1.swap_in()
        elif mode == "ema2":
            ema2.swap_in()
        total, n_seen = 0.0, 0
        for _captions, emb, series in val_loader:
            emb = emb.to(device).float()
            x_1 = series.to(device).float().unsqueeze(-1)  # (B, L, 1)
            loss = model.forward_cond(x_1, emb)["loss"]
            total += float(loss.item()) * x_1.size(0)
            n_seen += x_1.size(0)
        if mode == "ema1":
            ema1.swap_out()
        elif mode == "ema2":
            ema2.swap_out()
        model.train()
        return {"loss": total / max(n_seen, 1)}

    @torch.no_grad()
    def generate_normed(emb: torch.Tensor, mode: str, out_len: int) -> torch.Tensor:
        """Prior-only autoregressive sampling, decode to [0,1]-space series (B, 1, out_len)."""
        model.eval()
        if mode == "ema1":
            ema1.swap_in()
        elif mode == "ema2":
            ema2.swap_in()
        x_dec = model.sample(emb, out_len=out_len)  # (B, out_len, 1)
        if mode == "ema1":
            ema1.swap_out()
        elif mode == "ema2":
            ema2.swap_out()
        model.train()
        return x_dec.permute(0, 2, 1).float()  # (B, 1, out_len)

    def _to_aligner(x_bvl: torch.Tensor) -> torch.Tensor:
        if x_bvl.size(-1) < args.aligner_min_length:
            x_bvl = F.interpolate(x_bvl, size=args.aligner_min_length, mode="linear", align_corners=False)
        return x_bvl

    @torch.no_grad()
    def run_generation_metrics(step_now: int, epoch_now: int):
        if not args.enable_gen_eval or args.gen_eval_every_epochs <= 0:
            return
        if (epoch_now + 1) % args.gen_eval_every_epochs != 0:
            return
        if metric_loader is None or metric_embedder is None:
            return
        seed_all(args.metrics_seed + epoch_now)
        modes = ("online", "ema1", "ema2")
        mode_metrics = {}
        for m in modes:
            acc = ConTSGMetricsAccumulator(
                embedder=metric_embedder,
                cfg=MetricConfig(k=args.metrics_k, mdd_bins=args.metrics_mdd_bins,
                                 max_samples=args.metrics_max_samples, seed=args.metrics_seed + epoch_now),
            )
            for bi, (captions, emb, real_raw_bvl) in enumerate(metric_loader):
                if args.gen_eval_max_batches > 0 and bi >= args.gen_eval_max_batches:
                    break
                emb = emb.to(device).float()
                real_raw_bvl = real_raw_bvl.to(device).float()
                gen_normed_bvl = generate_normed(emb, m, out_len=real_raw_bvl.size(-1))
                gen_raw_bvl = full_ds.denormalize(gen_normed_bvl)  # back to raw scale for the aligner
                acc.update(
                    captions=captions,
                    real_series_bvl=_to_aligner(real_raw_bvl),
                    gen_series_bvl=_to_aligner(gen_raw_bvl),
                )
            mode_metrics[m] = acc.compute()
        payload = {"gen_eval/epoch": epoch_now}
        for m in modes:
            payload.update({f"gen_eval/{m}/{k}": v for k, v in mode_metrics[m].items()})
        wandb.log(payload, step=step_now)
        fid_key = "fidelity/fid"
        print(f"  [gen-eval] epoch {epoch_now}  "
              f"online_fid={mode_metrics['online'].get(fid_key, float('nan')):.4f}  "
              f"ema1_fid={mode_metrics['ema1'].get(fid_key, float('nan')):.4f}  "
              f"ema2_fid={mode_metrics['ema2'].get(fid_key, float('nan')):.4f}")

    best_val_ema2 = float("inf")

    def run_validation(step_now: int, epoch_now: int):
        nonlocal best_val_ema2
        if val_loader is None:
            return
        metrics = {m: eval_loss(m, args.val_seed) for m in ("online", "ema1", "ema2")}
        wandb.log({"val/epoch": epoch_now,
                   "val/online/loss": metrics["online"]["loss"],
                   "val/ema1/loss": metrics["ema1"]["loss"],
                   "val/ema2/loss": metrics["ema2"]["loss"]}, step=step_now)
        print(f"  [val] epoch={epoch_now} step={step_now} online={metrics['online']['loss']:.4f} "
              f"ema1={metrics['ema1']['loss']:.4f} ema2={metrics['ema2']['loss']:.4f}")
        if metrics["ema2"]["loss"] < best_val_ema2:
            best_val_ema2 = metrics["ema2"]["loss"]
            save_ckpt(args.save_best, args, model, opt, ema1, ema2, cfg, split_info, normalization,
                      extra={"val_loss_ema2": best_val_ema2, "step": step_now, "epoch": epoch_now})
            wandb.log({"val/best_loss_ema2": best_val_ema2}, step=step_now)
            print(f"        -> new best val/loss {best_val_ema2:.4f} saved -> {args.save_best}")

    # ---------- train ----------
    step = 0
    t0 = time.time()
    model.train()
    for epoch in range(args.epochs):
        for _captions, emb, x_1 in train_loader:
            emb = emb.to(device).float()
            x_1 = x_1.to(device).float().unsqueeze(-1)  # (B, L, 1)
            out = model.forward_cond(x_1, emb)
            loss = out["loss"]

            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            else:
                grad_norm = torch.tensor(float("nan"))
            opt.step()
            ema1.update()
            ema2.update()
            step += 1

            wandb.log({"train/loss": float(loss.item()),
                       "train/loss_rec_ts": float(out["loss_rec_ts"].item()),
                       "train/loss_rec_mov": float(out["loss_rec_mov"].item()),
                       "train/loss_kld": float(out["loss_kld"].item()),
                       "train/grad_norm": float(grad_norm),
                       "train/lr": opt.param_groups[0]["lr"], "train/epoch": epoch}, step=step)
            if step % args.log_every == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f} "
                      f"rec_ts={out['loss_rec_ts'].item():.4f} rec_mov={out['loss_rec_mov'].item():.4f} "
                      f"kld={out['loss_kld'].item():.4f} gn={float(grad_norm):.3f} ({time.time()-t0:.1f}s)")
            if args.eval_every_steps > 0 and step % args.eval_every_steps == 0:
                run_validation(step, epoch)

        run_validation(step, epoch)
        run_generation_metrics(step, epoch)

        if args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0:
            ckpt_path = args.save.with_name(f"{args.save.stem}_epoch{epoch + 1}{args.save.suffix}")
            save_ckpt(ckpt_path, args, model, opt, ema1, ema2, cfg, split_info, normalization,
                      extra={"step": step, "epoch": epoch})
            print(f"  [ckpt] epoch {epoch + 1} saved -> {ckpt_path}")

    save_ckpt(args.save, args, model, opt, ema1, ema2, cfg, split_info, normalization,
              extra={"final_step": step, "final_epoch": args.epochs - 1})
    print(f"saved final -> {args.save.resolve()}")
    if val_loader is not None:
        print(f"best val/loss = {best_val_ema2:.4f} -> {args.save_best.resolve()}")
    wandb.finish()


if __name__ == "__main__":
    main()
