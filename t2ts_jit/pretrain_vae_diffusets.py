"""Pretrain the DiffuSETS VAE (Stage-1 of the DiffuSETS baseline).

This is the DiffuSETS analogue of ``pretrain_vae_t2s.py``: same data, same
[0,1]-per-length normalization, same train/val split logic, so the DiffuSETS and
T2S latent baselines are directly comparable. The only differences are the model
(an SD-style conv VAE with a *tunable* downsampling factor instead of T2S's
length-adaptive interpolating VQVAE) and the loss (reconstruction + KL).

The VAE is mix-trained on lengths 24/48/96. Its latent is
``(B, latent_channels, L / downsample_factor)``; with the default factor 2 that is
length 12/24/48 for the three training lengths, leaving enough resolution for the
latent U-Net in Stage-2 (``train_cond_v7_diffusets.py``).

Example:
    python t2ts_jit/pretrain_vae_diffusets.py --source ETTm1 --epochs 2000 --mix-train
"""

from __future__ import annotations

import argparse
import ast
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from huggingface_hub import hf_hub_download
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "cond_jit_basics"))

from diffusets_modules import DiffuSETSVAE, vae_loss_fn  # noqa: E402

REPO_ID = "WinfredGe/TSFragment-600K"
SOURCES = ("ETTh1", "ETTh1s", "ETTm1", "airquality", "electricity", "exchangerate", "traffic")
LENGTHS = (24, 48, 96)
FACTOR_TO_NUMDOWN = {2: 1, 4: 2, 8: 3}


def _parse_floats(s) -> np.ndarray:
    if isinstance(s, (list, tuple, np.ndarray)):
        return np.asarray(s, dtype=np.float32)
    s = str(s).strip()
    if "," in s:
        return np.asarray(ast.literal_eval(s), dtype=np.float32)
    return np.asarray([float(x) for x in s.replace("[", "").replace("]", "").split()], dtype=np.float32)


class TSFragmentSeriesDataset(Dataset):
    """One (source, length) split, optionally min-max normalized to [0, 1].

    Identical row filtering / stat computation to ``pretrain_vae_t2s.py`` so the
    VAE holds out exactly the diffusion stage's val rows.
    """

    def __init__(self, source: str, length: int, cache_dir: Path,
                 normalize: str = "global", max_rows: int | None = None):
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}, expected one of {SOURCES}")
        if length not in LENGTHS:
            raise ValueError(f"unknown length {length!r}, expected one of {LENGTHS}")
        if normalize not in ("none", "global"):
            raise ValueError("normalize must be one of: none, global")

        fname = f"embedding_cleaned_{source}_{length}.csv"
        local = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=fname, cache_dir=str(cache_dir))
        df = pd.read_csv(local, usecols=["Text", "OT"], nrows=max_rows)
        df = df.dropna(subset=["Text", "OT"]).reset_index(drop=True)

        self.length = length
        self.normalize = normalize
        self.series = torch.from_numpy(np.stack([_parse_floats(s) for s in df["OT"]])).float()  # (N, L)
        if normalize == "global":
            self.global_min = float(self.series.min().item())
            self.global_max = float(self.series.max().item())
        else:
            self.global_min = None
            self.global_max = None

    def __len__(self) -> int:
        return self.series.size(0)

    def __getitem__(self, idx):
        raw = self.series[idx]
        if self.normalize == "global":
            span = max(self.global_max - self.global_min, 1e-6)
            return (raw - self.global_min) / span
        return raw


def collate_series(batch):
    """Group a mixed-length batch by length so each group can be stacked."""
    groups: dict[int, list[torch.Tensor]] = {}
    for x in batch:
        L = int(x.shape[-1])
        groups.setdefault(L, []).append(x)
    return {L: torch.stack(xs, dim=0) for L, xs in groups.items()}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source", type=str, default="ETTm1", choices=SOURCES)
    p.add_argument("--cache-dir", type=Path, default=Path("./hf_cache"))
    p.add_argument("--normalize", type=str, default="global", choices=["none", "global"],
                   help="Scale fed to the VAE. Must match the diffusion stage (default global -> [0,1]).")
    p.add_argument("--max-rows-per-length", type=int, default=None)
    p.add_argument("--mix-train", dest="mix_train", action="store_true",
                   help="Train on 24/48/96 concatenated.")
    p.add_argument("--no-mix-train", dest="mix_train", action="store_false")
    p.set_defaults(mix_train=True)
    p.add_argument("--single-length", type=int, default=24, choices=LENGTHS)

    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", type=Path, default=Path("./t2ts_jit/vae_ckpts_diffusets"))
    p.add_argument("--save-name", type=str, default=None)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--val-split-seed", type=int, default=None)

    # DiffuSETS VAE geometry. downsample-factor=2 -> latent length L/2 (the default
    # the DiffuSETS baseline is tuned for); 4/8 are also exposed (tunable).
    p.add_argument("--downsample-factor", type=int, default=2, choices=[2, 4, 8],
                   help="Sequence downsampling of the VAE (latent length = L / factor).")
    p.add_argument("--latent-channels", type=int, default=4)
    p.add_argument("--base-channels", type=int, default=128,
                   help="Must be a multiple of 32 (GroupNorm).")

    # KL annealing (off by default; reconstruction-dominant VAE like the original).
    p.add_argument("--kld-weight", type=float, default=1.0)
    p.add_argument("--kld-annealing", action="store_true", default=False)
    p.add_argument("--kld-warmup-epochs", type=int, default=300)
    p.add_argument("--kld-start-weight", type=float, default=0.01)

    p.add_argument("--wandb-project", default="diffusets-baseline-vae")
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
    if args.base_channels % 32 != 0:
        raise ValueError(f"--base-channels must be a multiple of 32 (got {args.base_channels}).")
    num_down = FACTOR_TO_NUMDOWN[args.downsample_factor]

    # ---------------- data ----------------
    lengths = list(LENGTHS) if args.mix_train else [args.single_length]
    dataset_desc = f"{args.source} mix(24,48,96)" if args.mix_train else f"{args.source} L={args.single_length}"
    per_len = {L: TSFragmentSeriesDataset(args.source, L, args.cache_dir, args.normalize, args.max_rows_per_length)
               for L in lengths}
    norm_per_length = {
        int(L): {"global_min": ds.global_min, "global_max": ds.global_max}
        for L, ds in per_len.items()
    }

    # ---- split: replicate train_cond_v7_diffusets.py's PER-LENGTH random_split exactly ----
    split_seed = args.seed if args.val_split_seed is None else args.val_split_seed
    split_per_length: dict[int, dict] = {}
    train_subsets, val_subsets = [], []
    n_total = n_train = n_val = 0
    for L in lengths:
        ds = per_len[L]
        n_l = len(ds)
        nv_l = int(round(n_l * args.val_ratio)) if args.val_ratio > 0 else 0
        nv_l = min(nv_l, max(n_l - 1, 0))
        nt_l = n_l - nv_l
        perm = torch.randperm(n_l, generator=torch.Generator().manual_seed(split_seed)).tolist()
        tr_idx, va_idx = perm[:nt_l], perm[nt_l:]
        split_per_length[int(L)] = {"n_total": n_l, "n_train": nt_l, "n_val": nv_l,
                                    "train_indices": tr_idx, "val_indices": va_idx}
        train_subsets.append(Subset(ds, tr_idx))
        if va_idx:
            val_subsets.append(Subset(ds, va_idx))
        n_total += n_l
        n_train += nt_l
        n_val += nv_l
        print(f"[split] L={L} n={n_l} train={nt_l} val={nv_l} (seed={split_seed})")

    train_ds = ConcatDataset(train_subsets)
    val_ds = ConcatDataset(val_subsets) if val_subsets else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, collate_fn=collate_series)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, drop_last=False, collate_fn=collate_series)

    run_name = (args.save_name
                or f"vae_{args.source}_{'mix' if args.mix_train else f'L{args.single_length}'}"
                   f"_ds{args.downsample_factor}_ep{args.epochs}")
    save_root = args.save_dir / run_name
    save_root.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        name=args.wandb_run_name or run_name, mode=args.wandb_mode,
        tags=args.wandb_tags, notes=args.wandb_notes,
        config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    )

    vae = DiffuSETSVAE(n_var=1, latent_channels=args.latent_channels,
                       base_channels=args.base_channels, num_down=num_down).to(device)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"[data] {dataset_desc} | total={n_total} train={n_train} val={n_val} | norm={args.normalize}")
    print(f"[norm] per-length stats: {norm_per_length}")
    print(f"[model] DiffuSETSVAE kwargs={vae.init_kwargs()} | params={sum(p.numel() for p in vae.parameters())}")
    print(f"[save] {save_root}")

    with torch.no_grad():
        probe = next(iter(per_len.values()))[0].unsqueeze(0).unsqueeze(-1).to(device).float()  # (1, L, 1)
        z, _, _ = vae.encode(probe)
        print(f"[latent] encoder output shape = {tuple(z.shape)} "
              f"(latent length = L/{args.downsample_factor})")

    def kld_weight(epoch: int) -> float:
        if not args.kld_annealing or epoch >= args.kld_warmup_epochs:
            return args.kld_weight
        progress = epoch / max(args.kld_warmup_epochs, 1)
        return args.kld_start_weight + (args.kld_weight - args.kld_start_weight) * progress

    def _save(path: Path, epoch: int, best_val: float):
        payload = {
            "state_dict": vae.state_dict(),
            "vae_kwargs": vae.init_kwargs(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val,
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "normalize": args.normalize,
            "norm_per_length": norm_per_length,
            "split_info": {"val_ratio": args.val_ratio, "split_seed": split_seed,
                           "n_total": n_total, "n_train": n_train, "n_val": n_val,
                           "per_length": split_per_length},
        }
        torch.save(payload, path)

    def _step(x_bl: torch.Tensor, epoch: int, train: bool):
        x = x_bl.unsqueeze(-1).float().to(device)  # (B, L, 1)
        recons, mean, log_var = vae.reconstruct(x)
        out = vae_loss_fn(recons, x, mean, log_var, kld_weight(epoch))
        if train:
            optimizer.zero_grad()
            out["loss"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(vae.parameters(), args.grad_clip)
            optimizer.step()
        return out

    step = 0
    t0 = time.time()
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        vae.train()
        losses, recons_e = [], []
        for batch_groups in train_loader:
            for x in batch_groups.values():
                out = _step(x, epoch, train=True)
                step += 1
                losses.append(float(out["loss"].item()))
                recons_e.append(float(out["mse"].item()))
                wandb.log({"train/loss": float(out["loss"].item()), "train/mse": float(out["mse"].item()),
                           "train/kld": float(out["kld"].item()), "train/kld_weight": kld_weight(epoch),
                           "train/epoch": epoch}, step=step)
                if step % args.log_every == 0:
                    print(f"epoch={epoch} step={step} loss={out['loss'].item():.6f} "
                          f"mse={out['mse'].item():.6f} kld={out['kld'].item():.4f} ({time.time()-t0:.1f}s)")

        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        epoch_recon = float(np.mean(recons_e)) if recons_e else float("nan")

        val_loss = val_recon = float("nan")
        if val_loader is not None:
            vae.eval()
            vl, vr = [], []
            with torch.no_grad():
                for batch_groups in val_loader:
                    for x in batch_groups.values():
                        out = _step(x, epoch, train=False)
                        vl.append(float(out["loss"].item()))
                        vr.append(float(out["mse"].item()))
            val_loss = float(np.mean(vl)) if vl else float("nan")
            val_recon = float(np.mean(vr)) if vr else float("nan")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save(save_root / "best_model.pth", epoch, best_val_loss)
                print(f"[best] epoch={epoch} val_loss={best_val_loss:.6f} -> {save_root / 'best_model.pth'}")

        wandb.log({"epoch/index": epoch, "epoch/train_loss": epoch_loss, "epoch/train_recon": epoch_recon,
                   "epoch/val_loss": val_loss, "epoch/val_recon": val_recon}, step=step)
        print(f"[epoch {epoch}] train_loss={epoch_loss:.6f} train_recon={epoch_recon:.6f} "
              f"val_loss={val_loss:.6f} val_recon={val_recon:.6f}")

    _save(save_root / "final_model.pth", args.epochs - 1, best_val_loss)
    if val_loader is None:
        _save(save_root / "best_model.pth", args.epochs - 1, best_val_loss)
    print(f"[done] best -> {save_root / 'best_model.pth'}")
    wandb.finish()


if __name__ == "__main__":
    main()
