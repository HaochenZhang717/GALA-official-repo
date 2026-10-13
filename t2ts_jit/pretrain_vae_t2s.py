"""Pretrain the T2S LA-VAE (length-adaptive conv autoencoder) for the baseline.

This is the Stage-1 of the T2S baseline that is being run inside ``t2ts_jit`` to
compare against the Aligner-V7 + jit2d method. It is refactored from
``baselines_repro/t2s_repro/train_vae_t2s_repro.py`` with one important
correctness change:

  * The series are **global min-max normalized to [0, 1] per length** before they
    reach the VAE, and the per-length (min, max) used for that scaling is stored
    in the checkpoint. The conditional diffusion stage (``train_cond_v7_t2s.py``)
    reuses exactly those stats so the *frozen* encoder always sees in-distribution
    inputs (the original repro trained the VAE on raw series but then fed [0,1]
    series into the frozen encoder during diffusion — an out-of-distribution
    mismatch).

The VAE is mix-trained on lengths 24/48/96 (T2S's length-adaptive design) and the
latent is hard-wired to shape (B, embedding_dim=64, 30); ``train_cond_v7_t2s.py``
asserts that geometry.

Example:
    python t2ts_jit/pretrain_vae_t2s.py --source ETTm1 --epochs 2000 --mix-train
"""

from __future__ import annotations

import argparse
import ast
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
_REPO_ROOT = _HERE.parent
_T2S_DIR = _REPO_ROOT / "baseline_codes" / "T2S"
sys.path.insert(0, str(_T2S_DIR))

from model.pretrained.vqvae import vqvae  # noqa: E402

REPO_ID = "WinfredGe/TSFragment-600K"
SOURCES = ("ETTh1", "ETTh1s", "ETTm1", "airquality", "electricity", "exchangerate", "traffic")
LENGTHS = (24, 48, 96)


def _parse_floats(s) -> np.ndarray:
    if isinstance(s, (list, tuple, np.ndarray)):
        return np.asarray(s, dtype=np.float32)
    s = str(s).strip()
    if "," in s:
        return np.asarray(ast.literal_eval(s), dtype=np.float32)
    return np.asarray([float(x) for x in s.replace("[", "").replace("]", "").split()], dtype=np.float32)


class TSFragmentSeriesDataset(Dataset):
    """One (source, length) split, optionally min-max normalized to [0, 1].

    Stores ``global_min`` / ``global_max`` (scalars over this length's OT matrix)
    so the diffusion stage can reuse the identical scaling.
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
        # Mirror train_cond_v7_t2s.py's row filtering EXACTLY (dropna on Text+OT, not
        # OT alone) so per-length indices line up with the diffusion stage's split and
        # the VAE can hold out precisely the diffusion val rows.
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
                   help="Train on 24/48/96 concatenated (T2S length-adaptive design).")
    p.add_argument("--no-mix-train", dest="mix_train", action="store_false",
                   help="Disable mixed-length training; use --single-length only.")
    p.set_defaults(mix_train=True)
    p.add_argument("--single-length", type=int, default=24, choices=LENGTHS)

    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", type=Path, default=Path("./t2ts_jit/vae_ckpts_t2s"))
    p.add_argument("--save-name", type=str, default=None,
                   help="Override the run subdir name (default vae_<source>_<mix|L..>_ep<epochs>).")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--val-split-seed", type=int, default=None)

    # vqvae geometry — embedding_dim=64 yields the (B, 64, 30) latent the DiT needs.
    p.add_argument("--block-hidden-size", type=int, default=128)
    p.add_argument("--num-residual-layers", type=int, default=2)
    p.add_argument("--res-hidden-size", type=int, default=256)
    p.add_argument("--embedding-dim", type=int, default=64)

    p.add_argument("--wandb-project", default="t2s-baseline-vae")
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

    # ---------------- data ----------------
    lengths = list(LENGTHS) if args.mix_train else [args.single_length]
    dataset_desc = f"{args.source} mix(24,48,96)" if args.mix_train else f"{args.source} L={args.single_length}"
    per_len = {L: TSFragmentSeriesDataset(args.source, L, args.cache_dir, args.normalize, args.max_rows_per_length)
               for L in lengths}

    # Per-length scaling stats the diffusion stage will reuse for exact consistency.
    norm_per_length = {
        int(L): {"global_min": ds.global_min, "global_max": ds.global_max}
        for L, ds in per_len.items()
    }

    # ---- split: replicate train_cond_v7_t2s.py's PER-LENGTH random_split exactly ----
    # The diffusion stage trains one model per (source, length) and holds out
    #   val = randperm(n_L, manual_seed(split_seed))[n_train_L:]
    # The VAE is a single mix-trained model shared by all lengths, so its val set is the
    # UNION of each length's diffusion val rows; its train set is everything else. This
    # guarantees every series in any diffusion val split is excluded from VAE training.
    # NOTE: split_seed / val_ratio / max_rows here MUST match the diffusion stage
    # (val_split_seed=0, val_ratio=0.1, max_rows=full).
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
        # identical to torch.utils.data.random_split's internal randperm for this length
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

    run_name = args.save_name or f"vae_{args.source}_{'mix' if args.mix_train else f'L{args.single_length}'}_ep{args.epochs}"
    save_root = args.save_dir / run_name
    save_root.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        name=args.wandb_run_name or run_name, mode=args.wandb_mode,
        tags=args.wandb_tags, notes=args.wandb_notes,
        config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    )

    model = vqvae(args).to(device)
    optimizer = model.configure_optimizers(lr=args.lr)

    print(f"[data] {dataset_desc} | total={n_total} train={n_train} val={n_val} | norm={args.normalize}")
    print(f"[norm] per-length stats: {norm_per_length}")
    print(f"[save] {save_root}")

    # Sanity: confirm the latent geometry the DiT requires.
    with torch.no_grad():
        probe = next(iter(per_len.values()))[0].unsqueeze(0).to(device).float()  # (1, L)
        z, _ = model.encoder(probe)
        print(f"[latent] encoder output shape = {tuple(z.shape)} (expect (1, 64, 30))")
        if int(z.shape[1]) != 64 or int(z.shape[2]) != 30:
            print("  WARNING: latent is not (64, 30); the T2S DiT denoiser will reject this VAE.")

    def _save(path: Path, epoch: int, best_val: float):
        payload = {
            "model": model,
            "state_dict": model.state_dict(),
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

    step = 0
    t0 = time.time()
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        losses, recons = [], []
        for batch_groups in train_loader:
            for x in batch_groups.values():
                x = x.float().to(device)
                loss, recon_error, _x_recon, _z = model.shared_eval(x, optimizer, "train")
                step += 1
                losses.append(float(loss.item()))
                recons.append(float(recon_error.item()))
                wandb.log({"train/loss": float(loss.item()), "train/recon_error": float(recon_error.item()),
                           "train/epoch": epoch}, step=step)
                if step % args.log_every == 0:
                    print(f"epoch={epoch} step={step} loss={loss.item():.6f} recon={recon_error.item():.6f} "
                          f"({time.time()-t0:.1f}s)")

        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        epoch_recon = float(np.mean(recons)) if recons else float("nan")

        val_loss = val_recon = float("nan")
        if val_loader is not None:
            model.eval()
            vl, vr = [], []
            with torch.no_grad():
                for batch_groups in val_loader:
                    for x in batch_groups.values():
                        x = x.float().to(device)
                        loss, recon_error, _r, _z = model.shared_eval(x, None, "val")
                        vl.append(float(loss.item()))
                        vr.append(float(recon_error.item()))
            val_loss = float(np.mean(vl)) if vl else float("nan")
            val_recon = float(np.mean(vr)) if vr else float("nan")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save(save_root / "best_model.pth", epoch, best_val_loss)
                print(f"[best] epoch={epoch} val_loss={best_val_loss:.6f} -> {save_root / 'best_model.pth'}")

        wandb.log({"epoch/index": epoch, "epoch/train_loss": epoch_loss, "epoch/train_recon_error": epoch_recon,
                   "epoch/val_loss": val_loss, "epoch/val_recon_error": val_recon}, step=step)
        print(f"[epoch {epoch}] train_loss={epoch_loss:.6f} train_recon={epoch_recon:.6f} "
              f"val_loss={val_loss:.6f} val_recon={val_recon:.6f}")

    # Always leave a final model even when there is no val split.
    _save(save_root / "final_model.pth", args.epochs - 1, best_val_loss)
    if val_loader is None:
        _save(save_root / "best_model.pth", args.epochs - 1, best_val_loss)
    print(f"[done] best -> {save_root / 'best_model.pth'}")
    wandb.finish()


if __name__ == "__main__":
    main()
