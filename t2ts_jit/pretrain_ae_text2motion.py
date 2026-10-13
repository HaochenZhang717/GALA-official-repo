"""Pretrain the Text2Motion movement autoencoder (Stage-1 of the baseline).

This mirrors ``pretrain_vae_t2s.py`` but for the Text2Motion backbone: instead of
the T2S length-adaptive VAE it trains the **MovementConvEncoder / MovementConvDecoder**
(conv down/up by ``unit_length=4``). Per the chosen recipe it trains **one AE per
(source, length)** (NOT mix-trained), so the Stage-2 conditional trainer
(``train_cond_v7_text2motion.py``) loads the matching ``(source, length)`` AE.

Consistency guarantees (identical to the T2S baseline):
  * Series are **global min-max normalized to [0, 1]** before the AE, and the
    ``(global_min, global_max)`` used for that scaling is stored in the checkpoint
    so the frozen encoder in Stage-2 sees in-distribution inputs.
  * Rows are filtered with ``dropna(["Text", "OT"])`` and the val split is the
    seeded ``random_split`` that ``train_cond_v7_text2motion.py`` (and v7) produce,
    so the AE never trains on a Stage-2 validation series (match
    ``--val-ratio`` / ``--val-split-seed`` / ``--max-rows``).

Example:
    python t2ts_jit/pretrain_ae_text2motion.py --source ETTm1 --length 24 --epochs 2000
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
from torch.utils.data import DataLoader, Dataset, random_split

_HERE = Path(__file__).resolve().parent
_JIT_DIR = _HERE
sys.path.insert(0, str(_JIT_DIR))

from cond_jit_basics.text2motion import MovementAutoEncoder, Text2MotionConfig  # noqa: E402

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
    """One (source, length) split, optionally global min-max normalized to [0, 1].

    Row filtering (``dropna`` on ``["Text", "OT"]``) mirrors the Stage-2 dataset
    exactly so seeded splits line up.
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
    return torch.stack(batch, dim=0)  # (B, L)


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
    p.add_argument("--length", type=int, default=24, choices=LENGTHS)
    p.add_argument("--cache-dir", type=Path, default=Path("./hf_cache"))
    p.add_argument("--normalize", type=str, default="global", choices=["none", "global"],
                   help="Scale fed to the AE. Must match the Stage-2 trainer (default global -> [0,1]).")
    p.add_argument("--max-rows", type=int, default=None)

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=270)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", type=Path, default=Path("./t2ts_jit/ae_ckpts_text2motion"))
    p.add_argument("--save-name", type=str, default=None,
                   help="Override the run subdir name (default ae_<source>_L<length>_ep<epochs>).")
    p.add_argument("--log-every", type=int, default=20)
    # Split params MUST match the Stage-2 trainer (val_ratio=0.1, val_split_seed=0).
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--val-split-seed", type=int, default=0)

    # movement AE geometry (Text2MotionConfig defaults)
    p.add_argument("--dim-movement-latent", type=int, default=512)
    p.add_argument("--dim-movement-enc-hidden", type=int, default=512)
    p.add_argument("--dim-movement-dec-hidden", type=int, default=512)
    p.add_argument("--unit-length", type=int, default=4)
    # Latent regularizers (Text2Motion DecompTrainerV3 defaults: 0.001 each).
    p.add_argument("--lambda-sparsity", type=float, default=0.001)
    p.add_argument("--lambda-smooth", type=float, default=0.001)

    p.add_argument("--wandb-project", default="text2motion-baseline-ae")
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

    if args.length % args.unit_length != 0:
        raise ValueError(f"length={args.length} must be divisible by unit_length={args.unit_length}")

    # ---------------- data ----------------
    ds = TSFragmentSeriesDataset(args.source, args.length, args.cache_dir, args.normalize, args.max_rows)
    norm_stats = {"global_min": ds.global_min, "global_max": ds.global_max}

    n_total = len(ds)
    n_val = int(round(n_total * args.val_ratio)) if args.val_ratio > 0 else 0
    n_val = min(n_val, max(n_total - 1, 0))
    n_train = n_total - n_val
    split_seed = args.val_split_seed
    if n_val > 0:
        train_ds, val_ds = random_split(ds, [n_train, n_val],
                                        generator=torch.Generator().manual_seed(split_seed))
        train_indices = list(train_ds.indices)
        val_indices = list(val_ds.indices)
    else:
        train_ds, val_ds = ds, None
        train_indices = list(range(n_total))
        val_indices = []
    print(f"[split] {args.source} L={args.length} n={n_total} train={n_train} val={n_val} (seed={split_seed})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, collate_fn=collate_series)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, drop_last=False, collate_fn=collate_series)

    run_name = args.save_name or f"ae_{args.source}_L{args.length}_ep{args.epochs}"
    save_root = args.save_dir / run_name
    save_root.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        name=args.wandb_run_name or run_name, mode=args.wandb_mode,
        tags=args.wandb_tags, notes=args.wandb_notes,
        config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    )

    # ---------------- model ----------------
    cfg = Text2MotionConfig(
        n_var=1,
        unit_length=args.unit_length,
        dim_movement_latent=args.dim_movement_latent,
        dim_movement_enc_hidden=args.dim_movement_enc_hidden,
        dim_movement_dec_hidden=args.dim_movement_dec_hidden,
        lambda_sparsity=args.lambda_sparsity,
        lambda_smooth=args.lambda_smooth,
    )
    model = MovementAutoEncoder(cfg).to(device)
    # Plain Adam to match the Text2Motion Decomp baseline (DecompTrainerV3 uses
    # optim.Adam with no weight decay and no gradient clipping); weight_decay
    # defaults to 0 so the optimizer matches the baseline exactly.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"[data] {args.source} L{args.length} | total={n_total} train={n_train} val={n_val} | norm={args.normalize}")
    print(f"[norm] {norm_stats}")
    print(f"[save] {save_root}")

    # Sanity: confirm the movement latent geometry (length L/unit_length).
    with torch.no_grad():
        probe = ds[0].unsqueeze(0).unsqueeze(-1).to(device).float()  # (1, L, 1)
        z = model.encode(probe)
        rec = model.decode(z)
        print(f"[latent] movement shape = {tuple(z.shape)} (expect (1, {args.length // args.unit_length}, "
              f"{args.dim_movement_latent})); recon shape = {tuple(rec.shape)}")

    def _save(path: Path, epoch: int, best_val: float):
        payload = {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "cfg": cfg.to_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val,
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "normalize": args.normalize,
            "norm_stats": norm_stats,
            "split_info": {"val_ratio": args.val_ratio, "split_seed": split_seed,
                           "n_total": n_total, "n_train": n_train, "n_val": n_val,
                           "train_indices": train_indices, "val_indices": val_indices},
        }
        torch.save(payload, path)

    step = 0
    t0 = time.time()
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x in train_loader:
            x = x.unsqueeze(-1).float().to(device)  # (B, L, 1)
            out = model(x)
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            losses.append(float(loss.item()))
            wandb.log({
                "train/loss": float(loss.item()),
                "train/loss_rec": float(out["loss_rec"].item()),
                "train/loss_sparsity": float(out["loss_sparsity"].item()),
                "train/loss_smooth": float(out["loss_smooth"].item()),
                "train/epoch": epoch,
            }, step=step)
            if step % args.log_every == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.6f} "
                      f"(rec={out['loss_rec'].item():.6f} "
                      f"sparsity={out['loss_sparsity'].item():.6f} "
                      f"smooth={out['loss_smooth'].item():.6f}) ({time.time()-t0:.1f}s)")

        epoch_loss = float(np.mean(losses)) if losses else float("nan")

        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            vl = []
            with torch.no_grad():
                for x in val_loader:
                    x = x.unsqueeze(-1).float().to(device)
                    vl.append(float(model(x)["loss"].item()))
            val_loss = float(np.mean(vl)) if vl else float("nan")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save(save_root / "best_model.pth", epoch, best_val_loss)
                print(f"[best] epoch={epoch} val_loss={best_val_loss:.6f} -> {save_root / 'best_model.pth'}")

        wandb.log({"epoch/index": epoch, "epoch/train_loss": epoch_loss, "epoch/val_loss": val_loss}, step=step)
        print(f"[epoch {epoch}] train_loss={epoch_loss:.6f} val_loss={val_loss:.6f}")

    _save(save_root / "final_model.pth", args.epochs - 1, best_val_loss)
    if val_loader is None:
        _save(save_root / "best_model.pth", args.epochs - 1, best_val_loss)
    print(f"[done] best -> {save_root / 'best_model.pth'}")
    wandb.finish()


if __name__ == "__main__":
    main()
