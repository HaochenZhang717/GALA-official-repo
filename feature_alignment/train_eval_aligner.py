"""
Train the independent evaluation aligner (see `eval_aligner.py`).

This is a scorer, not a method: it exists so FID / CTTP / JFTSD can be reported
in an embedding space that shares no weights, no architecture and no pretraining
pipeline with EmbeddingGemma-300M + Chronos-2.

Two things matter for the numbers to be defensible, and both are enforced here:

  1. The scorer is trained ONLY on the train split, using the same
     `--val-ratio` / `--val-split-seed` as the Stage-2 diffusion runs
     (`--val-ratio 0.1 --val-split-seed 0`). Generation eval happens on the val
     split, so the scorer never saw those windows.
  2. Checkpoint selection uses held-out retrieval accuracy only — nothing that
     involves a generator.

Typical launch (matches the ETTh1s / L48 ablation setting):

    torchrun --nproc_per_node=4 feature_alignment/train_eval_aligner.py \
        --source ETTh1s --length 48 --min-length 64 --max-rows none \
        --proj-dim 128 --batch-size 128 --epochs 30 --lr 2e-4 \
        --val-ratio 0.1 --val-split-seed 0 --device cuda \
        --save-best ./feature_alignment/results_eval_aligner/ETTh1s_L48/eval_aligner_best.pt
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler

from dataset import (
    LENGTHS,
    SOURCES,
    TSFragmentDataset,
    VERBALTS_DATASETS,
    VERBALTS_SPLITS,
    VerbalTSDataset,
    collate,
)
from eval_aligner import DEFAULT_TEXT_LORA_TARGETS, DEFAULT_TEXT_MODEL, EvalAligner

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def int_or_none(s: str) -> int | None:
    if s is None or s.strip().lower() in ("", "none", "null"):
        return None
    return int(s)


def is_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_ddp() -> tuple[int, int, int, bool]:
    if not is_torchrun():
        return 0, 0, 1, False
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size, True


def batch_series_hash(series: torch.Tensor) -> torch.Tensor:
    """Hash each sample series to an int64 id, for the multi-positive mask."""
    series_cpu = series.detach().cpu().contiguous()
    out = torch.empty(series_cpu.size(0), dtype=torch.long)
    for i in range(series_cpu.size(0)):
        digest = hashlib.sha1(series_cpu[i].numpy().tobytes()).digest()
        out[i] = int.from_bytes(digest[:8], byteorder="little", signed=False) & ((1 << 63) - 1)
    return out


def _series_tensor_from_dataset(ds) -> torch.Tensor:
    if isinstance(ds, Subset):
        base = ds.dataset
        if not hasattr(base, "series"):
            raise AttributeError("Subset base dataset is missing `.series`.")
        return base.series.index_select(0, torch.as_tensor(ds.indices, dtype=torch.long))
    if not hasattr(ds, "series"):
        raise AttributeError("Dataset is missing `.series`.")
    return ds.series


def compute_train_series_stats(ds) -> tuple[float, float]:
    """Global z-score stats. Train split only — using the full set would leak."""
    series = _series_tensor_from_dataset(ds).float()
    return float(series.mean().item()), max(float(series.std(unbiased=False).item()), 1e-5)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("dataset")
    g.add_argument("--dataset-backend", default="tsfragment", choices=["tsfragment", "verbalts"])
    g.add_argument("--source", default="ETTh1s", choices=SOURCES)
    g.add_argument("--length", type=int, default=48, choices=LENGTHS)
    g.add_argument("--verbalts-name", default="synthetic_u", choices=VERBALTS_DATASETS)
    g.add_argument("--verbalts-split", default="train", choices=VERBALTS_SPLITS)
    g.add_argument("--verbalts-val-split", default="valid", choices=VERBALTS_SPLITS + ("none",))
    g.add_argument("--verbalts-root", type=Path, default=Path("./VerbalTSDatasets"))
    g.add_argument("--min-length", type=int, default=64)
    g.add_argument("--max-rows", type=int_or_none, default=None)
    g.add_argument("--cache-dir", type=Path, default=Path("./hf_cache"))

    g = p.add_argument_group("text encoder")
    g.add_argument("--text-model-id", default=DEFAULT_TEXT_MODEL,
                   help="Must not share a family with the method's text encoder, nor with the "
                        "LLM that wrote the captions (Qwen3-8B).")
    g.add_argument("--text-dtype", default="float32", choices=list(DTYPES))
    g.add_argument("--text-pooling", default="cls", choices=["cls", "mean"],
                   help="'cls' for BGE checkpoints, 'mean' for e5 / sentence-transformers.")
    g.add_argument("--text-max-length", type=int, default=512)
    g.add_argument("--text-prefix", default="")
    g.add_argument("--text-lora-r", type=int, default=8)
    g.add_argument("--text-lora-alpha", type=int, default=16)
    g.add_argument("--text-lora-dropout", type=float, default=0.05)
    g.add_argument("--text-lora-target-modules", nargs="+", default=list(DEFAULT_TEXT_LORA_TARGETS))

    g = p.add_argument_group("time-series encoder (from scratch)")
    g.add_argument("--cnn-hidden-channels", type=int, default=256)
    g.add_argument("--cnn-kernel-size", type=int, default=3)
    g.add_argument("--cnn-dilations", nargs="+", type=int, default=[1, 2, 4, 8])
    g.add_argument("--cnn-dropout", type=float, default=0.1)

    g = p.add_argument_group("shared space")
    g.add_argument("--proj-dim", type=int, default=128,
                   help="Keep this well below the number of eval samples: FID/JFTSD estimate a "
                        "proj_dim (resp. 2*proj_dim) covariance and go ill-conditioned otherwise.")
    g.add_argument("--init-temperature", type=float, default=0.07)
    g.add_argument("--min-temperature", type=float, default=0.03)
    g.add_argument("--hf-token-env", default="HF_TOKEN")

    g = p.add_argument_group("training")
    g.add_argument("--batch-size", type=int, default=128, help="Per-rank under torchrun.")
    g.add_argument("--epochs", type=int, default=30)
    g.add_argument("--lr", type=float, default=2e-4)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--warmup-steps", type=int, default=200)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="cuda")
    g.add_argument("--matmul-tf32", action="store_true", default=True,
                   help="PyTorch defaults this off, which leaves fp32 matmuls at ~1/2 the speed "
                        "they could run at with no meaningful precision loss (fp32 accumulate).")
    g.add_argument("--no-matmul-tf32", dest="matmul_tf32", action="store_false")

    g = p.add_argument_group("validation")
    g.add_argument("--val-ratio", type=float, default=0.1,
                   help="Must match the Stage-2 diffusion runs so gen-eval windows stay held out.")
    g.add_argument("--val-split-seed", type=int, default=0,
                   help="Must match the diffusion runs' --val-split-seed.")
    g.add_argument("--eval-every-epochs", type=int, default=1)
    g.add_argument("--select-by", default="loss", choices=["loss", "r@1"],
                   help="Which held-out quantity picks the best checkpoint. Either way it is a "
                        "held-out quantity of the scorer alone — never anything involving a generator.")

    g = p.add_argument_group("io")
    g.add_argument("--log-every", type=int, default=20)
    g.add_argument("--save", type=Path, default=Path("./eval_aligner_ckpt.pt"))
    g.add_argument("--save-best", type=Path, default=Path("./eval_aligner_best.pt"))

    g = p.add_argument_group("wandb")
    g.add_argument("--wandb-project", default="text-ts-eval-aligner")
    g.add_argument("--wandb-entity", default=None)
    g.add_argument("--wandb-run-name", default=None)
    g.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    g.add_argument("--wandb-tags", nargs="+", default=[])
    g.add_argument("--wandb-notes", default=None)

    return p.parse_args()


@torch.no_grad()
def evaluate(model: EvalAligner, loader, device: torch.device) -> dict[str, float]:
    """Full-split retrieval on held-out data.

    Batch-level accuracy depends on batch size, so the report-worthy number is
    retrieval over the whole val split at once. Duplicate series count as
    positives for each other, matching the training objective.
    """
    was_training = model.training
    model.eval()

    text_chunks, ts_chunks, hash_chunks = [], [], []
    loss_sum, n_seen = 0.0, 0
    for captions, series in loader:
        series = series.to(device)
        series_hash = batch_series_hash(series).to(device)
        out = model(captions, series, series_hash=series_hash, gather=False)
        b = series.size(0)
        loss_sum += out["loss"].item() * b
        n_seen += b
        text_chunks.append(out["text_emb"].float().cpu())
        ts_chunks.append(out["ts_emb"].float().cpu())
        hash_chunks.append(series_hash.cpu())

    if was_training:
        model.train()
    if n_seen == 0:
        return {}

    text_all = torch.cat(text_chunks)
    ts_all = torch.cat(ts_chunks)
    hashes = torch.cat(hash_chunks)
    sim = text_all @ ts_all.t()  # (N, N)

    metrics = {"loss": loss_sum / n_seen, "n_val": float(n_seen)}
    for name, matrix in (("t2s", sim), ("s2t", sim.t())):
        order = matrix.argsort(dim=1, descending=True)
        retrieved = hashes[order]                     # (N, N) hashes in rank order
        hit = retrieved.eq(hashes[:, None])           # positives anywhere in the ranking
        for k in (1, 5, 10):
            metrics[f"r@{k}_{name}"] = hit[:, :k].any(dim=1).float().mean().item()
        # rank of the first positive, 1-indexed
        first = hit.float().argmax(dim=1) + 1
        metrics[f"median_rank_{name}"] = first.median().item()
    metrics["r@1_mean"] = 0.5 * (metrics["r@1_t2s"] + metrics["r@1_s2t"])
    return metrics


def save_ckpt(path: Path, model: EvalAligner, args, ts_stats: tuple[float, float], extra: dict | None = None) -> int:
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "aligner_trainable_state": state,
        # Buffers are not parameters; without these the encoder would silently
        # rebuild with mean=0/std=1 and every embedding would shift.
        "ts_stats": {"mean": ts_stats[0], "std": ts_stats[1]},
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return len(state)


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, is_distributed = setup_ddp()
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = args.matmul_tf32
    torch.backends.cudnn.allow_tf32 = args.matmul_tf32
    device = torch.device(f"cuda:{local_rank}") if is_distributed else torch.device(args.device)

    run = None
    if is_main:
        cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        cfg["world_size"] = world_size
        cfg["global_batch_size"] = args.batch_size * world_size
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            tags=args.wandb_tags,
            notes=args.wandb_notes,
            config=cfg,
        )

    # ---------------- dataset ----------------
    if args.dataset_backend == "tsfragment":
        ds = TSFragmentDataset(
            source=args.source,
            length=args.length,
            min_length=args.min_length,
            max_rows=args.max_rows,
            cache_dir=args.cache_dir,
        )
        n_total = len(ds)
        n_val = min(int(round(n_total * args.val_ratio)) if args.val_ratio > 0 else 0, n_total - 1)
        n_train = n_total - n_val
        split_seed = args.seed if args.val_split_seed is None else args.val_split_seed
        if n_val > 0:
            train_ds, val_ds = random_split(
                ds, [n_train, n_val], generator=torch.Generator().manual_seed(split_seed)
            )
        else:
            train_ds, val_ds = ds, None
        ds_name = f"{args.source}/L={args.length}"
    else:
        train_ds = VerbalTSDataset(
            dataset_name=args.verbalts_name,
            split=args.verbalts_split,
            min_length=args.min_length,
            max_rows=args.max_rows,
            data_root=args.verbalts_root,
        )
        val_ds = None
        if args.verbalts_val_split.lower() != "none":
            val_ds = VerbalTSDataset(
                dataset_name=args.verbalts_name,
                split=args.verbalts_val_split,
                min_length=args.min_length,
                max_rows=args.max_rows,
                data_root=args.verbalts_root,
            )
        ds = train_ds
        n_train, n_val = len(train_ds), (0 if val_ds is None else len(val_ds))
        n_total = n_train + n_val
        ds_name = f"{args.verbalts_name}/{args.verbalts_split}"

    ts_mean, ts_std = compute_train_series_stats(train_ds)
    if is_main:
        print(f"dataset: {ds_name} -> L={ds.upsampled_length}")
        print(f"size   : {n_total} ({n_train} train / {n_val} val, split_seed={args.val_split_seed})")
        print(f"z-score (train split only): mean={ts_mean:.6f} std={ts_std:.6f}")
        if args.proj_dim * 2 > max(n_val, 1):
            print(f"[warn] proj_dim={args.proj_dim} vs n_val={n_val}: JFTSD estimates a "
                  f"{2 * args.proj_dim}-dim covariance and will be ill-conditioned.")

    if is_distributed:
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True, seed=args.seed)
        loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate,
                            num_workers=args.num_workers, drop_last=True, pin_memory=True)
    else:
        sampler = None
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                            num_workers=args.num_workers, drop_last=True)

    # Validation runs on rank 0 over the whole split: retrieval metrics are only
    # meaningful against the full candidate pool, not a per-rank shard.
    val_loader = None
    if val_ds is not None and is_main:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                                num_workers=args.num_workers, drop_last=False)

    # ---------------- model ----------------
    hf_token = os.environ.get(args.hf_token_env) or None
    aligner = EvalAligner(
        text_model_id=args.text_model_id,
        proj_dim=args.proj_dim,
        text_dtype=DTYPES[args.text_dtype],
        device=device,
        init_temperature=args.init_temperature,
        min_temperature=args.min_temperature,
        hf_token=hf_token,
        text_lora_r=args.text_lora_r,
        text_lora_alpha=args.text_lora_alpha,
        text_lora_dropout=args.text_lora_dropout,
        text_lora_target_modules=tuple(args.text_lora_target_modules),
        text_pooling=args.text_pooling,
        text_max_length=args.text_max_length,
        text_prefix=args.text_prefix,
        cnn_hidden_channels=args.cnn_hidden_channels,
        cnn_kernel_size=args.cnn_kernel_size,
        cnn_dilations=tuple(args.cnn_dilations),
        cnn_dropout=args.cnn_dropout,
        ts_global_mean=ts_mean,
        ts_global_std=ts_std,
    )
    if is_main:
        aligner.print_trainable_parameters()

    if is_distributed:
        model = DDP(aligner, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        inner = model.module
    else:
        model = aligner
        inner = aligner

    opt = torch.optim.AdamW(inner.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(len(loader), 1)
    total_steps = max(steps_per_epoch * args.epochs, 1)

    def lr_lambda(step: int) -> float:
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159265 * progress)).item()))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    # Lower-is-better score, so both selection criteria share one comparison.
    best_score = float("inf")
    best_epoch = -1

    def run_validation(step_now: int, epoch_now: int) -> None:
        nonlocal best_score, best_epoch
        if is_main and val_loader is not None:
            metrics = evaluate(inner, val_loader, device)
            if metrics:
                wandb.log({f"val/{k}": v for k, v in metrics.items()} | {"val/epoch": epoch_now}, step=step_now)
                print(
                    f"  [val] epoch {epoch_now} loss={metrics['loss']:.4f} "
                    f"R@1(t2s)={metrics['r@1_t2s']:.4f} R@1(s2t)={metrics['r@1_s2t']:.4f} "
                    f"R@10(t2s)={metrics['r@10_t2s']:.4f} n={int(metrics['n_val'])}"
                )
                score = metrics["loss"] if args.select_by == "loss" else -metrics["r@1_mean"]
                if score < best_score:
                    best_score = score
                    best_epoch = epoch_now
                    n = save_ckpt(
                        args.save_best, inner, args, (ts_mean, ts_std),
                        extra={
                            "select_by": args.select_by,
                            "val_loss": metrics["loss"],
                            "val_r1_mean": metrics["r@1_mean"],
                            "step": step_now,
                            "epoch": epoch_now,
                        },
                    )
                    wandb.log({"val/best_loss": metrics["loss"], "val/best_r@1_mean": metrics["r@1_mean"]}, step=step_now)
                    print(
                        f"  [val] new best ({args.select_by}) loss={metrics['loss']:.4f} "
                        f"R@1={metrics['r@1_mean']:.4f} -> {args.save_best} ({n} tensors)"
                    )
        if is_distributed:
            dist.barrier()

    # ---------------- train ----------------
    model.train()
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for captions, series in loader:
            series = series.to(device)
            series_hash = batch_series_hash(series).to(device)
            out = model(captions, series, series_hash=series_hash, gather=True)

            opt.zero_grad()
            out["loss"].backward()
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(inner.trainable_parameters(), args.grad_clip)
            else:
                grad_norm = torch.tensor(float("nan"))
            opt.step()
            sched.step()
            step += 1

            if is_main:
                wandb.log(
                    {
                        "train/loss": out["loss"].item(),
                        "train/loss_t2s": out["loss_t2s"].item(),
                        "train/loss_s2t": out["loss_s2t"].item(),
                        "train/mean_cos": out["cosine"].mean().item(),
                        "train/acc_t2s": out["acc_t2s"].item(),
                        "train/acc_s2t": out["acc_s2t"].item(),
                        "train/temperature": 1.0 / out["logit_scale"].item(),
                        "train/grad_norm": float(grad_norm),
                        "train/lr": opt.param_groups[0]["lr"],
                        "train/epoch": epoch,
                    },
                    step=step,
                )
                if step % args.log_every == 0:
                    print(
                        f"epoch {epoch}  step {step:>5}  loss={out['loss'].item():.4f}  "
                        f"cos={out['cosine'].mean().item():+.4f}  "
                        f"acc_t2s={out['acc_t2s'].item():.3f}  acc_s2t={out['acc_s2t'].item():.3f}  "
                        f"tau={1.0 / out['logit_scale'].item():.3f}  ({time.time() - t0:.1f}s)"
                    )

        if args.eval_every_epochs > 0 and (epoch + 1) % args.eval_every_epochs == 0:
            run_validation(step, epoch)

    if is_main:
        n = save_ckpt(args.save, inner, args, (ts_mean, ts_std), extra={"step": step, "epochs": args.epochs})
        best_str = f"{best_score:.4f}" if args.select_by == "loss" else f"{-best_score:.4f}"
        print(f"saved -> {args.save.resolve()} ({n} tensors); "
              f"best val {args.select_by}={best_str} at epoch {best_epoch} -> {args.save_best.resolve()}")
        wandb.finish()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
