"""
Train the M3-JEPA baseline on one split of TSFragment-600K / VerbalTS.

M3-JEPA = MMoE cross-modal predictor + JEPA smooth-L1 regression loss +
symmetric single-positive InfoNCE, optimized over the two proxy tasks
(text2ts / ts2text) either alternately (faithful AGD) or jointly. Encoders are
the same as our own aligner (Gemma + Chronos-2, LoRA, global z-score norm) so
the comparison is apples-to-apples. No diffusion / downstream branch.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from dataset import (
    LENGTHS,
    SOURCES,
    TSFragmentDataset,
    VERBALTS_DATASETS,
    VERBALTS_SPLITS,
    VerbalTSDataset,
    collate,
)
from jepa_align import JepaEncoders, build_m3jepa

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


def _series_tensor_from_dataset(ds) -> torch.Tensor:
    if isinstance(ds, Subset):
        base = ds.dataset
        if not hasattr(base, "series"):
            raise AttributeError("Subset base dataset is missing `.series`.")
        indices = torch.as_tensor(ds.indices, dtype=torch.long)
        return base.series.index_select(0, indices)
    if not hasattr(ds, "series"):
        raise AttributeError("Dataset is missing `.series`.")
    return ds.series


def _compute_series_stats(ds) -> tuple[float, float]:
    series = _series_tensor_from_dataset(ds).float()
    global_mean = float(series.mean().item())
    global_std = max(float(series.std(unbiased=False).item()), 1e-5)
    return global_mean, global_std


def tasks_for_step(schedule: str, step: int) -> tuple[str, ...]:
    """Which proxy tasks to optimize this step.

    `alternate` flips text2ts/ts2text by step index (synced across ranks since
    every rank shares the same step counter). `joint` optimizes both each step.
    """
    if schedule == "joint":
        return ("text2ts", "ts2text")
    return ("text2ts",) if step % 2 == 0 else ("ts2text",)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("dataset")
    g.add_argument("--dataset-backend", default="tsfragment", choices=["tsfragment", "verbalts"])
    g.add_argument("--source", default="ETTh1", choices=SOURCES)
    g.add_argument("--length", type=int, default=24, choices=LENGTHS)
    g.add_argument("--verbalts-name", default="synthetic_u", choices=VERBALTS_DATASETS)
    g.add_argument("--verbalts-split", default="train", choices=VERBALTS_SPLITS)
    g.add_argument("--verbalts-val-split", default="valid", choices=VERBALTS_SPLITS + ("none",))
    g.add_argument(
        "--verbalts-root",
        type=Path,
        default=Path("./data/VerbalTSDatasets"),
    )
    g.add_argument("--min-length", type=int, default=64)
    g.add_argument("--max-rows", type=int_or_none, default=2000)
    g.add_argument("--cache-dir", type=Path, default=Path("./hf_cache"))

    g = p.add_argument_group("encoders")
    g.add_argument("--llm-id", default="google/embeddinggemma-300m")
    g.add_argument("--chronos-model-id", default="amazon/chronos-2")
    g.add_argument("--llm-dtype", default="bfloat16", choices=list(DTYPES))
    g.add_argument("--hf-token-env", default="HF_TOKEN")
    g.add_argument(
        "--tune-side",
        default="both",
        choices=["both", "text", "ts"],
        help="Which encoder side to LoRA-adapt; the frozen side stays fully frozen.",
    )
    g.add_argument("--text-lora-r", type=int, default=8)
    g.add_argument("--text-lora-alpha", type=int, default=16)
    g.add_argument("--text-lora-dropout", type=float, default=0.05)
    g.add_argument("--text-lora-target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    g.add_argument("--ts-lora-r", type=int, default=8)
    g.add_argument("--ts-lora-alpha", type=int, default=16)
    g.add_argument("--ts-lora-dropout", type=float, default=0.05)
    g.add_argument("--ts-lora-target-modules", nargs="+", default=["q", "k", "v", "o"])

    g = p.add_argument_group("m3-jepa predictor")
    g.add_argument("--num-experts", type=int, default=12)
    g.add_argument("--k-expert", type=int, default=4)
    g.add_argument("--hidden-size", type=int, default=512)
    g.add_argument("--dropout-rate", type=float, default=0.1)
    g.add_argument("--temperature", type=float, default=0.03)
    g.add_argument("--l2-weight", type=float, default=1.0)
    g.add_argument("--cl-weight", type=float, default=1.0)
    g.add_argument("--task-weight-text2ts", type=float, default=0.5)
    g.add_argument("--task-weight-ts2text", type=float, default=0.5)
    g.add_argument("--task-schedule", default="alternate", choices=["alternate", "joint"])

    g = p.add_argument_group("training")
    g.add_argument("--batch-size", type=int, default=8, help="Per-rank under torchrun.")
    g.add_argument("--epochs", type=int, default=1)
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--num-workers", type=int, default=0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="cpu")

    g = p.add_argument_group("validation")
    g.add_argument("--val-ratio", type=float, default=0.1)
    g.add_argument("--val-split-seed", type=int, default=None)

    g = p.add_argument_group("io")
    g.add_argument("--log-every", type=int, default=10)
    g.add_argument("--save", type=Path, default=Path("./m3jepa_ckpt.pt"))
    g.add_argument("--save-best", type=Path, default=Path("./m3jepa_best.pt"))

    g = p.add_argument_group("wandb")
    g.add_argument("--wandb-project", default="text-ts-m3jepa")
    g.add_argument("--wandb-entity", default=None)
    g.add_argument("--wandb-run-name", default=None)
    g.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    g.add_argument("--wandb-tags", nargs="+", default=[])
    g.add_argument("--wandb-notes", default=None)

    return p.parse_args()


@torch.no_grad()
def evaluate(model_inner, loader, device, schedule: str) -> dict[str, float]:
    model_inner.eval()
    keys = ("loss", "loss_pred", "loss_cl", "mean_cos", "acc_t2s", "acc_s2t")
    sums = {k: 0.0 for k in keys}
    n_seen = 0
    for captions, series in loader:
        series = series.to(device)
        # Always evaluate both directions so val metrics are schedule-independent.
        out = model_inner(captions, series, tasks=("text2ts", "ts2text"))
        b = series.size(0)
        sums["loss"] += out["loss"].item() * b
        sums["loss_pred"] += out["loss_pred"].item() * b
        sums["loss_cl"] += out["loss_cl"].item() * b
        sums["mean_cos"] += out["cosine"].mean().item() * b
        sums["acc_t2s"] += out["acc_t2s"].item() * b
        sums["acc_s2t"] += out["acc_s2t"].item() * b
        n_seen += b
    model_inner.train()
    n = max(n_seen, 1)
    return {k: v / n for k, v in sums.items()}


def save_ckpt(path: Path, inner_model, args, extra: dict | None = None) -> int:
    state = {n: p.detach().cpu() for n, p in inner_model.named_parameters() if p.requires_grad}
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_trainable_state": state,
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
    device = torch.device(f"cuda:{local_rank}") if is_distributed else torch.device(args.device)
    if is_main:
        print("[stage] setup complete")

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
        print("[stage] wandb initialized")

    # ---------------- dataset ----------------
    if is_main:
        print("[stage] building dataset")
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
            train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(split_seed))
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
        n_train = len(train_ds)
        n_val = 0 if val_ds is None else len(val_ds)
        n_total = n_train + n_val
        ds_name = f"{args.verbalts_name}/train={args.verbalts_split}, val={args.verbalts_val_split if val_ds is not None else 'none'}"

    ts_global_mean, ts_global_std = _compute_series_stats(train_ds)
    if is_main:
        print(f"dataset: {args.dataset_backend} / {ds_name} -> upsampled L={ds.upsampled_length}")
        print(f"size : {n_total} total ({n_train} train / {n_val} val)")
        print(f"global z-score: mean={ts_global_mean:.6f}, std={ts_global_std:.6f}")

    if is_distributed:
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True, seed=args.seed)
        loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    else:
        sampler = None
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers, drop_last=True)

    val_loader = None
    if val_ds is not None:
        if is_distributed:
            val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, collate_fn=collate, num_workers=args.num_workers, drop_last=False, pin_memory=True)
        else:
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers, drop_last=False)

    # ---------------- model ----------------
    hf_token = os.environ.get(args.hf_token_env) or None
    if is_main:
        print(f"[stage] building encoders (tune_side={args.tune_side})")
    encoders = JepaEncoders(
        llm_id=args.llm_id,
        chronos_model_id=args.chronos_model_id,
        llm_dtype=DTYPES[args.llm_dtype],
        device=device,
        hf_token=hf_token,
        tune_side=args.tune_side,
        text_lora_r=args.text_lora_r,
        text_lora_alpha=args.text_lora_alpha,
        text_lora_dropout=args.text_lora_dropout,
        text_lora_target_modules=tuple(args.text_lora_target_modules),
        ts_lora_r=args.ts_lora_r,
        ts_lora_alpha=args.ts_lora_alpha,
        ts_lora_dropout=args.ts_lora_dropout,
        ts_lora_target_modules=tuple(args.ts_lora_target_modules),
        ts_global_mean=ts_global_mean,
        ts_global_std=ts_global_std,
    )
    model = build_m3jepa(
        encoders=encoders,
        num_experts=args.num_experts,
        k_expert=args.k_expert,
        hidden_size=args.hidden_size,
        dropout_rate=args.dropout_rate,
        temperature=args.temperature,
        l2_weight=args.l2_weight,
        cl_weight=args.cl_weight,
        task_weight_text2ts=args.task_weight_text2ts,
        task_weight_ts2text=args.task_weight_ts2text,
    )
    model.to(device)
    if is_main:
        print("[stage] model ready")

    if is_distributed:
        ddp = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        inner = ddp.module
    else:
        ddp = model
        inner = model

    if is_main:
        n_trainable = sum(p.numel() for p in inner.trainable_parameters())
        n_total_p = sum(p.numel() for p in inner.parameters())
        print(f"trainable params: {n_trainable:,} / {n_total_p:,}")

    opt = torch.optim.AdamW(inner.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_loss = float("inf")

    def run_validation(step_now: int, epoch_now: int) -> None:
        nonlocal best_val_loss
        if val_loader is None:
            return
        metrics = evaluate(inner, val_loader, device, args.task_schedule)
        if not is_main:
            return
        wandb.log({f"val/{k}": v for k, v in metrics.items()} | {"val/epoch": epoch_now}, step=step_now)
        print(f"  [val] epoch {epoch_now} step {step_now:>5} loss={metrics['loss']:.4f} "
              f"acc_t2s={metrics['acc_t2s']:.3f} acc_s2t={metrics['acc_s2t']:.3f}")
        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]
            n_states = save_ckpt(args.save_best, inner, args, extra={"val_loss": best_val_loss, "step": step_now, "epoch": epoch_now})
            wandb.log({"val/best_loss": best_val_loss}, step=step_now)
            print(f"  [val] new best -> {args.save_best} ({n_states} tensors)")

    ddp.train()
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for captions, series in loader:
            series = series.to(device)
            tasks = tasks_for_step(args.task_schedule, step)
            out = ddp(captions, series, tasks=tasks)

            opt.zero_grad()
            out["loss"].backward()
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(inner.trainable_parameters(), args.grad_clip)
            else:
                grad_norm = torch.tensor(float("nan"))
            opt.step()
            step += 1

            if is_main:
                wandb.log(
                    {
                        "train/loss": out["loss"].item(),
                        "train/loss_pred": out["loss_pred"].item(),
                        "train/loss_cl": out["loss_cl"].item(),
                        "train/mean_cos": out["cosine"].mean().item(),
                        "train/acc_t2s": out["acc_t2s"].item(),
                        "train/acc_s2t": out["acc_s2t"].item(),
                        "train/grad_norm": float(grad_norm),
                        "train/lr": opt.param_groups[0]["lr"],
                        "train/epoch": epoch,
                    },
                    step=step,
                )
                if step % args.log_every == 0:
                    dt = time.time() - t0
                    print(
                        f"epoch {epoch}  step {step:>5}  task={'+'.join(tasks)}  "
                        f"loss={out['loss'].item():.4f}  pred={out['loss_pred'].item():.4f}  cl={out['loss_cl'].item():.4f}  "
                        f"cos={out['cosine'].mean().item():+.4f}  "
                        f"acc_t2s={out['acc_t2s'].item():.3f}  acc_s2t={out['acc_s2t'].item():.3f}  "
                        f"gn={float(grad_norm):.3f}  ({dt:.1f}s)"
                    )

        run_validation(step, epoch)

    if is_main:
        n_states = save_ckpt(args.save, inner, args, extra={"step": step, "epochs": args.epochs})
        print(f"saved trainable state -> {args.save.resolve()} ({n_states} tensors)")
        if run is not None and args.wandb_mode != "disabled" and args.save.exists():
            try:
                artifact = wandb.Artifact(name=f"m3jepa-ckpt-{run.id}", type="model")
                artifact.add_file(str(args.save))
                run.log_artifact(artifact)
            except Exception as e:
                print(f"[warn] wandb artifact upload skipped: {e}")
        wandb.finish()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
