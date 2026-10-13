"""
Stage-1 STRUCTURE baseline trainer.

Trains the STRUCTURE feature-alignment baseline (frozen Gemma + frozen Chronos-2,
a Transformer head + linear projector per side) on one split of TSFragment-600K /
VerbalTS, as a comparison point for the V7 method.

Objective (per the STRUCTURE paper, Gröger et al. NeurIPS 2025):
    loss = CLIP_contrastive(aligned_text, aligned_ts)            # diagonal InfoNCE
         + lambda(t) * 0.5 * ( structure_reg(orig_text, aligned_text)
                             + structure_reg(orig_ts,   aligned_ts) )
where `orig_*` are the FROZEN pre-head encoder features whose neighborhood
geometry the regularizer preserves, and lambda(t) follows a linear warmup.

There is NO diffusion term (that is the V7-side regularizer being compared
against).  The contrastive temperature is FIXED at --temperature (STRUCTURE uses
0.05) and logit_scale is not learned.

This is a thin fork of train_v7_ablation_ft.py: dataset handling, DDP, validation,
checkpointing and wandb logging mirror it, so the produced checkpoint
(`ft_method="head"` in its saved args) is loaded verbatim by the EXISTING stage-2
builder ablation_diffusion_df_n_arch/train_cond_v7_ablation_ft.py.

Nothing in the existing codebase or baseline repos is modified.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.data.distributed import DistributedSampler

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_COND_DIR = _REPO_ROOT / "t2ts_jit" / "cond_jit_basics"
# feature_alignment itself must be importable so `structure_align` (package) and
# `aligner_v7_ablation_ft` (imported by the subclass) resolve.
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_COND_DIR))

from aligner_v7 import _gather_features_with_grad
from dataset import (
    LENGTHS,
    SOURCES,
    TSFragmentDataset,
    VERBALTS_DATASETS,
    VERBALTS_SPLITS,
    VerbalTSDataset,
    collate,
)
from structure_align import Centering, DistanceFunction, StructureAligner
from structure_align.structure_loss import structure_reg

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def int_or_none(s: str) -> int | None:
    if s is None or s.strip().lower() in ("", "none", "null"):
        return None
    return int(s)


def str2bool(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "y", "t")


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


class StructureJointModel(torch.nn.Module):
    """Wraps the StructureAligner with the CLIP + STRUCTURE objective.

    `structure_lambda` is supplied per-step from the warmup schedule in main().
    The contrastive term is gathered across ranks (like V7); the STRUCTURE term is
    computed on the local per-rank batch (batch-level regularizer, as in the
    paper), so its gradients flow to the head/projector and are DDP-averaged.
    """

    def __init__(
        self,
        aligner: StructureAligner,
        temperature: float,
        structure_levels: int,
        structure_temperature: float,
        structure_weighting: str,
        structure_margin: float,
        structure_distance: DistanceFunction,
        structure_centering: Centering,
        structure_center_first: bool,
    ):
        super().__init__()
        self.aligner = aligner
        self.temperature = temperature
        self.structure_levels = structure_levels
        self.structure_temperature = structure_temperature
        self.structure_weighting = structure_weighting
        self.structure_margin = structure_margin
        self.structure_distance = structure_distance
        self.structure_centering = structure_centering
        self.structure_center_first = structure_center_first

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _structure_one_side(self, original: torch.Tensor, aligned: torch.Tensor) -> torch.Tensor:
        return structure_reg(
            original,
            aligned,
            levels=self.structure_levels,
            temperature=self.structure_temperature,
            margin=self.structure_margin,
            weighting=self.structure_weighting,
            distance_type=self.structure_distance,
            centering_type=self.structure_centering,
            center_first=self.structure_center_first,
        )

    def forward(self, captions, series_bvl: torch.Tensor, structure_lambda: float, gather: bool = True):
        text_aligned, text_orig = self.aligner.encode_text(captions, return_pre_proj=True)
        ts_aligned, ts_orig = self.aligner.encode_time_series(series_bvl, return_pre_proj=True)

        # ---- diagonal CLIP contrastive (fixed temperature) ----
        if gather:
            text_all = _gather_features_with_grad(text_aligned)
            ts_all = _gather_features_with_grad(ts_aligned)
        else:
            text_all = text_aligned
            ts_all = ts_aligned

        # Both sides are gathered in the same rank order, so the positive of
        # global row i is global column i -> labels = arange(N).
        logits = (text_all @ ts_all.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_t2s = F.cross_entropy(logits, labels)
        loss_s2t = F.cross_entropy(logits.t(), labels)
        loss_ctr = 0.5 * (loss_t2s + loss_s2t)

        # ---- STRUCTURE regularizer (local batch, per side) ----
        loss_struct_text = self._structure_one_side(text_orig, text_aligned)
        loss_struct_ts = self._structure_one_side(ts_orig, ts_aligned)
        loss_struct = 0.5 * (loss_struct_text + loss_struct_ts)

        total = loss_ctr + structure_lambda * loss_struct

        with torch.no_grad():
            cosine = (text_aligned * ts_aligned).sum(dim=-1)
            pred_t2s = logits.argmax(dim=-1)
            pred_s2t = logits.argmax(dim=0)
            acc_t2s = pred_t2s.eq(labels).float().mean()
            acc_s2t = pred_s2t.eq(labels).float().mean()
        return {
            "loss": total,
            "loss_contrastive": loss_ctr,
            "loss_structure": loss_struct,
            "loss_structure_text": loss_struct_text,
            "loss_structure_ts": loss_struct_ts,
            "loss_t2s": loss_t2s,
            "loss_s2t": loss_s2t,
            "cosine": cosine,
            "acc_t2s": acc_t2s,
            "acc_s2t": acc_s2t,
            "structure_lambda": torch.tensor(float(structure_lambda)),
            "text_emb": text_all,
        }


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

    g = p.add_argument_group("aligner")
    g.add_argument("--llm-id", default="google/embeddinggemma-300m")
    g.add_argument("--chronos-model-id", default="amazon/chronos-2")
    g.add_argument("--proj-dim", type=int, default=512)
    g.add_argument("--llm-dtype", default="bfloat16", choices=list(DTYPES))
    g.add_argument("--hf-token-env", default="HF_TOKEN")
    g.add_argument("--init-temperature", type=float, default=0.07)
    g.add_argument("--min-temperature", type=float, default=0.03)
    # Stored in saved_args so the existing stage-2 builder rebuilds the head
    # architecture. Pinned to "head" -- both backbones frozen.
    g.add_argument("--ft-method", default="head", choices=["head"])
    g.add_argument("--head-depth", type=int, default=1, help="Transformer-head blocks per side.")
    g.add_argument("--head-num-heads", type=int, default=12, help="Attention heads in the Transformer head.")
    g.add_argument("--head-ffn-dim", type=int, default=2560, help="Feed-forward dim in the Transformer head.")
    g.add_argument("--head-dropout", type=float, default=0.0)

    g = p.add_argument_group("objective")
    g.add_argument("--temperature", type=float, default=0.05, help="Fixed CLIP contrastive temperature (STRUCTURE uses 0.05).")
    g.add_argument("--structure-lambda", type=float, default=10.0, help="STRUCTURE weight (base, after warmup).")
    g.add_argument("--structure-levels", type=int, default=1, help="Multi-scale neighborhood levels.")
    g.add_argument("--structure-warmup-steps", type=int, default=1000, help="Linear warmup steps for structure-lambda.")
    g.add_argument("--structure-temperature", type=float, default=0.05, help="Temperature inside structure_reg (STRUCTURE ties this to the CLIP temperature).")
    g.add_argument("--structure-weighting", default="none", choices=["none", "inverse"])
    g.add_argument("--structure-margin", type=float, default=0.0)
    g.add_argument("--structure-centering", default="mean", choices=[c.name for c in Centering])
    g.add_argument("--structure-distance", default="cosine", choices=[d.name for d in DistanceFunction])
    g.add_argument("--structure-center-first", type=str2bool, default=False)

    g = p.add_argument_group("training")
    g.add_argument("--batch-size", type=int, default=8, help="Per-rank under torchrun.")
    g.add_argument("--epochs", type=int, default=1)
    g.add_argument("--lr", type=float, default=2e-5)
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--num-workers", type=int, default=0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="cpu")

    g = p.add_argument_group("validation")
    g.add_argument("--val-ratio", type=float, default=0.1)
    g.add_argument("--val-split-seed", type=int, default=None)
    g.add_argument("--eval-every-steps", type=int, default=0)

    g = p.add_argument_group("io")
    g.add_argument("--log-every", type=int, default=10)
    g.add_argument("--save", type=Path, default=Path("./aligner_structure_ckpt.pt"))
    g.add_argument("--save-best", type=Path, default=Path("./aligner_structure_best.pt"))

    g = p.add_argument_group("wandb")
    g.add_argument("--wandb-project", default="text-ts-aligner-v7")
    g.add_argument("--wandb-entity", default=None)
    g.add_argument("--wandb-run-name", default=None)
    g.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    g.add_argument("--wandb-tags", nargs="+", default=[])
    g.add_argument("--wandb-notes", default=None)

    return p.parse_args()


def lambda_at_step(step: int, base: float, warmup_steps: int) -> float:
    """Linear warmup of the structure weight, matching STRUCTURE's CLIPLoss.step()."""
    if warmup_steps <= 0:
        return base
    return base * min(1.0, step / warmup_steps)


@torch.no_grad()
def evaluate(model_inner: StructureJointModel, loader, device: torch.device, structure_lambda: float) -> dict[str, float]:
    model_inner.eval()
    keys = ("loss", "loss_contrastive", "loss_structure", "loss_structure_text", "loss_structure_ts", "loss_t2s", "loss_s2t", "mean_cos", "acc_t2s", "acc_s2t")
    sums = {k: 0.0 for k in keys}
    n_seen = 0
    for captions, series in loader:
        series = series.to(device)
        out = model_inner(captions, series, structure_lambda=structure_lambda, gather=True)
        b = series.size(0)
        sums["loss"] += out["loss"].item() * b
        sums["loss_contrastive"] += out["loss_contrastive"].item() * b
        sums["loss_structure"] += out["loss_structure"].item() * b
        sums["loss_structure_text"] += out["loss_structure_text"].item() * b
        sums["loss_structure_ts"] += out["loss_structure_ts"].item() * b
        sums["loss_t2s"] += out["loss_t2s"].item() * b
        sums["loss_s2t"] += out["loss_s2t"].item() * b
        sums["mean_cos"] += out["cosine"].mean().item() * b
        sums["acc_t2s"] += out["acc_t2s"].item() * b
        sums["acc_s2t"] += out["acc_s2t"].item() * b
        n_seen += b
    model_inner.train()
    n = max(n_seen, 1)
    return {k: v / n for k, v in sums.items()}


def save_ckpt(path: Path, inner_model: StructureJointModel, args, extra: dict | None = None) -> dict[str, int]:
    aligner_state = {n: p.detach().cpu() for n, p in inner_model.aligner.named_parameters() if p.requires_grad}
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        # Key name matches what the stage-2 loader (_load_v7_ckpt) expects.
        "aligner_trainable_state": aligner_state,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return {"aligner": len(aligner_state)}


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, is_distributed = setup_ddp()
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}") if is_distributed else torch.device(args.device)
    if is_main:
        print("[stage] setup complete (STRUCTURE baseline, ft_method=head)")

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
        wandb.log({"system/started": 1, "system/world_size": world_size}, step=0)
        print("[stage] wandb initialized")

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
        print("[stage] dataset ready")
        print(f"dataset: {args.dataset_backend} / {ds_name} -> upsampled L={ds.upsampled_length}")
        print(f"size : {n_total} total ({n_train} train / {n_val} val)")
        print(f"global z-score for aligner: mean={ts_global_mean:.6f}, std={ts_global_std:.6f}")

    if is_distributed:
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True, seed=args.seed)
        loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    else:
        sampler = None
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers, drop_last=True)
    if is_main:
        print("[stage] train loader ready")

    val_loader = None
    if val_ds is not None:
        if is_distributed:
            val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, collate_fn=collate, num_workers=args.num_workers, drop_last=False, pin_memory=True)
        else:
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers, drop_last=False)
    if is_main:
        print("[stage] val loader ready")

    hf_token = os.environ.get(args.hf_token_env) or None
    if is_main:
        print("[stage] building STRUCTURE aligner (frozen encoders + Transformer head)")
    aligner = StructureAligner(
        llm_id=args.llm_id,
        chronos_model_id=args.chronos_model_id,
        proj_dim=args.proj_dim,
        llm_dtype=DTYPES[args.llm_dtype],
        device=device,
        init_temperature=args.init_temperature,
        min_temperature=args.min_temperature,
        hf_token=hf_token,
        head_depth=args.head_depth,
        head_num_heads=args.head_num_heads,
        head_ffn_dim=args.head_ffn_dim,
        head_dropout=args.head_dropout,
        ts_global_mean=ts_global_mean,
        ts_global_std=ts_global_std,
    )
    if is_main:
        print("[stage] aligner ready")
        aligner.print_trainable_parameters()

    joint = StructureJointModel(
        aligner=aligner,
        temperature=args.temperature,
        structure_levels=args.structure_levels,
        structure_temperature=args.structure_temperature,
        structure_weighting=args.structure_weighting,
        structure_margin=args.structure_margin,
        structure_distance=DistanceFunction[args.structure_distance],
        structure_centering=Centering[args.structure_centering],
        structure_center_first=args.structure_center_first,
    )
    joint.to(device)

    if is_distributed:
        model = DDP(joint, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        inner = model.module
    else:
        model = joint
        inner = joint

    if is_main:
        n_trainable = sum(p.numel() for p in inner.trainable_parameters())
        n_total_p = sum(p.numel() for p in inner.parameters())
        print(f"trainable params: {n_trainable:,} / {n_total_p:,}")

    opt = torch.optim.AdamW(inner.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_loss = float("inf")

    def run_validation(step_now: int, epoch_now: int, lam_now: float) -> None:
        nonlocal best_val_loss
        if val_loader is None:
            return
        metrics = evaluate(inner, val_loader, device, structure_lambda=lam_now)
        if not is_main:
            return
        wandb.log({f"val/{k}": v for k, v in metrics.items()} | {"val/epoch": epoch_now}, step=step_now)
        print(f"  [val] epoch {epoch_now} step {step_now:>5} loss={metrics['loss']:.4f}")
        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]
            n_states = save_ckpt(args.save_best, inner, args, extra={"val_loss": best_val_loss, "step": step_now, "epoch": epoch_now})
            wandb.log({"val/best_loss": best_val_loss}, step=step_now)
            print(f"  [val] new best -> {args.save_best} (aligner={n_states['aligner']})")

    model.train()
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for captions, series in loader:
            series = series.to(device)
            lam = lambda_at_step(step, args.structure_lambda, args.structure_warmup_steps)
            out = model(captions, series, structure_lambda=lam, gather=True)

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
                        "train/loss_contrastive": out["loss_contrastive"].item(),
                        "train/loss_structure": out["loss_structure"].item(),
                        "train/loss_structure_text": out["loss_structure_text"].item(),
                        "train/loss_structure_ts": out["loss_structure_ts"].item(),
                        "train/loss_t2s": out["loss_t2s"].item(),
                        "train/loss_s2t": out["loss_s2t"].item(),
                        "train/mean_cos": out["cosine"].mean().item(),
                        "train/acc_t2s": out["acc_t2s"].item(),
                        "train/acc_s2t": out["acc_s2t"].item(),
                        "train/structure_lambda": lam,
                        "train/temperature": args.temperature,
                        "train/grad_norm": float(grad_norm),
                        "train/lr": opt.param_groups[0]["lr"],
                        "train/epoch": epoch,
                    },
                    step=step,
                )
                if step % args.log_every == 0:
                    dt = time.time() - t0
                    print(
                        f"epoch {epoch}  step {step:>5}  "
                        f"loss={out['loss'].item():.4f}  ctr={out['loss_contrastive'].item():.4f}  "
                        f"struct={out['loss_structure'].item():.4f}  lam={lam:.3f}  "
                        f"cos={out['cosine'].mean().item():+.4f}  "
                        f"acc_t2s={out['acc_t2s'].item():.3f}  acc_s2t={out['acc_s2t'].item():.3f}  "
                        f"gn={float(grad_norm):.3f}  ({dt:.1f}s)"
                    )

        lam_end = lambda_at_step(step, args.structure_lambda, args.structure_warmup_steps)
        run_validation(step, epoch, lam_end)

    if is_main:
        n_states = save_ckpt(args.save, inner, args, extra={"step": step, "epochs": args.epochs})
        print(f"saved trainable state -> {args.save.resolve()} (aligner={n_states['aligner']})")
        if run is not None and args.wandb_mode != "disabled" and args.save.exists():
            try:
                artifact = wandb.Artifact(name=f"aligner-structure-ckpt-{run.id}", type="model")
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
