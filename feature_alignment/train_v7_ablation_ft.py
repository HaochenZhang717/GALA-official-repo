"""
Train the stage-1 fine-tuning-method ablation aligner
(TextToTimeSeriesAlignerV7AblationFT) on one split of TSFragment-600K.

This is a thin fork of train_v7.py. The ONLY differences are:
  * it builds TextToTimeSeriesAlignerV7AblationFT instead of ...AlignerV7;
  * it adds the `--ft-method` ablation dimension plus the Transformer-head and
    full-FT (gradient-checkpointing) options.

Everything else -- dataset handling, DDP, the contrastive+diffusion V7JointModel,
validation, checkpointing, and wandb logging -- is identical to train_v7.py, so
the diffusion-on/off arm is still controlled by --diffusion-weight.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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
sys.path.insert(0, str(_COND_DIR))

from aligner_v7 import _gather_features_with_grad, _gather_no_grad
from aligner_v7_ablation_ft import TextToTimeSeriesAlignerV7AblationFT
from dataset import (
    LENGTHS,
    SOURCES,
    TSFragmentDataset,
    VERBALTS_DATASETS,
    VERBALTS_SPLITS,
    VerbalTSDataset,
    collate,
)
from denoiser import Denoiser

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
    """Hash each sample series to an int64 id for multi-positive contrastive loss."""
    series_cpu = series.detach().cpu().contiguous()
    b = series_cpu.size(0)
    out = torch.empty(b, dtype=torch.long)
    for i in range(b):
        arr = series_cpu[i].numpy()
        digest = hashlib.sha1(arr.tobytes()).digest()
        val = int.from_bytes(digest[:8], byteorder="little", signed=False)
        out[i] = val & ((1 << 63) - 1)
    return out


def series_to_xts(series: torch.Tensor) -> torch.Tensor:
    """(B, V, L) -> (B, L, V) for cond_jit_basics Denoiser."""
    return series.permute(0, 2, 1).contiguous()


def resize_series_length(series: torch.Tensor, length: int | None) -> torch.Tensor:
    """Resize only the diffusion target length; aligner inputs stay untouched."""
    if length is None or series.size(-1) == length:
        return series
    return F.interpolate(series.float(), size=length, mode="linear", align_corners=False)


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


def _compute_series_stats(ds) -> tuple[float, float, float, float]:
    series = _series_tensor_from_dataset(ds).float()
    global_mean = float(series.mean().item())
    global_std = max(float(series.std(unbiased=False).item()), 1e-5)
    global_min = float(series.min().item())
    global_max = float(series.max().item())
    return global_mean, global_std, global_min, global_max


class V7JointModel(torch.nn.Module):
    def __init__(
        self,
        aligner: TextToTimeSeriesAlignerV7AblationFT,
        denoiser: Denoiser,
        diffusion_weight: float,
        global_min: float,
        global_max: float,
        diffusion_length: int | None,
    ):
        super().__init__()
        self.aligner = aligner
        self.denoiser = denoiser
        self.diffusion_weight = diffusion_weight
        dev = self.aligner.device
        self.register_buffer("global_min", torch.tensor(float(global_min), dtype=torch.float32, device=dev))
        self.register_buffer("global_max", torch.tensor(float(global_max), dtype=torch.float32, device=dev))
        self.diffusion_length = diffusion_length

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _normalize_to_minus1_1(self, x: torch.Tensor) -> torch.Tensor:
        denom = (self.global_max - self.global_min).clamp(min=1e-12)
        x01 = (x - self.global_min) / denom
        return x01.clamp(0.0, 1.0) * 2.0 - 1.0

    def forward(self, captions, series_bvl: torch.Tensor, series_hash: torch.Tensor, gather: bool = True):
        # Single aligner pass for both branches.
        text_emb = self.aligner.encode_text(captions)
        ts_emb = self.aligner.encode_time_series(series_bvl)

        if series_hash is None:
            series_hash = torch.arange(text_emb.size(0), device=text_emb.device, dtype=torch.long)
        else:
            series_hash = series_hash.to(text_emb.device, dtype=torch.long)

        if gather:
            text_all = _gather_features_with_grad(text_emb)
            ts_all = _gather_features_with_grad(ts_emb)
            hash_all = _gather_no_grad(series_hash)
        else:
            text_all = text_emb
            ts_all = ts_emb
            hash_all = series_hash

        logit_scale = self.aligner.logit_scale.clamp(max=self.aligner.max_logit_scale).exp()
        sim = text_all @ ts_all.t()
        logits = logit_scale * sim
        pos_mask = hash_all[:, None].eq(hash_all[None, :])

        loss_t2s = self.aligner._multi_positive_ce(logits, pos_mask)
        loss_s2t = self.aligner._multi_positive_ce(logits.t(), pos_mask.t())
        loss_ctr = 0.5 * (loss_t2s + loss_s2t)

        # diffusion target uses globally scaled series in [-1, 1]
        series_diff = resize_series_length(series_bvl, self.diffusion_length)
        x_ts = series_to_xts(self._normalize_to_minus1_1(series_diff.float()))
        loss_diff = self.denoiser(x_ts, text_emb)

        total = loss_ctr + self.diffusion_weight * loss_diff
        with torch.no_grad():
            cosine = sim.diagonal()
            pred_t2s = logits.argmax(dim=-1)
            pred_s2t = logits.argmax(dim=0)
            acc_t2s = hash_all[pred_t2s].eq(hash_all).float().mean()
            acc_s2t = hash_all[pred_s2t].eq(hash_all).float().mean()
        return {
            "loss": total,
            "loss_contrastive": loss_ctr,
            "loss_diffusion": loss_diff,
            "loss_t2s": loss_t2s,
            "loss_s2t": loss_s2t,
            "cosine": cosine,
            "acc_t2s": acc_t2s,
            "acc_s2t": acc_s2t,
            "logit_scale": logit_scale.detach(),
            "num_positive_pairs": pos_mask.float().sum().detach(),
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
    g.add_argument(
        "--tune-side",
        default="both",
        choices=["both"],
        help="This ablation runs only under tune_side='both'.",
    )
    g.add_argument("--text-lora-r", type=int, default=8)
    g.add_argument("--text-lora-alpha", type=int, default=16)
    g.add_argument("--text-lora-dropout", type=float, default=0.05)
    g.add_argument("--text-lora-target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    g.add_argument("--ts-lora-r", type=int, default=8)
    g.add_argument("--ts-lora-alpha", type=int, default=16)
    g.add_argument("--ts-lora-dropout", type=float, default=0.05)
    g.add_argument("--ts-lora-target-modules", nargs="+", default=["q", "k", "v", "o"])

    g = p.add_argument_group("fine-tuning method (ablation dimension)")
    g.add_argument(
        "--ft-method",
        default="lora",
        choices=["lora", "head", "full"],
        help="How to fine-tune both encoders: 'lora' (LoRA adapters), 'head' "
        "(frozen backbone + appended Transformer head), or 'full' (full "
        "fine-tuning of both backbones).",
    )
    g.add_argument("--head-depth", type=int, default=1, help="Transformer-head blocks per side (ft-method=head).")
    g.add_argument("--head-num-heads", type=int, default=12, help="Attention heads in the Transformer head.")
    g.add_argument("--head-ffn-dim", type=int, default=2560, help="Feed-forward dim in the Transformer head.")
    g.add_argument("--head-dropout", type=float, default=0.0)
    g.add_argument(
        "--grad-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing on both backbones (ft-method=full).",
    )

    g = p.add_argument_group("diffusion")
    g.add_argument("--diffusion-weight", type=float, default=0.1)
    g.add_argument("--diffusion-length", type=int_or_none, default=64)
    g.add_argument("--img-size", type=int, default=8)
    g.add_argument("--patch-size", type=int, default=2)
    g.add_argument("--in-channels", type=int, default=1)
    g.add_argument("--hidden-channels", type=int, default=64)
    g.add_argument("--depth", type=int, default=2)
    g.add_argument("--num-heads", type=int, default=4)
    g.add_argument("--attn-dropout", type=float, default=0.0)
    g.add_argument("--proj-dropout", type=float, default=0.0)
    g.add_argument("--bottleneck-dim", type=int, default=64)
    g.add_argument("--p-mean", type=float, default=0.0)
    g.add_argument("--p-std", type=float, default=1.0)
    g.add_argument("--t-eps", type=float, default=1e-3)
    g.add_argument("--noise-scale", type=float, default=1.0)
    g.add_argument("--ema-decay1", type=float, default=0.9999)
    g.add_argument("--ema-decay2", type=float, default=0.999)
    g.add_argument("--sampling-method", default="heun", choices=["euler", "heun"])
    g.add_argument("--num-sampling-steps", type=int, default=50)
    g.add_argument("--interval-min", type=float, default=0.0)
    g.add_argument("--interval-max", type=float, default=1.0)

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
    g.add_argument("--eval-every-steps", type=int, default=0)

    g = p.add_argument_group("io")
    g.add_argument("--log-every", type=int, default=10)
    g.add_argument("--save", type=Path, default=Path("./aligner_v7_ckpt.pt"))
    g.add_argument("--save-best", type=Path, default=Path("./aligner_v7_best.pt"))

    g = p.add_argument_group("wandb")
    g.add_argument("--wandb-project", default="text-ts-aligner-v7")
    g.add_argument("--wandb-entity", default=None)
    g.add_argument("--wandb-run-name", default=None)
    g.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    g.add_argument("--wandb-tags", nargs="+", default=[])
    g.add_argument("--wandb-notes", default=None)

    return p.parse_args()


def build_denoiser(args: argparse.Namespace, text_embed_dim: int, seq_len: int, device: torch.device) -> Denoiser:
    ns = SimpleNamespace(
        text_embed_dim=text_embed_dim,
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        depth=args.depth,
        num_heads=args.num_heads,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        bottleneck_dim=args.bottleneck_dim,
        seq_len=seq_len,
        P_mean=args.p_mean,
        P_std=args.p_std,
        t_eps=args.t_eps,
        noise_scale=args.noise_scale,
        ema_decay1=args.ema_decay1,
        ema_decay2=args.ema_decay2,
        sampling_method=args.sampling_method,
        num_sampling_steps=args.num_sampling_steps,
        interval_min=args.interval_min,
        interval_max=args.interval_max,
    )
    return Denoiser(ns).to(device)


@torch.no_grad()
def evaluate(model_inner: V7JointModel, loader, device: torch.device) -> dict[str, float]:
    model_inner.eval()
    sums = {k: 0.0 for k in ("loss", "loss_contrastive", "loss_diffusion", "loss_t2s", "loss_s2t", "mean_cos", "acc_t2s", "acc_s2t", "mean_pos_per_anchor")}
    n_seen = 0
    for captions, series in loader:
        series = series.to(device)
        series_hash = batch_series_hash(series).to(device)
        out = model_inner(captions, series, series_hash=series_hash, gather=True)
        b = series.size(0)
        n_anchor = max(int(out["text_emb"].size(0)), 1)
        mean_pos_per_anchor = out["num_positive_pairs"].item() / n_anchor
        sums["loss"] += out["loss"].item() * b
        sums["loss_contrastive"] += out["loss_contrastive"].item() * b
        sums["loss_diffusion"] += out["loss_diffusion"].item() * b
        sums["loss_t2s"] += out["loss_t2s"].item() * b
        sums["loss_s2t"] += out["loss_s2t"].item() * b
        sums["mean_cos"] += out["cosine"].mean().item() * b
        sums["acc_t2s"] += out["acc_t2s"].item() * b
        sums["acc_s2t"] += out["acc_s2t"].item() * b
        sums["mean_pos_per_anchor"] += mean_pos_per_anchor * b
        n_seen += b
    model_inner.train()
    n = max(n_seen, 1)
    return {k: v / n for k, v in sums.items()}


def save_ckpt(path: Path, inner_model: V7JointModel, args, extra: dict | None = None) -> dict[str, int]:
    aligner_state = {n: p.detach().cpu() for n, p in inner_model.aligner.named_parameters() if p.requires_grad}
    denoiser_state = {n: p.detach().cpu() for n, p in inner_model.denoiser.named_parameters() if p.requires_grad}
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "aligner_trainable_state": aligner_state,
        "denoiser_trainable_state": denoiser_state,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return {"aligner": len(aligner_state), "denoiser": len(denoiser_state)}


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, is_distributed = setup_ddp()
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}") if is_distributed else torch.device(args.device)
    if is_main:
        print(f"[stage] setup complete (ft_method={args.ft_method})")

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
    ts_global_mean, ts_global_std, global_min, global_max = _compute_series_stats(train_ds)
    if is_main:
        print("[stage] dataset ready")

    if is_main:
        print(f"dataset: {args.dataset_backend} / {ds_name} -> upsampled L={ds.upsampled_length}")
        print(f"size : {n_total} total ({n_train} train / {n_val} val)")
        print(f"global z-score for aligner: mean={ts_global_mean:.6f}, std={ts_global_std:.6f}")
        print(f"global scale for diffusion target: min={global_min:.6f}, max={global_max:.6f}")

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
        print(f"[stage] building aligner v7 ablation-ft (ft_method={args.ft_method}, tune_side={args.tune_side})")
    aligner = TextToTimeSeriesAlignerV7AblationFT(
        llm_id=args.llm_id,
        chronos_model_id=args.chronos_model_id,
        proj_dim=args.proj_dim,
        llm_dtype=DTYPES[args.llm_dtype],
        device=device,
        init_temperature=args.init_temperature,
        min_temperature=args.min_temperature,
        hf_token=hf_token,
        tune_side=args.tune_side,
        ft_method=args.ft_method,
        text_lora_r=args.text_lora_r,
        text_lora_alpha=args.text_lora_alpha,
        text_lora_dropout=args.text_lora_dropout,
        text_lora_target_modules=tuple(args.text_lora_target_modules),
        ts_lora_r=args.ts_lora_r,
        ts_lora_alpha=args.ts_lora_alpha,
        ts_lora_dropout=args.ts_lora_dropout,
        ts_lora_target_modules=tuple(args.ts_lora_target_modules),
        head_depth=args.head_depth,
        head_num_heads=args.head_num_heads,
        head_ffn_dim=args.head_ffn_dim,
        head_dropout=args.head_dropout,
        grad_checkpointing=args.grad_checkpointing,
        ts_global_mean=ts_global_mean,
        ts_global_std=ts_global_std,
    )
    if is_main:
        print("[stage] aligner ready")
        aligner.print_trainable_parameters()
    denoiser_seq_len = ds.upsampled_length if args.diffusion_length is None else args.diffusion_length
    denoiser = build_denoiser(args, text_embed_dim=args.proj_dim, seq_len=denoiser_seq_len, device=device)
    if is_main:
        print(f"[stage] denoiser ready (diffusion_length={denoiser_seq_len})")
    joint = V7JointModel(
        aligner=aligner,
        denoiser=denoiser,
        diffusion_weight=args.diffusion_weight,
        global_min=global_min,
        global_max=global_max,
        diffusion_length=args.diffusion_length,
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

    def run_validation(step_now: int, epoch_now: int) -> None:
        nonlocal best_val_loss
        if val_loader is None:
            return
        metrics = evaluate(inner, val_loader, device)
        if not is_main:
            return
        wandb.log({f"val/{k}": v for k, v in metrics.items()} | {"val/epoch": epoch_now}, step=step_now)
        print(f"  [val] epoch {epoch_now} step {step_now:>5} loss={metrics['loss']:.4f}")
        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]
            n_states = save_ckpt(args.save_best, inner, args, extra={"val_loss": best_val_loss, "step": step_now, "epoch": epoch_now})
            wandb.log({"val/best_loss": best_val_loss}, step=step_now)
            print(f"  [val] new best -> {args.save_best} (aligner={n_states['aligner']} denoiser={n_states['denoiser']})")

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

            n_anchor = max(int(out["text_emb"].size(0)), 1)
            mean_pos_per_anchor = out["num_positive_pairs"].item() / n_anchor

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
                        "train/loss_diffusion": out["loss_diffusion"].item(),
                        "train/loss_t2s": out["loss_t2s"].item(),
                        "train/loss_s2t": out["loss_s2t"].item(),
                        "train/mean_cos": out["cosine"].mean().item(),
                        "train/acc_t2s": out["acc_t2s"].item(),
                        "train/acc_s2t": out["acc_s2t"].item(),
                        "train/mean_pos_per_anchor": mean_pos_per_anchor,
                        "train/logit_scale": out["logit_scale"].item(),
                        "train/temperature": 1.0 / out["logit_scale"].item(),
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
                        f"loss={out['loss'].item():.4f}  ctr={out['loss_contrastive'].item():.4f}  diff={out['loss_diffusion'].item():.4f}  "
                        f"cos={out['cosine'].mean().item():+.4f}  "
                        f"acc_t2s={out['acc_t2s'].item():.3f}  acc_s2t={out['acc_s2t'].item():.3f}  "
                        f"tau={1.0 / out['logit_scale'].item():.3f}  gn={float(grad_norm):.3f}  ({dt:.1f}s)"
                    )

        run_validation(step, epoch)

    if is_main:
        n_states = save_ckpt(args.save, inner, args, extra={"step": step, "epochs": args.epochs})
        print(f"saved trainable state -> {args.save.resolve()} (aligner={n_states['aligner']} denoiser={n_states['denoiser']})")
        if run is not None and args.wandb_mode != "disabled" and args.save.exists():
            try:
                artifact = wandb.Artifact(name=f"aligner-v7-ablft-ckpt-{run.id}", type="model")
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
