"""
Standalone generation-quality read-out for the auxiliary denoiser g_xi from
the ORIGINAL (frozen, sequential) two-stage pipeline's Stage-1 aligner
checkpoint.

Why this script exists
-----------------------
feature_alignment/train_v7.py's V7JointModel already trains a lightweight
conditional denoiser g_xi alongside the Stage-1 contrastive aligner, using the
SAME flow-matching loss as the real Stage-2 generator (L_stage1 = L_align +
lambda_dw * L_gen(g_xi)). Its trained weights are saved in every Stage-1
checkpoint (`denoiser_trainable_state` in aligner_v7_best.pt) -- but nothing
in the existing pipeline ever samples from g_xi or scores it against
FID/CTTP/JFTSD; train_v7.py's own evaluate() only reports the diffusion
*loss*, never generation quality. That is exactly the gap this script fills:
it needs NO retraining, only rebuilding g_xi's architecture from the
checkpoint's saved args and loading its already-trained weights, then running
the SAME ConTSGMetricsAccumulator gen-eval protocol used everywhere else
(scored against the SAME fixed "both" aligner used as the metric embedder for
the main pipeline / every ablation, so numbers are directly comparable).

If g_xi's standalone FID/CTTP/JFTSD is already close to the final two-stage
model's, Stage 2 buys little beyond what Stage 1's own auxiliary loss already
achieves; if far, that gap is itself worth reporting.

(For the auxiliary denoiser trained by the JOINT single-stage ablation in
this same folder, see train_cond_v7_ablation_joint.py's own gen_eval/aux/*
metrics -- computed live during that run, not from this script.)

Usage
-----
    python ablation_diffusion_joint/eval_gxi_standalone.py \
        --source ETTh1s --length 48 \
        --metrics-aligner-ckpt feature_alignment/results_aligner_v7/ETTh1s_L48/aligner_v7_best.pt

--stage1-ckpt defaults to feature_alignment/results_aligner_v7/{source}_L{length}/aligner_v7_best.pt
(the checkpoint actually used as ALIGNER_CKPT/METRICS_ALIGNER_CKPT by the main
pipeline). --metrics-aligner-ckpt defaults to the SAME checkpoint (it doubles
as both g_xi's source and the fixed metric embedder, matching how the main
pipeline scores itself against its own aligner) but can be pointed at a
different checkpoint if desired.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_T2TS_DIR = _REPO_ROOT / "t2ts_jit"
_FA_DIR = _REPO_ROOT / "feature_alignment"
sys.path.insert(0, str(_T2TS_DIR))
sys.path.insert(0, str(_FA_DIR))

from aligner_v7 import TextToTimeSeriesAlignerV7  # noqa: E402
from train_v7 import build_denoiser  # noqa: E402
from dataset import TSFragmentDataset, SOURCES, LENGTHS, collate  # noqa: E402
from train_cond_v7 import AlignerV7MetricEmbedder  # noqa: E402
from contsg_v4_metrics import ConTSGMetricsAccumulator, MetricConfig, seed_all  # noqa: E402

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def int_or_none(v) -> int | None:
    if v is None or str(v).strip().lower() in ("", "none", "null"):
        return None
    return int(v)


def upsample_for_aligner(series_bvl: torch.Tensor, min_length: int) -> torch.Tensor:
    if min_length <= 0 or series_bvl.size(-1) >= min_length:
        return series_bvl
    return F.interpolate(series_bvl, size=min_length, mode="linear", align_corners=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, choices=SOURCES)
    p.add_argument("--length", type=int, required=True, choices=LENGTHS)
    p.add_argument("--stage1-ckpt", type=Path, default=None,
                   help="g_xi source checkpoint. Defaults to "
                        "feature_alignment/results_aligner_v7/{source}_L{length}/aligner_v7_best.pt")
    p.add_argument("--metrics-aligner-ckpt", type=Path, default=None,
                   help="Fixed reference aligner for FID/CTTP/JFTSD. Defaults to --stage1-ckpt.")
    p.add_argument("--cache-dir", type=Path, default=Path("./hf_cache"))
    p.add_argument("--max-rows", type=int_or_none, default=None)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--min-length", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--gen-eval-max-batches", type=int, default=0, help="0 means full val set.")
    p.add_argument("--metrics-k", type=int, default=5)
    p.add_argument("--metrics-mdd-bins", type=int, default=32)
    p.add_argument("--metrics-max-samples", type=int, default=0)
    p.add_argument("--metrics-seed", type=int, default=123)
    p.add_argument("--hf-token-env", default="HF_TOKEN")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=None,
                   help="Append a JSON line here (defaults to <stage1-ckpt dir>/gxi_standalone_metrics.txt).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    hf_token = os.environ.get(args.hf_token_env) or None

    stage1_ckpt = args.stage1_ckpt
    if stage1_ckpt is None:
        stage1_ckpt = _FA_DIR / "results_aligner_v7" / f"{args.source}_L{args.length}" / "aligner_v7_best.pt"
    metrics_ckpt = args.metrics_aligner_ckpt if args.metrics_aligner_ckpt is not None else stage1_ckpt
    out_path = args.out if args.out is not None else stage1_ckpt.resolve().parent / "gxi_standalone_metrics.txt"

    print(f"[gxi-eval] stage1 ckpt (g_xi source) = {stage1_ckpt}")
    print(f"[gxi-eval] metrics embedder ckpt     = {metrics_ckpt}")

    payload = torch.load(str(stage1_ckpt), map_location="cpu", weights_only=False)
    saved_args = payload["args"]
    denoiser_state = payload.get("denoiser_trainable_state")
    if not denoiser_state:
        raise KeyError(
            f"{stage1_ckpt} has no (non-empty) 'denoiser_trainable_state' -- was this trained with "
            "feature_alignment/train_v7.py's V7JointModel (--diffusion-weight > 0)?"
        )

    # ---------- reconstruct the EXACT Stage-1 train/val split ----------
    ds = TSFragmentDataset(
        source=args.source, length=args.length,
        min_length=int(saved_args.get("min_length", args.min_length)),
        max_rows=args.max_rows, cache_dir=args.cache_dir,
    )
    n_total = len(ds)
    val_ratio = float(saved_args.get("val_ratio", args.val_ratio))
    n_val = min(int(round(n_total * val_ratio)) if val_ratio > 0 else 0, n_total - 1)
    n_train = n_total - n_val
    if n_val <= 0:
        raise RuntimeError("val_ratio produced an empty val split; cannot score gen-eval.")
    seed_arg = int(saved_args.get("seed", 0))
    val_split_seed = saved_args.get("val_split_seed")
    split_seed = seed_arg if val_split_seed is None else int(val_split_seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(split_seed))
    print(f"[gxi-eval] split: {n_total} total ({n_train} train / {n_val} val), split_seed={split_seed}")

    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=collate, num_workers=0, drop_last=False)

    # ---------- rebuild g_xi's text encoder (the Stage-1 aligner) ----------
    train_series = ds.series.index_select(0, torch.as_tensor(train_ds.indices, dtype=torch.long)).float()
    ts_mean = float(train_series.mean().item())
    ts_std = max(float(train_series.std(unbiased=False).item()), 1e-5)

    llm_dtype = DTYPES.get(saved_args.get("llm_dtype", "bfloat16"), torch.bfloat16)
    aligner = TextToTimeSeriesAlignerV7(
        llm_id=saved_args["llm_id"], chronos_model_id=saved_args["chronos_model_id"],
        proj_dim=int(saved_args["proj_dim"]), llm_dtype=llm_dtype, device=device,
        init_temperature=float(saved_args["init_temperature"]), min_temperature=float(saved_args["min_temperature"]),
        hf_token=hf_token, tune_side=saved_args.get("tune_side", "both"),
        text_lora_r=int(saved_args.get("text_lora_r", 8)), text_lora_alpha=int(saved_args.get("text_lora_alpha", 16)),
        text_lora_dropout=float(saved_args.get("text_lora_dropout", 0.05)),
        text_lora_target_modules=tuple(saved_args.get("text_lora_target_modules", ("q_proj", "k_proj", "v_proj", "o_proj"))),
        ts_lora_r=int(saved_args.get("ts_lora_r", 8)), ts_lora_alpha=int(saved_args.get("ts_lora_alpha", 16)),
        ts_lora_dropout=float(saved_args.get("ts_lora_dropout", 0.05)),
        ts_lora_target_modules=tuple(saved_args.get("ts_lora_target_modules", ("q", "k", "v", "o"))),
        ts_global_mean=ts_mean, ts_global_std=ts_std,
    )
    aligner_state = payload.get("aligner_trainable_state")
    if aligner_state:
        model_state = aligner.state_dict()
        filtered = {k: v for k, v in aligner_state.items()
                    if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)}
        missing, unexpected = aligner.load_state_dict(filtered, strict=False)
        print(f"[gxi-eval] loaded aligner weights (matched={len(filtered)} missing={len(missing)} unexpected={len(unexpected)})")
    aligner.eval()
    for p in aligner.parameters():
        p.requires_grad_(False)

    # ---------- rebuild g_xi (Denoiser) from the EXACT saved hyperparameters ----------
    diffusion_length = int_or_none(saved_args.get("diffusion_length", 64))
    seq_len = diffusion_length if diffusion_length is not None else ds.upsampled_length
    aux_ns = argparse.Namespace(
        img_size=int(saved_args.get("img_size", 8)), patch_size=int(saved_args.get("patch_size", 2)),
        in_channels=int(saved_args.get("in_channels", 1)), hidden_channels=int(saved_args.get("hidden_channels", 64)),
        depth=int(saved_args.get("depth", 2)), num_heads=int(saved_args.get("num_heads", 4)),
        attn_dropout=float(saved_args.get("attn_dropout", 0.0)), proj_dropout=float(saved_args.get("proj_dropout", 0.0)),
        bottleneck_dim=int(saved_args.get("bottleneck_dim", 64)),
        p_mean=float(saved_args.get("p_mean", 0.0)), p_std=float(saved_args.get("p_std", 1.0)),
        t_eps=float(saved_args.get("t_eps", 1e-3)), noise_scale=float(saved_args.get("noise_scale", 1.0)),
        ema_decay1=float(saved_args.get("ema_decay1", 0.9999)), ema_decay2=float(saved_args.get("ema_decay2", 0.999)),
        sampling_method=saved_args.get("sampling_method", "heun"),
        num_sampling_steps=int(saved_args.get("num_sampling_steps", 50)),
        interval_min=float(saved_args.get("interval_min", 0.0)), interval_max=float(saved_args.get("interval_max", 1.0)),
    )
    g_xi = build_denoiser(aux_ns, text_embed_dim=int(saved_args["proj_dim"]), seq_len=seq_len, device=device)
    missing, unexpected = g_xi.load_state_dict(denoiser_state, strict=False)
    print(f"[gxi-eval] loaded g_xi weights (missing={len(missing)} unexpected={len(unexpected)})")
    g_xi.eval()
    for p in g_xi.parameters():
        p.requires_grad_(False)

    # g_xi's own diffusion-target normalization: mirrors train_v7.py's
    # _compute_series_stats(train_ds) exactly (train split, upsampled to seq_len).
    train_series_up = train_series
    if train_series_up.size(-1) < seq_len:
        train_series_up = F.interpolate(
            train_series_up.unsqueeze(1), size=seq_len, mode="linear", align_corners=False
        ).squeeze(1)
    aux_min = float(train_series_up.min().item())
    aux_max = float(train_series_up.max().item())
    print(f"[gxi-eval] g_xi normalization stats: min={aux_min:.6f} max={aux_max:.6f} seq_len={seq_len}")

    # ---------- fixed reference aligner for FID/CTTP/JFTSD scoring ----------
    metric_embedder = AlignerV7MetricEmbedder(
        ckpt_path=metrics_ckpt, device=device, hf_token=hf_token,
        ts_global_mean=ts_mean, ts_global_std=ts_std,
    )

    # ---------- generate + score ----------
    seed_all(args.metrics_seed)
    acc = ConTSGMetricsAccumulator(
        embedder=metric_embedder,
        cfg=MetricConfig(
            k=args.metrics_k, mdd_bins=args.metrics_mdd_bins,
            max_samples=args.metrics_max_samples, seed=args.metrics_seed,
        ),
    )
    n_seen = 0
    with torch.no_grad():
        for bi, (captions, series) in enumerate(loader):
            if args.gen_eval_max_batches > 0 and bi >= args.gen_eval_max_batches:
                break
            real_bvl = series.to(device).float()
            text_emb = aligner.encode_text(captions)
            gen_blv = g_xi.generate(text_emb).float()
            gen_bvl = gen_blv.permute(0, 2, 1).contiguous()
            gen_bvl = ((gen_bvl + 1.0) * 0.5) * (aux_max - aux_min) + aux_min
            real_for_align = upsample_for_aligner(real_bvl, args.min_length)
            gen_for_align = upsample_for_aligner(gen_bvl, args.min_length)
            acc.update(captions=captions, real_series_bvl=real_for_align, gen_series_bvl=gen_for_align)
            n_seen += len(captions)
    metrics = acc.compute()

    print(f"[gxi-eval] standalone g_xi generation metrics (n_scored={n_seen}):")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    rec = {
        "kind": "gxi_standalone", "source": args.source, "length": args.length,
        "stage1_ckpt": str(stage1_ckpt), "metrics_aligner_ckpt": str(metrics_ckpt),
        "n_scored": n_seen, **metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[gxi-eval] appended -> {out_path}")


if __name__ == "__main__":
    main()
