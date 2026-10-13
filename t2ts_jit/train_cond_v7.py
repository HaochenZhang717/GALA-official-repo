"""
Train a conditional time-series diffusion model with an Aligner V7 text encoder.

This wraps `t2ts_jit/train_cond_v6.py` and swaps the V6-specific aligner pieces
for their V7 equivalents, including restoring the train-split global z-score
statistics saved in V7 checkpoints.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Subset, random_split

import train_cond_v6 as _v6
from aligner_v7 import TextToTimeSeriesAlignerV7

_LATEST_V7_STATS: tuple[float, float] | None = None


def _load_v7_ckpt(ckpt_path: Path) -> tuple[dict, dict]:
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    saved_args = payload["args"]
    ckpt_state = payload.get("aligner_trainable_state", payload.get("trainable_state"))
    if ckpt_state is None:
        raise KeyError("checkpoint missing aligner_trainable_state/trainable_state")
    return saved_args, ckpt_state


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


def _compute_v7_stats_from_args(args: argparse.Namespace) -> tuple[float, float]:
    split_seed = args.seed if args.val_split_seed is None else args.val_split_seed

    if args.dataset_backend == "tsfragment":
        ds = _v6.TSFragmentDataset(
            source=args.source,
            length=args.length,
            min_length=args.min_length,
            max_rows=args.max_rows,
            cache_dir=args.cache_dir,
        )
        n_total = len(ds)
        n_val = int(round(n_total * args.val_ratio)) if args.val_ratio > 0 else 0
        n_val = min(n_val, n_total - 1)
        n_train = n_total - n_val
        if n_val > 0:
            train_ds, _val_ds = random_split(
                ds,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(split_seed),
            )
        else:
            train_ds = ds
        return _compute_series_stats(train_ds)

    train_ds = _v6.VerbalTSDataset(
        dataset_name=args.verbalts_name,
        split=args.verbalts_split,
        min_length=args.min_length,
        max_rows=args.max_rows,
        data_root=args.verbalts_root,
    )
    return _compute_series_stats(train_ds)


def _build_v7_aligner(
    saved_args: dict,
    ckpt_state: dict,
    device: torch.device,
    hf_token: str | None,
    ts_global_mean: float,
    ts_global_std: float,
):
    llm_dtype = _v6.DTYPES.get(saved_args.get("llm_dtype", "bfloat16"), torch.bfloat16)
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
    filtered = {
        k: v for k, v in ckpt_state.items()
        if (k in model_state and tuple(model_state[k].shape) == tuple(v.shape))
    }
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return model, filtered, missing, unexpected


class AlignerV7TextEncoder(nn.Module):
    """Finetuned TextToTimeSeriesAlignerV7 used as a frozen text encoder."""

    def __init__(
        self,
        ckpt_path: Path,
        device: torch.device,
        hf_token: str | None = None,
        ts_global_mean: float | None = None,
        ts_global_std: float | None = None,
    ):
        super().__init__()
        # Fall back to the train-split stats computed when the V7 text/metric
        # encoder was first built. Callers like the metrics-encoder sanity-check
        # construct this without passing stats; defaulting to (0, 1) here would
        # disable GlobalZScoreNorm and silently tank retrieval accuracy.
        if ts_global_mean is None or ts_global_std is None:
            if _LATEST_V7_STATS is None:
                raise RuntimeError(
                    "V7 text-encoder stats are unavailable. Build the V7 text/metric "
                    "encoder first or pass ts_global_mean/ts_global_std explicitly."
                )
            ts_global_mean, ts_global_std = _LATEST_V7_STATS
        self._ckpt_path = Path(ckpt_path)
        self._hf_token = hf_token
        self._ts_global_mean = float(ts_global_mean)
        self._ts_global_std = float(ts_global_std)
        self._saved_args, ckpt_state = _load_v7_ckpt(self._ckpt_path)
        self.aligner, filtered, missing, unexpected = _build_v7_aligner(
            self._saved_args,
            ckpt_state,
            device=device,
            hf_token=hf_token,
            ts_global_mean=self._ts_global_mean,
            ts_global_std=self._ts_global_std,
        )
        print(
            "[AlignerV7TextEncoder] loaded state_dict "
            f"(matched={len(filtered)} missing={len(missing)} unexpected={len(unexpected)}) "
            f"stats(mean={self._ts_global_mean:.6f}, std={self._ts_global_std:.6f})"
        )
        for p in self.aligner.parameters():
            p.requires_grad_(False)
        self.aligner.eval()
        self._out_dim = int(self.aligner.text_proj.out_features)

    @property
    def out_dim(self) -> int:
        return self._out_dim

    @torch.no_grad()
    def encode(self, captions) -> torch.Tensor:
        return self.aligner.encode_text(list(captions))

    def build_retrieval_model(self, device: torch.device) -> nn.Module:
        saved_args, ckpt_state = _load_v7_ckpt(self._ckpt_path)
        model, _filtered, _missing, _unexpected = _build_v7_aligner(
            saved_args,
            ckpt_state,
            device=device,
            hf_token=self._hf_token,
            ts_global_mean=self._ts_global_mean,
            ts_global_std=self._ts_global_std,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model


class AlignerV7MetricEmbedder:
    """Load a full TextToTimeSeriesAlignerV7 model for metric embedding."""

    def __init__(
        self,
        ckpt_path: Path,
        device: torch.device,
        hf_token: str | None = None,
        ts_global_mean: float | None = None,
        ts_global_std: float | None = None,
    ):
        if ts_global_mean is None or ts_global_std is None:
            if _LATEST_V7_STATS is None:
                raise RuntimeError(
                    "V7 metric embedder stats are unavailable. Build the V7 text encoder first "
                    "or pass ts_global_mean/ts_global_std explicitly."
                )
            ts_global_mean, ts_global_std = _LATEST_V7_STATS
        saved_args, ckpt_state = _load_v7_ckpt(Path(ckpt_path))
        self.model, filtered, missing, unexpected = _build_v7_aligner(
            saved_args,
            ckpt_state,
            device=device,
            hf_token=hf_token,
            ts_global_mean=ts_global_mean,
            ts_global_std=ts_global_std,
        )
        print(
            "[AlignerV7MetricEmbedder] loaded state_dict "
            f"(matched={len(filtered)} missing={len(missing)} unexpected={len(unexpected)}) "
            f"stats(mean={float(ts_global_mean):.6f}, std={float(ts_global_std):.6f})"
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = torch.device(device)

    @torch.no_grad()
    def encode_text(self, captions) -> torch.Tensor:
        return self.model.encode_text(list(captions))

    @torch.no_grad()
    def encode_series(self, series_bvl: torch.Tensor) -> torch.Tensor:
        return self.model.encode_time_series(series_bvl)


def _parse_args_v7():
    orig_parse_args = _v6.parse_args

    def parse_args():
        argv = list(sys.argv)
        for i, tok in enumerate(argv):
            if tok == "--text-encoder" and i + 1 < len(argv) and argv[i + 1] == "aligner-v7":
                argv[i + 1] = "aligner-v6"
            elif tok == "--metrics-aligner-v7-ckpt":
                argv[i] = "--metrics-aligner-ckpt"
        old_argv = sys.argv
        sys.argv = argv
        try:
            args = orig_parse_args()
        finally:
            sys.argv = old_argv
        args.requested_text_encoder = "aligner-v7" if args.text_encoder == "aligner-v6" else args.text_encoder
        return args

    return parse_args


def _build_text_encoder_v7(args: argparse.Namespace, device: torch.device, hf_token: str | None) -> nn.Module:
    global _LATEST_V7_STATS
    if args.text_encoder == "aligner-v6":
        if args.aligner_ckpt is None:
            raise ValueError("--aligner-ckpt is required when --text-encoder=aligner-v7.")
        ts_global_mean, ts_global_std = _compute_v7_stats_from_args(args)
        _LATEST_V7_STATS = (ts_global_mean, ts_global_std)
        print(
            "[train_cond_v7] computed aligner stats from current train split: "
            f"mean={ts_global_mean:.6f} std={ts_global_std:.6f}"
        )
        return AlignerV7TextEncoder(
            ckpt_path=args.aligner_ckpt,
            device=device,
            hf_token=hf_token,
            ts_global_mean=ts_global_mean,
            ts_global_std=ts_global_std,
        )
    # Non-aligner conditioning encoders (e.g. --text-encoder=frozen-longclip for
    # the BRIDGE baseline) still need train-split series stats for the V7 metric
    # embedder used in gen-eval. Those stats are independent of the text encoder,
    # so compute them here whenever a metric aligner ckpt is configured.
    if getattr(args, "metrics_aligner_ckpt", None) is not None:
        ts_global_mean, ts_global_std = _compute_v7_stats_from_args(args)
        _LATEST_V7_STATS = (ts_global_mean, ts_global_std)
        print(
            "[train_cond_v7] computed V7 metric-embedder stats from current train split: "
            f"mean={ts_global_mean:.6f} std={ts_global_std:.6f}"
        )
    return _ORIG_BUILD_TEXT_ENCODER(args, device=device, hf_token=hf_token)


def _save_ckpt_v7(path: Path, inner: nn.Module, args: argparse.Namespace,
                  text_encoder_kind: str, text_embed_dim: int, seq_len: int,
                  normalization: dict, split_info: dict,
                  extra: dict | None = None) -> int:
    args_for_save = argparse.Namespace(**vars(args))
    if getattr(args_for_save, "requested_text_encoder", None) == "aligner-v7":
        args_for_save.text_encoder = "aligner-v7"
    if text_encoder_kind == "aligner-v6":
        text_encoder_kind = "aligner-v7"
    return _ORIG_SAVE_CKPT(
        path,
        inner,
        args_for_save,
        text_encoder_kind,
        text_embed_dim,
        seq_len,
        normalization,
        split_info,
        extra=extra,
    )


def _wandb_init_v7(*args, **kwargs):
    config = kwargs.get("config")
    if isinstance(config, dict) and config.get("text_encoder") == "aligner-v6":
        config = dict(config)
        config["text_encoder"] = "aligner-v7"
        kwargs["config"] = config
    return _ORIG_WANDB_INIT(*args, **kwargs)


def main() -> None:
    _v6.TEXT_ENCODERS = tuple(dict.fromkeys([*_v6.TEXT_ENCODERS, "aligner-v7"]))
    _v6.AlignerV6TextEncoder = AlignerV7TextEncoder
    _v6.AlignerV6MetricEmbedder = AlignerV7MetricEmbedder
    _v6.build_text_encoder = _build_text_encoder_v7
    _v6.parse_args = _parse_args_v7()
    _v6.save_ckpt = _save_ckpt_v7
    _v6.wandb.init = _wandb_init_v7
    _v6.__doc__ = __doc__
    _v6.main()


_ORIG_SAVE_CKPT = _v6.save_ckpt
_ORIG_WANDB_INIT = _v6.wandb.init
_ORIG_BUILD_TEXT_ENCODER = _v6.build_text_encoder


if __name__ == "__main__":
    main()
