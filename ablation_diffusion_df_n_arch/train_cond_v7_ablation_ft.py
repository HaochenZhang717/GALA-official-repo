"""
Stage-2 driver for the fine-tuning-METHOD x diffusion-arm ablation.

Freeze a stage-1 ft-method ablation aligner (ft_method in {lora, head, full},
each trained with or without the stage-1 diffusion loss) and train the
conditional diffusion model EXACTLY like the main V7 pipeline, teeing the
generation-eval metrics into a plain-text JSONL file so they can be read back
offline without touching W&B.

Why this file is not just train_cond_v7_ablation.py
---------------------------------------------------
The upstream stage-2 builder (`t2ts_jit/train_cond_v7.py::_build_v7_aligner`)
always reconstructs a *standard* `TextToTimeSeriesAlignerV7`, which only knows
the LoRA-or-frozen architecture. That is fine for `ft_method=lora` checkpoints,
but a `head` checkpoint has extra `text_head.*`/`ts_head.*` modules and a `full`
checkpoint stores the entire (non-PEFT) backbone -- in both cases the saved keys
do NOT match a standard aligner, so `strict=False` would silently drop them and
the fine-tuning would be lost.

So, in addition to the gen-eval TXT tee (identical to train_cond_v7_ablation.py),
this driver swaps the *conditioning* text encoder for one that rebuilds
`TextToTimeSeriesAlignerV7AblationFT` using the `ft_method`/head/LoRA fields
recorded in the stage-1 checkpoint's `args`. The metric embedder is left
untouched (it stays the FIXED main "both" aligner) so FID/adherence remain
comparable across all variants.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_T2TS_DIR = _REPO_ROOT / "t2ts_jit"
_FEATURE_ALIGN_DIR = _REPO_ROOT / "feature_alignment"
# train_cond_v6/v7 live in t2ts_jit and perform their own feature_alignment/
# sys.path wiring on import; we add feature_alignment too so we can import the
# ablation-ft aligner class directly.
sys.path.insert(0, str(_T2TS_DIR))
sys.path.insert(0, str(_FEATURE_ALIGN_DIR))

import train_cond_v6 as _v6  # noqa: E402

# IMPORTANT: wandb.init() REBINDS wandb.log (the run's real logger replaces the
# module-level function), so any tee installed before init is silently wiped out
# -- that bug left every test_metrics.txt with only its `meta` header. Wrap the
# real wandb.init *before* importing train_cond_v7 (which snapshots the current
# _v6.wandb.init into its module-level _ORIG_WANDB_INIT) so the gen-eval tee is
# re-installed on wandb.log AFTER each init, once the real logger is in place.
_REAL_WANDB_INIT = _v6.wandb.init


def _wandb_init_then_tee(*args, **kwargs):
    run = _REAL_WANDB_INIT(*args, **kwargs)
    _install_gen_eval_tee()  # defined below; bound at call time (post-import)
    return run


_v6.wandb.init = _wandb_init_then_tee

import train_cond_v7 as _v7  # noqa: E402


# --------------------------------------------------------------------------- #
# gen-eval TXT tee (identical behavior to ablation_diffusion/train_cond_v7_ablation.py)
# --------------------------------------------------------------------------- #
def _argv_value(flag: str) -> str | None:
    """Read a value passed as `--flag value` or `--flag=value` from sys.argv."""
    argv = sys.argv
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


def _resolve_txt_path() -> Path:
    save = _argv_value("--save")
    if save:
        return Path(save).resolve().parent / "test_metrics.txt"
    return _HERE / "test_metrics.txt"


_TXT_PATH = _resolve_txt_path()


def _append_record(rec: dict) -> None:
    _TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_TXT_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _install_gen_eval_tee() -> None:
    """Wrap wandb.log so every `gen_eval/*` payload is also appended to TXT."""
    orig_log = _v6.wandb.log

    def tee_log(data=None, *args, **kwargs):
        try:
            if isinstance(data, dict) and any(k.startswith("gen_eval/") for k in data):
                step = kwargs.get("step")
                if step is None and args:
                    step = args[0]
                rec: dict = {"kind": "gen_eval", "step": step}
                for k, v in data.items():
                    if k == "gen_eval/epoch":
                        rec["epoch"] = v
                    elif k.startswith("gen_eval/"):
                        rec[k[len("gen_eval/"):]] = v
                _append_record(rec)
        except Exception as e:  # never let TXT logging crash training
            print(f"[ablation-txt] WARN failed to write gen-eval metrics: {e}")
        return orig_log(data, *args, **kwargs)

    _v6.wandb.log = tee_log


# --------------------------------------------------------------------------- #
# ft-method-aware conditioning aligner
# --------------------------------------------------------------------------- #
def _build_ablation_ft_aligner(
    saved_args: dict,
    ckpt_state: dict,
    device: torch.device,
    hf_token: str | None,
    ts_global_mean: float,
    ts_global_std: float,
):
    """Rebuild the EXACT stage-1 aligner architecture from the checkpoint args.

    Mirrors t2ts_jit/train_cond_v7.py::_build_v7_aligner but instantiates
    TextToTimeSeriesAlignerV7AblationFT with the saved ft_method/head/LoRA
    config, so head/full checkpoints load their trained weights correctly.
    """
    from aligner_v7_ablation_ft import TextToTimeSeriesAlignerV7AblationFT

    llm_dtype = _v6.DTYPES.get(saved_args.get("llm_dtype", "bfloat16"), torch.bfloat16)
    ft_method = saved_args.get("ft_method", "lora")
    model = TextToTimeSeriesAlignerV7AblationFT(
        llm_id=saved_args["llm_id"],
        chronos_model_id=saved_args["chronos_model_id"],
        proj_dim=int(saved_args["proj_dim"]),
        llm_dtype=llm_dtype,
        device=device,
        init_temperature=float(saved_args["init_temperature"]),
        min_temperature=float(saved_args["min_temperature"]),
        hf_token=hf_token,
        tune_side="both",
        ft_method=ft_method,
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
        head_depth=int(saved_args.get("head_depth", 1)),
        head_num_heads=int(saved_args.get("head_num_heads", 12)),
        head_ffn_dim=int(saved_args.get("head_ffn_dim", 2560)),
        head_dropout=float(saved_args.get("head_dropout", 0.0)),
        grad_checkpointing=False,  # inference / frozen conditioning encoder
        ts_global_mean=float(ts_global_mean),
        ts_global_std=float(ts_global_std),
    )
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in ckpt_state.items()
        if (k in model_state and tuple(model_state[k].shape) == tuple(v.shape))
    }
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    # Guard: every saved trainable tensor must land somewhere. If not, the
    # architecture we rebuilt does not match how stage-1 was trained.
    if len(filtered) != len(ckpt_state):
        dropped = sorted(set(ckpt_state) - set(filtered))
        print(
            f"[AblationFT][WARN] ft_method={ft_method}: only loaded "
            f"{len(filtered)}/{len(ckpt_state)} stage-1 trainable tensors; "
            f"dropped {len(dropped)} (first few: {dropped[:5]}). "
            "This usually means the rebuilt architecture is wrong."
        )
    return model, filtered, missing, unexpected


class AblationFTTextEncoder(nn.Module):
    """Frozen stage-1 ft-method aligner used as the conditioning text encoder.

    Drop-in replacement for t2ts_jit/train_cond_v7.py::AlignerV7TextEncoder that
    rebuilds the ablation-ft architecture instead of the standard aligner.
    """

    def __init__(
        self,
        ckpt_path: Path,
        device: torch.device,
        hf_token: str | None = None,
        ts_global_mean: float | None = None,
        ts_global_std: float | None = None,
    ):
        super().__init__()
        if ts_global_mean is None or ts_global_std is None:
            if _v7._LATEST_V7_STATS is None:
                raise RuntimeError(
                    "V7 text-encoder stats are unavailable. Build the V7 text/metric "
                    "encoder first or pass ts_global_mean/ts_global_std explicitly."
                )
            ts_global_mean, ts_global_std = _v7._LATEST_V7_STATS
        self._ckpt_path = Path(ckpt_path)
        self._hf_token = hf_token
        self._ts_global_mean = float(ts_global_mean)
        self._ts_global_std = float(ts_global_std)
        self._saved_args, ckpt_state = _v7._load_v7_ckpt(self._ckpt_path)
        self.aligner, filtered, missing, unexpected = _build_ablation_ft_aligner(
            self._saved_args,
            ckpt_state,
            device=device,
            hf_token=hf_token,
            ts_global_mean=self._ts_global_mean,
            ts_global_std=self._ts_global_std,
        )
        print(
            "[AblationFTTextEncoder] "
            f"ft_method={self._saved_args.get('ft_method', 'lora')} loaded state_dict "
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
        saved_args, ckpt_state = _v7._load_v7_ckpt(self._ckpt_path)
        model, _filtered, _missing, _unexpected = _build_ablation_ft_aligner(
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


def main() -> None:
    # Self-describing header (valid JSON) so each run is parseable.
    _append_record({
        "kind": "meta",
        "source": _argv_value("--source"),
        "length": _argv_value("--length"),
        "aligner_ckpt": _argv_value("--aligner-ckpt"),
        "metrics_aligner_ckpt": _argv_value("--metrics-aligner-ckpt"),
        "save": _argv_value("--save"),
    })
    print(f"[ablation-txt] gen-eval metrics -> {_TXT_PATH}")
    # The gen-eval tee is installed by _wandb_init_then_tee right AFTER
    # wandb.init() (see the import-time wrapper above). Installing it here would
    # be wiped out by wandb.init() rebinding wandb.log.
    # Swap the conditioning text encoder for the ft-method-aware one. This rebinds
    # the module global that both train_cond_v7._build_text_encoder_v7 and
    # train_cond_v7.main() read, so the conditioning aligner is reconstructed with
    # the correct ft_method. The metric embedder (AlignerV7MetricEmbedder) is left
    # as-is so it stays the fixed main "both" reference.
    _v7.AlignerV7TextEncoder = AblationFTTextEncoder
    # Hand off to the unmodified V7 conditional-diffusion training entrypoint.
    _v7.main()


if __name__ == "__main__":
    main()
