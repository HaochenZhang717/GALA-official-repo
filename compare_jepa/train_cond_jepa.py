"""
Stage-2 driver for the M3-JEPA baseline (self-contained).

Freeze the stage-1 M3-JEPA aligner (Gemma + Chronos-2 with LoRA, MMoE predictor,
trained by `feature_alignment/train_jepa.py`) and train the conditional diffusion
model EXACTLY like the main V7 pipeline, teeing the generation-eval metrics into a
plain-text JSONL file so they can be read back offline without touching W&B.

Design — identical to compare_structure/train_cond_structure.py
---------------------------------------------------------------
The conditional-diffusion training is handed off to the UNMODIFIED V7 entrypoint
`t2ts_jit/train_cond_v7.py` (which wraps train_cond_v6.py). The ONLY thing this
driver changes is the *conditioning* text encoder: `train_cond_v7.AlignerV7TextEncoder`
is rebound to a `JepaTextEncoder` that rebuilds the genuine M3-JEPA aligner and
exposes the aligned shared-space text embedding. The metric embedder
(`AlignerV7MetricEmbedder`) is left untouched — it stays the FIXED main "both" V7
aligner — so FID/adherence remain comparable across all baselines.

Conditioning vector
-------------------
M3-JEPA's analogue of V7's `encode_text` is the text representation in the aligned
CL space, i.e. `normalize(cl_text_emb(pooled_text))`, dim = stage-1 `hidden_size`.
That is what conditions the diffusion model here.
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
# train_cond_v6/v7 live in t2ts_jit and do their own feature_alignment/ sys.path
# wiring; we add feature_alignment too so we can import the jepa_align package.
sys.path.insert(0, str(_T2TS_DIR))
sys.path.insert(0, str(_FEATURE_ALIGN_DIR))

import train_cond_v6 as _v6  # noqa: E402

# IMPORTANT: wandb.init() REBINDS wandb.log, so any tee installed before init is
# wiped. Wrap wandb.init BEFORE importing train_cond_v7 (which snapshots the
# current _v6.wandb.init into its _ORIG_WANDB_INIT) so the gen-eval tee is
# re-installed on wandb.log AFTER each init. Mirrors train_cond_structure.py.
_REAL_WANDB_INIT = _v6.wandb.init


def _wandb_init_then_tee(*args, **kwargs):
    run = _REAL_WANDB_INIT(*args, **kwargs)
    _install_gen_eval_tee()
    return run


_v6.wandb.init = _wandb_init_then_tee

import train_cond_v7 as _v7  # noqa: E402

from jepa_align import JepaEncoders, build_m3jepa  # noqa: E402


# --------------------------------------------------------------------------- #
# gen-eval TXT tee (identical behaviour to the structure driver)
# --------------------------------------------------------------------------- #
def _argv_value(flag: str) -> str | None:
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
            print(f"[jepa-txt] WARN failed to write gen-eval metrics: {e}")
        return orig_log(data, *args, **kwargs)

    _v6.wandb.log = tee_log


# --------------------------------------------------------------------------- #
# M3-JEPA conditioning aligner
# --------------------------------------------------------------------------- #
def _load_jepa_ckpt(ckpt_path: Path) -> tuple[dict, dict]:
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    saved_args = payload["args"]
    # M3-JEPA stage-1 ckpts use `model_trainable_state`. Stay lenient and also
    # accept the V7 keys: train_cond_v6 reuses the conditioning-encoder class
    # (rebound to JepaTextEncoder) to sanity-check the FIXED V7 metrics ckpt, so
    # this loader is occasionally handed a V7-format checkpoint. We must not
    # crash there — the rebuilt model simply shape-mismatches and is discarded
    # (the real metric embedder is the untouched AlignerV7MetricEmbedder).
    ckpt_state = (
        payload.get("model_trainable_state")
        or payload.get("aligner_trainable_state")
        or payload.get("trainable_state")
    )
    if ckpt_state is None:
        raise KeyError(
            "checkpoint missing model_trainable_state/aligner_trainable_state/trainable_state"
        )
    return saved_args, ckpt_state


def _build_jepa_aligner(
    saved_args: dict,
    ckpt_state: dict,
    device: torch.device,
    hf_token: str | None,
    ts_global_mean: float,
    ts_global_std: float,
):
    """Rebuild the EXACT stage-1 M3-JEPA aligner from the checkpoint args."""
    llm_dtype = _v6.DTYPES.get(saved_args.get("llm_dtype", "bfloat16"), torch.bfloat16)
    encoders = JepaEncoders(
        llm_id=saved_args["llm_id"],
        chronos_model_id=saved_args["chronos_model_id"],
        llm_dtype=llm_dtype,
        device=device,
        hf_token=hf_token,
        tune_side=saved_args.get("tune_side", "both"),
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
    model = build_m3jepa(
        encoders=encoders,
        num_experts=int(saved_args.get("num_experts", 12)),
        k_expert=int(saved_args.get("k_expert", 4)),
        hidden_size=int(saved_args.get("hidden_size", 512)),
        dropout_rate=float(saved_args.get("dropout_rate", 0.1)),
        temperature=float(saved_args.get("temperature", 0.03)),
        l2_weight=float(saved_args.get("l2_weight", 1.0)),
        cl_weight=float(saved_args.get("cl_weight", 1.0)),
        task_weight_text2ts=float(saved_args.get("task_weight_text2ts", 0.5)),
        task_weight_ts2text=float(saved_args.get("task_weight_ts2text", 0.5)),
    )
    # build_m3jepa constructs the CL projection heads (cl_text_emb / cl_ts_emb)
    # and the other top-level M3JEPA params on CPU; only `encoders` was placed on
    # `device`. Move the whole module to `device` so encode_text/encode_time_series
    # don't hit a cuda/cpu mismatch. .to(device) preserves param dtypes (e.g. the
    # bf16 LLM inside encoders) and is idempotent for already-on-device submodules.
    model = model.to(device)
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in ckpt_state.items()
        if (k in model_state and tuple(model_state[k].shape) == tuple(v.shape))
    }
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if len(filtered) != len(ckpt_state):
        dropped = sorted(set(ckpt_state) - set(filtered))
        print(
            f"[JEPA][WARN] only loaded {len(filtered)}/{len(ckpt_state)} stage-1 "
            f"trainable tensors; dropped {len(dropped)} (first few: {dropped[:5]}). "
            "This usually means the rebuilt architecture does not match the checkpoint."
        )
    return model, filtered, missing, unexpected


class _JepaRetrievalModel(nn.Module):
    """V7-aligner-compatible retrieval wrapper over a frozen M3-JEPA aligner.

    train_cond_v6.evaluate_aligner_retrieval() calls the retrieval model as
    `model(captions, series, gather=False)` and expects a dict with
    acc_t2s / acc_s2t / cosine (the V7 aligner forward contract). M3JEPA.forward
    has a different signature, so we adapt: compute retrieval directly in the
    M3-JEPA shared CL space, exactly as the V7 aligner does on its shared space.
    """

    def __init__(self, aligner: nn.Module):
        super().__init__()
        self.aligner = aligner

    @torch.no_grad()
    def forward(self, captions, series, series_hash=None, gather: bool = False, **_kwargs) -> dict:
        text = self.aligner.encode_text(list(captions))
        ts = self.aligner.encode_time_series(series)
        sim = text @ ts.t()
        cosine = sim.diagonal()
        labels = torch.arange(sim.size(0), device=sim.device)
        acc_t2s = sim.argmax(dim=1).eq(labels).float().mean()
        acc_s2t = sim.argmax(dim=0).eq(labels).float().mean()
        return {"acc_t2s": acc_t2s, "acc_s2t": acc_s2t, "cosine": cosine}


class JepaTextEncoder(nn.Module):
    """Frozen stage-1 M3-JEPA aligner used as the conditioning text encoder.

    Drop-in replacement for t2ts_jit/train_cond_v7.py::AlignerV7TextEncoder. The
    public interface (out_dim / encode / build_retrieval_model) is identical; the
    conditioning vector is the aligned shared-space text embedding.
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
                    "Aligner stats are unavailable. Build the text/metric encoder "
                    "first or pass ts_global_mean/ts_global_std explicitly."
                )
            ts_global_mean, ts_global_std = _v7._LATEST_V7_STATS
        self._ckpt_path = Path(ckpt_path)
        self._hf_token = hf_token
        self._ts_global_mean = float(ts_global_mean)
        self._ts_global_std = float(ts_global_std)
        self._saved_args, ckpt_state = _load_jepa_ckpt(self._ckpt_path)
        self.aligner, filtered, missing, unexpected = _build_jepa_aligner(
            self._saved_args,
            ckpt_state,
            device=device,
            hf_token=hf_token,
            ts_global_mean=self._ts_global_mean,
            ts_global_std=self._ts_global_std,
        )
        print(
            "[JepaTextEncoder] loaded state_dict "
            f"(matched={len(filtered)} missing={len(missing)} unexpected={len(unexpected)}) "
            f"stats(mean={self._ts_global_mean:.6f}, std={self._ts_global_std:.6f})"
        )
        for p in self.aligner.parameters():
            p.requires_grad_(False)
        self.aligner.eval()
        self._out_dim = int(self.aligner.cl_text_emb.out_features)

    @property
    def out_dim(self) -> int:
        return self._out_dim

    @torch.no_grad()
    def encode(self, captions) -> torch.Tensor:
        return self.aligner.encode_text(list(captions))

    def build_retrieval_model(self, device: torch.device) -> nn.Module:
        saved_args, ckpt_state = _load_jepa_ckpt(self._ckpt_path)
        model, _filtered, _missing, _unexpected = _build_jepa_aligner(
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
        return _JepaRetrievalModel(model)


def main() -> None:
    _append_record({
        "kind": "meta",
        "baseline": "m3jepa",
        "source": _argv_value("--source"),
        "length": _argv_value("--length"),
        "aligner_ckpt": _argv_value("--aligner-ckpt"),
        "metrics_aligner_ckpt": _argv_value("--metrics-aligner-ckpt"),
        "save": _argv_value("--save"),
    })
    print(f"[jepa-txt] gen-eval metrics -> {_TXT_PATH}")
    # Swap the conditioning text encoder for the M3-JEPA one. This rebinds the
    # module global that train_cond_v7._build_text_encoder_v7 reads, so the
    # conditioning aligner is reconstructed as an M3-JEPA aligner. The metric
    # embedder (AlignerV7MetricEmbedder) is left as the fixed main "both" V7 ref.
    _v7.AlignerV7TextEncoder = JepaTextEncoder
    _v7.main()


if __name__ == "__main__":
    main()
