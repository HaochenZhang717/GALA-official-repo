"""
Stage-2 driver for the "NO stage-1 feature alignment" ablation.

Question this ablation answers
------------------------------
The main pipeline first runs stage-1 contrastive feature alignment (text<->TS)
to produce a frozen aligner, then freezes it and trains the conditional
diffusion model on top. This ablation SKIPS stage-1 entirely and trains the
conditioning text encoder JOINTLY with the diffusion model, so we can measure
what we lose by not aligning first. Two text encoders are compared:

  * frozen  : frozen embeddinggemma-300m backbone + a TRAINABLE projector
              (the denoiser's TextConditioner adapter). Use --text-encoder
              frozen-gemma. Nothing new is needed -- the upstream pipeline
              already caches the frozen embeddings and trains the projector.
  * lora    : LoRA-adapted embeddinggemma-300m (base frozen) + a trainable
              projector, trained end-to-end by the diffusion loss. Use
              --text-encoder tunable-gemma; this driver swaps the upstream
              TunableGemmaTextEncoder (which is a FULL fine-tune) for the LoRA
              variant defined below.

Why this works without touching t2ts_jit
----------------------------------------
`train_cond_v6.main()` already has a joint-training path keyed on
`--text-encoder == "tunable-gemma"`:
  * it DISABLES the text-embedding cache and encodes captions on-the-fly with
    gradients (train_cond_v6.py: `use_cached_text_emb`),
  * it adds the text encoder's trainable params to the optimizer with a separate
    LR (`--tunable-gemma-lr`) and includes them in grad-clip.
So registering a LoRA encoder under the `tunable-gemma` name (by rebinding
`train_cond_v6.TunableGemmaTextEncoder`, which `build_text_encoder` resolves at
call time) reuses that entire path unchanged. The `frozen` arm needs no swap.

The metric embedder used for FID / adherence is left untouched -- it stays the
FIXED main "both" V7 aligner (`--metrics-aligner-ckpt`), so the numbers are
directly comparable to the main pipeline and the other stage-2 ablations.

gen-eval metrics are teed to <save_dir>/test_metrics.txt exactly like
ablation_diffusion_df_n_arch/train_cond_v7_ablation_ft.py.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_T2TS_DIR = _REPO_ROOT / "t2ts_jit"
# train_cond_v6/v7 live in t2ts_jit and perform their own feature_alignment/
# sys.path wiring on import; we only need t2ts_jit/ on the path here.
sys.path.insert(0, str(_T2TS_DIR))

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
# gen-eval TXT tee (identical behavior to ablation_diffusion_df_n_arch)
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
# LoRA Gemma conditioning encoder (trained jointly with the diffusion model)
# --------------------------------------------------------------------------- #
class LoRAGemmaTextEncoder(nn.Module):
    """LoRA-adapted embeddinggemma-300m + trainable projector, trained JOINTLY
    with the diffusion model (NO stage-1 feature alignment).

    Drop-in replacement for `train_cond_v6.TunableGemmaTextEncoder`: identical
    constructor signature (model_id, dtype, device, proj_dim, hf_token) and the
    same encode() contract (mean-pool -> project -> L2-normalize), so the
    upstream build_text_encoder() / on-the-fly-encode / optimizer paths -- all
    keyed on `--text-encoder tunable-gemma` -- pick it up unchanged. The ONLY
    difference vs TunableGemmaTextEncoder is that the Gemma backbone is
    LoRA-adapted (base frozen) instead of fully fine-tuned, so only the LoRA
    adapters + projector receive gradients.

    The tokenization / prefix / pooling mirror FrozenGemmaTextEncoder exactly so
    the `frozen` and `lora` arms differ ONLY in whether the backbone is adapted.

    LoRA hyper-params are read from the environment (set by the run script) so no
    new CLI flag is added to the upstream argparse:
        LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES (comma-separated).
    Defaults match the stage-1 V7 aligner text-side LoRA config.
    """

    def __init__(self, model_id, dtype, device, proj_dim, hf_token=None):
        super().__init__()
        from peft import LoraConfig, get_peft_model

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        text_base = AutoModel.from_pretrained(model_id, torch_dtype=dtype, token=hf_token)

        r = int(os.environ.get("LORA_R", "8"))
        alpha = int(os.environ.get("LORA_ALPHA", "16"))
        dropout = float(os.environ.get("LORA_DROPOUT", "0.05"))
        targets_env = os.environ.get("LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj")
        target_modules = [t.strip() for t in targets_env.split(",") if t.strip()]
        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
        )
        # get_peft_model freezes the base weights and marks only the LoRA
        # adapters trainable.
        self.model = get_peft_model(text_base, lora_cfg)
        self.model.to(device)
        self.model.train()

        # ---------- gradient checkpointing on the Gemma backbone ----------
        # Unlike the `frozen` arm (which caches text embeddings), the `lora` arm
        # forwards the full embeddinggemma backbone every step and keeps all of
        # its activations for backprop. At bs=512 that peak OOMs a single 94GB
        # GPU. Gradient checkpointing trades recompute for a large activation
        # saving. Enabled by default; set LORA_GRAD_CKPT=0 to disable.
        if os.environ.get("LORA_GRAD_CKPT", "1") == "1":
            base = self.model.get_base_model()
            if hasattr(base, "config"):
                base.config.use_cache = False
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                # older transformers without the gradient_checkpointing_kwargs arg
                self.model.gradient_checkpointing_enable()
            # This driver runs a hand-written training loop (no HF Trainer), so
            # we must enable input grads ourselves: the checkpointed segments
            # otherwise receive only frozen-embedding / integer-id inputs that do
            # not require grad, and no gradient reaches the LoRA adapters.
            self.model.enable_input_require_grads()
            print("[LoRAGemmaTextEncoder] gradient checkpointing enabled (use_reentrant=False)")

        hidden = int(text_base.config.hidden_size)
        # Trainable projector to the conditioning dim (created in fp32).
        self.text_proj = nn.Linear(hidden, int(proj_dim)).to(device)
        self._device = torch.device(device)
        self._out_dim = int(proj_dim)

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(
            f"[LoRAGemmaTextEncoder] r={r} alpha={alpha} dropout={dropout} "
            f"targets={target_modules} proj_dim={proj_dim} "
            f"trainable {n_train:,} / {n_total:,} params"
        )

    @property
    def out_dim(self) -> int:
        return self._out_dim

    @staticmethod
    def _format(text: str) -> str:
        return f"task: search result | query: {text}"

    def encode(self, captions) -> torch.Tensor:
        texts = [self._format(c) for c in captions]
        batch = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=2047
        )
        batch = {k: v.to(self._device) for k, v in batch.items()}
        out = self.model(**batch)
        last_hidden = out.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        emb = self.text_proj(pooled.float())
        return F.normalize(emb, p=2, dim=1)


def main() -> None:
    # Self-describing header (valid JSON) so each run is parseable.
    _append_record({
        "kind": "meta",
        "ablation": "no_stage1_alignment",
        "text_encoder": _argv_value("--text-encoder"),
        "source": _argv_value("--source"),
        "length": _argv_value("--length"),
        "metrics_aligner_ckpt": _argv_value("--metrics-aligner-ckpt"),
        "tunable_proj_dim": _argv_value("--tunable-proj-dim"),
        "tunable_gemma_lr": _argv_value("--tunable-gemma-lr"),
        "lora": {
            "r": os.environ.get("LORA_R"),
            "alpha": os.environ.get("LORA_ALPHA"),
            "dropout": os.environ.get("LORA_DROPOUT"),
            "target_modules": os.environ.get("LORA_TARGET_MODULES"),
        },
        "save": _argv_value("--save"),
    })
    print(f"[ablation-txt] gen-eval metrics -> {_TXT_PATH}")
    # The gen-eval tee is installed by _wandb_init_then_tee right AFTER
    # wandb.init() (see the import-time wrapper above). Installing it here would
    # be wiped out by wandb.init() rebinding wandb.log.
    #
    # Register the LoRA Gemma encoder under the `tunable-gemma` name. This only
    # takes effect when --text-encoder=tunable-gemma (the `lora` arm); the
    # `frozen` arm uses --text-encoder=frozen-gemma and is unaffected. We rebind
    # the module global that train_cond_v6.build_text_encoder() resolves at call
    # time, so the upstream joint-training path builds the LoRA encoder instead
    # of the full fine-tune.
    _v6.TunableGemmaTextEncoder = LoRAGemmaTextEncoder
    # Hand off to the unmodified V7 conditional-diffusion training entrypoint.
    _v7.main()


if __name__ == "__main__":
    main()
