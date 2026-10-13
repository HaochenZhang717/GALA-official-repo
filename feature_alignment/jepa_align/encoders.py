"""
JepaEncoders — text + time-series encoders for the M3-JEPA baseline.

Reuses the V7 encoder setup so the M3-JEPA baseline is apples-to-apples with our
own aligner:
  - text side : google/embeddinggemma-300m, LoRA-adapted (base frozen), mean-pooled.
  - ts side   : amazon/chronos-2, LoRA-adapted (base frozen), Chronos instance norm
                replaced by dataset-level GlobalZScoreNorm, global token pooled.

Unlike the V7 aligner, the encoders expose the *raw pooled hidden* vectors
(before any shared-space projection / normalization). M3-JEPA needs them because
its JEPA regression target is the raw target-modality embedding, and its MMoE
predictor maps across the two raw dimensions.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

from chronos import Chronos2Pipeline


class GlobalZScoreNorm(nn.Module):
    """Dataset-level z-score normalization, drop-in for Chronos instance_norm."""

    def __init__(self, global_mean: float, global_std: float, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("global_mean", torch.tensor(float(global_mean), dtype=torch.float32))
        self.register_buffer("global_std", torch.tensor(max(float(global_std), eps), dtype=torch.float32))

    def forward(
        self, x: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if loc_scale is None:
            view_shape = [1] * x.ndim
            loc = self.global_mean.to(device=x.device, dtype=torch.float32).view(*view_shape)
            scale = self.global_std.to(device=x.device, dtype=torch.float32).view(*view_shape)
        else:
            loc, scale = loc_scale
            loc = loc.to(device=x.device, dtype=torch.float32)
            scale = scale.to(device=x.device, dtype=torch.float32)
        scale = torch.clamp(scale, min=self.eps)
        return ((x - loc) / scale).to(orig_dtype), (loc, scale)

    def inverse(self, x: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        loc, scale = loc_scale
        x = x * scale.to(device=x.device, dtype=torch.float32) + loc.to(device=x.device, dtype=torch.float32)
        return x.to(orig_dtype)


class JepaEncoders(nn.Module):
    """Text (Gemma) + time-series (Chronos-2) encoders returning raw pooled hidden."""

    def __init__(
        self,
        llm_id: str = "google/embeddinggemma-300m",
        chronos_model_id: str = "amazon/chronos-2",
        chronos_pipeline: Chronos2Pipeline | None = None,
        llm_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
        hf_token: str | None = None,
        tune_side: str = "both",
        text_lora_r: int = 8,
        text_lora_alpha: int = 16,
        text_lora_dropout: float = 0.05,
        text_lora_target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
        ts_lora_r: int = 8,
        ts_lora_alpha: int = 16,
        ts_lora_dropout: float = 0.05,
        ts_lora_target_modules: Sequence[str] = ("q", "k", "v", "o"),
        ts_global_mean: float = 0.0,
        ts_global_std: float = 1.0,
    ):
        super().__init__()
        self._device = torch.device(device)

        if tune_side not in ("both", "text", "ts"):
            raise ValueError(f"tune_side must be one of 'both'|'text'|'ts', got {tune_side!r}")
        self.tune_side = tune_side
        self._tune_text = tune_side in ("both", "text")
        self._tune_ts = tune_side in ("both", "ts")

        # ---------- text encoder (LoRA-adapted or frozen) ----------
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id, token=hf_token)
        text_base = AutoModel.from_pretrained(llm_id, torch_dtype=llm_dtype, token=hf_token)
        if self._tune_text:
            text_lora_cfg = LoraConfig(
                r=text_lora_r,
                lora_alpha=text_lora_alpha,
                lora_dropout=text_lora_dropout,
                target_modules=list(text_lora_target_modules),
                bias="none",
            )
            self.text_encoder = get_peft_model(text_base, text_lora_cfg)
        else:
            text_base.requires_grad_(False)
            self.text_encoder = text_base
        self.text_encoder.to(self._device)
        self.text_encoder.train() if self._tune_text else self.text_encoder.eval()
        self.text_dim = int(text_base.config.hidden_size)

        # ---------- Chronos-2 encoder (LoRA-adapted or frozen) ----------
        if chronos_pipeline is None:
            chronos_pipeline = Chronos2Pipeline.from_pretrained(
                chronos_model_id, device_map=str(self._device), token=hf_token
            )
        self.chronos_pipeline = chronos_pipeline
        ts_base = chronos_pipeline.model
        if self._tune_ts:
            ts_lora_cfg = LoraConfig(
                r=ts_lora_r,
                lora_alpha=ts_lora_alpha,
                lora_dropout=ts_lora_dropout,
                target_modules=list(ts_lora_target_modules),
                bias="none",
            )
            try:
                self.ts_model = get_peft_model(ts_base, ts_lora_cfg)
            except ValueError as e:
                if "Target modules" not in str(e):
                    raise
                auto_targets = self._infer_ts_lora_targets(ts_base)
                if not auto_targets:
                    raise
                print(
                    "[warn] provided ts_lora_target_modules not found in Chronos model; "
                    f"fallback to auto-detected targets: {auto_targets}"
                )
                ts_lora_cfg = LoraConfig(
                    r=ts_lora_r,
                    lora_alpha=ts_lora_alpha,
                    lora_dropout=ts_lora_dropout,
                    target_modules=auto_targets,
                    bias="none",
                )
                self.ts_model = get_peft_model(ts_base, ts_lora_cfg)
            base_ts_model = self.ts_model.get_base_model() if hasattr(self.ts_model, "get_base_model") else self.ts_model
        else:
            ts_base.requires_grad_(False)
            self.ts_model = ts_base
            base_ts_model = ts_base
        if not hasattr(base_ts_model, "instance_norm"):
            raise AttributeError(
                "Chronos base model is missing `instance_norm`; cannot install GlobalZScoreNorm."
            )
        base_ts_model.instance_norm = GlobalZScoreNorm(
            global_mean=ts_global_mean,
            global_std=ts_global_std,
        )
        self.ts_model.to(self._device)
        self.ts_model.train() if self._tune_ts else self.ts_model.eval()
        self.ts_dim = int(ts_base.model_dim)

        self.to(self._device)

    @property
    def device(self) -> torch.device:
        return self._device

    def train(self, mode: bool = True):
        super().train(mode)
        # A frozen encoder is always kept in eval mode so its features stay
        # deterministic (no dropout) regardless of the outer train()/eval() call.
        if not self._tune_text:
            self.text_encoder.eval()
        if not self._tune_ts:
            self.ts_model.eval()
        return self

    def print_trainable_parameters(self) -> None:
        if hasattr(self.text_encoder, "print_trainable_parameters"):
            print("[text_encoder] LoRA trainable parameters:")
            self.text_encoder.print_trainable_parameters()
        if hasattr(self.ts_model, "print_trainable_parameters"):
            print("[ts_model] LoRA trainable parameters:")
            self.ts_model.print_trainable_parameters()

    @staticmethod
    def _format_query(text: str) -> str:
        return f"task: search result | query: {text}"

    @staticmethod
    def _infer_ts_lora_targets(ts_model: nn.Module) -> list[str]:
        """Infer likely attention projection module names for Chronos/T5-like encoders."""
        suffixes = set()
        for name, module in ts_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            leaf = name.split(".")[-1]
            if leaf in {"q", "k", "v", "o", "q_proj", "k_proj", "v_proj", "o_proj"}:
                suffixes.add(leaf)
        order = ["q", "k", "v", "o", "q_proj", "k_proj", "v_proj", "o_proj"]
        return [x for x in order if x in suffixes]

    def encode_text_raw(self, captions: Sequence[str]) -> torch.Tensor:
        """captions -> (B, text_dim) mean-pooled raw hidden (float32)."""
        texts = [self._format_query(c) for c in captions]
        batch = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=2047)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.text_encoder(**batch)
        last_hidden = out.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled.float()

    def encode_ts_raw(self, series: torch.Tensor) -> torch.Tensor:
        """series: (B, V, L) -> (B, ts_dim) global-token pooled raw hidden (float32)."""
        if series.device != self.device:
            series = series.to(self.device)
        series = series.float()
        bsz, n_var, seq_len = series.shape

        context = series.reshape(bsz * n_var, seq_len)
        group_ids = torch.arange(bsz, device=self.device).repeat_interleave(n_var)

        encoder_outputs, *_ = self.ts_model.encode(context=context, group_ids=group_ids)
        hidden = encoder_outputs[0]

        # Match V4/V7 global-token choice.
        global_tok = hidden[:, -2, :]
        global_tok = global_tok.reshape(bsz, n_var, -1).mean(dim=1)
        return global_tok.float()
