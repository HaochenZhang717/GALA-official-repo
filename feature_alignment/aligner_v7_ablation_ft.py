"""
TextToTimeSeriesAlignerV7AblationFT

Stage-1 ablation aligner that varies *how* the two encoders are fine-tuned,
while holding the V7 design otherwise fixed (same contrastive objective, same
GlobalZScoreNorm, intended to run under tune_side='both').

Three fine-tuning methods, selected by `ft_method`:

  * "lora"  : both backbones frozen, LoRA adapters trained, projectors trained.
              (== the current main V7 method; reproduces aligner_v7 behavior.)
  * "head"  : both backbones FULLY frozen, a small Transformer head is appended
              on each side's pre-pool token sequence, head + projectors trained.
              Defaults are parameter-matched to LoRA r=64 (~12.6M total).
  * "full"  : both backbones fully trainable (no LoRA, no head), projectors
              trained. Optional gradient checkpointing for memory.

This module does NOT modify aligner_v7.py; it imports its reusable helpers and
re-implements the encode/forward paths with the extra `ft_method` branching.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer

from chronos import Chronos2Pipeline

# Reuse V7 building blocks verbatim so the two files cannot drift apart.
from aligner_v7 import (
    GlobalZScoreNorm,
    _gather_features_with_grad,
    _gather_no_grad,
)

FT_METHODS = ("lora", "head", "full")


class TextToTimeSeriesAlignerV7AblationFT(nn.Module):
    def __init__(
        self,
        llm_id: str = "google/embeddinggemma-300m",
        chronos_model_id: str = "amazon/chronos-2",
        chronos_pipeline: Chronos2Pipeline | None = None,
        proj_dim: int = 512,
        llm_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
        init_temperature: float = 0.07,
        min_temperature: float = 0.03,
        hf_token: str | None = None,
        tune_side: str = "both",
        # ---- fine-tuning method selector (the ablation dimension) ----
        ft_method: str = "lora",
        # ---- LoRA hyper-params (used when ft_method == "lora") ----
        text_lora_r: int = 8,
        text_lora_alpha: int = 16,
        text_lora_dropout: float = 0.05,
        text_lora_target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
        ts_lora_r: int = 8,
        ts_lora_alpha: int = 16,
        ts_lora_dropout: float = 0.05,
        ts_lora_target_modules: Sequence[str] = ("q", "k", "v", "o"),
        # ---- Transformer head hyper-params (used when ft_method == "head") ----
        head_depth: int = 1,
        head_num_heads: int = 12,
        head_ffn_dim: int = 2560,
        head_dropout: float = 0.0,
        # ---- full fine-tuning options (used when ft_method == "full") ----
        grad_checkpointing: bool = False,
        # ---- global z-score (V7 invariant) ----
        ts_global_mean: float = 0.0,
        ts_global_std: float = 1.0,
    ):
        super().__init__()
        self._device = torch.device(device)

        if tune_side != "both":
            # This ablation is defined only for tune_side='both'; we keep the
            # arg for signature parity but refuse the single-side combinations
            # (they belong to the separate tune-side ablation).
            raise ValueError(
                "TextToTimeSeriesAlignerV7AblationFT only supports tune_side='both'; "
                f"got {tune_side!r}. Use aligner_v7.TextToTimeSeriesAlignerV7 for "
                "the single-side ablation."
            )
        self.tune_side = "both"

        if ft_method not in FT_METHODS:
            raise ValueError(f"ft_method must be one of {FT_METHODS}, got {ft_method!r}")
        self.ft_method = ft_method
        self.head_depth = head_depth
        self.head_num_heads = head_num_heads
        self.head_ffn_dim = head_ffn_dim
        self.head_dropout = head_dropout
        self.grad_checkpointing = grad_checkpointing

        # ---------- text encoder ----------
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id, token=hf_token)
        text_base = AutoModel.from_pretrained(llm_id, torch_dtype=llm_dtype, token=hf_token)
        if ft_method == "lora":
            text_lora_cfg = LoraConfig(
                r=text_lora_r,
                lora_alpha=text_lora_alpha,
                lora_dropout=text_lora_dropout,
                target_modules=list(text_lora_target_modules),
                bias="none",
            )
            self.text_encoder = get_peft_model(text_base, text_lora_cfg)
        elif ft_method == "head":
            text_base.requires_grad_(False)
            self.text_encoder = text_base
        else:  # full
            text_base.requires_grad_(True)
            self.text_encoder = text_base
        self.text_encoder.to(self._device)
        text_dim = int(text_base.config.hidden_size)

        # ---------- Chronos-2 encoder ----------
        if chronos_pipeline is None:
            chronos_pipeline = Chronos2Pipeline.from_pretrained(
                chronos_model_id, device_map=str(self._device), token=hf_token
            )
        self.chronos_pipeline = chronos_pipeline
        ts_base = chronos_pipeline.model
        if ft_method == "lora":
            ts_targets = list(ts_lora_target_modules)
            ts_lora_cfg = LoraConfig(
                r=ts_lora_r,
                lora_alpha=ts_lora_alpha,
                lora_dropout=ts_lora_dropout,
                target_modules=ts_targets,
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
        elif ft_method == "head":
            ts_base.requires_grad_(False)
            self.ts_model = ts_base
            base_ts_model = ts_base
        else:  # full
            ts_base.requires_grad_(True)
            self.ts_model = ts_base
            base_ts_model = ts_base
        if not hasattr(base_ts_model, "instance_norm"):
            raise AttributeError(
                "Chronos base model is missing `instance_norm`; cannot install GlobalZScoreNorm for V7."
            )
        base_ts_model.instance_norm = GlobalZScoreNorm(
            global_mean=ts_global_mean,
            global_std=ts_global_std,
        )
        self.ts_model.to(self._device)
        ts_dim = int(ts_base.model_dim)

        # ---------- Transformer heads (only for ft_method == "head") ----------
        self.text_head: nn.Module | None = None
        self.ts_head: nn.Module | None = None
        if ft_method == "head":
            self.text_head = self._make_head(text_dim).to(self._device)
            self.ts_head = self._make_head(ts_dim).to(self._device)

        # ---------- optional gradient checkpointing (full FT) ----------
        if ft_method == "full" and grad_checkpointing:
            for tag, module in (("text", self.text_encoder), ("ts", self.ts_model)):
                if hasattr(module, "gradient_checkpointing_enable"):
                    try:
                        if hasattr(module, "config"):
                            module.config.use_cache = False
                        module.gradient_checkpointing_enable()
                        print(f"[grad-ckpt] enabled on {tag} backbone")
                    except Exception as e:
                        print(f"[warn] gradient checkpointing not enabled on {tag}: {e}")
                else:
                    print(f"[warn] {tag} backbone has no gradient_checkpointing_enable()")

        # ---------- projectors to shared space ----------
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.ts_proj = nn.Linear(ts_dim, proj_dim)

        # ---------- learnable logit scale ----------
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))
        self.min_temperature = min_temperature
        self.max_logit_scale = math.log(1.0 / min_temperature)

        self.to(self._device)
        # Apply the correct train/eval regime for frozen backbones.
        self.train()

    # ------------------------------------------------------------------ utils
    def _make_head(self, d_model: int) -> nn.Module:
        """A small pre-norm Transformer encoder appended after a frozen backbone."""
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.head_num_heads,
            dim_feedforward=self.head_ffn_dim,
            dropout=self.head_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        return nn.TransformerEncoder(layer, num_layers=self.head_depth)

    @property
    def device(self) -> torch.device:
        return next(self.text_proj.parameters()).device

    def train(self, mode: bool = True):
        super().train(mode)
        # For "lora" and "full" the backbones follow `mode` (LoRA dropout / full
        # FT both want train mode). For "head" the backbones are frozen and kept
        # in eval so their features stay deterministic; only heads/projectors
        # follow `mode`.
        if self.ft_method == "head":
            self.text_encoder.eval()
            self.ts_model.eval()
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def print_trainable_parameters(self) -> None:
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[aligner-ft={self.ft_method}] trainable {n_train:,} / {n_total:,}")
        if hasattr(self.text_encoder, "print_trainable_parameters"):
            print("[text_encoder] LoRA trainable parameters:")
            self.text_encoder.print_trainable_parameters()
        if hasattr(self.ts_model, "print_trainable_parameters"):
            print("[ts_model] LoRA trainable parameters:")
            self.ts_model.print_trainable_parameters()
        if self.text_head is not None:
            n_th = sum(p.numel() for p in self.text_head.parameters())
            n_ts = sum(p.numel() for p in self.ts_head.parameters())
            print(f"[heads] text_head={n_th:,}  ts_head={n_ts:,}  total={n_th + n_ts:,}")

    @staticmethod
    def _format_query(text: str) -> str:
        return f"task: search result | query: {text}"

    @staticmethod
    def _infer_ts_lora_targets(ts_model: nn.Module) -> list[str]:
        suffixes = set()
        for name, module in ts_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            leaf = name.split(".")[-1]
            if leaf in {"q", "k", "v", "o", "q_proj", "k_proj", "v_proj", "o_proj"}:
                suffixes.add(leaf)
        order = ["q", "k", "v", "o", "q_proj", "k_proj", "v_proj", "o_proj"]
        return [x for x in order if x in suffixes]

    # --------------------------------------------------------------- encoders
    def encode_text(self, captions: Sequence[str]) -> torch.Tensor:
        texts = [self._format_query(c) for c in captions]
        batch = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=2047)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.text_encoder(**batch)
        last_hidden = out.last_hidden_state
        attn = batch["attention_mask"]
        if self.ft_method == "head":
            # Transformer head runs in fp32 on the (frozen) token sequence.
            pad_mask = attn == 0  # (B, T) True where padded -> ignored by attention
            last_hidden = self.text_head(last_hidden.float(), src_key_padding_mask=pad_mask)
        mask = attn.unsqueeze(-1).to(last_hidden.dtype)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        emb = self.text_proj(pooled.float())
        return F.normalize(emb, dim=-1)

    def encode_time_series(self, series: torch.Tensor) -> torch.Tensor:
        """series: (B, V, L) -> (B, proj_dim)."""
        if series.device != self.device:
            series = series.to(self.device)
        series = series.float()
        bsz, n_var, seq_len = series.shape

        context = series.reshape(bsz * n_var, seq_len)
        group_ids = torch.arange(bsz, device=self.device).repeat_interleave(n_var)

        encoder_outputs, *_ = self.ts_model.encode(context=context, group_ids=group_ids)
        hidden = encoder_outputs[0]

        if self.ft_method == "head":
            hidden = self.ts_head(hidden.float())

        # Match V4/V7 global-token choice: the reg token sits at position -2.
        global_tok = hidden[:, -2, :]
        global_tok = global_tok.reshape(bsz, n_var, -1).mean(dim=1)

        emb = self.ts_proj(global_tok)
        return F.normalize(emb, dim=-1)

    @staticmethod
    def _multi_positive_ce(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=1)
        pos_count = positive_mask.sum(dim=1).clamp(min=1)
        loss_i = -(log_prob * positive_mask.float()).sum(dim=1) / pos_count.float()
        return loss_i.mean()

    def forward(
        self,
        captions: Sequence[str],
        time_series: torch.Tensor,
        series_hash: torch.Tensor | None = None,
        gather: bool = True,
    ) -> dict:
        text_emb = self.encode_text(captions)
        ts_emb = self.encode_time_series(time_series)

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

        logit_scale = self.logit_scale.clamp(max=self.max_logit_scale).exp()
        sim = text_all @ ts_all.t()
        logits = logit_scale * sim
        pos_mask = hash_all[:, None].eq(hash_all[None, :])

        loss_t2s = self._multi_positive_ce(logits, pos_mask)
        loss_s2t = self._multi_positive_ce(logits.t(), pos_mask.t())
        loss = 0.5 * (loss_t2s + loss_s2t)

        with torch.no_grad():
            cosine = sim.diagonal()
            pred_t2s = logits.argmax(dim=-1)
            pred_s2t = logits.argmax(dim=0)
            acc_t2s = hash_all[pred_t2s].eq(hash_all).float().mean()
            acc_s2t = hash_all[pred_s2t].eq(hash_all).float().mean()

        return {
            "loss": loss,
            "loss_t2s": loss_t2s,
            "loss_s2t": loss_s2t,
            "cosine": cosine,
            "acc_t2s": acc_t2s,
            "acc_s2t": acc_s2t,
            "logit_scale": logit_scale.detach(),
            "num_positive_pairs": pos_mask.float().sum().detach(),
            "text_emb": text_all,
            "ts_emb": ts_all,
        }
