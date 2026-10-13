"""
TextToTimeSeriesAlignerV6

V6 design (relative to V4):
- Text side: Gemma embedding model adapted with LoRA (base weights frozen).
- Time-series side: Chronos-2 encoder adapted with LoRA (base weights frozen).
- Shared-space objective: same symmetric multi-positive InfoNCE as V4.

This keeps the training/inference interface close to `aligner_v4.py` while
reducing trainable parameters and overfitting risk on smaller TSFragment splits.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.distributed.nn.functional import all_gather as _all_gather_with_grad
from transformers import AutoModel, AutoTokenizer

from chronos import Chronos2Pipeline


def _gather_features_with_grad(t: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return t
    world_size = dist.get_world_size()
    if world_size == 1:
        return t
    return torch.cat(_all_gather_with_grad(t.contiguous()), dim=0)


def _gather_no_grad(t: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return t
    world_size = dist.get_world_size()
    if world_size == 1:
        return t
    gathered = [torch.empty_like(t) for _ in range(world_size)]
    dist.all_gather(gathered, t.contiguous())
    return torch.cat(gathered, dim=0)


class TextToTimeSeriesAlignerV6(nn.Module):
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
        text_lora_r: int = 8,
        text_lora_alpha: int = 16,
        text_lora_dropout: float = 0.05,
        text_lora_target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
        ts_lora_r: int = 8,
        ts_lora_alpha: int = 16,
        ts_lora_dropout: float = 0.05,
        ts_lora_target_modules: Sequence[str] = ("q", "k", "v", "o"),
    ):
        super().__init__()
        self._device = torch.device(device)

        # ---------- LoRA-tuned text encoder ----------
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id, token=hf_token)
        text_base = AutoModel.from_pretrained(llm_id, torch_dtype=llm_dtype, token=hf_token)
        text_lora_cfg = LoraConfig(
            r=text_lora_r,
            lora_alpha=text_lora_alpha,
            lora_dropout=text_lora_dropout,
            target_modules=list(text_lora_target_modules),
            bias="none",
        )
        self.text_encoder = get_peft_model(text_base, text_lora_cfg)
        self.text_encoder.to(self._device)
        self.text_encoder.train()
        text_dim = int(text_base.config.hidden_size)

        # ---------- LoRA-tuned Chronos-2 encoder ----------
        if chronos_pipeline is None:
            chronos_pipeline = Chronos2Pipeline.from_pretrained(
                chronos_model_id, device_map=str(self._device), token=hf_token
            )
        self.chronos_pipeline = chronos_pipeline
        ts_base = chronos_pipeline.model
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
        self.ts_model.to(self._device)
        self.ts_model.train()
        ts_dim = int(ts_base.model_dim)

        # ---------- projectors to shared space ----------
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.ts_proj = nn.Linear(ts_dim, proj_dim)

        # ---------- learnable logit scale ----------
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))
        self.min_temperature = min_temperature
        self.max_logit_scale = math.log(1.0 / min_temperature)

        self.to(self._device)

    @property
    def device(self) -> torch.device:
        return next(self.text_proj.parameters()).device

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

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

    def encode_text(self, captions: Sequence[str]) -> torch.Tensor:
        texts = [self._format_query(c) for c in captions]
        batch = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=2047)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.text_encoder(**batch)
        last_hidden = out.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        emb = self.text_proj(pooled.float())
        return F.normalize(emb, dim=-1)

    def encode_time_series(self, series: torch.Tensor) -> torch.Tensor:
        """series: (B, V, L) -> (B, proj_dim), trainable through Chronos LoRA."""
        if series.device != self.device:
            series = series.to(self.device)
        series = series.float()
        bsz, n_var, seq_len = series.shape

        context = series.reshape(bsz * n_var, seq_len)
        group_ids = torch.arange(bsz, device=self.device).repeat_interleave(n_var)

        encoder_outputs, *_ = self.ts_model.encode(context=context, group_ids=group_ids)
        hidden = encoder_outputs[0]

        # Match V4 global-token choice.
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
