"""
TextToTimeSeriesAlignerV4

V3 design:
- Text side: full fine-tuning of Gemma embedding model (mean-pooled sentence embedding).
- Time-series side: full fine-tuning of Chronos-2 encoder.
- TS representation: use Chronos encoder output's second-to-last token
  (the requested global token), then mean-pool across variates.
- Alignment objective: symmetric InfoNCE (CLIP-style) with learnable temperature.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
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


class TextToTimeSeriesAlignerV4(nn.Module):
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
    ):
        super().__init__()
        self._device = torch.device(device)

        # ---------- trainable text encoder ----------
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id, token=hf_token)
        self.text_encoder = AutoModel.from_pretrained(llm_id, torch_dtype=llm_dtype, token=hf_token)
        self.text_encoder.to(self._device)
        self.text_encoder.train()
        for p in self.text_encoder.parameters():
            p.requires_grad_(True)
        text_dim = int(self.text_encoder.config.hidden_size)

        # ---------- trainable Chronos-2 encoder ----------
        if chronos_pipeline is None:
            chronos_pipeline = Chronos2Pipeline.from_pretrained(
                chronos_model_id, device_map=str(self._device), token=hf_token
            )
        self.chronos_pipeline = chronos_pipeline
        self.ts_model = chronos_pipeline.model
        for p in self.ts_model.parameters():
            p.requires_grad_(True)
        ts_dim = int(self.ts_model.model_dim)

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

    @staticmethod
    def _format_query(text: str) -> str:
        # Keep the same prompt style used in embeddinggemma_minimal.py
        return f"task: search result | query: {text}"

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
        """series: (B, V, L) -> (B, proj_dim), trainable through Chronos."""
        if series.device != self.device:
            series = series.to(self.device)
        series = series.float()
        B, V, L = series.shape

        # Chronos encode expects flattened context: (B*V, L) + group_ids.
        context = series.reshape(B * V, L)
        group_ids = torch.arange(B, device=self.device).repeat_interleave(V)

        encoder_outputs, *_ = self.ts_model.encode(context=context, group_ids=group_ids)
        # encoder_outputs[0]: (B*V, n_tokens, d_model)
        hidden = encoder_outputs[0]

        # Requested global token choice: second-to-last token.
        global_tok = hidden[:, -2, :]  # (B*V, d_model)
        global_tok = global_tok.reshape(B, V, -1).mean(dim=1)  # (B, d_model)

        emb = self.ts_proj(global_tok)
        return F.normalize(emb, dim=-1)

    @staticmethod
    def _multi_positive_ce(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
        """Row-wise CE with potentially multiple positives per anchor.

        logits: (N, N)
        positive_mask: bool (N, N), positive_mask[i, j]=True if j is a positive for i.
        """
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
        text_emb = self.encode_text(captions)         # frozen text tower + trainable text_proj
        ts_emb = self.encode_time_series(time_series) # full-finetune Chronos + ts_proj

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
        pos_mask = hash_all[:, None].eq(hash_all[None, :])  # (N, N)

        loss_t2s = self._multi_positive_ce(logits, pos_mask)
        loss_s2t = self._multi_positive_ce(logits.t(), pos_mask.t())
        loss = 0.5 * (loss_t2s + loss_s2t)

        with torch.no_grad():
            cosine = sim.diagonal()
            pred_t2s = logits.argmax(dim=-1)
            pred_s2t = logits.argmax(dim=0)
            # Group-aware accuracy: prediction is correct if it lands on any
            # sample with the same time-series hash (multi-caption positives).
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
