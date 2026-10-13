"""
M3JEPA — M3-JEPA baseline adapted to our text<->time-series feature alignment.

The two modalities are text (captions) and time series. Two proxy tasks,
`text2ts` and `ts2text`, are optimized (alternately or jointly). For each task:
  - the MMoE predictor maps the source-modality raw embedding to the target
    modality's latent space; a smooth-L1 JEPA regression loss pulls the
    prediction towards the true target embedding;
  - a symmetric single-positive InfoNCE loss aligns the two modalities in a
    shared CL space.

Training produces aligned shared-space embeddings (via the modality CL heads),
evaluated by cross-modal retrieval accuracy — directly comparable to our own
aligner. There is no separate downstream / generative stage.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import JepaEncoders
from .losses import smooth_l1_prediction_loss, symmetric_contrastive_loss
from .mmoe import MoEPredictor, create_mmoe_predictor

TASKS = ("text2ts", "ts2text")


class M3JEPA(nn.Module):
    def __init__(
        self,
        encoders: JepaEncoders,
        num_experts: int = 12,
        k_expert: int = 4,
        hidden_size: int = 512,
        dropout_rate: float = 0.1,
        temperature: float = 0.03,
        l2_weight: float = 1.0,
        cl_weight: float = 1.0,
        task_weight_text2ts: float = 0.5,
        task_weight_ts2text: float = 0.5,
    ):
        super().__init__()
        self.encoders = encoders
        text_dim = encoders.text_dim
        ts_dim = encoders.ts_dim

        self.predictor: MoEPredictor = create_mmoe_predictor(
            num_experts=num_experts,
            text_dim=text_dim,
            ts_dim=ts_dim,
            hidden_size=hidden_size,
            k=k_expert,
            drop=dropout_rate,
        )

        # Modality embeddings added to the source before prediction (M3-JEPA's
        # text_to_image_embedding / image_to_text_embedding analogue).
        self.text2ts_embedding = nn.Parameter(torch.empty(1, text_dim))
        self.ts2text_embedding = nn.Parameter(torch.empty(1, ts_dim))

        # Modality CL heads -> shared retrieval space (used for the contrastive
        # target and for evaluation retrieval).
        self.cl_text_emb = nn.Linear(text_dim, hidden_size, bias=True)
        self.cl_ts_emb = nn.Linear(ts_dim, hidden_size, bias=True)

        self.temperature = temperature
        self.l2_weight = l2_weight
        self.cl_weight = cl_weight
        self.task_weights = {"text2ts": task_weight_text2ts, "ts2text": task_weight_ts2text}

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.kaiming_uniform_(self.text2ts_embedding, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.ts2text_embedding, nonlinearity="relu")
        for layer in (self.cl_text_emb, self.cl_ts_emb):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ------------------------------------------------------------------ #
    def _encode(self, captions: Sequence[str], series: torch.Tensor):
        text_h = self.encoders.encode_text_raw(captions)
        ts_h = self.encoders.encode_ts_raw(series)
        return text_h, ts_h

    def shared_embeddings(self, text_h: torch.Tensor, ts_h: torch.Tensor):
        """Normalized shared-space embeddings for cross-modal retrieval."""
        text_shared = F.normalize(self.cl_text_emb(text_h), dim=-1)
        ts_shared = F.normalize(self.cl_ts_emb(ts_h), dim=-1)
        return text_shared, ts_shared

    def encode_text(self, captions: Sequence[str]) -> torch.Tensor:
        """captions -> normalized shared-space text embedding (B, hidden_size).

        This is the M3-JEPA analogue of V7's `encode_text`: the text representation
        in the aligned CL space, used both as the stage-2 conditioning vector and
        for cross-modal retrieval.
        """
        text_h = self.encoders.encode_text_raw(captions)
        return F.normalize(self.cl_text_emb(text_h), dim=-1)

    def encode_time_series(self, series: torch.Tensor) -> torch.Tensor:
        """series (B, V, L) -> normalized shared-space ts embedding (B, hidden_size)."""
        ts_h = self.encoders.encode_ts_raw(series)
        return F.normalize(self.cl_ts_emb(ts_h), dim=-1)

    def _task_loss(self, task: str, text_h: torch.Tensor, ts_h: torch.Tensor):
        if task == "text2ts":
            src = text_h + self.text2ts_embedding
            target = ts_h
            cl_target = self.cl_ts_emb(ts_h)
        elif task == "ts2text":
            src = ts_h + self.ts2text_embedding
            target = text_h
            cl_target = self.cl_text_emb(text_h)
        else:
            raise ValueError(f"Invalid task: {task!r}. Expected one of {TASKS}.")

        pred, cl_src = self.predictor(src, task)
        l2 = smooth_l1_prediction_loss(pred, target)
        cl = symmetric_contrastive_loss(cl_src, cl_target, self.temperature)
        loss = self.task_weights[task] * (self.l2_weight * l2 + self.cl_weight * cl)
        return loss, l2, cl

    def forward(self, captions: Sequence[str], series: torch.Tensor, tasks: Sequence[str]) -> dict:
        """Encode once, accumulate loss over `tasks` (one for alternate, two for joint)."""
        if isinstance(tasks, str):
            tasks = (tasks,)
        text_h, ts_h = self._encode(captions, series)

        total = text_h.new_zeros(())
        l2_sum = text_h.new_zeros(())
        cl_sum = text_h.new_zeros(())
        for task in tasks:
            loss, l2, cl = self._task_loss(task, text_h, ts_h)
            total = total + loss
            l2_sum = l2_sum + l2
            cl_sum = cl_sum + cl
        n = max(len(tasks), 1)

        with torch.no_grad():
            text_shared, ts_shared = self.shared_embeddings(text_h, ts_h)
            sim = text_shared @ ts_shared.t()
            cosine = sim.diagonal()
            labels = torch.arange(sim.size(0), device=sim.device)
            acc_t2s = sim.argmax(dim=1).eq(labels).float().mean()
            acc_s2t = sim.argmax(dim=0).eq(labels).float().mean()

        return {
            "loss": total,
            "loss_pred": (l2_sum / n).detach(),
            "loss_cl": (cl_sum / n).detach(),
            "cosine": cosine,
            "acc_t2s": acc_t2s,
            "acc_s2t": acc_s2t,
            "tasks": tuple(tasks),
        }


def build_m3jepa(
    encoders: JepaEncoders,
    num_experts: int = 12,
    k_expert: int = 4,
    hidden_size: int = 512,
    dropout_rate: float = 0.1,
    temperature: float = 0.03,
    l2_weight: float = 1.0,
    cl_weight: float = 1.0,
    task_weight_text2ts: float = 0.5,
    task_weight_ts2text: float = 0.5,
) -> M3JEPA:
    return M3JEPA(
        encoders=encoders,
        num_experts=num_experts,
        k_expert=k_expert,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        temperature=temperature,
        l2_weight=l2_weight,
        cl_weight=cl_weight,
        task_weight_text2ts=task_weight_text2ts,
        task_weight_ts2text=task_weight_ts2text,
    )
