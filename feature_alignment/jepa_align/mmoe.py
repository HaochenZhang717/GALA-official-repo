"""
MoEPredictor — Multi-gate Mixture-of-Experts cross-modal predictor for M3-JEPA.

Adapted from the original M3-JEPA `src/models/MMoE.py` (Geely AI Lab, Apache-2.0)
with two changes for our text<->time-series setting:
  - operates on pooled single vectors (B, dim) instead of token sequences
    (B, seq_len, dim), since our encoders already pool to one vector per sample;
  - the two modalities are `text` (text_dim) and `ts` (ts_dim), and the proxy
    tasks are `text2ts` and `ts2text`.

For each task the predictor returns two outputs:
  - an L2/regression prediction in the *target* modality's raw dimension, and
  - a contrastive (CL) embedding in the shared `hidden_size` space.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

TASKS = ("text2ts", "ts2text")


class MoEPredictor(nn.Module):
    def __init__(
        self,
        num_experts: int,
        text_dim: int,
        ts_dim: int,
        hidden_size: int = 256,
        k: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.k = min(k, num_experts)
        self.hidden_size = hidden_size

        # Layer normalization for the two output branches.
        self.l2_norm = nn.LayerNorm(hidden_size)
        self.cl_norm = nn.LayerNorm(hidden_size)

        # Expert networks (shared across tasks; gating disentangles them).
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.Dropout(dropout),
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(hidden_size, num_experts)

        # Source-modality input projections into the shared expert space.
        self.text_embed = nn.Linear(text_dim, hidden_size, bias=True)
        self.ts_embed = nn.Linear(ts_dim, hidden_size, bias=True)

        # L2 prediction heads (-> raw target-modality dim).
        self.text_to_ts = nn.Linear(hidden_size, ts_dim, bias=True)
        self.ts_to_text = nn.Linear(hidden_size, text_dim, bias=True)

        # Contrastive-learning heads (-> shared hidden space).
        self.cl_text_to_ts = nn.Linear(hidden_size, hidden_size, bias=True)
        self.cl_ts_to_text = nn.Linear(hidden_size, hidden_size, bias=True)

        # Disentangled task embeddings for the two output branches.
        self.cl_task_embedding = nn.Parameter(torch.empty(1, hidden_size))
        self.l2_task_embedding = nn.Parameter(torch.empty(1, hidden_size))

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        def init_linear(layer: nn.Module) -> None:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        for expert in self.experts:
            expert.apply(init_linear)
        for module in [
            self.gate, self.text_embed, self.ts_embed,
            self.text_to_ts, self.ts_to_text,
            self.cl_text_to_ts, self.cl_ts_to_text,
        ]:
            init_linear(module)
        nn.init.kaiming_uniform_(self.cl_task_embedding, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.l2_task_embedding, nonlinearity="relu")

    def forward(self, x: torch.Tensor, task: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, src_dim) -> (l2_pred (B, tgt_dim), cl_emb (B, hidden_size))."""
        if task not in TASKS:
            raise ValueError(f"Invalid task: {task!r}. Expected one of {TASKS}.")

        x = F.gelu(self._apply_task_embedding(x, task))

        l2_input = x + self.l2_task_embedding
        cl_input = x + self.cl_task_embedding

        l2_output = self.l2_norm(self._forward_moe(l2_input))
        l2_output = F.gelu(l2_output) + l2_input

        cl_output = self.cl_norm(self._forward_moe(cl_input))
        cl_output = F.gelu(cl_output) + cl_input

        return self._apply_task_output(l2_output, cl_output, task)

    def _forward_moe(self, x: torch.Tensor) -> torch.Tensor:
        """Top-k gated mixture over experts. x: (B, hidden_size)."""
        gate_output = self.gate(x)
        topk_values, topk_indices = torch.topk(F.softmax(gate_output, dim=-1), self.k, dim=-1)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (B, E, H)
        idx = topk_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1))  # (B, k, H)
        selected = torch.gather(expert_outputs, 1, idx)  # (B, k, H)
        weighted = selected * topk_values.unsqueeze(-1)
        return weighted.sum(dim=1)  # (B, H)

    def _apply_task_embedding(self, x: torch.Tensor, task: str) -> torch.Tensor:
        if task == "text2ts":
            return self.text_embed(x)
        return self.ts_embed(x)

    def _apply_task_output(
        self, l2_output: torch.Tensor, cl_output: torch.Tensor, task: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if task == "text2ts":
            return self.text_to_ts(l2_output), self.cl_text_to_ts(cl_output)
        return self.ts_to_text(l2_output), self.cl_ts_to_text(cl_output)


def create_mmoe_predictor(
    num_experts: int,
    text_dim: int,
    ts_dim: int,
    hidden_size: int = 256,
    k: int = 2,
    drop: float = 0.0,
) -> MoEPredictor:
    return MoEPredictor(
        num_experts=num_experts,
        text_dim=text_dim,
        ts_dim=ts_dim,
        hidden_size=hidden_size,
        k=k,
        dropout=drop,
    )
