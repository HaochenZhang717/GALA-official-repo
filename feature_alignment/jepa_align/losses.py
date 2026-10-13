"""
Loss functions for the M3-JEPA baseline.

Faithful to the original M3-JEPA `src/loss_function.py`:
  - JEPA regression term: smooth-L1 between predicted and target embedding.
  - Symmetric single-positive InfoNCE in the shared CL space (batch diagonal).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def smooth_l1_prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """JEPA regression loss: predictor(source) should match the target embedding."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    return F.smooth_l1_loss(pred, target)


def contrastive_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float = 0.03) -> torch.Tensor:
    """Single-positive InfoNCE: row i of `pred` matches row i of `target`."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    pred = F.normalize(pred, dim=1)
    target = F.normalize(target, dim=1)
    logits = torch.matmul(pred, target.t()) / temperature
    labels = torch.arange(pred.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def symmetric_contrastive_loss(
    src_embed: torch.Tensor, tgt_embed: torch.Tensor, temperature: float = 0.03
) -> torch.Tensor:
    """Average of the two directional single-positive InfoNCE losses."""
    s2t = contrastive_loss(src_embed, tgt_embed, temperature)
    t2s = contrastive_loss(tgt_embed, src_embed, temperature)
    return 0.5 * (s2t + t2s)
