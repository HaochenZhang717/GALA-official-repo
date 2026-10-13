"""
STRUCTURE regularizer, ported verbatim (modulo imports) from the official
STRUCTURE repo so this stage-1 baseline does NOT depend on that repo at runtime.

Source: baseline_codes/STRUCTURE/src/loss/clip_loss.py
        baseline_codes/STRUCTURE/src/utils/utils.py  (safe_normalize)

Paper: "With Limited Data for Multimodal Alignment, Let the STRUCTURE Guide You"
       (Gröger, Wen, Le, Brbić; NeurIPS 2025), arXiv:2506.16895

The regularizer preserves the within-modality neighborhood geometry of the
*original* (frozen-encoder) embeddings inside the *aligned* (post-head) space,
by matching multi-scale random-walk transition matrices via Jensen-Shannon
divergence.  Only `structure_reg` and its direct dependencies are kept here; the
CLIP contrastive term lives in the training pipeline (train_structure.py).
"""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# safe_normalize / are_embeddings_normalized
#   (ported from STRUCTURE/src/utils/utils.py)
# ---------------------------------------------------------------------------
def are_embeddings_normalized(embeddings: torch.Tensor, epsilon: float = 1e-6) -> bool:
    """True iff every row already has (approximately) unit L2 norm."""
    norms = torch.norm(embeddings, p=2, dim=-1)
    return torch.all((norms - 1.0).abs() < epsilon).item()


def safe_normalize(
    embeddings: torch.Tensor, p: int = 2, dim: int = -1, epsilon: float = 1e-12
) -> torch.Tensor:
    """Normalize embeddings only if they aren't already normalized."""
    if are_embeddings_normalized(embeddings):
        return embeddings
    return torch.nn.functional.normalize(embeddings, p=p, dim=dim, eps=epsilon)


# ---------------------------------------------------------------------------
# centering / distance options
#   (ported from STRUCTURE/src/loss/clip_loss.py)
# ---------------------------------------------------------------------------
class Centering(Enum):
    none = 0
    mean = 1
    standard = 2


def center_embeddings(
    embeddings: torch.Tensor,
    centering_type: Centering,
) -> torch.Tensor:
    if centering_type == Centering.none:
        return embeddings
    elif centering_type == Centering.mean:
        return embeddings - embeddings.mean(0, keepdim=True)
    elif centering_type == Centering.standard:
        return (embeddings - embeddings.mean(0, keepdim=True)) / embeddings.std(
            0, keepdim=True
        )


class DistanceFunction(Enum):
    cosine = 0
    rbf = 1
    spearman = 2


def compute_similarity(
    embeddings: torch.Tensor,
    distance_type: DistanceFunction,
    temperature=None,
    gamma=None,
) -> torch.Tensor:
    if distance_type == DistanceFunction.cosine:
        assert temperature is not None
        return (embeddings @ embeddings.T) / temperature
    elif distance_type == DistanceFunction.rbf:
        assert gamma is not None
        distances = torch.cdist(embeddings, embeddings, p=2)
        return torch.exp(-gamma * distances.pow(2))
    elif distance_type == DistanceFunction.spearman:
        assert temperature is not None
        ranks = embeddings.argsort(dim=-1).argsort(dim=-1).float()
        ranks = F.normalize(ranks, p=2, dim=-1)
        return (ranks @ ranks.T) / temperature


# ---------------------------------------------------------------------------
# STRUCTURE regularizer
# ---------------------------------------------------------------------------
def structure_reg(
    original_embeddings: torch.Tensor,
    aligned_embeddings: torch.Tensor,
    levels: int = 3,
    temperature: float = 0.1,
    gamma: float = 1.0,
    margin: float = 0.0,
    eps: float = 1e-12,
    weighting: str = "inverse",
    distance_type: DistanceFunction = DistanceFunction.cosine,
    centering_type: Centering = Centering.mean,
    center_first: bool = False,
):
    """
    Compute structure regularizer loss between original and aligned embeddings.
    Preserves relationships at multiple scales.
    """
    with torch.amp.autocast("cuda", enabled=False):
        # normalize embeddings for numerical stability
        if center_first:
            original_embeddings = center_embeddings(
                embeddings=original_embeddings,
                centering_type=centering_type,
            )
            aligned_embeddings = center_embeddings(
                embeddings=aligned_embeddings,
                centering_type=centering_type,
            )
        original_norm = safe_normalize(original_embeddings, p=2, dim=-1)
        aligned_norm = safe_normalize(aligned_embeddings, p=2, dim=-1)
        if not center_first:
            original_norm = center_embeddings(
                embeddings=original_norm,
                centering_type=centering_type,
            )
            aligned_norm = center_embeddings(
                embeddings=aligned_norm,
                centering_type=centering_type,
            )
        # compute similarity matrices with temperature scaling
        original_sim = compute_similarity(
            embeddings=original_norm,
            distance_type=distance_type,
            temperature=temperature,
            gamma=gamma,
        )
        aligned_sim = compute_similarity(
            embeddings=aligned_norm,
            distance_type=distance_type,
            temperature=temperature,
            gamma=gamma,
        )
        # apply robust softmax with stable log-sum-exp trick
        total_loss = 0
        for level in range(1, levels + 1):
            original_structure = torch.matrix_power(
                F.softmax(original_sim, dim=-1),
                level,
            )
            aligned_structure = torch.matrix_power(
                F.softmax(aligned_sim, dim=-1),
                level,
            )
            # use Jensen-Shannon divergence instead of KL (more stable)
            m = 0.5 * (original_structure + aligned_structure)
            js_div = 0.5 * (
                F.kl_div(
                    (aligned_structure + eps).log(), m + eps, reduction="batchmean"
                )
                + F.kl_div(
                    (original_structure + eps).log(), m + eps, reduction="batchmean"
                )
            )
            # add a margin in order to keep focusing on big improvements rather than small
            js_div = F.relu(js_div - margin)
            # apply scaling to keep loss magnitudes reasonable
            if weighting == "none":
                scaled_loss = js_div
            elif weighting == "inverse":
                # lower weight for higher levels
                scaled_loss = js_div * (1.0 / level)
            else:
                raise ValueError(f"Unknown weighting: {weighting}")
            total_loss += scaled_loss
    return total_loss / levels
