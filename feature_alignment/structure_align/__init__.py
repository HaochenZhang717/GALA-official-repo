"""STRUCTURE stage-1 feature-alignment baseline (frozen encoders + head)."""

from .structure_aligner import StructureAligner
from .structure_loss import (
    Centering,
    DistanceFunction,
    structure_reg,
)

__all__ = [
    "StructureAligner",
    "structure_reg",
    "Centering",
    "DistanceFunction",
]
