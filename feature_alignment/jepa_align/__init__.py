"""M3-JEPA baseline adapted to text<->time-series feature alignment."""

from .encoders import GlobalZScoreNorm, JepaEncoders
from .losses import (
    contrastive_loss,
    smooth_l1_prediction_loss,
    symmetric_contrastive_loss,
)
from .m3jepa import M3JEPA, build_m3jepa
from .mmoe import MoEPredictor, create_mmoe_predictor

__all__ = [
    "GlobalZScoreNorm",
    "JepaEncoders",
    "MoEPredictor",
    "create_mmoe_predictor",
    "M3JEPA",
    "build_m3jepa",
    "smooth_l1_prediction_loss",
    "contrastive_loss",
    "symmetric_contrastive_loss",
]
