"""Bridge model submodules for ConTSG.

This package contains the core components for the Bridge diffusion model:
- utils: Utility functions (timestep embedding, beta schedules, checkpointing, etc.)
- attention: Attention mechanisms (CrossAttention, Spatial1DTransformer)
- conditioning_mlp: Text-latent fusion module (ConditioningMLP)
- prototype: Domain-Unified Prototyper for latent representations
- unet: Main UNet architecture with timestep embedding and cross-attention
"""

from .utils import (
    Return,
    Return_grad,
    Return_grad_full,
    Return_grad_cfg,
    exists,
    default,
    checkpoint,
    timestep_embedding,
    zero_module,
    normalization,
    conv_nd,
    linear,
    avg_pool_nd,
    make_beta_schedule,
    make_ddim_timesteps,
    make_ddim_sampling_parameters,
    extract_into_tensor,
    instantiate_from_config,
)

from .attention import (
    CrossAttention,
    BasicTransformerBlock,
    Spatial1DTransformer,
    FeedForward,
    GEGLU,
)

from .conditioning_mlp import ConditioningMLP

from .prototype import (
    DomainUnifiedPrototyper,
    ResBlockTime,
    View,
)

from .unet import (
    UNetModel,
    Bridge,
    TimestepBlock,
    TimestepEmbedSequential,
    ResBlock,
    AttentionBlock,
    Upsample,
    Downsample,
    QKVAttention,
    QKVAttentionLegacy,
)

__all__ = [
    # Utils
    "Return",
    "Return_grad",
    "Return_grad_full",
    "Return_grad_cfg",
    "exists",
    "default",
    "checkpoint",
    "timestep_embedding",
    "zero_module",
    "normalization",
    "conv_nd",
    "linear",
    "avg_pool_nd",
    "make_beta_schedule",
    "make_ddim_timesteps",
    "make_ddim_sampling_parameters",
    "extract_into_tensor",
    "instantiate_from_config",
    # Attention
    "CrossAttention",
    "BasicTransformerBlock",
    "Spatial1DTransformer",
    "FeedForward",
    "GEGLU",
    # Conditioning
    "ConditioningMLP",
    # Prototype
    "DomainUnifiedPrototyper",
    "ResBlockTime",
    "View",
    # UNet
    "UNetModel",
    "Bridge",
    "TimestepBlock",
    "TimestepEmbedSequential",
    "ResBlock",
    "AttentionBlock",
    "Upsample",
    "Downsample",
    "QKVAttention",
    "QKVAttentionLegacy",
]
