"""
VerbalTS — vendored single-file copy of the diffusion-side architecture from
``baselines_repro/verbalts_repro``.

This file collects everything ``VerbalTSDenoiser`` (in ``denoiser.py``) needs:
- The VerbalTS network (multi-patch encode, residual blocks with adaLN/cross-
  attention/add conditioning, multi-patch decode + Linear(K,1) mixer).
- ``SideEncoder_Var`` (time-position + per-variate side embeddings).
- ``TextProjectorMVarMScaleMStep`` (cross-attention queries that map a text
  token sequence into per-variate / per-scale / per-diffusion-stage tokens).
- DDPM forward + DDIM reverse samplers (slim).

The interface lives in ``denoiser.VerbalTSDenoiser`` and matches ``Denoiser``:
``forward(x_ts, text_embeds) -> loss`` and ``generate(text_embeds) -> samples``.

Tweaks relative to the original ``verbalts_repro`` source:
- ``SideEncoder_Var.var_ids`` is now a buffer (moves with ``.to(device)``);
  ``time_embedding`` reads its device from the input instead of a saved
  ``self.device`` field.
- Samplers' ``self.beta/alpha/alpha_bar/...`` are stored as plain tensors with
  ``.to(device)`` called inside the sampler's ``.to()`` override so they
  follow the parent module.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- small helpers -----------------------------

def get_torch_trans(heads: int = 8, layers: int = 1, channels: int = 64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", batch_first=True
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def get_torch_cross_trans(heads: int = 8, layers: int = 1, channels: int = 64):
    encoder_layer = nn.TransformerDecoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", batch_first=True
    )
    return nn.TransformerDecoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels: int, out_channels: int, kernel_size: int):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


# ----------------------------- diffusion step embedding -----------------------------

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps: int, embedding_dim: int = 128, projection_dim: int | None = None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step: torch.Tensor) -> torch.Tensor:
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    @staticmethod
    def _build_embedding(num_steps: int, dim: float = 64) -> torch.Tensor:
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)


# ----------------------------- continuous-t embedders -----------------------------

class TimestepEmbedder(nn.Module):
    """Continuous-t embedder (sinusoidal frequencies + 2-layer MLP), mirroring
    ``TimestepEmbedder`` in ``model_jit``. Vendored here so ``verbalts.py``
    stays self-contained.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        """t: (N,) float. Returns (N, dim)."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class DiffusionEmbeddingContinuous(nn.Module):
    """Drop-in replacement for ``DiffusionEmbedding`` that accepts continuous
    float t (any range; matches the linear-path / flow-matching convention used
    by ``Denoiser``). Output shape ``(B, projection_dim)`` matches the original
    so ``ResidualBlock.diffusion_projection`` is unchanged.
    """
    def __init__(self, embedding_dim: int, projection_dim: int | None = None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = TimestepEmbedder.timestep_embedding(t, self.embedding_dim)
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x


# ----------------------------- patch / decode -----------------------------

class TsPatchEmbedding(nn.Module):
    def __init__(self, L_patch_len: int, channels: int, d_model: int, dropout: float):
        super().__init__()
        self.L_patch_len = L_patch_len
        self.padding_patch_layer = nn.ReplicationPad2d((0, L_patch_len, 0, 0))
        self.value_embedding = nn.Sequential(
            nn.Linear(L_patch_len * channels, d_model),
            nn.ReLU(),
        )

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        if x_in.shape[-1] % self.L_patch_len:
            x = self.padding_patch_layer(x_in)
        else:
            x = x_in
        x = x.unfold(dimension=3, size=self.L_patch_len, step=self.L_patch_len)
        B, C, n_var, Nl, Pl = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous().reshape(B, n_var, Nl, Pl * C)
        x = self.value_embedding(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class SidePatchEmbedding(nn.Module):
    def __init__(self, L_patch_len: int, channels: int, d_model: int, dropout: float):
        super().__init__()
        self.L_patch_len = L_patch_len
        self.padding_patch_layer = nn.ReplicationPad2d((0, L_patch_len, 0, 0))
        self.value_embedding = nn.Sequential(nn.Linear(L_patch_len * channels, d_model))

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        if x_in.shape[-1] % self.L_patch_len:
            x = self.padding_patch_layer(x_in)
        else:
            x = x_in
        x = x.unfold(dimension=3, size=self.L_patch_len, step=self.L_patch_len)
        B, C, n_var, Nl, Pl = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous().reshape(B, n_var, Nl, Pl * C)
        x = self.value_embedding(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PatchDecoder(nn.Module):
    def __init__(self, L_patch_len: int, d_model: int, channels: int):
        super().__init__()
        self.L_patch_len = L_patch_len
        self.channels = channels
        self.linear = nn.Linear(d_model, L_patch_len * channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, n_var, Nl = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.linear(x)
        x = x.reshape(B, n_var, Nl, self.L_patch_len, self.channels).permute(0, 4, 1, 2, 3).contiguous()
        x = x.reshape(B, self.channels, n_var, Nl * self.L_patch_len)
        return x


# ----------------------------- residual block -----------------------------

class ResidualBlock(nn.Module):
    def __init__(self, side_dim: int, channels: int, diffusion_embedding_dim: int, nheads: int, condition_type: str = "add"):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.side_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        if condition_type == "add":
            pass
        elif condition_type == "cross_attention":
            self.condition_cross_attention = get_torch_cross_trans(heads=nheads, layers=1, channels=channels)
        elif condition_type == "adaLN":
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 3 * channels, bias=True),
            )

    def forward_time(self, y, base_shape, attention_mask=None):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(0, 2, 1), mask=attention_mask).permute(0, 2, 1)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape, attention_mask=None):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(0, 2, 1), mask=attention_mask).permute(0, 2, 1)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def foward_cross_attention(self, y, cond, attetion_mask=None):
        B, channel, K, L = y.shape
        y = y.reshape(B, channel, K, L).permute(0, 2, 3, 1).reshape(B * K, L, channel)
        cond = cond.reshape(B, channel, K, L).permute(0, 2, 3, 1).reshape(B * K, L, channel)
        y = self.condition_cross_attention(tgt=y, memory=cond, memory_mask=attetion_mask).permute(0, 2, 1)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3)
        return y

    @staticmethod
    def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale) + shift

    def forward(self, x, side_emb, attr_emb, diffusion_emb, attention_mask=None, condition_type: str = "add"):
        if condition_type == "add":
            x = x + attr_emb
        elif condition_type == "cross_attention":
            x = self.foward_cross_attention(x, attr_emb, attetion_mask=attention_mask)
        elif condition_type == "adaLN":
            gama, beta, alpha = self.adaLN_modulation(attr_emb.permute(0, 2, 3, 1)).chunk(3, dim=-1)
            gama = gama.permute(0, 3, 1, 2)
            beta = beta.permute(0, 3, 1, 2)
            alpha = alpha.permute(0, 3, 1, 2)

        B, channel, K, L = x.shape
        base_shape = x.shape

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1).unsqueeze(-1)
        y = x + diffusion_emb
        if condition_type == "adaLN":
            y = self.modulate(y, gama, beta)

        y = self.forward_time(y, base_shape, attention_mask)
        y = self.forward_feature(y, base_shape, None)

        if condition_type == "adaLN":
            y = y.reshape(B, channel, K, L)
            y = alpha * y
            y = y.reshape(B, channel, K * L)

        y = y.reshape(B, channel, K * L)
        y = self.mid_projection(y)

        _, side_dim, _, _ = side_emb.shape
        side_emb = side_emb.reshape(B, side_dim, K * L)
        side_emb = self.side_projection(side_emb)
        y = y + side_emb

        gate, filter_ = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter_)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


# ----------------------------- side encoder -----------------------------

class SideEncoder_Var(nn.Module):
    """Per-(variate, time) side features. Stateless w.r.t. the diffusion step."""
    def __init__(self, num_var: int, var_emb: int, time_emb: int):
        super().__init__()
        self.num_var = num_var
        self.var_emb_dim = var_emb
        self.time_emb_dim = time_emb
        self.var_emb = nn.Embedding(num_embeddings=num_var, embedding_dim=var_emb)
        self.register_buffer("var_ids", torch.arange(num_var), persistent=False)
        self.total_emb_dim = var_emb + time_emb

    @staticmethod
    def time_embedding(pos: torch.Tensor, d_model: int = 128) -> torch.Tensor:
        # pos: (B, L) floats. Returns (B, L, d_model).
        device = pos.device
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=device, dtype=pos.dtype)
        position = pos.unsqueeze(2)
        div_term = 1.0 / torch.pow(
            10000.0, torch.arange(0, d_model, 2, device=device).to(pos.dtype) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, tp: torch.Tensor) -> torch.Tensor:
        """tp: (B, L) -> side_emb: (B, total_emb_dim, num_var, L)."""
        B, L = tp.shape
        time_emb = self.time_embedding(tp, self.time_emb_dim)              # (B, L, time_emb)
        time_emb = time_emb.unsqueeze(2).expand(-1, -1, self.num_var, -1)  # (B, L, V, time_emb)
        var_emb = self.var_emb(self.var_ids)                                # (V, var_emb)
        var_emb = var_emb.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)    # (B, L, V, var_emb)
        side = torch.cat([time_emb, var_emb], dim=-1)                       # (B, L, V, total)
        return side.permute(0, 3, 2, 1)                                     # (B, total, V, L)


# ----------------------------- text projector -----------------------------

class TextProjectorMVarMScaleMStep(nn.Module):
    """Cross-attention queries mapping a text token sequence into
    (variate, scale, diffusion-stage) tokens, matching the original
    ``var_scale_diffstep_multi`` projector.

    Input attr: (B, T, dim_in) — token sequence. If your text encoder produces
    a single pooled vector (B, D), unsqueeze to (B, 1, D) at the call site.
    Output: (B, n_var, n_scale, dim_out)
    """
    def __init__(self, n_var: int, n_scale: int, n_steps: int, n_stages: int, dim_in: int = 128, dim_out: int = 128):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_var = n_var
        self.seg_size = n_steps // n_stages + 1
        self.var_emb = nn.Parameter(torch.zeros((1, n_var, dim_in)))
        self.scale_emb = nn.Parameter(torch.zeros((1, n_scale, dim_in)))
        self.step_emb = nn.Parameter(torch.zeros((1, n_stages, dim_in)))
        var_cross_attn_layer = nn.TransformerDecoderLayer(d_model=dim_in, nhead=8, dim_feedforward=64, activation="gelu", batch_first=True)
        self.var_cross_attn = nn.TransformerDecoder(var_cross_attn_layer, num_layers=2)
        scale_cross_attn_layer = nn.TransformerDecoderLayer(d_model=dim_in, nhead=8, dim_feedforward=64, activation="gelu", batch_first=True)
        self.scale_cross_attn = nn.TransformerDecoder(scale_cross_attn_layer, num_layers=2)
        step_cross_attn_layer = nn.TransformerDecoderLayer(d_model=dim_in, nhead=8, dim_feedforward=64, activation="gelu", batch_first=True)
        self.step_cross_attn = nn.TransformerDecoder(step_cross_attn_layer, num_layers=2)
        self.proj_out = nn.Linear(self.dim_in, self.dim_out)

    def forward(self, attr: torch.Tensor, diffusion_step: torch.Tensor, attr_mask: torch.Tensor | None = None) -> torch.Tensor:
        memory_key_padding_mask = None
        if attr_mask is not None:
            memory_key_padding_mask = (attr_mask == 0)
        B = attr.shape[0]

        var_emb = self.var_emb.expand([B, -1, -1])
        mvar_attr = self.var_cross_attn(tgt=var_emb, memory=attr, memory_key_padding_mask=memory_key_padding_mask)
        mvar_attr = mvar_attr[:, :, None, :]                                # (B, n_var, 1, dim_in)

        scale_emb = self.scale_emb.expand([B, -1, -1])
        mscale_attr = self.scale_cross_attn(tgt=scale_emb, memory=attr, memory_key_padding_mask=memory_key_padding_mask)
        mscale_attr = mscale_attr[:, None, :, :].expand([-1, self.n_var, -1, -1])  # (B, n_var, n_scale, dim_in)

        step_emb = self.step_emb.expand([B, -1, -1])
        mstep_attr = self.step_cross_attn(tgt=step_emb, memory=attr, memory_key_padding_mask=memory_key_padding_mask)
        indices = diffusion_step // self.seg_size
        indices = indices[:, None, None]
        mstep_attr = torch.gather(mstep_attr, dim=1, index=indices.expand([-1, -1, mstep_attr.shape[-1]]))
        mstep_attr = mstep_attr[:, None, :, :].expand([-1, self.n_var, -1, -1])    # (B, n_var, 1, dim_in)

        mix_attr = mvar_attr + mscale_attr + mstep_attr
        return self.proj_out(mix_attr)


class TextProjectorMVarMScaleContinuous(nn.Module):
    """Continuous-t variant of ``TextProjectorMVarMScaleMStep``. The original
    step branch was:
        n_stages learned query tokens -> cross_attn(memory=text) -> gather by
        ``diffusion_step // seg_size``
    Here the n_stages learned tokens + the discrete gather are replaced by a
    single ``TimestepEmbedder(t)``-derived query token, which is then fed
    through ``step_cross_attn`` over the same text memory. This keeps the
    step path's "pull text content conditioned on t" semantics, just with a
    continuous t in place of a discretized stage bucket. The var- and scale-
    branches are unchanged.

    Input attr: (B, T, dim_in). t: (B,) float. Output: (B, n_var, n_scale, dim_out).
    """
    def __init__(self, n_var: int, n_scale: int, dim_in: int = 128, dim_out: int = 128):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_var = n_var
        self.var_emb = nn.Parameter(torch.zeros((1, n_var, dim_in)))
        self.scale_emb = nn.Parameter(torch.zeros((1, n_scale, dim_in)))
        var_cross_attn_layer = nn.TransformerDecoderLayer(d_model=dim_in, nhead=8, dim_feedforward=64, activation="gelu", batch_first=True)
        self.var_cross_attn = nn.TransformerDecoder(var_cross_attn_layer, num_layers=2)
        scale_cross_attn_layer = nn.TransformerDecoderLayer(d_model=dim_in, nhead=8, dim_feedforward=64, activation="gelu", batch_first=True)
        self.scale_cross_attn = nn.TransformerDecoder(scale_cross_attn_layer, num_layers=2)
        step_cross_attn_layer = nn.TransformerDecoderLayer(d_model=dim_in, nhead=8, dim_feedforward=64, activation="gelu", batch_first=True)
        self.step_cross_attn = nn.TransformerDecoder(step_cross_attn_layer, num_layers=2)
        self.t_embedder = TimestepEmbedder(hidden_size=dim_in)
        self.proj_out = nn.Linear(self.dim_in, self.dim_out)

    def forward(self, attr: torch.Tensor, t: torch.Tensor, attr_mask: torch.Tensor | None = None) -> torch.Tensor:
        memory_key_padding_mask = None
        if attr_mask is not None:
            memory_key_padding_mask = (attr_mask == 0)
        B = attr.shape[0]

        var_emb = self.var_emb.expand([B, -1, -1])
        mvar_attr = self.var_cross_attn(tgt=var_emb, memory=attr, memory_key_padding_mask=memory_key_padding_mask)
        mvar_attr = mvar_attr[:, :, None, :]                                       # (B, n_var, 1, dim_in)

        scale_emb = self.scale_emb.expand([B, -1, -1])
        mscale_attr = self.scale_cross_attn(tgt=scale_emb, memory=attr, memory_key_padding_mask=memory_key_padding_mask)
        mscale_attr = mscale_attr[:, None, :, :].expand([-1, self.n_var, -1, -1])  # (B, n_var, n_scale, dim_in)

        t_query = self.t_embedder(t).unsqueeze(1)                                  # (B, 1, dim_in) — one query per batch
        mstep_attr = self.step_cross_attn(tgt=t_query, memory=attr, memory_key_padding_mask=memory_key_padding_mask)
        mstep_attr = mstep_attr[:, None, :, :].expand([-1, self.n_var, -1, -1])    # (B, n_var, 1, dim_in)

        mix_attr = mvar_attr + mscale_attr + mstep_attr
        return self.proj_out(mix_attr)


# ----------------------------- the VerbalTS net -----------------------------

class VerbalTS(nn.Module):
    """Main diffusion network (multi-patch encode, residual blocks, multi-patch decode + mixer).

    forward args:
      x_raw:        (B, inputdim, n_var, L)  -- inputdim is 1 in our wrapper
      tp:           (B, L)                   -- time-position indices (arange(L) is fine)
      attr_emb_raw: (B, n_var, multipatch_num, channels)  when ``text_projector`` contains "scale"
                    otherwise (B, n_var, 1, channels) and the spatial broadcast is uniform.
                    Pass None to disable text condition (uses zeros).
      diffusion_step: (B,) long
    returns:
      pred:  (B, n_var, L)
      info:  {} (kept for parity with the original)
    """
    def __init__(self, config: dict, inputdim: int = 1):
        super().__init__()
        self.config = config
        self.n_var = config["n_var"]
        self.channels = config["channels"]
        self.multipatch_num = config["multipatch_num"]
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.side_encoder = SideEncoder_Var(
            num_var=config["side"]["num_var"],
            var_emb=config["side"]["var_emb"],
            time_emb=config["side"]["time_emb"],
        )
        side_dim = self.side_encoder.total_emb_dim

        self.attention_mask_type = config["attention_mask_type"]
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.ts_downsample = nn.ModuleList([])
        self.side_downsample = nn.ModuleList([])
        self.patch_decoder = nn.ModuleList([])

        for i in range(self.multipatch_num):
            self.ts_downsample.append(
                TsPatchEmbedding(
                    L_patch_len=config["base_patch"] * config["L_patch_len"] ** i,
                    channels=inputdim,
                    d_model=self.channels,
                    dropout=0,
                )
            )
            self.patch_decoder.append(
                PatchDecoder(
                    L_patch_len=config["base_patch"] * config["L_patch_len"] ** i,
                    d_model=self.channels,
                    channels=1,
                )
            )
            self.side_downsample.append(
                SidePatchEmbedding(
                    L_patch_len=config["base_patch"] * config["L_patch_len"] ** i,
                    channels=side_dim,
                    d_model=side_dim,
                    dropout=0,
                )
            )
        self.multipatch_mixer = nn.Linear(self.multipatch_num, 1)
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=side_dim,
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    condition_type=config["condition_type"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x_raw: torch.Tensor, tp: torch.Tensor, attr_emb_raw: torch.Tensor | None, diffusion_step: torch.Tensor):
        B_raw, inputdim, n_var, L = x_raw.shape
        side_emb_raw = self.side_encoder(tp)
        diffusion_emb = self.diffusion_embedding(diffusion_step)

        x_list = []
        side_list = []
        scale_length = []
        for i in range(self.multipatch_num):
            x = self.ts_downsample[i](x_raw)
            side_emb = self.side_downsample[i](side_emb_raw)
            x_list.append(x)
            side_list.append(side_emb)
            scale_length.append(x.shape[-1])

        if self.attention_mask_type == "full" or attr_emb_raw is None:
            attention_mask = None
        elif self.attention_mask_type == "parallel":
            attention_mask = self.get_mask(0, [x_list[i].shape[-1] for i in range(len(x_list))], device=x_raw.device)
        else:
            raise ValueError(f"unknown attention_mask_type: {self.attention_mask_type!r}")

        x_in = torch.cat(x_list, dim=-1)
        side_in = torch.cat(side_list, dim=-1)

        if attr_emb_raw is None:
            attr_emb = torch.zeros_like(x_in)
        else:
            if "scale" in self.config["text_projector"]:
                assert len(scale_length) == attr_emb_raw.shape[2], (
                    f"attr_emb_raw scale dim ({attr_emb_raw.shape[2]}) != multipatch_num ({len(scale_length)})"
                )
                mscale_attr_list = []
                for i in range(len(scale_length)):
                    tmp_scale_attr = attr_emb_raw[:, :, i:i + 1, :].expand([-1, -1, scale_length[i], -1])
                    mscale_attr_list.append(tmp_scale_attr)
                attr_emb = torch.cat(mscale_attr_list, dim=2)
            else:
                attr_emb = attr_emb_raw.expand([-1, -1, x_in.shape[-1], -1])
            attr_emb = attr_emb.permute(0, 3, 1, 2)

        B, _, Nk, Nl = x_in.shape
        _x_in = x_in
        skip = []
        for layer in self.residual_layers:
            x_in, skip_connection = layer(
                x_in + _x_in, side_in, attr_emb, diffusion_emb,
                attention_mask=attention_mask, condition_type=self.config["condition_type"],
            )
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))

        x = x.reshape(B, self.channels, Nk * Nl)
        x = self.output_projection1(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, Nk, Nl)

        start_id = 0
        all_out = []
        for i in range(len(x_list)):
            x_out = x[:, :, :, start_id:start_id + x_list[i].shape[-1]]
            x_out = self.patch_decoder[i](x_out)
            x_out = x_out[:, :, :, :L]
            all_out.append(x_out)
            start_id += x_list[i].shape[-1]

        all_out = torch.cat(all_out, dim=1)
        all_out = self.multipatch_mixer(all_out.permute(0, 2, 3, 1).contiguous())
        all_out = all_out.reshape((B_raw, n_var, L))
        return all_out, {}

    @staticmethod
    def get_mask(attr_len: int, len_list: Sequence[int], device: torch.device | str = "cuda:0") -> torch.Tensor:
        total_len = sum(len_list) + attr_len
        mask = torch.zeros(total_len, total_len, device=device) - float("inf")
        mask[:attr_len, :] = 0
        mask[:, :attr_len] = 0
        start_id = attr_len
        for i in range(len(len_list)):
            mask[start_id:start_id + len_list[i], start_id:start_id + len_list[i]] = 0
            start_id += len_list[i]
        return mask


class VerbalTSContinuous(VerbalTS):
    """``VerbalTS`` with its discrete ``DiffusionEmbedding`` swapped for the
    continuous-t variant. Multi-patch encode/decode, residual blocks, side
    encoder, mixer — all identical to ``VerbalTS``. The only behavioral change
    is that ``forward`` now expects ``diffusion_step`` to be a float tensor
    (e.g. t in [0, 1]) instead of a long index.
    """
    def __init__(self, config: dict, inputdim: int = 1):
        cfg = dict(config)
        cfg.setdefault("num_steps", 1)  # consumed by the parent's DiffusionEmbedding which we overwrite below
        super().__init__(cfg, inputdim)
        self.diffusion_embedding = DiffusionEmbeddingContinuous(
            embedding_dim=config["diffusion_embedding_dim"],
        )


# ----------------------------- samplers (DDPM forward + DDIM reverse) -----------------------------

class _SamplerBase(nn.Module):
    """Holds the (num_steps,) β/α/ᾱ coefficient tensors as non-persistent buffers
    so they move with ``.to(device)`` of the parent module.
    """
    def __init__(self, num_steps: int = 50, beta_start: float = 1e-4, beta_end: float = 0.5, schedule: str = "quad"):
        super().__init__()
        self.num_steps = num_steps
        if schedule == "quad":
            beta_np = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps, dtype=np.float32) ** 2
        elif schedule == "linear":
            beta_np = np.linspace(beta_start, beta_end, num_steps, dtype=np.float32)
        else:
            raise ValueError(f"unknown schedule {schedule!r}")
        alpha_np = 1 - beta_np
        alpha_bar_np = np.cumprod(alpha_np)

        beta = torch.tensor(beta_np, dtype=torch.float32).reshape(num_steps, 1, 1)
        alpha = torch.tensor(alpha_np, dtype=torch.float32).reshape(num_steps, 1, 1)
        alpha_bar = torch.tensor(alpha_bar_np, dtype=torch.float32).reshape(num_steps, 1, 1)
        self.register_buffer("beta", beta, persistent=False)
        self.register_buffer("alpha", alpha, persistent=False)
        self.register_buffer("alpha_bar", alpha_bar, persistent=False)


class DDPMForward(_SamplerBase):
    """Only the q(x_t | x_0) noising path; we don't need DDPM reverse."""
    def __init__(self, num_steps: int = 50, beta_start: float = 1e-4, beta_end: float = 0.5, schedule: str = "quad"):
        super().__init__(num_steps, beta_start, beta_end, schedule)
        self.register_buffer("alpha_bar_sqrt", self.alpha_bar ** 0.5, persistent=False)
        self.register_buffer("one_minus_alpha_bar_sqrt", (1 - self.alpha_bar) ** 0.5, persistent=False)

    def forward(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """x0, noise: (B, V, L); t: (B,) long. Returns noisy_x (B, V, L)."""
        return self.alpha_bar_sqrt[t] * x0 + self.one_minus_alpha_bar_sqrt[t] * noise


class DDIMReverse(_SamplerBase):
    """Deterministic DDIM reverse step + ``predict_x0`` helper."""
    def __init__(self, num_steps: int = 50, beta_start: float = 1e-4, beta_end: float = 0.5, schedule: str = "quad"):
        super().__init__(num_steps, beta_start, beta_end, schedule)
        self.register_buffer("alpha_bar_sqrt", self.alpha_bar ** 0.5, persistent=False)
        self.register_buffer("alpha_bar_sqrt_inverse", 1.0 / (self.alpha_bar ** 0.5), persistent=False)
        self.register_buffer("one_minus_alpha_bar_sqrt", (1 - self.alpha_bar) ** 0.5, persistent=False)
        numerator = 1 - self.alpha_bar[:-1]
        denominator = 1 - self.alpha_bar[1:]
        sigma_sq = numerator / denominator * self.beta[1:]
        self.register_buffer("reverse_coef2_determin", (1 - self.alpha_bar[:-1]) ** 0.5, persistent=False)
        self.register_buffer("sigma", sigma_sq ** 0.5, persistent=False)

    def predict_x0(self, xt: torch.Tensor, pred_noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        coef1 = self.one_minus_alpha_bar_sqrt[t]
        coef2 = self.alpha_bar_sqrt_inverse[t]
        return (xt - coef1 * pred_noise) * coef2

    def reverse(self, xt: torch.Tensor, pred_noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Deterministic DDIM step. When t==0, return ``predict_x0`` directly."""
        pred_x0 = self.predict_x0(xt, pred_noise, t)
        mask = (t == 0).unsqueeze(-1).unsqueeze(-1)
        coef1 = self.alpha_bar_sqrt[t - 1]
        coef2 = self.reverse_coef2_determin[t - 1]
        x_prev = coef1 * pred_x0 + coef2 * pred_noise
        x_prev = torch.where(mask, pred_x0, x_prev)
        return x_prev
