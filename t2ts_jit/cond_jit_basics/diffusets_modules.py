"""DiffuSETS building blocks for the t2ts_jit baseline.

Ported from ``baseline_codes/ConTSG-Bench/contsg/models/diffusets.py`` with the
ConTSG Lightning / Registry dependencies stripped out, and two adaptations for
this repo's *short* univariate series (lengths 24/48/96):

  * **Tunable VAE downsampling.** The original SD-style VAE downsamples the
    sequence by a fixed factor of 8 (three stride-2 convs). On length-24 series
    that leaves a latent of length 3, which the latent U-Net cannot downsample
    further. ``DiffuSETSVAE`` exposes ``num_down`` (default 1 -> factor 2) so the
    latent stays long enough (L/2 = 12/24/48).
  * **Self-contained diffusion.** ``GaussianDiffusion`` holds the DDPM/DDIM
    buffers and the ``q_sample`` / reverse-step math that previously lived in
    ConTSG's ``DiffusionMixin`` and ``DiffuSETS._build_model``.

The VAE consumes/produces ``(B, L, C)`` and its latent is ``(B, latent_channels,
L / 2**num_down)``; the U-Net denoises that latent conditioned on a single text
embedding token. Patient metadata (age/sex/HR) from the original ECG model is
dropped — this is text-only conditioning for general time series.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import kl_divergence
from torch.distributions.normal import Normal


# =============================================================================
# VAE Components (verbatim from ConTSG except where noted)
# =============================================================================
class VAESelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: Tensor, causal_mask: bool = False) -> Tensor:
        batch_size, seq_len, d_embed = x.shape
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)
        weight = q @ k.transpose(-1, -2)
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        output = weight @ v
        output = output.transpose(1, 2).reshape(batch_size, seq_len, d_embed)
        return self.out_proj(output)


class VAEAttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = VAESelfAttention(1, channels)

    def forward(self, x: Tensor) -> Tensor:
        residue = x
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        return x + residue


class VAEResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residue)


def _channel_schedule(base_channels: int, num_down: int, max_channels: int = 512) -> list[int]:
    """Channel widths per resolution level: base, then doubling (capped)."""
    chs = [base_channels]
    c = base_channels
    for _ in range(num_down):
        c = min(c * 2, max_channels)
        chs.append(c)
    return chs


class VAEEncoder(nn.Module):
    """(B, L, C) -> z, mean, log_var each (B, latent_channels, L / 2**num_down).

    ``num_down`` stride-2 convolutions set the downsampling factor (ConTSG hard
    wires 3 -> factor 8; here it is tunable, default 1 -> factor 2).
    """

    def __init__(self, in_channels: int, latent_channels: int = 4,
                 base_channels: int = 128, num_down: int = 1):
        super().__init__()
        chs = _channel_schedule(base_channels, num_down)
        self.num_down = num_down
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, chs[0], kernel_size=3, padding=1),
            VAEResidualBlock(chs[0], chs[0]),
            VAEResidualBlock(chs[0], chs[0]),
        ]
        for i in range(num_down):
            layers.append(nn.Conv1d(chs[i], chs[i], kernel_size=3, stride=2, padding=0))
            layers.append(VAEResidualBlock(chs[i], chs[i + 1]))
            layers.append(VAEResidualBlock(chs[i + 1], chs[i + 1]))
        top = chs[-1]
        layers += [
            VAEResidualBlock(top, top),
            VAEAttentionBlock(top),
            VAEResidualBlock(top, top),
            nn.GroupNorm(32, top),
            nn.SiLU(),
            nn.Conv1d(top, 2 * latent_channels, kernel_size=3, padding=1),
            nn.Conv1d(2 * latent_channels, 2 * latent_channels, kernel_size=1, padding=0),
        ]
        self.layers = nn.ModuleList(layers)
        self.latent_channels = latent_channels

    def forward(self, x: Tensor, noise: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        x = x.transpose(1, 2)  # (B, C, L)
        for layer in self.layers:
            if isinstance(layer, nn.Conv1d) and layer.stride[0] == 2:
                x = F.pad(x, (0, 1))
            x = layer(x)
        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.clamp(log_var, -30, 20)
        stdev = (log_var * 0.5).exp()
        if noise is None:
            noise = torch.randn_like(stdev)
        z = mean + stdev * noise
        z = z * 0.18215  # SD legacy scale factor
        return z, mean, log_var


class VAEDecoder(nn.Module):
    """(B, latent_channels, L/2**num_down) -> (B, L, C)."""

    def __init__(self, out_channels: int, latent_channels: int = 4,
                 base_channels: int = 128, num_down: int = 1):
        super().__init__()
        chs = _channel_schedule(base_channels, num_down)
        top = chs[-1]
        layers: list[nn.Module] = [
            nn.Conv1d(latent_channels, latent_channels, kernel_size=1, padding=0),
            nn.Conv1d(latent_channels, top, kernel_size=3, padding=1),
            VAEResidualBlock(top, top),
            VAEAttentionBlock(top),
            VAEResidualBlock(top, top),
        ]
        for i in range(num_down, 0, -1):
            in_c, out_c = chs[i], chs[i - 1]
            layers.append(nn.Upsample(scale_factor=2))
            layers.append(nn.Conv1d(in_c, in_c, kernel_size=3, padding=1))
            layers.append(VAEResidualBlock(in_c, out_c))
            layers.append(VAEResidualBlock(out_c, out_c))
        bottom = chs[0]
        layers += [
            nn.GroupNorm(32, bottom),
            nn.SiLU(),
            nn.Conv1d(bottom, out_channels, kernel_size=3, padding=1),
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, z: Tensor) -> Tensor:
        x = z / 0.18215
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2)  # (B, L, C)


class DiffuSETSVAE(nn.Module):
    """Encoder + decoder wrapper with length padding to a multiple of 2**num_down."""

    def __init__(self, n_var: int = 1, latent_channels: int = 4,
                 base_channels: int = 128, num_down: int = 1):
        super().__init__()
        self.n_var = n_var
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.num_down = num_down
        self.pad_multiple = 2 ** num_down
        self.encoder = VAEEncoder(n_var, latent_channels, base_channels, num_down)
        self.decoder = VAEDecoder(n_var, latent_channels, base_channels, num_down)

    def init_kwargs(self) -> dict:
        return {
            "n_var": self.n_var,
            "latent_channels": self.latent_channels,
            "base_channels": self.base_channels,
            "num_down": self.num_down,
        }

    def _pad(self, x_blc: Tensor) -> Tuple[Tensor, int]:
        orig_len = x_blc.shape[1]
        pad = (-orig_len) % self.pad_multiple
        if pad:
            x_blc = F.pad(x_blc, (0, 0, 0, pad), mode="replicate")
        return x_blc, orig_len

    def encode(self, x_blc: Tensor, noise: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        x_blc, _ = self._pad(x_blc)
        return self.encoder(x_blc, noise)

    def decode(self, z: Tensor, orig_len: int) -> Tensor:
        x = self.decoder(z)
        if x.shape[1] > orig_len:
            x = x[:, :orig_len, :]
        elif x.shape[1] < orig_len:
            x = F.pad(x.transpose(1, 2), (0, orig_len - x.shape[1]), mode="replicate").transpose(1, 2)
        return x

    def reconstruct(self, x_blc: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        orig_len = x_blc.shape[1]
        z, mean, log_var = self.encode(x_blc)
        recons = self.decode(z, orig_len)
        return recons, mean, log_var

    def latent_length(self, seq_len: int) -> int:
        padded = seq_len + ((-seq_len) % self.pad_multiple)
        return padded // self.pad_multiple


def vae_loss_fn(recons: Tensor, x: Tensor, mu: Tensor, log_var: Tensor,
                kld_weight: float = 1.0) -> dict:
    """VAE loss: per-sample summed MSE + KL(q(z|x) || N(0, I)). Mirrors ConTSG."""
    mse = F.mse_loss(recons, x, reduction="sum").div(x.size(0))
    q_z = Normal(mu, (log_var * 0.5).exp())
    p_z = Normal(torch.zeros_like(mu), torch.ones_like(log_var))
    kld = kl_divergence(q_z, p_z).sum(1).mean()
    return {"loss": mse + kld_weight * kld, "mse": mse.detach(), "kld": kld.detach()}


# =============================================================================
# U-Net Diffusion Components (verbatim from ConTSG)
# =============================================================================
class DiffusionTimeEmbedding(nn.Module):
    def __init__(self, num_steps: int, n_channels: int = 4, dim_embed: int = 64, dim_latent: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(dim_embed, dim_latent)
        self.fc2 = nn.Linear(dim_latent, n_channels)
        t = torch.arange(num_steps) + 1
        half_dim = dim_embed // 2
        emb = 10.0 / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        self.register_buffer("embeddings", emb)

    def forward(self, t: Tensor) -> Tensor:
        emb = self.embeddings[t]
        out = self.fc1(emb)
        out = F.mish(out)
        out = self.fc2(out)
        return out


class UNetSelfAttention(nn.Module):
    def __init__(self, channels: int, hidden_dim: int = 16, num_heads: int = 4):
        super().__init__()
        self.to_q = nn.Linear(channels, hidden_dim)
        self.to_k = nn.Linear(channels, hidden_dim)
        self.to_v = nn.Linear(channels, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, channels)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        x_norm = self.norm(x)
        q = self.to_q(x_norm)
        k = self.to_k(x_norm)
        v = self.to_v(x_norm)
        h, _ = self.attention(q, k, v)
        return self.to_out(h) + x


class UNetCrossAttention(nn.Module):
    def __init__(self, channels: int, hidden_dim: int = 16, num_heads: int = 4, text_embed_dim: int = 768):
        super().__init__()
        self.to_q = nn.Linear(channels, hidden_dim)
        self.to_k = nn.Linear(channels, hidden_dim)
        self.to_v = nn.Linear(channels, hidden_dim)
        self.to_out = nn.Linear(hidden_dim, channels)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.value_combiner = nn.Linear(text_embed_dim, channels)
        self.cond_norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor, text_embed: Optional[Tensor]) -> Tensor:
        x_norm = self.norm(x)
        q = self.to_q(x_norm)
        if text_embed is not None and text_embed.numel() > 0:
            combined = text_embed
        else:
            combined = torch.zeros(x.shape[0], 1, self.value_combiner.in_features,
                                   device=x.device, dtype=x.dtype)
        combined = self.value_combiner(combined)
        combined = combined.repeat(1, x.shape[1] // combined.shape[1], 1)
        combined = self.cond_norm(combined)
        k = self.to_k(combined)
        v = self.to_v(combined)
        h, _ = self.attention(q, k, v)
        return self.to_out(h) + x


class UNetBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, num_steps: int, kernel_size: int = 5,
                 n_heads: int = 4, hidden_dim: int = 16, text_embed_dim: int = 768):
        super().__init__()
        n_shortcut = (n_inputs + n_outputs) // 2
        self.pre_shortcut = nn.Conv1d(n_inputs, n_shortcut, kernel_size, padding="same")
        self.shortcut_conv = nn.Conv1d(n_shortcut, n_shortcut, 1, padding="same")
        self.post_shortcut = nn.Conv1d(n_shortcut, n_outputs, kernel_size, padding="same")
        self.norm1 = nn.GroupNorm(1, n_shortcut)
        self.norm2 = nn.GroupNorm(1, n_outputs)
        self.res_conv = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()
        self.self_attn = UNetSelfAttention(n_shortcut, hidden_dim, n_heads)
        self.cross_attn = UNetCrossAttention(n_shortcut, hidden_dim, n_heads, text_embed_dim)
        self.attn_norm = nn.LayerNorm(n_shortcut)
        self.time_emb = DiffusionTimeEmbedding(num_steps, n_inputs)

    def forward(self, x: Tensor, t: Tensor, text_embed: Optional[Tensor]) -> Tensor:
        initial_x = x
        t_emb = self.time_emb(t).unsqueeze(-1)
        x = x + t_emb
        shortcut = self.pre_shortcut(x)
        shortcut = self.norm1(shortcut)
        shortcut = F.mish(shortcut)
        shortcut = self.shortcut_conv(shortcut)
        shortcut = shortcut.transpose(-1, -2)
        shortcut = self.self_attn(shortcut)
        shortcut = self.cross_attn(shortcut, text_embed)
        shortcut = self.attn_norm(shortcut)
        shortcut = shortcut.transpose(-1, -2)
        out = self.post_shortcut(shortcut)
        out = self.norm2(out)
        out = F.mish(out)
        return out + self.res_conv(initial_x)


class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, num_steps: int, **kwargs):
        super().__init__()
        self.down = nn.Conv1d(n_outputs, n_outputs, 3, stride=2, padding=1)
        self.block = UNetBlock(n_inputs, n_outputs, num_steps, **kwargs)

    def forward(self, x: Tensor, t: Tensor, text_embed: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        h = self.block(x, t, text_embed)
        return h, self.down(h)


class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, num_steps: int,
                 up_dim: Optional[int] = None, **kwargs):
        super().__init__()
        self.block = UNetBlock(n_inputs, n_outputs, num_steps, **kwargs)
        up_ch = up_dim if up_dim is not None else n_inputs // 2
        self.up = nn.ConvTranspose1d(up_ch, up_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: Tensor, h: Optional[Tensor], t: Tensor, text_embed: Optional[Tensor]) -> Tensor:
        x = self.up(x)
        if h is not None:
            if x.shape[2] != h.shape[2]:
                target = h.shape[2]
                if x.shape[2] < target:
                    pad = target - x.shape[2]
                    x = F.pad(x, (pad // 2, pad - pad // 2), mode="replicate")
                else:
                    diff = x.shape[2] - target
                    x = x[:, :, diff // 2: diff // 2 + target]
            x = torch.cat([x, h], dim=1)
        return self.block(x, t, text_embed)


class BottleneckNet(nn.Module):
    def __init__(self, n_channels: int, num_steps: int, n_heads: int = 4,
                 hidden_dim: int = 16, text_embed_dim: int = 768):
        super().__init__()
        self.time_emb = DiffusionTimeEmbedding(num_steps, n_channels)
        self.conv1 = nn.Conv1d(n_channels, n_channels, 3, padding="same")
        self.conv1_2 = nn.Conv1d(n_channels, n_channels, 3, padding="same")
        self.conv2 = nn.Conv1d(n_channels, n_channels, 3, padding="same")
        self.norm1 = nn.GroupNorm(1, n_channels)
        self.norm2 = nn.GroupNorm(1, n_channels)
        self.attn_norm = nn.LayerNorm(n_channels)
        self.self_attn = UNetSelfAttention(n_channels, hidden_dim, n_heads)
        self.cross_attn = UNetCrossAttention(n_channels, hidden_dim, n_heads, text_embed_dim)

    def forward(self, x: Tensor, t: Tensor, text_embed: Optional[Tensor]) -> Tensor:
        out = x + self.time_emb(t).unsqueeze(-1)
        out = self.conv1(out)
        out = self.norm1(out)
        out = F.mish(out)
        out = self.conv1_2(out)
        out = out.transpose(-1, -2)
        out = self.self_attn(out)
        out = self.cross_attn(out, text_embed)
        out = self.attn_norm(out)
        out = out.transpose(-1, -2)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.mish(out)
        return x + out


class DiffuSETSUNet(nn.Module):
    """Latent U-Net: predicts noise on (B, n_channels, L_latent)."""

    def __init__(self, num_steps: int = 50, kernel_size: int = 3, num_levels: int = 3,
                 n_channels: int = 4, text_embed_dim: int = 128, n_heads: int = 8):
        super().__init__()
        self.num_levels = num_levels
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        input_ch: list[int] = []
        output_ch: list[int] = []
        for i in range(num_levels - 1):
            input_ch.append(n_channels * 2 ** i)
        for i in range(num_levels - 1):
            input_ch.append(2 * input_ch[num_levels - i - 2])
        for i in range(num_levels - 1):
            output_ch.append(2 * input_ch[i])
        for i in range(num_levels - 1):
            output_ch.append(output_ch[num_levels - i - 2] // 2)
        for i in range(num_levels - 2):
            k = 2 * (num_levels - 1) - i - 1
            input_ch[k] += output_ch[i]
        hidden_state = [ch * 2 for ch in input_ch]

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(num_levels - 1):
            self.down_blocks.append(
                DownsamplingBlock(input_ch[i], output_ch[i], num_steps, kernel_size=kernel_size,
                                  n_heads=n_heads, hidden_dim=hidden_state[i], text_embed_dim=text_embed_dim)
            )
        self.bottleneck = BottleneckNet(input_ch[num_levels], num_steps, n_heads=4,
                                        hidden_dim=32, text_embed_dim=text_embed_dim)
        i = num_levels - 1
        self.up_blocks.append(
            UpsamplingBlock(input_ch[i], output_ch[i], num_steps, kernel_size=kernel_size,
                            up_dim=input_ch[i], n_heads=n_heads, hidden_dim=hidden_state[i],
                            text_embed_dim=text_embed_dim)
        )
        for i in range(num_levels, 2 * num_levels - 2):
            self.up_blocks.append(
                UpsamplingBlock(input_ch[i], output_ch[i], num_steps, kernel_size=kernel_size,
                                n_heads=n_heads, hidden_dim=hidden_state[i], text_embed_dim=text_embed_dim)
            )
        self.output_conv = nn.Sequential(
            nn.Conv1d(output_ch[-1], n_channels, 3, padding="same"),
            nn.Mish(),
            nn.Conv1d(n_channels, n_channels, 1, padding="same"),
        )

    def forward(self, x: Tensor, t: Tensor, text_embed: Optional[Tensor] = None) -> Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)
        if text_embed is not None and text_embed.dim() == 2:
            text_embed = text_embed.unsqueeze(1)
        shortcuts: list[Tensor] = []
        out = x
        for block in self.down_blocks:
            h, out = block(out, t, text_embed)
            shortcuts.append(h)
        shortcuts.pop()
        out = self.bottleneck(out, t, text_embed)
        out = self.up_blocks[0](out, None, t, text_embed)
        for idx, block in enumerate(self.up_blocks[1:]):
            out = block(out, shortcuts[-1 - idx], t, text_embed)
        return self.output_conv(out)


# =============================================================================
# Gaussian diffusion schedule + sampling (DDPM / DDIM)
# =============================================================================
class GaussianDiffusion(nn.Module):
    """DDPM forward + DDPM/DDIM reverse buffers (latent-space, eps-prediction).

    Reverse-step formulas mirror ConTSG ``DiffuSETS.generate``; CFG is applied by
    the caller on the predicted noise before each step.
    """

    def __init__(self, num_steps: int = 100, schedule: str = "linear",
                 beta_start: float = 0.00085, beta_end: float = 0.0120):
        super().__init__()
        self.num_steps = int(num_steps)
        if schedule == "linear":
            betas = torch.linspace(float(beta_start), float(beta_end), num_steps)
        elif schedule == "quad":
            betas = torch.linspace(float(beta_start) ** 0.5, float(beta_end) ** 0.5, num_steps) ** 2
        elif schedule == "cosine":
            steps = num_steps + 1
            x = torch.linspace(0, num_steps, steps)
            ac = torch.cos(((x / num_steps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            ac = ac / ac[0]
            betas = torch.clip(1 - (ac[1:] / ac[:-1]), 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown noise schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("alpha_bar_sqrt", torch.sqrt(alphas_cumprod))
        self.register_buffer("alpha_bar_sqrt_inverse", 1.0 / torch.sqrt(alphas_cumprod))
        self.register_buffer("reverse_coef1", 1.0 / torch.sqrt(alphas))
        self.register_buffer("reverse_coef2", (1.0 - alphas) / torch.sqrt(1.0 - alphas_cumprod))
        sigma_sq = (1.0 - alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:]) * betas[1:]
        self.register_buffer("sigma", torch.sqrt(sigma_sq))
        self.register_buffer("reverse_coef2_ddim", torch.sqrt(1.0 - alphas_cumprod[:-1] - sigma_sq))
        self.register_buffer("reverse_coef2_ddim_determin", torch.sqrt(1.0 - alphas_cumprod[:-1]))

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sa = self.sqrt_alphas_cumprod[t]
        soma = self.sqrt_one_minus_alphas_cumprod[t]
        while sa.dim() < x_start.dim():
            sa = sa.unsqueeze(-1)
            soma = soma.unsqueeze(-1)
        return sa * x_start + soma * noise

    def ddpm_step(self, x: Tensor, eps: Tensor, t_idx: int) -> Tensor:
        out = self.reverse_coef1[t_idx] * (x - self.reverse_coef2[t_idx] * eps)
        if t_idx > 0:
            out = out + self.sigma[t_idx - 1] * torch.randn_like(x)
        return out

    def ddim_step(self, x: Tensor, eps: Tensor, t_idx: int, determin: bool = True) -> Tensor:
        pred_z0 = (x - self.sqrt_one_minus_alphas_cumprod[t_idx] * eps) * self.alpha_bar_sqrt_inverse[t_idx]
        if t_idx == 0:
            return pred_z0
        coef1 = self.alpha_bar_sqrt[t_idx - 1]
        if determin:
            coef2 = self.reverse_coef2_ddim_determin[t_idx - 1]
            out = coef1 * pred_z0 + coef2 * eps
        else:
            coef2 = self.reverse_coef2_ddim[t_idx - 1]
            out = coef1 * pred_z0 + coef2 * eps + self.sigma[t_idx - 1] * torch.randn_like(x)
        return out
