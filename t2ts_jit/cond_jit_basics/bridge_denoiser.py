"""BridgeDenoiser: adapts the ConTSG ``Bridge`` model to the train_cond_v6
denoiser interface (``forward`` / ``generate`` / EMA).

Bridge is a *data-space* diffusion baseline (no VAE) from

    "BRIDGE: Bootstrapping Text to Control Time-Series Generation via
     Multi-Agent Iterative Optimization and Diffusion Modeling" (ICML 2025).

It pairs a 1D UNet (``UNetModel``) with a ``DomainUnifiedPrototyper`` that
extracts domain-level *prototypes* from an example time series; the prototypes
+ assignment ``mask`` are injected via cross-attention, and the (pooled) text
embedding is fused into the prototype context by a gated ``ConditioningMLP``.
Classifier-free guidance drops the *prototype* context to a learnable null
embedding (this is Bridge's own CFG; it is independent of v6's TextConditioner).

This adapter mirrors ``VerbalTSDenoiser`` so it slots into v6's
``build_denoiser`` dispatch as ``--model-type=bridge``. Two Bridge-specific
quirks are reproduced faithfully from the original code path:

* **Example time series.** During training Bridge uses *the target series
  itself* as the prototype source (self-conditioning); see
  ``contsg/data/datamodule.py`` ("Use the same sample as the prototype source
  during training"). During generation the original evaluator grabs a batch of
  real training series as examples. Since v6's ``generate(text_embeds)`` carries
  no series, we cache a pool of training series via ``set_example_pool`` and draw
  from it (truncate / tile to the batch size) exactly like the original
  evaluator.

* **Prototyper mask.** ``DomainUnifiedPrototyper`` returns the raw orthogonal
  latents as the context and the assignment logits as a separate ``mask`` (the
  upstream code does not multiply the mask back in). We keep this as-is for
  faithfulness.

NOTE on clipping: the original Bridge clamps the predicted x0 to ``[-1, 1]``,
which assumes data in that range. v6 series are z-score normalized and routinely
exceed +/-1, so ``clip_denoised`` defaults to **False** here.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bridge_modules.unet import UNetModel
from bridge_modules.prototype import DomainUnifiedPrototyper


class BridgeDenoiser(nn.Module):
    """Bridge diffusion denoiser with a v6-compatible forward/generate/EMA API.

    Expected fields on the ``args`` ``SimpleNamespace`` (see
    ``train_cond_v6.build_denoiser``):
      text_embed_dim, seq_len, in_channels (n_var),
      latent_dim, num_latents, latent_unit, prototype_dim,
      model_channels, num_res_blocks, attention_resolutions, channel_mult,
      dropout, num_heads, num_head_channels, transformer_depth, fusion_type,
      use_cfg, cond_drop_prob, cfg_scale,
      diffusion_steps, beta_start, beta_end, schedule ("linear"|"quad"|"cosine"),
      sampler ("ddim"|"ddpm"), ddim_steps, eta, clip_denoised,
      ema_decay1, ema_decay2.
    """

    def __init__(self, args):
        super().__init__()

        self.n_var = int(args.in_channels)
        self.seq_len = int(args.seq_len)

        # ---- diffusion schedule ----
        self.num_steps = int(args.diffusion_steps)
        self.beta_start = float(args.beta_start)
        self.beta_end = float(args.beta_end)
        self.schedule = str(args.schedule)

        # ---- prototype / conditioning dims ----
        self.repre_emb_channels = int(args.latent_dim)
        self.num_latents = int(args.num_latents)
        self.latent_unit = int(args.latent_unit)
        prototype_dim = int(getattr(args, "prototype_dim", 0) or self.repre_emb_channels)
        context_dim = self.repre_emb_channels  # cross-attn context dim == prototype latent dim

        # ---- CFG ----
        self.use_cfg = bool(args.use_cfg)
        self.cond_drop_prob = float(args.cond_drop_prob)
        self.cfg_scale = float(getattr(args, "cfg_scale", 1.0))

        # ---- sampling ----
        self.sampler = str(getattr(args, "sampler", "ddim"))
        self.ddim_steps = int(getattr(args, "ddim_steps", 0) or self.num_steps)
        self.eta = float(getattr(args, "eta", 0.0))
        self.clip_denoised = bool(getattr(args, "clip_denoised", False))

        # ---- cond_stage_model: DomainUnifiedPrototyper ----
        self.cond_stage_model = DomainUnifiedPrototyper(
            dim=prototype_dim,
            window=self.seq_len,
            num_latents=self.num_latents,
            num_channels=self.n_var,
            latent_dim=self.repre_emb_channels,
            bn=True,
        )

        # ---- UNet backbone ----
        text_dim = int(args.text_embed_dim)
        self.model = UNetModel(
            seq_len=self.seq_len,
            in_channels=self.n_var,
            model_channels=int(args.model_channels),
            out_channels=self.n_var,
            num_res_blocks=int(args.num_res_blocks),
            attention_resolutions=list(args.attention_resolutions),
            dropout=float(args.dropout),
            channel_mult=list(args.channel_mult),
            conv_resample=True,
            dims=1,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=int(args.num_heads),
            num_head_channels=int(args.num_head_channels),
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=False,
            use_spatial_transformer=True,
            transformer_depth=int(args.transformer_depth),
            context_dim=context_dim,
            text_dim=text_dim,
            fusion_type=str(args.fusion_type),
            legacy=True,
            repre_emb_channels=self.repre_emb_channels,
            latent_unit=self.latent_unit,
            use_cfg=self.use_cfg,
            cond_drop_prob=self.cond_drop_prob,
            use_pam=False,
        )

        self._setup_diffusion_schedule()

        # Example pool for generation (plain attribute, NOT a buffer: keeps it out
        # of DDP buffer-broadcast and state_dict; re-derived from train data each run).
        self._example_pool: Tensor | None = None  # (N, L, C)

        # ---- EMA (mirrors VerbalTSDenoiser) ----
        self.ema_decay1 = float(args.ema_decay1)
        self.ema_decay2 = float(args.ema_decay2)
        self._n_ema = 0
        for p in self.parameters():
            self.register_buffer(f"_ema1_{self._n_ema}", p.detach().clone())
            self.register_buffer(f"_ema2_{self._n_ema}", p.detach().clone())
            self._n_ema += 1

    # ------------------------------------------------------------------ schedule
    def _setup_diffusion_schedule(self) -> None:
        if self.schedule == "linear":
            betas = torch.linspace(self.beta_start, self.beta_end, self.num_steps)
        elif self.schedule == "quad":
            betas = torch.linspace(
                self.beta_start ** 0.5, self.beta_end ** 0.5, self.num_steps
            ) ** 2
        elif self.schedule == "cosine":
            betas = self._cosine_beta_schedule(self.num_steps)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    @staticmethod
    def _cosine_beta_schedule(num_steps: int, s: float = 0.008) -> Tensor:
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps)
        alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    # ------------------------------------------------------------------ example pool
    @torch.no_grad()
    def set_example_pool(self, series_blc: Tensor) -> None:
        """Cache a pool of (normalized) training series for prototype extraction
        at generation time. Expects shape (N, L, C). Stored on CPU; moved to the
        compute device on demand inside ``generate``."""
        if series_blc.dim() != 3:
            raise ValueError(
                f"example pool must be (N, L, C); got {tuple(series_blc.shape)}"
            )
        self._example_pool = series_blc.detach().to("cpu").float().contiguous()

    def _draw_examples(self, batch_size: int, device) -> Tensor:
        """Return (B, C, L) examples, truncating / tiling the pool to ``batch_size``
        (mirrors the original Bridge evaluator's size-matching)."""
        if self._example_pool is None:
            raise RuntimeError(
                "BridgeDenoiser.generate requires an example pool. Call "
                "set_example_pool(train_series) after constructing the denoiser."
            )
        pool = self._example_pool  # (N, L, C)
        n = pool.size(0)
        if n >= batch_size:
            ex = pool[:batch_size]
        else:
            reps = (batch_size + n - 1) // n
            ex = pool.repeat(reps, 1, 1)[:batch_size]
        ex = ex.to(device)
        return ex.permute(0, 2, 1).contiguous()  # (B, C, L)

    # ------------------------------------------------------------------ diffusion math
    def _q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        sa = self.sqrt_alphas_cumprod[t]
        soma = self.sqrt_one_minus_alphas_cumprod[t]
        while sa.dim() < x_start.dim():
            sa = sa.unsqueeze(-1)
            soma = soma.unsqueeze(-1)
        return sa * x_start + soma * noise

    def _predict_x0_from_noise(self, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        sr = self.sqrt_recip_alphas_cumprod[t]
        srm1 = self.sqrt_recipm1_alphas_cumprod[t]
        while sr.dim() < x_t.dim():
            sr = sr.unsqueeze(-1)
            srm1 = srm1.unsqueeze(-1)
        return sr * x_t - srm1 * noise

    def _predict_noise(
        self, x: Tensor, t: Tensor, c: Tensor, mask, text_embedding
    ) -> Tensor:
        """Noise prediction with Bridge's own CFG (drops the *prototype* context
        to the learnable null embedding for the unconditional branch)."""
        if self.use_cfg and self.cfg_scale != 1.0:
            out = self.model.forward_with_cfg(
                x=x, timesteps=t, context=c, mask=mask, y=None,
                cfg_scale=self.cfg_scale, text_embedding=text_embedding,
            )
        else:
            out = self.model(
                x=x, timesteps=t, context=c, mask=mask, y=None,
                cond_drop_prob=0.0, text_embedding=text_embedding,
            )
        return out.pred

    # ------------------------------------------------------------------ training
    def forward(self, x_ts: Tensor, text_embeds: Tensor) -> Tensor:
        """x_ts: (B, L, C); text_embeds: (B, D). Returns scalar noise-MSE loss.

        Training uses the target series itself as the prototype example
        (self-conditioning), matching the original Bridge data pipeline.
        """
        if text_embeds is not None and text_embeds.dim() == 3:
            # Bridge fuses a *pooled* (B, D) text embedding; collapse token dim.
            text_embeds = text_embeds.mean(dim=1)

        x = x_ts.permute(0, 2, 1).contiguous()  # (B, C, L)
        B = x.size(0)
        device = x.device

        c, mask = self.cond_stage_model(x)  # example == target series

        t = torch.randint(0, self.num_steps, (B,), device=device).long()
        noise = torch.randn_like(x)
        noisy = self._q_sample(x, t, noise)

        out = self.model(
            x=noisy,
            timesteps=t,
            context=c,
            mask=mask,
            y=None,
            cond_drop_prob=self.cond_drop_prob if self.training else 0.0,
            text_embedding=text_embeds,
        )
        return F.mse_loss(out.pred, noise)

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def generate(self, text_embeds: Tensor) -> Tensor:
        """Returns generated series (B, L, C). Prototype examples are drawn from
        the cached training pool (set via ``set_example_pool``)."""
        if text_embeds is not None and text_embeds.dim() == 3:
            text_embeds = text_embeds.mean(dim=1)

        B = text_embeds.size(0)
        device = text_embeds.device

        example = self._draw_examples(B, device)  # (B, C, L)
        c, mask = self.cond_stage_model(example)

        x = torch.randn(B, self.n_var, self.seq_len, device=device)
        if self.sampler == "ddpm":
            x = self._ddpm_sample(x, c, mask, text_embeds)
        elif self.sampler == "ddim":
            x = self._ddim_sample(x, c, mask, text_embeds, self.ddim_steps, self.eta)
        else:
            raise ValueError(f"Unknown sampler: {self.sampler}")

        return x.permute(0, 2, 1).contiguous()  # (B, L, C)

    @torch.no_grad()
    def _ddpm_sample(self, x, c, mask, text_embedding) -> Tensor:
        B, device = x.size(0), x.device
        for t_int in reversed(range(self.num_steps)):
            t = torch.full((B,), t_int, device=device, dtype=torch.long)
            pred_noise = self._predict_noise(x, t, c, mask, text_embedding)
            x_recon = self._predict_x0_from_noise(x, t, pred_noise)
            if self.clip_denoised:
                x_recon = torch.clamp(x_recon, -1.0, 1.0)

            coef1 = self.posterior_mean_coef1[t]
            coef2 = self.posterior_mean_coef2[t]
            var = self.posterior_variance[t]
            while coef1.dim() < x.dim():
                coef1 = coef1.unsqueeze(-1)
                coef2 = coef2.unsqueeze(-1)
                var = var.unsqueeze(-1)
            mean = coef1 * x_recon + coef2 * x
            noise = torch.randn_like(x) if t_int > 0 else torch.zeros_like(x)
            x = mean + torch.sqrt(var) * noise
        return x

    @torch.no_grad()
    def _ddim_sample(self, x, c, mask, text_embedding, steps, eta) -> Tensor:
        B, device = x.size(0), x.device
        timesteps = np.linspace(0, self.num_steps - 1, steps, dtype=int).tolist()
        timesteps = list(reversed(timesteps))

        for i, t_int in enumerate(timesteps):
            t = torch.full((B,), t_int, device=device, dtype=torch.long)
            pred_noise = self._predict_noise(x, t, c, mask, text_embedding)
            x_0 = self._predict_x0_from_noise(x, t, pred_noise)
            if self.clip_denoised:
                x_0 = torch.clamp(x_0, -1.0, 1.0)

            t_prev = timesteps[i + 1] if i < len(timesteps) - 1 else 0
            alpha = self.alphas_cumprod[t_int]
            alpha_prev = (
                self.alphas_cumprod[t_prev]
                if t_prev > 0
                else torch.tensor(1.0, device=device)
            )
            sigma = (
                eta
                * torch.sqrt((1 - alpha_prev) / (1 - alpha))
                * torch.sqrt(1 - alpha / alpha_prev)
            )
            dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * pred_noise
            noise = torch.randn_like(x) if sigma > 0 else torch.zeros_like(x)
            x = torch.sqrt(alpha_prev) * x_0 + dir_xt + sigma * noise
        return x

    # ------------------------------------------------------------------ EMA
    @property
    def ema_params1(self) -> list[Tensor]:
        return [getattr(self, f"_ema1_{i}") for i in range(self._n_ema)]

    @property
    def ema_params2(self) -> list[Tensor]:
        return [getattr(self, f"_ema2_{i}") for i in range(self._n_ema)]

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
