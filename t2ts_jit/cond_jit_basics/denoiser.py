import torch
import torch.nn as nn
from model_jit import JiT, JiT1D, MultiResolutionJiT1D
from img_transformations import DelayEmbedder
from verbalts import (
    DDIMReverse,
    DDPMForward,
    TextProjectorMVarMScaleContinuous,
    TextProjectorMVarMScaleMStep,
    VerbalTS,
    VerbalTSContinuous,
)

class Denoiser(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.model_type = getattr(args, "model_type", "jit2d")
        num_register_tokens = getattr(args, "num_register_tokens", 0)

        if self.model_type == "jit2d":
            self.net = JiT(
                text_embed_dim=args.text_embed_dim,
                input_size=args.img_size,
                patch_size=args.patch_size,
                in_channels=args.in_channels,
                hidden_size=args.hidden_channels,
                depth=args.depth,
                num_heads=args.num_heads,
                mlp_ratio=4.0,
                attn_drop=args.attn_dropout,
                proj_drop=args.proj_dropout,
                bottleneck_dim=args.bottleneck_dim,
                num_register_tokens=num_register_tokens,
            )
            self.embedder = DelayEmbedder(
                device="cuda" if torch.cuda.is_available() else "cpu",
                seq_len=args.seq_len,
                delay=getattr(args, "delay", None) or args.img_size,
                embedding=args.img_size,
            )
            self.img_size = args.img_size
        elif self.model_type == "jit1d":
            self.net = JiT1D(
                text_embed_dim=args.text_embed_dim,
                input_size=args.seq_len,
                patch_size=args.patch_size,
                in_channels=args.in_channels,
                hidden_size=args.hidden_channels,
                depth=args.depth,
                num_heads=args.num_heads,
                mlp_ratio=4.0,
                attn_drop=args.attn_dropout,
                proj_drop=args.proj_dropout,
                bottleneck_dim=args.bottleneck_dim,
                num_register_tokens=num_register_tokens,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type!r}. Choose 'jit2d' or 'jit1d'.")

        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self._n_ema = 0
        for p in self.parameters():
            self.register_buffer(f"_ema1_{self._n_ema}", p.detach().clone())
            self.register_buffer(f"_ema2_{self._n_ema}", p.detach().clone())
            self._n_ema += 1

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps


    @property
    def ema_params1(self) -> list[torch.Tensor]:
        # Resolved fresh on every access so it picks up the post-`.to(device)`
        # buffer tensors (the buffer attribute is rebound by `Module._apply`;
        # a cached Python list would go stale).
        return [getattr(self, f"_ema1_{i}") for i in range(self._n_ema)]

    @property
    def ema_params2(self) -> list[torch.Tensor]:
        return [getattr(self, f"_ema2_{i}") for i in range(self._n_ema)]

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(
            self,
            x_ts: torch.Tensor,
            text_embeds: torch.Tensor,
        ):

        if self.model_type == "jit2d":
            x, x_mask = self.embedder.ts_to_img(x_ts, pad=True, mask=0, return_pad_mask=True)

            t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
            e = torch.randn_like(x) * self.noise_scale

            z = x_mask * (t * x + (1 - t) * e)
            v = x_mask * (x - z) / (1 - t).clamp_min(self.t_eps)

            x_pred = x_mask * self.net.forward(z, t.flatten(), text_embeds)
            v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

            loss = (v - v_pred) ** 2
            loss = (loss.sum(dim=(1,2,3)) / x_mask.sum(dim=(1,2,3))).mean()

        else:  # jit1d
            x = x_ts.transpose(1, 2).contiguous()  # (B, L, F) -> (B, C=F, L)

            t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
            e = torch.randn_like(x) * self.noise_scale

            z = t * x + (1 - t) * e
            v = (x - z) / (1 - t).clamp_min(self.t_eps)

            x_pred = self.net.forward(z, t.flatten(), text_embeds)
            v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

            loss = ((v - v_pred) ** 2).mean()

        return loss

    @torch.no_grad()
    def generate(self, text_embeds: torch.Tensor):
        """Generate samples conditioned only on study-group labels."""
        device = text_embeds.device
        bsz = text_embeds.size(0)

        if self.model_type == "jit2d":
            z = self.noise_scale * torch.randn(
                bsz, self.net.in_channels, self.img_size, self.img_size, device=device,
            )
            t_expand = lambda s: s.view(1, 1, 1, 1).expand(bsz, 1, 1, 1)
        else:  # jit1d
            z = self.noise_scale * torch.randn(
                bsz, self.net.in_channels, self.net.input_size, device=device,
            )
            t_expand = lambda s: s.view(1, 1, 1).expand(bsz, 1, 1)

        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        for i in range(self.steps - 1):
            t = t_expand(timesteps[i])
            t_next = t_expand(timesteps[i + 1])
            z = stepper(z, t, t_next, text_embeds)

        # last step euler
        z = self._euler_step(z, t_expand(timesteps[-2]), t_expand(timesteps[-1]), text_embeds)

        if self.model_type == "jit2d":
            return self.embedder.img_to_ts(z)
        else:
            return z.transpose(1, 2).contiguous()  # (B, C, L) -> (B, L, F)

    @torch.no_grad()
    def _forward_sample(self, z, t, text_embeds):
        x_cond = self.net(z, t.flatten(), text_embeds)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)
        return v_cond

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, text_embeds):
        v_pred = self._forward_sample(z, t, text_embeds)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, text_embeds):
        v_pred_t = self._forward_sample(z, t, text_embeds)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, text_embeds)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)


class MultiResolutionDenoiser(nn.Module):
    """Denoiser wrapping ``MultiResolutionJiT1D`` with per-scale text tokens.

    Loss math mirrors ``Denoiser`` in ``jit1d`` mode (linear flow matching
    with sigmoid-of-normal t sampling). The only change is that the inner
    network accepts ``text_embeds`` shape (B, K, text_embed_dim) where
    K == len(args.patch_sizes).
    """
    def __init__(self, args):
        super().__init__()
        num_register_tokens = getattr(args, "num_register_tokens", 0)
        self.net = MultiResolutionJiT1D(
            text_embed_dim=args.text_embed_dim,
            input_size=args.seq_len,
            patch_sizes=list(args.patch_sizes),
            in_channels=args.in_channels,
            hidden_size=args.hidden_channels,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=4.0,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            bottleneck_dim=args.bottleneck_dim,
            num_register_tokens=num_register_tokens,
        )

        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self._n_ema = 0
        for p in self.parameters():
            self.register_buffer(f"_ema1_{self._n_ema}", p.detach().clone())
            self.register_buffer(f"_ema2_{self._n_ema}", p.detach().clone())
            self._n_ema += 1

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps

    @property
    def ema_params1(self) -> list[torch.Tensor]:
        return [getattr(self, f"_ema1_{i}") for i in range(self._n_ema)]

    @property
    def ema_params2(self) -> list[torch.Tensor]:
        return [getattr(self, f"_ema2_{i}") for i in range(self._n_ema)]

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x_ts: torch.Tensor, text_embeds: torch.Tensor):
        """
        x_ts:        (B, L, F)   F == in_channels
        text_embeds: (B, K, D)   K == multipatch_num
        returns:     scalar loss
        """
        x = x_ts.transpose(1, 2).contiguous()  # (B, L, F) -> (B, F, L)

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net.forward(z, t.flatten(), text_embeds)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        loss = ((v - v_pred) ** 2).mean()
        return loss

    @torch.no_grad()
    def generate(self, text_embeds: torch.Tensor):
        """text_embeds: (B, K, D). Returns (B, L, F)."""
        device = text_embeds.device
        bsz = text_embeds.size(0)
        z = self.noise_scale * torch.randn(
            bsz, self.net.in_channels, self.net.input_size, device=device,
        )
        t_expand = lambda s: s.view(1, 1, 1).expand(bsz, 1, 1)
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        for i in range(self.steps - 1):
            t = t_expand(timesteps[i])
            t_next = t_expand(timesteps[i + 1])
            z = stepper(z, t, t_next, text_embeds)

        # last step euler
        z = self._euler_step(z, t_expand(timesteps[-2]), t_expand(timesteps[-1]), text_embeds)
        return z.transpose(1, 2).contiguous()  # (B, F, L) -> (B, L, F)

    @torch.no_grad()
    def _forward_sample(self, z, t, text_embeds):
        x_cond = self.net(z, t.flatten(), text_embeds)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)
        return v_cond

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, text_embeds):
        v_pred = self._forward_sample(z, t, text_embeds)
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, text_embeds):
        v_pred_t = self._forward_sample(z, t, text_embeds)
        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, text_embeds)
        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)


class VerbalTSDenoiser(nn.Module):
    """Denoiser wrapping ``VerbalTS`` (DDPM-trained) with the same forward /
    generate / EMA interface as ``Denoiser`` and ``MultiResolutionDenoiser``.

    Internals:
    - Training:  discrete DDPM noise prediction (predict epsilon, MSE loss).
    - Sampling:  deterministic DDIM (hardcoded; parent's ``sampling_method``
                 argument is ignored for this denoiser).
    - ``tp`` synthesized internally as ``arange(L)``.
    - ``n_var`` taken from ``args.in_channels``.
    - Text condition uses ``TextProjectorMVarMScaleMStep`` (diffstep-aware)
      to map a text token sequence ``(B, T, D)`` -> ``(B, n_var, multipatch_num, channels)``.
      ``forward`` / ``generate`` accept ``text_embeds`` shape ``(B, D)`` (pooled
      cache) or ``(B, T, D)`` (per-token features); ``(B, D)`` is internally
      treated as a length-1 sequence.

    Args expected on the ``SimpleNamespace`` ``args``:
      text_embed_dim, seq_len, in_channels,
      channels, multipatch_num, base_patch, L_patch_len, layers, nheads,
      diffusion_embedding_dim, var_emb, time_emb,
      attention_mask_type ("full" | "parallel"),
      condition_type ("adaLN" | "add" | "cross_attention"),
      num_steps, beta_start, beta_end, schedule ("quad" | "linear"),
      text_projector_stages, textemb_hidden_dim, text_emb_dim,
      ema_decay1, ema_decay2, noise_scale.

    Text path (matches verbalts_repro's TrainableTextProjector +
    TextProjectorMVarMScaleMStep):

        CLIP (B, T, text_embed_dim)
          -> text_pre_projector: Linear(text_embed_dim, textemb_hidden_dim)
                                 -> LayerNorm
                                 -> LeakyReLU(0.2)
                                 -> Linear(textemb_hidden_dim, text_emb_dim)
        -> (B, T, text_emb_dim)
        -> TextProjectorMVarMScaleMStep(dim_in=text_emb_dim, dim_out=channels)
        -> (B, n_var, multipatch_num, channels)
    """
    def __init__(self, args):
        super().__init__()
        n_var = args.in_channels
        channels = args.channels
        multipatch_num = args.multipatch_num
        num_steps = args.num_steps
        textemb_hidden_dim = int(args.textemb_hidden_dim)
        text_emb_dim = int(args.text_emb_dim)

        diff_cfg = {
            "n_var": n_var,
            "channels": channels,
            "multipatch_num": multipatch_num,
            "num_steps": num_steps,
            "diffusion_embedding_dim": args.diffusion_embedding_dim,
            "base_patch": args.base_patch,
            "L_patch_len": args.L_patch_len,
            "layers": args.layers,
            "nheads": args.nheads,
            "condition_type": args.condition_type,
            "attention_mask_type": args.attention_mask_type,
            "text_projector": "var_scale_diffstep_multi",
            "side": {
                "num_var": n_var,
                "var_emb": args.var_emb,
                "time_emb": args.time_emb,
            },
        }
        self.net = VerbalTS(diff_cfg, inputdim=1)

        # Mirrors verbalts_repro's ``TrainableTextProjector``: compresses raw
        # CLIP features (typically 768d) to a low-dim caption representation
        # before the var/scale/step cross-attention, so the latter operates
        # at d=text_emb_dim instead of d=text_embed_dim.
        self.text_pre_projector = nn.Sequential(
            nn.Linear(int(args.text_embed_dim), textemb_hidden_dim),
            nn.LayerNorm(textemb_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(textemb_hidden_dim, text_emb_dim),
        )

        self.text_projector = TextProjectorMVarMScaleMStep(
            n_var=n_var,
            n_scale=multipatch_num,
            n_steps=num_steps,
            n_stages=args.text_projector_stages,
            dim_in=text_emb_dim,
            dim_out=channels,
        )

        self.ddpm_forward = DDPMForward(num_steps, args.beta_start, args.beta_end, args.schedule)
        self.ddim_reverse = DDIMReverse(num_steps, args.beta_start, args.beta_end, args.schedule)

        self.num_steps = num_steps
        self.n_var = n_var
        self.seq_len = args.seq_len
        self.noise_scale = args.noise_scale  # accepted for parity; DDPM does not require it

        # EMA (mirrors Denoiser / MultiResolutionDenoiser)
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self._n_ema = 0
        for p in self.parameters():
            self.register_buffer(f"_ema1_{self._n_ema}", p.detach().clone())
            self.register_buffer(f"_ema2_{self._n_ema}", p.detach().clone())
            self._n_ema += 1

    @property
    def ema_params1(self) -> list[torch.Tensor]:
        return [getattr(self, f"_ema1_{i}") for i in range(self._n_ema)]

    @property
    def ema_params2(self) -> list[torch.Tensor]:
        return [getattr(self, f"_ema2_{i}") for i in range(self._n_ema)]

    @staticmethod
    def _prepare_text(text_embeds: torch.Tensor) -> torch.Tensor:
        """(B, D) -> (B, 1, D); (B, T, D) passes through."""
        if text_embeds.dim() == 2:
            return text_embeds.unsqueeze(1)
        if text_embeds.dim() == 3:
            return text_embeds
        raise ValueError(f"text_embeds must be 2D (B,D) or 3D (B,T,D); got {tuple(text_embeds.shape)}")

    @staticmethod
    def _build_tp(B: int, L: int, device, dtype=torch.float32) -> torch.Tensor:
        return torch.arange(L, device=device, dtype=dtype).unsqueeze(0).expand(B, -1)

    def forward(self, x_ts: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        x_ts:        (B, L, F)   with F == in_channels == n_var
        text_embeds: (B, D) or (B, T, D)
        returns:     scalar DDPM noise-prediction loss
        """
        x = x_ts.transpose(1, 2).contiguous()                            # (B, V, L)
        B, V, L = x.shape
        device = x.device

        t = torch.randint(0, self.num_steps, (B,), device=device)        # (B,) long
        noise = torch.randn_like(x)                                      # (B, V, L)
        noisy = self.ddpm_forward(x, t, noise)                           # (B, V, L)

        attr_seq = self._prepare_text(text_embeds)                       # (B, T, text_embed_dim)
        attr_seq = self.text_pre_projector(attr_seq)                     # (B, T, text_emb_dim)
        attr_emb_raw = self.text_projector(attr_seq, t)                  # (B, V, K, ch)

        tp = self._build_tp(B, L, device)                                # (B, L) float
        x_raw = noisy.unsqueeze(1)                                       # (B, 1, V, L)
        pred_noise, _ = self.net(x_raw, tp, attr_emb_raw, t)             # (B, V, L)

        return ((noise - pred_noise) ** 2).mean()

    @torch.no_grad()
    def generate(self, text_embeds: torch.Tensor) -> torch.Tensor:
        """Deterministic DDIM sampling. Returns (B, L, F)."""
        attr_seq = self._prepare_text(text_embeds)                       # (B, T, text_embed_dim)
        attr_seq = self.text_pre_projector(attr_seq)                     # (B, T, text_emb_dim)
        B = attr_seq.size(0)
        device = attr_seq.device
        V, L = self.n_var, self.seq_len
        x = self.noise_scale * torch.randn(B, V, L, device=device)
        tp = self._build_tp(B, L, device)

        for step in range(self.num_steps - 1, -1, -1):
            t = torch.full((B,), step, dtype=torch.long, device=device)
            attr_emb_raw = self.text_projector(attr_seq, t)
            x_raw = x.unsqueeze(1)
            pred_noise, _ = self.net(x_raw, tp, attr_emb_raw, t)
            x = self.ddim_reverse.reverse(x, pred_noise, t)

        return x.transpose(1, 2).contiguous()                            # (B, L, F)

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)


class VerbalTSJiTDenoiser(nn.Module):
    """VerbalTS backbone (multi-patch encode/decode + residual blocks + side
    encoder + mixer) trained with the JiT-style linear flow-matching path and
    x-pred + v-loss objective. Sampling uses euler / heun (parameters from
    ``args.sampling_method`` / ``args.num_sampling_steps``).

    Backbone differences vs ``VerbalTSDenoiser``:
    - Uses ``VerbalTSContinuous`` (continuous-t ``DiffusionEmbedding``).
    - Uses ``TextProjectorMVarMScaleContinuous`` (continuous-t step branch via
      ``TimestepEmbedder`` instead of a discrete ``gather``).
    - No DDPM/DDIM samplers; no ``num_steps`` / ``beta_*`` / ``schedule`` /
      ``text_projector_stages`` args.

    Text path (mirrors ``VerbalTSDenoiser``):
        CLIP (B, T, text_embed_dim)
          -> text_pre_projector: Linear -> LayerNorm -> LeakyReLU -> Linear
          -> (B, T, text_emb_dim)
          -> TextProjectorMVarMScaleContinuous(dim_in=text_emb_dim, dim_out=channels)
          -> (B, n_var, multipatch_num, channels)

    Args expected on the ``SimpleNamespace`` ``args``:
      text_embed_dim, seq_len, in_channels,
      channels, multipatch_num, base_patch, L_patch_len, layers, nheads,
      diffusion_embedding_dim, var_emb, time_emb,
      attention_mask_type ("full" | "parallel"),
      condition_type ("adaLN" | "add" | "cross_attention"),
      textemb_hidden_dim, text_emb_dim,
      P_mean, P_std, t_eps, noise_scale,
      sampling_method ("euler" | "heun"), num_sampling_steps,
      ema_decay1, ema_decay2.
    """
    def __init__(self, args):
        super().__init__()
        n_var = args.in_channels
        channels = args.channels
        multipatch_num = args.multipatch_num
        textemb_hidden_dim = int(args.textemb_hidden_dim)
        text_emb_dim = int(args.text_emb_dim)

        diff_cfg = {
            "n_var": n_var,
            "channels": channels,
            "multipatch_num": multipatch_num,
            "diffusion_embedding_dim": args.diffusion_embedding_dim,
            "base_patch": args.base_patch,
            "L_patch_len": args.L_patch_len,
            "layers": args.layers,
            "nheads": args.nheads,
            "condition_type": args.condition_type,
            "attention_mask_type": args.attention_mask_type,
            "text_projector": "var_scale_diffstep_multi",
            "side": {
                "num_var": n_var,
                "var_emb": args.var_emb,
                "time_emb": args.time_emb,
            },
        }
        self.net = VerbalTSContinuous(diff_cfg, inputdim=1)

        # Same CLIP-feature compression as VerbalTSDenoiser, so the var/scale
        # cross-attention runs at d=text_emb_dim instead of d=text_embed_dim.
        self.text_pre_projector = nn.Sequential(
            nn.Linear(int(args.text_embed_dim), textemb_hidden_dim),
            nn.LayerNorm(textemb_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(textemb_hidden_dim, text_emb_dim),
        )

        self.text_projector = TextProjectorMVarMScaleContinuous(
            n_var=n_var,
            n_scale=multipatch_num,
            dim_in=text_emb_dim,
            dim_out=channels,
        )

        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        self.n_var = n_var
        self.seq_len = args.seq_len

        # EMA (mirrors the other denoisers)
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self._n_ema = 0
        for p in self.parameters():
            self.register_buffer(f"_ema1_{self._n_ema}", p.detach().clone())
            self.register_buffer(f"_ema2_{self._n_ema}", p.detach().clone())
            self._n_ema += 1

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps

    @property
    def ema_params1(self) -> list[torch.Tensor]:
        return [getattr(self, f"_ema1_{i}") for i in range(self._n_ema)]

    @property
    def ema_params2(self) -> list[torch.Tensor]:
        return [getattr(self, f"_ema2_{i}") for i in range(self._n_ema)]

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    @staticmethod
    def _prepare_text(text_embeds: torch.Tensor) -> torch.Tensor:
        if text_embeds.dim() == 2:
            return text_embeds.unsqueeze(1)
        if text_embeds.dim() == 3:
            return text_embeds
        raise ValueError(f"text_embeds must be 2D (B,D) or 3D (B,T,D); got {tuple(text_embeds.shape)}")

    @staticmethod
    def _build_tp(B: int, L: int, device, dtype=torch.float32) -> torch.Tensor:
        return torch.arange(L, device=device, dtype=dtype).unsqueeze(0).expand(B, -1)

    def forward(self, x_ts: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        x_ts:        (B, L, F)   with F == in_channels == n_var
        text_embeds: (B, D) or (B, T, D)
        returns:     scalar v-loss
        """
        x = x_ts.transpose(1, 2).contiguous()                               # (B, V, L)
        B, V, L = x.shape
        device = x.device

        t = self.sample_t(B, device=device).view(-1, *([1] * (x.ndim - 1)))  # (B, 1, 1)
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = (x - z) / (1 - t).clamp_min(self.t_eps)

        t_flat = t.flatten()                                                # (B,) float
        attr_seq = self._prepare_text(text_embeds)                          # (B, T, text_embed_dim)
        attr_seq = self.text_pre_projector(attr_seq)                        # (B, T, text_emb_dim)
        attr_emb_raw = self.text_projector(attr_seq, t_flat)                # (B, V, K, ch)
        tp = self._build_tp(B, L, device)                                   # (B, L)
        x_pred, _ = self.net(z.unsqueeze(1), tp, attr_emb_raw, t_flat)      # (B, V, L)

        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
        return ((v - v_pred) ** 2).mean()

    @torch.no_grad()
    def generate(self, text_embeds: torch.Tensor) -> torch.Tensor:
        """text_embeds: (B, D) or (B, T, D). Returns (B, L, F)."""
        attr_seq = self._prepare_text(text_embeds)                          # (B, T, text_embed_dim)
        attr_seq = self.text_pre_projector(attr_seq)                        # (B, T, text_emb_dim)
        B = attr_seq.size(0)
        device = attr_seq.device
        V, L = self.n_var, self.seq_len

        z = self.noise_scale * torch.randn(B, V, L, device=device)
        t_expand = lambda s: s.view(1, 1, 1).expand(B, 1, 1)
        timesteps = torch.linspace(0.0, 1.0, self.steps + 1, device=device)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        for i in range(self.steps - 1):
            t = t_expand(timesteps[i])
            t_next = t_expand(timesteps[i + 1])
            z = stepper(z, t, t_next, attr_seq)

        # last step euler
        z = self._euler_step(z, t_expand(timesteps[-2]), t_expand(timesteps[-1]), attr_seq)
        return z.transpose(1, 2).contiguous()                               # (B, L, F)

    @torch.no_grad()
    def _forward_sample(self, z, t, attr_seq):
        B, V, L = z.shape
        t_flat = t.flatten()
        attr_emb_raw = self.text_projector(attr_seq, t_flat)
        tp = self._build_tp(B, L, z.device)
        x_pred, _ = self.net(z.unsqueeze(1), tp, attr_emb_raw, t_flat)
        return (x_pred - z) / (1 - t).clamp_min(self.t_eps)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, attr_seq):
        v_pred = self._forward_sample(z, t, attr_seq)
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, attr_seq):
        v_pred_t = self._forward_sample(z, t, attr_seq)
        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, attr_seq)
        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)

