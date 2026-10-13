"""Text2Motion (CVPR'22) backbone — vendored, ConTSG-free single-file copy.

This is the architecture used by the Text2Motion baseline inside ``t2ts_jit``. It
is lifted from ``baseline_codes/ConTSG-Bench/contsg/models/text2motion.py`` but
stripped of the ConTSG ``ExperimentConfig`` / ``Registry`` / Lightning
(``BaseGeneratorModule``) dependencies so it can be driven by the plain two-stage
trainers that mirror the T2S baseline:

  * Stage-1 (``pretrain_ae_text2motion.py``): train the **MovementConvEncoder /
    MovementConvDecoder** autoencoder (conv down/up by ``unit_length=4``) on the
    [0,1]-normalized univariate series. Analogous to the T2S LA-VAE.
  * Stage-2 (``train_cond_v7_text2motion.py``): freeze the movement AE and train
    the **text-conditioned autoregressive CVAE generator** (prior / posterior GRU
    + attention + movement-latent decoder) with KL + reconstruction losses.

Differences vs the ConTSG adaptation:
  * Hyper-parameters are passed via the plain :class:`Text2MotionConfig` dataclass
    instead of ``Text2MotionModelConfig``.
  * The text condition is a single pooled embedding (default 128-d OpenAI
    ``TextEmbedding``, matching the T2S baseline). ``word_hids`` is that single
    token, so the attention layer attends over one key — faithful to the ConTSG
    adaptation's ``_encode_text``.
  * The movement AE is split into a standalone :class:`MovementAutoEncoder`
    (used by Stage-1) and re-used inside :class:`Text2Motion` (Stage-2) under the
    SAME submodule names (``mov_enc`` / ``mov_dec``) so Stage-1 weights load
    directly via :meth:`Text2Motion.load_movement_ae`.

Series convention everywhere here is ``(B, L, C)`` with ``C = n_var`` (the
trainers feed univariate series as ``(B, L, 1)``).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================ config ===========
@dataclass
class Text2MotionConfig:
    """Plain hyper-parameter holder (ConTSG ``Text2MotionModelConfig`` defaults,
    adapted to univariate series + a 128-d pooled text embedding)."""

    # data / movement autoencoder
    n_var: int = 1
    unit_length: int = 4  # conv down/up factor; this backbone requires 4
    dim_movement_latent: int = 512
    dim_movement_enc_hidden: int = 512
    dim_movement_dec_hidden: int = 512

    # text-conditioned latent generator
    text_input_dim: int = 128  # OpenAI TextEmbedding width (matches T2S)
    text_latent_dim: int = 512
    dim_att_vec: int = 512
    dim_z: int = 128
    n_layers_pri: int = 1
    n_layers_pos: int = 1
    n_layers_dec: int = 1
    dim_pri_hidden: int = 1024
    dim_pos_hidden: int = 1024
    dim_dec_hidden: int = 1024

    # losses / training behaviour
    lambda_rec_ts: float = 1.0
    lambda_rec_mov: float = 1.0
    lambda_kld: float = 0.005
    teacher_forcing_ratio: float = 0.4
    detach_movement_latents: bool = True
    # Stage-1 movement-AE latent regularizers (Text2Motion DecompTrainerV3 defaults).
    lambda_sparsity: float = 0.001
    lambda_smooth: float = 0.001

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Text2MotionConfig":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in fields})


# ============================================================ helpers ==========
def _init_weight(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.xavier_uniform_(module.weight)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


def _reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def _kl_gauss(mu1: Tensor, logvar1: Tensor, mu2: Tensor, logvar2: Tensor) -> Tensor:
    """KL( N(mu1, var1) || N(mu2, var2) ), summed and averaged over the leading dim."""
    sigma1 = torch.exp(0.5 * logvar1)
    sigma2 = torch.exp(0.5 * logvar2)
    kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (
        2 * torch.exp(logvar2)
    ) - 0.5
    return kld.sum() / mu1.shape[0]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 300) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, pos: Tensor) -> Tensor:
        return self.pe[pos]


# ============================================================ movement AE ======
class MovementConvEncoder(nn.Module):
    """``(B, L, C) -> (B, L/4, output_size)`` (two stride-2 convs)."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.apply(_init_weight)

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs.permute(0, 2, 1)
        y = self.main(x).permute(0, 2, 1)
        return self.out_net(y)


class MovementConvDecoder(nn.Module):
    """``(B, L/4, input_size) -> (B, L, output_size)`` (two stride-2 deconvs)."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(input_size, hidden_size, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(hidden_size, output_size, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.apply(_init_weight)

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs.permute(0, 2, 1)
        y = self.main(x).permute(0, 2, 1)
        return self.out_net(y)


def _build_mov_enc(cfg: Text2MotionConfig) -> MovementConvEncoder:
    return MovementConvEncoder(cfg.n_var, cfg.dim_movement_enc_hidden, cfg.dim_movement_latent)


def _build_mov_dec(cfg: Text2MotionConfig) -> MovementConvDecoder:
    return MovementConvDecoder(cfg.dim_movement_latent, cfg.dim_movement_dec_hidden, cfg.n_var)


class MovementAutoEncoder(nn.Module):
    """Stage-1 module: the movement conv autoencoder only.

    Submodule names (``mov_enc`` / ``mov_dec``) match :class:`Text2Motion` so the
    Stage-1 ``state_dict`` loads straight into the Stage-2 model.
    """

    def __init__(self, cfg: Text2MotionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mov_enc = _build_mov_enc(cfg)
        self.mov_dec = _build_mov_dec(cfg)

    def encode(self, ts: Tensor) -> Tensor:
        return self.mov_enc(ts)

    def decode(self, movements: Tensor) -> Tensor:
        return self.mov_dec(movements)

    def forward(self, ts: Tensor) -> Dict[str, Tensor]:
        """``ts`` is ``(B, L, C)``. Decomp-baseline loss: L1 reconstruction plus
        the two latent regularizers from Text2Motion ``DecompTrainerV3``:
        L1 sparsity ``mean|z|`` and temporal smoothness ``L1(z[:, 1:], z[:, :-1])``
        over the ``(B, L/unit_length, latent)`` movement latents."""
        movements = self.mov_enc(ts)
        recon = self.mov_dec(movements)
        loss_rec = F.l1_loss(recon, ts)
        loss_sparsity = movements.abs().mean()
        loss_smooth = (
            F.l1_loss(movements[:, 1:], movements[:, :-1])
            if movements.size(1) > 1
            else movements.new_zeros(())
        )
        loss = (
            loss_rec
            + self.cfg.lambda_sparsity * loss_sparsity
            + self.cfg.lambda_smooth * loss_smooth
        )
        return {
            "loss": loss,
            "loss_rec": loss_rec,
            "loss_sparsity": loss_sparsity,
            "loss_smooth": loss_smooth,
        }


# ============================================================ generator nets ===
class TextDecoder(nn.Module):
    """Prior / posterior GRU producing a Gaussian latent ``z`` per movement step."""

    def __init__(self, text_size: int, input_size: int, output_size: int, hidden_size: int, n_layers: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.n_layers = int(n_layers)
        self.emb = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for _ in range(self.n_layers)])
        self.z2init = nn.Linear(text_size, hidden_size * n_layers)
        self.positional_encoder = PositionalEncoding(hidden_size)
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.apply(_init_weight)

    def get_init_hidden(self, latent: Tensor) -> List[Tensor]:
        hidden = self.z2init(latent)
        hidden = torch.split(hidden, self.hidden_size, dim=-1)
        return list(hidden)

    def forward(self, inputs: Tensor, hidden: List[Tensor], p: Tensor) -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
        x_in = self.emb(inputs)
        x_in = x_in + self.positional_encoder(p).to(inputs.device).detach()
        h_in = x_in
        for i in range(self.n_layers):
            hidden[i] = self.gru[i](h_in, hidden[i])
            h_in = hidden[i]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = _reparameterize(mu, logvar)
        return z, mu, logvar, hidden


class TextVAEDecoder(nn.Module):
    """Movement-latent decoder GRU."""

    def __init__(self, text_size: int, input_size: int, output_size: int, hidden_size: int, n_layers: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.n_layers = int(n_layers)
        self.emb = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.z2init = nn.Linear(text_size, hidden_size * n_layers)
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for _ in range(self.n_layers)])
        self.positional_encoder = PositionalEncoding(hidden_size)
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )
        self.apply(_init_weight)

    def get_init_hidden(self, latent: Tensor) -> List[Tensor]:
        hidden = self.z2init(latent)
        hidden = torch.split(hidden, self.hidden_size, dim=-1)
        return list(hidden)

    def forward(self, inputs: Tensor, last_pred: Tensor, hidden: List[Tensor], p: Tensor) -> Tuple[Tensor, List[Tensor]]:
        h_in = self.emb(inputs)
        h_in = h_in + self.positional_encoder(p).to(inputs.device).detach()
        for i in range(self.n_layers):
            hidden[i] = self.gru[i](h_in, hidden[i])
            h_in = hidden[i]
        pose_pred = self.output(h_in)
        _ = last_pred
        return pose_pred, hidden


class AttLayer(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, value_dim: int) -> None:
        super().__init__()
        self.W_q = nn.Linear(query_dim, value_dim)
        self.W_k = nn.Linear(key_dim, value_dim, bias=False)
        self.W_v = nn.Linear(key_dim, value_dim)
        self.softmax = nn.Softmax(dim=1)
        self.dim = int(value_dim)
        self.apply(_init_weight)

    def forward(self, query: Tensor, key_mat: Tensor) -> Tuple[Tensor, Tensor]:
        query_vec = self.W_q(query).unsqueeze(-1)
        val_set = self.W_v(key_mat)
        key_set = self.W_k(key_mat)
        weights = torch.matmul(key_set, query_vec) / math.sqrt(self.dim)
        co_weights = self.softmax(weights)
        values = val_set * co_weights
        pred = values.sum(dim=1)
        return pred, co_weights


# ============================================================ full model =======
class Text2Motion(nn.Module):
    """Stage-2 model: frozen movement AE + text-conditioned autoregressive CVAE.

    Training stages mirror the ConTSG ``set_stage`` behaviour:
      * ``pretrain``: only the movement AE is trainable (not used by Stage-2;
        provided for completeness / parity).
      * ``finetune``: AE frozen, the conditional generator is trainable.
    """

    SUPPORTED_STAGES = ("pretrain", "finetune")

    def __init__(self, cfg: Text2MotionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._ae_frozen = False  # set by freeze_movement_ae / set_stage('finetune')

        # movement autoencoder (same names as MovementAutoEncoder)
        self.mov_enc = _build_mov_enc(cfg)
        self.mov_dec = _build_mov_dec(cfg)

        # text projection: pooled text embedding -> text latent
        self.text_proj = nn.Linear(cfg.text_input_dim, cfg.text_latent_dim)

        self.att_layer = AttLayer(
            query_dim=cfg.dim_dec_hidden,
            key_dim=cfg.text_latent_dim,
            value_dim=cfg.dim_att_vec,
        )
        self.seq_pri = TextDecoder(
            text_size=cfg.text_latent_dim,
            input_size=cfg.dim_movement_latent + cfg.dim_att_vec,
            output_size=cfg.dim_z,
            hidden_size=cfg.dim_pri_hidden,
            n_layers=cfg.n_layers_pri,
        )
        self.seq_post = TextDecoder(
            text_size=cfg.text_latent_dim,
            input_size=cfg.dim_movement_latent + cfg.dim_movement_latent + cfg.dim_att_vec,
            output_size=cfg.dim_z,
            hidden_size=cfg.dim_pos_hidden,
            n_layers=cfg.n_layers_pos,
        )
        self.seq_dec = TextVAEDecoder(
            text_size=cfg.text_latent_dim,
            input_size=cfg.dim_movement_latent + cfg.dim_att_vec + cfg.dim_z,
            output_size=cfg.dim_movement_latent,
            hidden_size=cfg.dim_dec_hidden,
            n_layers=cfg.n_layers_dec,
        )

    # ----------------------------------------------------------- stage utils ---
    @property
    def _generator_modules(self) -> List[nn.Module]:
        return [self.text_proj, self.att_layer, self.seq_pri, self.seq_post, self.seq_dec]

    def load_movement_ae(self, ae_state_dict: dict, strict: bool = True) -> None:
        """Load Stage-1 ``MovementAutoEncoder`` weights into ``mov_enc``/``mov_dec``."""
        enc_sd = {k[len("mov_enc."):]: v for k, v in ae_state_dict.items() if k.startswith("mov_enc.")}
        dec_sd = {k[len("mov_dec."):]: v for k, v in ae_state_dict.items() if k.startswith("mov_dec.")}
        if not enc_sd or not dec_sd:
            raise KeyError("AE state_dict has no 'mov_enc.'/'mov_dec.' keys; got "
                           f"{list(ae_state_dict)[:4]}...")
        self.mov_enc.load_state_dict(enc_sd, strict=strict)
        self.mov_dec.load_state_dict(dec_sd, strict=strict)

    def freeze_movement_ae(self) -> None:
        for p in self.mov_enc.parameters():
            p.requires_grad = False
        for p in self.mov_dec.parameters():
            p.requires_grad = False
        self._ae_frozen = True
        self.mov_enc.eval()
        self.mov_dec.eval()

    def train(self, mode: bool = True) -> "Text2Motion":
        """Keep the frozen movement AE in eval mode (its encoder has dropout) even
        when the rest of the model is switched to train mode."""
        super().train(mode)
        if self._ae_frozen:
            self.mov_enc.eval()
            self.mov_dec.eval()
        return self

    def set_stage(self, stage: str) -> None:
        if stage not in self.SUPPORTED_STAGES:
            raise ValueError(f"Unknown stage: {stage}. Supported: {self.SUPPORTED_STAGES}")
        if stage == "pretrain":
            self._ae_frozen = False
            for p in self.mov_enc.parameters():
                p.requires_grad = True
            for p in self.mov_dec.parameters():
                p.requires_grad = True
            for m in self._generator_modules:
                for p in m.parameters():
                    p.requires_grad = False
        else:  # finetune
            self.freeze_movement_ae()
            for m in self._generator_modules:
                for p in m.parameters():
                    p.requires_grad = True

    # ----------------------------------------------------------- core pieces ---
    def _encode_text(self, text_emb: Tensor) -> Tuple[Tensor, Tensor]:
        text_latent = self.text_proj(text_emb.float())
        word_hids = text_latent.unsqueeze(1)  # single text token
        return word_hids, text_latent

    def _initial_movement_token(self, batch_size: int, device: torch.device) -> Tensor:
        zeros = torch.zeros((batch_size, self.cfg.unit_length, self.cfg.n_var), device=device)
        init = self.mov_enc(zeros)
        if init.shape[1] != 1:
            raise ValueError(f"Expected init movement length=1, got {tuple(init.shape)}")
        return init.squeeze(1)

    def _generate_movements(
        self,
        word_hids: Tensor,
        text_latent: Tensor,
        mov_len: int,
        m_lens: Tensor,
        use_posterior: bool,
        movements_gt: Optional[Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> Dict[str, Tensor]:
        cfg = self.cfg
        B = text_latent.shape[0]
        device = text_latent.device

        hidden_pri = self.seq_pri.get_init_hidden(text_latent)
        hidden_dec = self.seq_dec.get_init_hidden(text_latent)

        hidden_pos: Optional[List[Tensor]] = None
        if use_posterior:
            hidden_pos = self.seq_post.get_init_hidden(text_latent)
            if movements_gt is None:
                raise ValueError("movements_gt must be provided when use_posterior=True")

        mov_in = self._initial_movement_token(B, device)
        teacher_force = use_posterior and (random.random() < teacher_forcing_ratio)

        mus_pri: List[Tensor] = []
        logvars_pri: List[Tensor] = []
        mus_pos: List[Tensor] = []
        logvars_pos: List[Tensor] = []
        fake_mov_batch: List[Tensor] = []

        for i in range(mov_len):
            att_vec, _ = self.att_layer(hidden_dec[-1], word_hids)
            tta = m_lens // cfg.unit_length - i

            pri_in = torch.cat([mov_in, att_vec], dim=-1)
            z_pri, mu_pri, logvar_pri, hidden_pri = self.seq_pri(pri_in, hidden_pri, tta)
            mus_pri.append(mu_pri)
            logvars_pri.append(logvar_pri)

            if use_posterior:
                assert movements_gt is not None and hidden_pos is not None
                mov_tgt = movements_gt[:, i]
                pos_in = torch.cat([mov_in, mov_tgt, att_vec], dim=-1)
                z_pos, mu_pos, logvar_pos, hidden_pos = self.seq_post(pos_in, hidden_pos, tta)
                mus_pos.append(mu_pos)
                logvars_pos.append(logvar_pos)
                z = z_pos
            else:
                z = z_pri
                mov_tgt = None

            dec_in = torch.cat([mov_in, att_vec, z], dim=-1)
            fake_mov, hidden_dec = self.seq_dec(dec_in, mov_in, hidden_dec, tta)
            fake_mov_batch.append(fake_mov.unsqueeze(1))

            if teacher_force and mov_tgt is not None:
                mov_in = mov_tgt.detach()
            else:
                mov_in = fake_mov.detach()

        out: Dict[str, Tensor] = {
            "fake_movements": torch.cat(fake_mov_batch, dim=1),
            "mu_pri": torch.cat(mus_pri, dim=0),
            "logvar_pri": torch.cat(logvars_pri, dim=0),
        }
        if use_posterior:
            out["mu_pos"] = torch.cat(mus_pos, dim=0)
            out["logvar_pos"] = torch.cat(logvars_pos, dim=0)
        return out

    # ----------------------------------------------------------- train/gen -----
    def forward_cond(self, ts: Tensor, text_emb: Tensor) -> Dict[str, Tensor]:
        """Stage-2 conditional training. ``ts`` is ``(B, L, C)``."""
        cfg = self.cfg
        ts = ts.float()
        if ts.dim() != 3:
            raise ValueError(f"Expected ts to be (B, L, C), got {tuple(ts.shape)}")
        B, L, _ = ts.shape
        if L % cfg.unit_length != 0:
            raise ValueError(f"seq_length must be divisible by unit_length={cfg.unit_length}, got L={L}")

        with torch.no_grad():
            movements = self.mov_enc(ts)  # AE is frozen in finetune
        word_hids, text_latent = self._encode_text(text_emb)

        mov_len = movements.shape[1]
        m_lens = torch.full((B,), L, device=ts.device, dtype=torch.long)
        movements_gt = movements.detach() if cfg.detach_movement_latents else movements

        outputs = self._generate_movements(
            word_hids=word_hids,
            text_latent=text_latent,
            mov_len=mov_len,
            m_lens=m_lens,
            use_posterior=True,
            movements_gt=movements_gt,
            teacher_forcing_ratio=cfg.teacher_forcing_ratio,
        )

        fake_movements = outputs["fake_movements"]
        fake_ts = self.mov_dec(fake_movements)

        loss_rec_ts = F.smooth_l1_loss(fake_ts, ts)
        loss_rec_mov = F.smooth_l1_loss(fake_movements, movements_gt)
        loss_kld = _kl_gauss(outputs["mu_pos"], outputs["logvar_pos"], outputs["mu_pri"], outputs["logvar_pri"])
        loss = cfg.lambda_rec_ts * loss_rec_ts + cfg.lambda_rec_mov * loss_rec_mov + cfg.lambda_kld * loss_kld
        return {
            "loss": loss,
            "loss_rec_ts": loss_rec_ts,
            "loss_rec_mov": loss_rec_mov,
            "loss_kld": loss_kld,
        }

    @torch.no_grad()
    def sample(self, text_emb: Tensor, out_len: int) -> Tensor:
        """Prior-only autoregressive generation. Returns ``(B, out_len, C)``."""
        cfg = self.cfg
        if out_len % cfg.unit_length != 0:
            raise ValueError(f"out_len must be divisible by unit_length={cfg.unit_length}, got {out_len}")
        B = text_emb.shape[0]
        device = text_emb.device
        mov_len = out_len // cfg.unit_length
        word_hids, text_latent = self._encode_text(text_emb)
        m_lens = torch.full((B,), out_len, device=device, dtype=torch.long)
        gen = self._generate_movements(
            word_hids=word_hids,
            text_latent=text_latent,
            mov_len=mov_len,
            m_lens=m_lens,
            use_posterior=False,
        )
        return self.mov_dec(gen["fake_movements"])  # (B, out_len, C)
