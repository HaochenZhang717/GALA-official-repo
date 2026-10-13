"""
StructureAligner -- stage-1 feature-alignment baseline following STRUCTURE
(Gröger et al., NeurIPS 2025), used as a comparison point for the V7 method.

Design (matches the user's "faithful STRUCTURE" decision):
  * Both unimodal encoders (Gemma text, Chronos-2 time series) are FULLY FROZEN.
  * A small Transformer head + a linear projector per side form the trainable
    alignment layer -- i.e. exactly the `ft_method="head"` architecture of
    TextToTimeSeriesAlignerV7AblationFT.  We therefore SUBCLASS that aligner and
    pin ft_method="head", so the produced checkpoint is loaded verbatim by the
    EXISTING stage-2 builder
    (ablation_diffusion_df_n_arch/train_cond_v7_ablation_ft.py), which already
    rebuilds the head architecture from saved_args["ft_method"].

The only addition over the base class is `return_pre_proj`: the encode methods
can also return the PRE-head, frozen-encoder pooled feature.  That fixed feature
is the "original" embedding whose neighborhood geometry the STRUCTURE regularizer
preserves inside the aligned (post-head + projector) space.

IMPORTANT: the `aligned` branch below is kept byte-identical to the base
class's head-mode encode_text / encode_time_series, so the conditioning text
embedding produced here at stage-1 equals the one stage-2 reconstructs from the
checkpoint.  Do not let the two drift.

This file does NOT modify any existing code; it only imports and subclasses.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from aligner_v7_ablation_ft import TextToTimeSeriesAlignerV7AblationFT


class StructureAligner(TextToTimeSeriesAlignerV7AblationFT):
    """Frozen dual-encoder aligner with a Transformer head per side (ft=head),
    exposing the pre-head frozen feature for the STRUCTURE regularizer."""

    def __init__(
        self,
        llm_id: str = "google/embeddinggemma-300m",
        chronos_model_id: str = "amazon/chronos-2",
        proj_dim: int = 512,
        llm_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
        init_temperature: float = 0.07,
        min_temperature: float = 0.03,
        hf_token: str | None = None,
        head_depth: int = 1,
        head_num_heads: int = 12,
        head_ffn_dim: int = 2560,
        head_dropout: float = 0.0,
        ts_global_mean: float = 0.0,
        ts_global_std: float = 1.0,
    ):
        # Pin ft_method="head": both backbones frozen, Transformer head + linear
        # projector trained on each side. (No LoRA, no full fine-tuning.)
        super().__init__(
            llm_id=llm_id,
            chronos_model_id=chronos_model_id,
            proj_dim=proj_dim,
            llm_dtype=llm_dtype,
            device=device,
            init_temperature=init_temperature,
            min_temperature=min_temperature,
            hf_token=hf_token,
            tune_side="both",
            ft_method="head",
            head_depth=head_depth,
            head_num_heads=head_num_heads,
            head_ffn_dim=head_ffn_dim,
            head_dropout=head_dropout,
            ts_global_mean=ts_global_mean,
            ts_global_std=ts_global_std,
        )
        # STRUCTURE uses a FIXED contrastive temperature, so logit_scale is not
        # learned here. Freezing keeps it out of the optimizer / trainable state.
        self.logit_scale.requires_grad_(False)

    # ------------------------------------------------------------------
    # encode_text: returns aligned (post-head+proj) and optionally the
    # pre-head frozen pooled feature used as the STRUCTURE "original".
    # ------------------------------------------------------------------
    def encode_text(
        self, captions: Sequence[str], return_pre_proj: bool = False
    ):
        texts = [self._format_query(c) for c in captions]
        batch = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=2047
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.text_encoder(**batch)
        last_hidden = out.last_hidden_state
        attn = batch["attention_mask"]

        # ---- aligned branch (identical to base head-mode encode_text) ----
        pad_mask = attn == 0  # (B, T) True where padded -> ignored by attention
        head_hidden = self.text_head(last_hidden.float(), src_key_padding_mask=pad_mask)
        mask = attn.unsqueeze(-1).to(head_hidden.dtype)
        pooled = (head_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        emb = self.text_proj(pooled.float())
        aligned = F.normalize(emb, dim=-1)

        if not return_pre_proj:
            return aligned

        # ---- STRUCTURE original: mean-pool of the RAW (pre-head) hidden ----
        mask0 = attn.unsqueeze(-1).to(last_hidden.dtype)
        orig_pooled = (last_hidden * mask0).sum(dim=1) / mask0.sum(dim=1).clamp(min=1e-9)
        return aligned, orig_pooled.float()

    # ------------------------------------------------------------------
    # encode_time_series: same structure for the time-series side.
    # ------------------------------------------------------------------
    def encode_time_series(
        self, series: torch.Tensor, return_pre_proj: bool = False
    ):
        if series.device != self.device:
            series = series.to(self.device)
        series = series.float()
        bsz, n_var, seq_len = series.shape

        context = series.reshape(bsz * n_var, seq_len)
        group_ids = torch.arange(bsz, device=self.device).repeat_interleave(n_var)

        encoder_outputs, *_ = self.ts_model.encode(context=context, group_ids=group_ids)
        hidden = encoder_outputs[0]

        # ---- aligned branch (identical to base head-mode encode_time_series) ----
        head_hidden = self.ts_head(hidden.float())
        global_tok = head_hidden[:, -2, :]
        global_tok = global_tok.reshape(bsz, n_var, -1).mean(dim=1)
        emb = self.ts_proj(global_tok)
        aligned = F.normalize(emb, dim=-1)

        if not return_pre_proj:
            return aligned

        # ---- STRUCTURE original: pre-head global token (position -2) ----
        orig_tok = hidden[:, -2, :]
        orig_tok = orig_tok.reshape(bsz, n_var, -1).mean(dim=1)
        return aligned, orig_tok.float()
