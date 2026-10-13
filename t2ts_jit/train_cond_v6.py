"""
Train a conditional time-series diffusion model (JiT-style denoiser) on one
dataset split (TSFragment or VerbalTS).

The denoiser (`t2ts_jit/cond_jit_basics/denoiser.py::Denoiser`) is conditioned
on a single text-feature vector per sample. Two text-encoder choices:

  --text-encoder frozen-gemma
      Frozen google/embeddinggemma-300m via raw transformers, mean-pooled and
      L2-normalized. Mirrors feature_alignment/embeddinggemma_minimal.py.

  --text-encoder aligner-v6 --aligner-ckpt <path>
      The finetuned TextToTimeSeriesAlignerV6 from feature_alignment/train_v6.py.
      The checkpoint stores trainable weights plus the args used to build the
      aligner; this script reconstructs a same-shape V6 aligner and uses
      `encode_text()` as the text condition encoder.

In both cases the text encoder is frozen — only the denoiser is trained.

NOTE: `t2ts_jit/cond_jit_basics/model_jit.py` currently has two bugs marked
with TODO(bug) comments (text_embedder dim mismatch + a reference to a
nonexistent label_embedder). Fix them before launching training.

Launch single-GPU:
    python t2ts_jit/train_cond_v6.py [args...]

Launch multi-GPU (DDP) with torchrun:
    torchrun --standalone --nproc_per_node=4 t2ts_jit/train_cond_v6.py [args...]

Under torchrun, --batch-size is the PER-RANK batch size; effective global
batch is batch_size * world_size.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModel, AutoTokenizer, CLIPTextConfig, CLIPTextModelWithProjection

# ---------- repo path wiring ----------
_HERE = Path(__file__).resolve().parent              # .../t2ts_jit
_REPO_ROOT = _HERE.parent                            # .../TextToTimeSeries
_COND_DIR = _HERE / "cond_jit_basics"
_FA_DIR = _REPO_ROOT / "feature_alignment"

# denoiser.py uses bare `from model_jit import JiT`, so cond_jit_basics/ must
# be on sys.path; same story for feature_alignment/.
sys.path.insert(0, str(_COND_DIR))
sys.path.insert(0, str(_FA_DIR))

from denoiser import Denoiser, VerbalTSDenoiser, VerbalTSJiTDenoiser  # noqa: E402
from bridge_denoiser import BridgeDenoiser  # noqa: E402
from contsg_v4_metrics import ConTSGMetricsAccumulator, MetricConfig, seed_all  # noqa: E402
from dataset import (  # noqa: E402
    LENGTHS,
    SOURCES,
    TSFragmentDataset,
    VERBALTS_DATASETS,
    VERBALTS_SPLITS,
    VerbalTSDataset,
    collate,
)


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

TEXT_ENCODERS = ("frozen-gemma", "tunable-gemma", "aligner-v6", "frozen-longclip")
NORMALIZE_MODES = ("per-sample", "global", "none")
TEXT_NORM_MODES = ("none", "layernorm")


# ======================================================= normalized dataset ===
class NormalizedTSDataset(Dataset):
    """Wrap TSFragmentDataset to min-max rescale each series to [-1, 1].

    Diffusion / flow-matching training is best-behaved when inputs sit in a
    bounded range of roughly the same scale as the noise prior. Raw TSFragment
    series have arbitrary amplitude, so we rescale via

        x_norm = 2 * (x - x_min) / (x_max - x_min) - 1

    before feeding the denoiser.

      mode='per-sample' (default): each series is rescaled with its OWN
        (min, max). Model learns shapes only; generated outputs live on
        [-1, 1] and must be remapped back to physical units with a
        separately recorded per-sample (min, max) (not stored here — the
        per-sample stats are lost by design in this mode).

      mode='global': a single (min, max) is computed once over the entire
        base dataset and applied uniformly. Stats are exposed as
        `.global_min` / `.global_max` and saved in the checkpoint so a
        sampler can invert the scaling.

      mode='none': pass-through.

    Returns (caption, series) with `series` shape (1, L), matching the base
    dataset's interface so the existing `collate` works unchanged.
    """

    def __init__(self, base: Dataset, mode: str = "per-sample", eps: float = 1e-6):
        if mode not in NORMALIZE_MODES:
            raise ValueError(f"unknown normalize mode {mode!r}; expected one of {NORMALIZE_MODES}")
        self.base = base
        self.mode = mode
        self.eps = eps
        if mode == "global":
            # `base.series` is (N, L'); compute scalar min/max over all data.
            # Done once up-front instead of streaming, so it's fast even on
            # the 600K-row split (one .min()/.max() over a contiguous tensor).
            series = base.series
            self.global_min = float(series.min().item())
            self.global_max = float(series.max().item())
            if self.global_max - self.global_min < eps:
                raise ValueError(
                    "base dataset has near-constant amplitude "
                    f"(max - min = {self.global_max - self.global_min}); cannot rescale to [-1, 1]."
                )
        else:
            self.global_min = None
            self.global_max = None

    # Proxy the upsampled_length attribute so callers reading it through the
    # wrapper see the same value as on the underlying TSFragmentDataset.
    @property
    def upsampled_length(self) -> int:
        return self.base.upsampled_length

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        caption, series = self.base[idx]    # series: (1, L)
        if self.mode == "per-sample":
            mn = series.min()
            mx = series.max()
            span = (mx - mn).clamp_min(self.eps)
            series = 2.0 * (series - mn) / span - 1.0
        elif self.mode == "global":
            span = self.global_max - self.global_min   # scalar, > eps by check above
            series = 2.0 * (series - self.global_min) / span - 1.0
        return caption, series


class CachedTextEmbeddingDataset(Dataset):
    """Cache text embeddings for all samples once, then serve from CPU memory.

    Returned sample format:
      (caption: str, text_emb: Tensor[D], series: Tensor[1, L])
    """

    def __init__(
        self,
        base: Dataset,
        text_encoder: nn.Module,
        device: torch.device,
        batch_size: int = 256,
        num_workers: int = 0,
        is_main: bool = True,
    ):
        self.base = base
        self.global_min = getattr(base, "global_min", None)
        self.global_max = getattr(base, "global_max", None)
        self.normalize_mode = getattr(base, "mode", None)
        self._text_emb_cpu: torch.Tensor | None = None
        self._build_cache(
            text_encoder=text_encoder,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            is_main=is_main,
        )

    @property
    def upsampled_length(self) -> int:
        return self.base.upsampled_length

    def __len__(self) -> int:
        return len(self.base)

    def __getattr__(self, name):
        # Delegate unknown attributes (e.g., global_min/global_max/mode) to base dataset.
        return getattr(self.base, name)

    def __getitem__(self, idx: int):
        caption, series = self.base[idx]
        text_emb = self._text_emb_cpu[idx]
        return caption, text_emb, series

    @torch.no_grad()
    def _build_cache(
        self,
        text_encoder: nn.Module,
        device: torch.device,
        batch_size: int,
        num_workers: int,
        is_main: bool,
    ) -> None:
        loader = DataLoader(
            self.base,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=num_workers,
            drop_last=False,
        )
        if is_main:
            print(f"[text-cache] encoding {len(self.base)} captions...")
        chunks: list[torch.Tensor] = []
        was_training = text_encoder.training
        text_encoder.eval()
        for captions, _series in loader:
            emb = text_encoder.encode(captions).to(device=device, dtype=torch.float32)
            chunks.append(emb.detach().cpu())
        if was_training:
            text_encoder.train()
        self._text_emb_cpu = torch.cat(chunks, dim=0)
        if is_main:
            print(f"[text-cache] done: shape={tuple(self._text_emb_cpu.shape)}")


def collate_cached(batch):
    captions = [x[0] for x in batch]
    text_emb = torch.stack([x[1] for x in batch], dim=0)
    series = torch.stack([x[2] for x in batch], dim=0)
    return captions, text_emb, series


class IndexedCachedTextEmbeddingDataset(Dataset):
    """GPU-resident text-embedding cache with index-based lookup.

    Mirrors verbalts_repro's ``CachedRawTextEmbeddingDataset``: instead of
    returning the precomputed embedding from ``__getitem__`` (which forces
    every DataLoader worker to pickle / unpickle a per-sample ``(T, D)``
    tensor for unpooled CLIP tokens — CPU-RAM-quadratic in dataset size),
    ``__getitem__`` returns only the integer index. The full embedding bank
    lives on the GPU in low precision (default bfloat16), and the training
    step looks the batch up via ``lookup(idx)``.

    Cost comparison for unpooled CLIP-L tokens (T=248, D=768):
      CachedTextEmbeddingDataset (CPU float32): N * 248 * 768 * 4 B = ~762 KB / sample
      IndexedCachedTextEmbeddingDataset (GPU bf16): N * 248 * 768 * 2 B = ~381 KB / sample
    For 100K samples that's ~76 GB CPU vs ~38 GB GPU — and GPU lookup also
    avoids the DataLoader worker pickling overhead.

    Returns from ``__getitem__``:  ``(caption: str, idx: int, series: Tensor[1, L])``
    """

    def __init__(
        self,
        base: Dataset,
        text_encoder: nn.Module,
        device: torch.device,
        batch_size: int = 256,
        num_workers: int = 0,
        is_main: bool = True,
        storage_dtype: torch.dtype = torch.bfloat16,
    ):
        self.base = base
        self.global_min = getattr(base, "global_min", None)
        self.global_max = getattr(base, "global_max", None)
        self.normalize_mode = getattr(base, "mode", None)
        self._device = torch.device(device)
        self._storage_dtype = storage_dtype
        self._text_emb_dev: torch.Tensor | None = None
        self._build_cache(
            text_encoder=text_encoder,
            batch_size=batch_size,
            num_workers=num_workers,
            is_main=is_main,
        )

    @property
    def upsampled_length(self) -> int:
        return self.base.upsampled_length

    def __getattr__(self, name):
        # Delegate unknown attributes to base dataset, like CachedTextEmbeddingDataset.
        return getattr(self.base, name)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        caption, series = self.base[idx]
        return caption, idx, series

    @torch.no_grad()
    def lookup(self, idx: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Fetch the cached embeddings for a batch of indices and cast to ``dtype``."""
        idx = idx.to(self._device, dtype=torch.long)
        return self._text_emb_dev[idx].to(dtype=dtype)

    @torch.no_grad()
    def _build_cache(
        self,
        text_encoder: nn.Module,
        batch_size: int,
        num_workers: int,
        is_main: bool,
    ) -> None:
        loader = DataLoader(
            self.base,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=num_workers,
            drop_last=False,
        )
        if is_main:
            print(
                f"[text-cache] encoding {len(self.base)} captions on {self._device} "
                f"in {self._storage_dtype} (indexed / GPU-resident)..."
            )
        chunks: list[torch.Tensor] = []
        was_training = text_encoder.training
        text_encoder.eval()
        for captions, _series in loader:
            emb = text_encoder.encode(captions)
            emb = emb.to(device=self._device, dtype=self._storage_dtype).detach()
            chunks.append(emb)
        if was_training:
            text_encoder.train()
        self._text_emb_dev = torch.cat(chunks, dim=0)
        if is_main:
            shape = tuple(self._text_emb_dev.shape)
            mem_gb = self._text_emb_dev.numel() * self._text_emb_dev.element_size() / (1024 ** 3)
            print(
                f"[text-cache] done: shape={shape} dtype={self._text_emb_dev.dtype} "
                f"device={self._text_emb_dev.device} ({mem_gb:.2f} GB)"
            )


def collate_cached_indexed(batch):
    captions = [x[0] for x in batch]
    idx = torch.tensor([x[1] for x in batch], dtype=torch.long)
    series = torch.stack([x[2] for x in batch], dim=0)
    return captions, idx, series


def int_or_none(s: str) -> int | None:
    if s is None or s.strip().lower() in ("", "none", "null"):
        return None
    return int(s)


def is_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_ddp() -> tuple[int, int, int, bool]:
    if not is_torchrun():
        return 0, 0, 1, False
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size, True


# ============================================================ text encoders ===
class FrozenGemmaTextEncoder(nn.Module):
    """Frozen embeddinggemma-300m: mean-pool + L2-normalize.

    Wraps the caption with the "task: search result | query: ..." prefix that
    embedding-gemma was trained with, matching feature_alignment/embeddinggemma_minimal.py.
    """

    def __init__(self, model_id: str, dtype: torch.dtype, device: torch.device, hf_token: str | None = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype, token=hf_token)
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._device = torch.device(device)
        self._out_dim = int(self.model.config.hidden_size)

    @property
    def out_dim(self) -> int:
        return self._out_dim

    @staticmethod
    def _format(text: str) -> str:
        return f"task: search result | query: {text}"

    @torch.no_grad()
    def encode(self, captions) -> torch.Tensor:
        texts = [self._format(c) for c in captions]
        batch = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=2047
        )
        batch = {k: v.to(self._device) for k, v in batch.items()}
        out = self.model(**batch)
        last_hidden = out.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(pooled.float(), p=2, dim=1)


class FrozenLongCLIPTextEncoder(nn.Module):
    """Frozen LongCLIP text encoder.

    ``output_type`` selects between two formats — both interchangeable with
    other text encoders downstream as long as ``TextConditioner`` /
    ``DenoiserWithTextConditioner`` are 3D-aware:

      "cls" (default): pooled + CLIP-projected ``text_embeds`` (B, projection_dim),
          L2-normalized.  Used by jit2d / jit1d.

      "all": per-token ``last_hidden_state`` (B, T, hidden_size), NOT
          normalized.  ``out_dim = hidden_size`` (= 768 for ViT-L). Used by
          --model-type=verbalts so ``TextProjectorMVarMScaleMStep`` sees a
          real token sequence to cross-attend against.

    LongCLIP-GmP variants extend CLIP's positional embeddings to 248 tokens
    (vs. 77 in stock CLIP); we mirror the verbalts_repro setup by overriding
    ``max_position_embeddings`` when the model id looks like LongCLIP.
    """

    def __init__(
        self,
        model_id: str,
        dtype: torch.dtype,
        device: torch.device,
        hf_token: str | None = None,
        output_type: str = "cls",
    ):
        super().__init__()
        if output_type not in ("cls", "all"):
            raise ValueError(f"output_type must be 'cls' or 'all', got {output_type!r}")
        self.output_type = output_type
        clip_config = CLIPTextConfig.from_pretrained(model_id, token=hf_token)
        if "LongCLIP" in model_id or "Longclip" in model_id:
            clip_config.max_position_embeddings = 248
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.model = CLIPTextModelWithProjection.from_pretrained(
            model_id, config=clip_config, torch_dtype=dtype, token=hf_token,
        )
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._device = torch.device(device)
        self._max_length = int(self.model.config.max_position_embeddings)
        if output_type == "cls":
            self._out_dim = int(self.model.config.projection_dim)
        else:  # "all"
            self._out_dim = int(self.model.config.hidden_size)

    @property
    def out_dim(self) -> int:
        return self._out_dim

    @torch.no_grad()
    def encode(self, captions) -> torch.Tensor:
        batch = self.tokenizer(
            list(captions),
            padding="max_length",
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"].to(self._device)
        out = self.model(input_ids=input_ids)
        if self.output_type == "cls":
            return F.normalize(out.text_embeds.float(), p=2, dim=1)
        return out.last_hidden_state.float()  # (B, T, hidden_size)


class TunableGemmaTextEncoder(nn.Module):
    """Trainable embeddinggemma-300m: mean-pool + L2-normalize."""

    def __init__(
        self,
        model_id: str,
        dtype: torch.dtype,
        device: torch.device,
        proj_dim: int,
        hf_token: str | None = None,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype, token=hf_token)
        self.model.to(device)
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad_(True)
        hidden = int(self.model.config.hidden_size)
        self.text_proj = nn.Linear(hidden, int(proj_dim)).to(device)
        self._device = torch.device(device)
        self._out_dim = int(proj_dim)

    @property
    def out_dim(self) -> int:
        return self._out_dim

    @staticmethod
    def _format(text: str) -> str:
        return f"task: search result | query: {text}"

    def encode(self, captions) -> torch.Tensor:
        texts = [self._format(c) for c in captions]
        batch = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=2047
        )
        batch = {k: v.to(self._device) for k, v in batch.items()}
        out = self.model(**batch)
        last_hidden = out.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        emb = self.text_proj(pooled.float())
        return F.normalize(emb, p=2, dim=1)


class AlignerV6TextEncoder(nn.Module):
    """Finetuned TextToTimeSeriesAlignerV6 used as a frozen text encoder.

    Rebuilds the V6 aligner from checkpoint args and loads matching trainable
    aligner tensors from checkpoint.
    """

    def __init__(self, ckpt_path: Path, device: torch.device, hf_token: str | None = None):
        super().__init__()
        # Lazy import: pulls in `chronos` only when this mode is selected.
        from aligner_v6 import TextToTimeSeriesAlignerV6

        self._ckpt_path = Path(ckpt_path)
        self._saved_args = None
        self._hf_token = hf_token
        payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        saved_args = payload["args"]
        self._saved_args = saved_args
        ckpt_state = payload.get("aligner_trainable_state", payload.get("trainable_state"))
        if ckpt_state is None:
            raise KeyError("checkpoint missing aligner_trainable_state/trainable_state")
        llm_dtype = DTYPES.get(saved_args.get("llm_dtype", "bfloat16"), torch.bfloat16)

        self.aligner = TextToTimeSeriesAlignerV6(
            llm_id=saved_args["llm_id"],
            chronos_model_id=saved_args["chronos_model_id"],
            proj_dim=int(saved_args["proj_dim"]),
            llm_dtype=llm_dtype,
            device=device,
            init_temperature=float(saved_args["init_temperature"]),
            min_temperature=float(saved_args["min_temperature"]),
            hf_token=hf_token,
            text_lora_r=int(saved_args.get("text_lora_r", 8)),
            text_lora_alpha=int(saved_args.get("text_lora_alpha", 16)),
            text_lora_dropout=float(saved_args.get("text_lora_dropout", 0.05)),
            text_lora_target_modules=tuple(saved_args.get("text_lora_target_modules", ("q_proj", "k_proj", "v_proj", "o_proj"))),
            ts_lora_r=int(saved_args.get("ts_lora_r", 8)),
            ts_lora_alpha=int(saved_args.get("ts_lora_alpha", 16)),
            ts_lora_dropout=float(saved_args.get("ts_lora_dropout", 0.05)),
            ts_lora_target_modules=tuple(saved_args.get("ts_lora_target_modules", ("q", "k", "v", "o"))),
        )

        model_state = self.aligner.state_dict()
        filtered = {
            k: v for k, v in ckpt_state.items()
            if (k in model_state and tuple(model_state[k].shape) == tuple(v.shape))
        }
        # missing, unexpected = self.aligner.load_state_dict(filtered, strict=False)
        # print(
        #     "[AlignerV6TextEncoder] loaded keys "
        #     f"{len(filtered)}/{len(model_state)}; missing={len(missing)} unexpected={len(unexpected)}"
        # )
        missing, unexpected = self.aligner.load_state_dict(filtered, strict=False)
        print(
            "[AlignerV6TextEncoder] loaded state_dict "
            f"(matched={len(filtered)} missing={len(missing)} unexpected={len(unexpected)})"
        )
        for p in self.aligner.parameters():
            p.requires_grad_(False)
        self.aligner.eval()

        self._out_dim = int(self.aligner.text_proj.out_features)

    @property
    def out_dim(self) -> int:
        return self._out_dim

    @torch.no_grad()
    def encode(self, captions) -> torch.Tensor:
        return self.aligner.encode_text(list(captions))

    def build_retrieval_model(self, device: torch.device) -> nn.Module:
        """Build a full aligner (with Chronos) for retrieval sanity-check."""
        from aligner_v6 import TextToTimeSeriesAlignerV6

        payload = torch.load(str(self._ckpt_path), map_location="cpu", weights_only=False)
        saved_args = payload["args"]
        ckpt_state = payload.get("aligner_trainable_state", payload.get("trainable_state"))
        if ckpt_state is None:
            raise KeyError("checkpoint missing aligner_trainable_state/trainable_state")
        llm_dtype = DTYPES.get(saved_args.get("llm_dtype", "bfloat16"), torch.bfloat16)
        model = TextToTimeSeriesAlignerV6(
            llm_id=saved_args["llm_id"],
            chronos_model_id=saved_args["chronos_model_id"],
            proj_dim=int(saved_args["proj_dim"]),
            llm_dtype=llm_dtype,
            device=device,
            init_temperature=float(saved_args["init_temperature"]),
            min_temperature=float(saved_args["min_temperature"]),
            hf_token=self._hf_token,
            text_lora_r=int(saved_args.get("text_lora_r", 8)),
            text_lora_alpha=int(saved_args.get("text_lora_alpha", 16)),
            text_lora_dropout=float(saved_args.get("text_lora_dropout", 0.05)),
            text_lora_target_modules=tuple(saved_args.get("text_lora_target_modules", ("q_proj", "k_proj", "v_proj", "o_proj"))),
            ts_lora_r=int(saved_args.get("ts_lora_r", 8)),
            ts_lora_alpha=int(saved_args.get("ts_lora_alpha", 16)),
            ts_lora_dropout=float(saved_args.get("ts_lora_dropout", 0.05)),
            ts_lora_target_modules=tuple(saved_args.get("ts_lora_target_modules", ("q", "k", "v", "o"))),
        )
        model_state = model.state_dict()
        filtered = {
            k: v for k, v in ckpt_state.items()
            if (k in model_state and tuple(model_state[k].shape) == tuple(v.shape))
        }
        model.load_state_dict(filtered, strict=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model


class AlignerV6MetricEmbedder:
    """Load a full TextToTimeSeriesAlignerV6 model for metric embedding."""

    def __init__(self, ckpt_path: Path, device: torch.device, hf_token: str | None = None):
        from aligner_v6 import TextToTimeSeriesAlignerV6

        payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        args = payload["args"]
        ckpt_state = payload.get("aligner_trainable_state", payload.get("trainable_state"))
        if ckpt_state is None:
            raise KeyError("checkpoint missing aligner_trainable_state/trainable_state")

        self.model = TextToTimeSeriesAlignerV6(
            llm_id=args["llm_id"],
            chronos_model_id=args["chronos_model_id"],
            proj_dim=int(args["proj_dim"]),
            llm_dtype=getattr(torch, args.get("llm_dtype", "bfloat16")),
            device=device,
            init_temperature=float(args["init_temperature"]),
            min_temperature=float(args["min_temperature"]),
            hf_token=hf_token,
            text_lora_r=int(args.get("text_lora_r", 8)),
            text_lora_alpha=int(args.get("text_lora_alpha", 16)),
            text_lora_dropout=float(args.get("text_lora_dropout", 0.05)),
            text_lora_target_modules=tuple(args.get("text_lora_target_modules", ("q_proj", "k_proj", "v_proj", "o_proj"))),
            ts_lora_r=int(args.get("ts_lora_r", 8)),
            ts_lora_alpha=int(args.get("ts_lora_alpha", 16)),
            ts_lora_dropout=float(args.get("ts_lora_dropout", 0.05)),
            ts_lora_target_modules=tuple(args.get("ts_lora_target_modules", ("q", "k", "v", "o"))),
        )
        model_state = self.model.state_dict()
        filtered = {
            k: v for k, v in ckpt_state.items()
            if (k in model_state and tuple(model_state[k].shape) == tuple(v.shape))
        }
        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        print(
            "[AlignerV6MetricEmbedder] loaded state_dict "
            f"(matched={len(filtered)} missing={len(missing)} unexpected={len(unexpected)})"
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = torch.device(device)

    @torch.no_grad()
    def encode_text(self, captions) -> torch.Tensor:
        return self.model.encode_text(list(captions))

    @torch.no_grad()
    def encode_series(self, series_bvl: torch.Tensor) -> torch.Tensor:
        return self.model.encode_time_series(series_bvl)


# ====================================================================== args ===
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- dataset ----
    g = p.add_argument_group("dataset")
    g.add_argument("--dataset-backend", default="tsfragment", choices=["tsfragment", "verbalts"])
    g.add_argument("--source", default="ETTh1", choices=SOURCES)
    g.add_argument("--length", type=int, default=24, choices=LENGTHS)
    g.add_argument("--verbalts-name", default="synthetic_u", choices=VERBALTS_DATASETS)
    g.add_argument("--verbalts-split", default="train", choices=VERBALTS_SPLITS)
    g.add_argument("--verbalts-val-split", default="valid", choices=VERBALTS_SPLITS + ("none",))
    g.add_argument(
        "--verbalts-root",
        type=Path,
        default=Path("./data/VerbalTSDatasets"),
    )
    g.add_argument("--min-length", type=int, default=64,
                   help="Linearly upsample series shorter than this.")
    g.add_argument("--max-rows", type=int_or_none, default=2000,
                   help="Cap rows loaded from the CSV. Pass 'none' to load all rows.")
    g.add_argument("--cache-dir", type=Path, default=Path("./hf_cache"))
    g.add_argument("--normalize", default="per-sample", choices=NORMALIZE_MODES,
                   help="min-max rescale each series to [-1, 1] before training. "
                        "'per-sample' uses each series's own (min, max); "
                        "'global' uses train-split (min, max) (saved in the checkpoint); "
                        "'none' disables rescaling.")

    # ---- text encoder ----
    g = p.add_argument_group("text encoder")
    g.add_argument("--text-encoder", default="aligner-v6", choices=TEXT_ENCODERS)
    g.add_argument("--llm-id", default="google/embeddinggemma-300m",
                   help="Used by --text-encoder=frozen-gemma. The aligner mode "
                        "reads llm_id from the checkpoint instead.")
    g.add_argument("--llm-dtype", default="bfloat16", choices=list(DTYPES),
                   help="dtype for the frozen-gemma / frozen-longclip encoder.")
    g.add_argument("--longclip-model-id", default="zer0int/LongCLIP-GmP-ViT-L-14",
                   help="Used by --text-encoder=frozen-longclip.")
    g.add_argument("--longclip-output-type", default="cls", choices=["cls", "all"],
                   help="'cls': pooled CLIP-projected text_embeds (B, projection_dim), L2-normalized "
                        "(jit2d/jit1d standard). 'all': per-token last_hidden_state (B, T, hidden_size), "
                        "needed by --model-type=verbalts to feed TextProjectorMVarMScaleMStep a real "
                        "token sequence to cross-attend against.")
    g.add_argument("--aligner-ckpt", type=Path, default=None,
                   help="Path to a TextToTimeSeriesAlignerV6 checkpoint "
                        "(required for --text-encoder=aligner-v6).")
    g.add_argument("--hf-token-env", default="HF_TOKEN",
                   help="Env var name holding the Hugging Face token (gated models).")
    g.add_argument("--text-cache-batch-size", type=int, default=256,
                   help="Batch size used when precomputing text embeddings into CPU memory.")
    g.add_argument("--text-cond-norm", default="layernorm", choices=TEXT_NORM_MODES,
                   help="Normalization applied to cached text embeddings before denoiser.")
    g.add_argument("--text-adapter-hidden-dim", type=int, default=512,
                   help="Hidden dim of text adapter MLP (<=0 disables adapter).")
    g.add_argument("--text-adapter-dropout", type=float, default=0.0,
                   help="Dropout used in text adapter MLP.")
    g.add_argument("--cond-drop-prob", type=float, default=0.0,
                   help="Per-sample probability of replacing the text embedding with a learnable "
                        "null embedding during training (classifier-free guidance prep). "
                        "0.0 disables. Inference is always conditional (no CFG yet).")
    g.add_argument("--tunable-proj-dim", type=int, default=512,
                   help="Projector output dim for --text-encoder=tunable-gemma.")

    # ---- denoiser (JiT) ----
    g = p.add_argument_group("denoiser")
    g.add_argument("--model-type", default="jit2d",
                   choices=["jit2d", "jit1d", "verbalts", "verbalts_jit", "bridge"],
                   help="jit2d: delay-embed the series into a 2D image and use JiT (linear flow matching). "
                        "jit1d: feed the raw (B, L, F) series to JiT1D (linear flow matching). "
                        "verbalts: multi-patch VerbalTS architecture trained with discrete DDPM; "
                        "uses the --verbalts-* args below (most jit/diffusion args are ignored). "
                        "verbalts_jit: same VerbalTS backbone but trained with the JiT-style linear "
                        "flow-matching objective (x-pred + v-loss). Uses the --verbalts-* backbone "
                        "args plus --p-mean/--p-std/--t-eps/--sampling-method/--num-sampling-steps; "
                        "--verbalts-num-steps / --verbalts-beta-* / --verbalts-schedule / "
                        "--verbalts-text-projector-stages are ignored. "
                        "bridge: BRIDGE (ICML 2025) data-space DDPM with a DomainUnifiedPrototyper + "
                        "1D UNet cross-attention and gated text fusion. Uses the --bridge-* args below; "
                        "most jit/diffusion args are ignored. Loss is noise-MSE (not comparable to jit "
                        "v-MSE). Requires pooled 2D text embeddings.")
    g.add_argument("--img-size", type=int, default=8,
                   help="[jit2d only] Side length of the delay-embedded image. Must satisfy "
                        "delay*(img_size-1) + img_size >= seq_len (the upsampled series must fit). "
                        "Ignored for jit1d.")
    g.add_argument("--delay", type=int, default=None,
                   help="[jit2d only] Column stride used by DelayEmbedder. "
                        "Defaults to --img-size (non-overlapping columns). "
                        "delay < img_size yields overlapping columns; delay > img_size leaves gaps. "
                        "Must satisfy delay*(img_size-1) + img_size >= seq_len.")
    g.add_argument("--patch-size", type=int, default=2)
    g.add_argument("--in-channels", type=int, default=1,
                   help="Number of variates (TSFragment-600K is univariate -> 1).")
    g.add_argument("--hidden-channels", type=int, default=384)
    g.add_argument("--depth", type=int, default=8)
    g.add_argument("--num-heads", type=int, default=6)
    g.add_argument("--attn-dropout", type=float, default=0.0)
    g.add_argument("--proj-dropout", type=float, default=0.0)
    g.add_argument("--bottleneck-dim", type=int, default=128)
    g.add_argument("--num-register-tokens", type=int, default=0,
                   help="Number of register tokens prepended to the JiT/JiT1D token sequence.")

    # ---- diffusion / flow-matching ----
    g = p.add_argument_group("diffusion")
    g.add_argument("--p-mean", type=float, default=0.0,
                   help="Mean of the Gaussian over logit(t); t = sigmoid(N(p_mean, p_std)).")
    g.add_argument("--p-std", type=float, default=1.0)
    g.add_argument("--t-eps", type=float, default=1e-3,
                   help="Lower clamp on (1 - t) when computing the velocity target.")
    g.add_argument("--noise-scale", type=float, default=1.0)
    g.add_argument("--ema-decay1", type=float, default=0.9999)
    g.add_argument("--ema-decay2", type=float, default=0.999)
    g.add_argument("--sampling-method", default="heun", choices=["euler", "heun"])
    g.add_argument("--num-sampling-steps", type=int, default=50)
    g.add_argument("--interval-min", type=float, default=0.0)
    g.add_argument("--interval-max", type=float, default=1.0)

    # ---- verbalts (only when --model-type=verbalts) ----
    g = p.add_argument_group("verbalts (only when --model-type=verbalts)")
    g.add_argument("--verbalts-channels", type=int, default=64,
                   help="VerbalTS internal d_model (channel width of patch tokens).")
    g.add_argument("--verbalts-multipatch-num", type=int, default=3,
                   help="Number of parallel patch scales K.")
    g.add_argument("--verbalts-base-patch", type=int, default=2,
                   help="Smallest patch length; scale i uses patch_len = base_patch * (L_patch_len ** i).")
    g.add_argument("--verbalts-L-patch-len", type=int, default=2,
                   help="Multiplicative growth factor for patch length across scales.")
    g.add_argument("--verbalts-layers", type=int, default=4,
                   help="Number of ResidualBlocks in the VerbalTS backbone.")
    g.add_argument("--verbalts-nheads", type=int, default=8,
                   help="Attention heads in VerbalTS time/feature self-attention.")
    g.add_argument("--verbalts-diff-embed-dim", type=int, default=128,
                   help="DiffusionEmbedding output dim (sinusoidal+MLP).")
    g.add_argument("--verbalts-var-emb", type=int, default=16,
                   help="Per-variate embedding dim used by SideEncoder_Var.")
    g.add_argument("--verbalts-time-emb", type=int, default=128,
                   help="Time-position sinusoidal dim used by SideEncoder_Var.")
    g.add_argument("--verbalts-attention-mask-type", default="full", choices=["full", "parallel"],
                   help="full: attention across all scales; parallel: block-diagonal (no cross-scale).")
    g.add_argument("--verbalts-condition-type", default="adaLN",
                   choices=["adaLN", "add", "cross_attention"],
                   help="How attr_emb is injected into each ResidualBlock.")
    g.add_argument("--verbalts-num-steps", type=int, default=50,
                   help="Discrete DDPM training steps. Sampling walks all of these in reverse (DDIM deterministic).")
    g.add_argument("--verbalts-beta-start", type=float, default=1e-4)
    g.add_argument("--verbalts-beta-end", type=float, default=0.5)
    g.add_argument("--verbalts-schedule", default="quad", choices=["quad", "linear"],
                   help="DDPM beta schedule.")
    g.add_argument("--verbalts-text-projector-stages", type=int, default=5,
                   help="Number of diffusion-step stages for TextProjectorMVarMScaleMStep. "
                        "n_stages divides --verbalts-num-steps into bins; one learnable step-query per bin.")
    g.add_argument("--verbalts-textemb-hidden-dim", type=int, default=512,
                   help="Hidden dim of the TrainableTextProjector-style MLP that compresses raw CLIP "
                        "features before TextProjectorMVarMScaleMStep. Mirrors verbalts_repro's "
                        "TrainableTextProjector(Linear(text_embed_dim, textemb_hidden_dim) -> LayerNorm "
                        "-> LeakyReLU -> Linear(textemb_hidden_dim, text_emb_dim)).")
    g.add_argument("--verbalts-text-emb-dim", type=int, default=64,
                   help="Output dim of the text pre-projector; also the dim_in of "
                        "TextProjectorMVarMScaleMStep's var/scale/step cross-attention queries.")

    # ---- bridge (only when --model-type=bridge) ----
    g = p.add_argument_group("bridge (only when --model-type=bridge)")
    g.add_argument("--bridge-latent-dim", type=int, default=32,
                   help="Prototype latent dim (repre_emb_channels); also the UNet cross-attn context_dim.")
    g.add_argument("--bridge-num-latents", type=int, default=16,
                   help="Number of learnable orthogonal prototype vectors in DomainUnifiedPrototyper.")
    g.add_argument("--bridge-latent-unit", type=int, default=1,
                   help="Bridge latent_unit (cond_emb_channels = latent_dim * latent_unit when CFG on).")
    g.add_argument("--bridge-prototype-dim", type=int, default=0,
                   help="Prototyper internal conv width. 0 -> fall back to --bridge-latent-dim.")
    g.add_argument("--bridge-model-channels", type=int, default=64,
                   help="Base channel count of the 1D UNet backbone.")
    g.add_argument("--bridge-num-res-blocks", type=int, default=2,
                   help="ResBlocks per UNet level.")
    g.add_argument("--bridge-attention-resolutions", type=int, nargs="+", default=[1, 2, 4],
                   help="Downsample rates at which (cross-)attention is applied.")
    g.add_argument("--bridge-channel-mult", type=int, nargs="+", default=[1, 2, 4, 4],
                   help="UNet channel multiplier per level. Downsamples by 2 (len-1) times, so "
                        "seq_len must be divisible by 2**(len-1). Shorten for short series.")
    g.add_argument("--bridge-dropout", type=float, default=0.0)
    g.add_argument("--bridge-num-heads", type=int, default=8,
                   help="Attention heads in UNet (Spatial1DTransformer / AttentionBlock).")
    g.add_argument("--bridge-num-head-channels", type=int, default=-1,
                   help="Fixed channels per head (overrides --bridge-num-heads when >0).")
    g.add_argument("--bridge-transformer-depth", type=int, default=1,
                   help="Depth of each Spatial1DTransformer cross-attention block.")
    g.add_argument("--bridge-fusion-type", default="gated_add",
                   choices=["gated_add", "add", "concat"],
                   help="ConditioningMLP text<->prototype fusion mode.")
    g.add_argument("--bridge-use-cfg", action="store_true", default=True,
                   help="Enable BRIDGE's own prototype-context classifier-free guidance.")
    g.add_argument("--bridge-no-cfg", dest="bridge_use_cfg", action="store_false",
                   help="Disable BRIDGE CFG (no null prototype embedding).")
    g.add_argument("--bridge-cond-drop-prob", type=float, default=0.5,
                   help="Train-time probability of dropping the prototype context to the null embedding.")
    g.add_argument("--bridge-cfg-scale", type=float, default=1.0,
                   help="Sampling CFG scale. 1.0 = no guidance; >1 amplifies the prototype signal.")
    # Diffusion schedule defaults match ConTSG's bridge_*.yaml (50 quad steps,
    # beta 1e-4 -> 0.5), which also coincide with this repo's VerbalTS settings.
    g.add_argument("--bridge-diffusion-steps", type=int, default=50,
                   help="Number of discrete DDPM training steps.")
    g.add_argument("--bridge-beta-start", type=float, default=1e-4)
    g.add_argument("--bridge-beta-end", type=float, default=0.5)
    g.add_argument("--bridge-schedule", default="quad", choices=["linear", "quad", "cosine"],
                   help="DDPM beta schedule.")
    g.add_argument("--bridge-sampler", default="ddim", choices=["ddim", "ddpm"],
                   help="Generation sampler.")
    g.add_argument("--bridge-ddim-steps", type=int, default=0,
                   help="DDIM reverse steps. 0 -> use --bridge-diffusion-steps.")
    g.add_argument("--bridge-eta", type=float, default=0.0,
                   help="DDIM eta (0 = deterministic).")
    g.add_argument("--bridge-clip-denoised", action="store_true", default=False,
                   help="Clamp predicted x0 to [-1, 1]. Default OFF: v6 series are z-score "
                        "normalized and routinely exceed +/-1, so clipping would corrupt them.")

    # ---- optim / loop ----
    g = p.add_argument_group("training")
    g.add_argument("--batch-size", type=int, default=64,
                   help="PER-RANK batch size when running under torchrun.")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--tunable-gemma-lr", type=float, default=None,
                   help="Optional separate LR for Gemma params when --text-encoder=tunable-gemma. "
                        "If unset, falls back to --lr.")
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--grad-clip", type=float, default=1.0,
                   help="Max grad norm; set <=0 to disable.")
    g.add_argument("--num-workers", type=int, default=0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="cpu",
                   help="Ignored under torchrun (per-rank cuda device is set automatically).")

    # ---- validation ----
    g = p.add_argument_group("validation")
    g.add_argument("--val-ratio", type=float, default=0.1,
                   help="Fraction of the dataset reserved for validation. 0 disables.")
    g.add_argument("--val-split-seed", type=int, default=None,
                   help="Seed for the train/val split. Defaults to --seed.")
    g.add_argument("--val-seed", type=int, default=12345,
                   help="Fixed RNG seed used inside evaluate() so diffusion val "
                        "loss is comparable across epochs (the loss is stochastic "
                        "in t and the noise sample).")
    g.add_argument("--eval-every-steps", type=int, default=0,
                   help="Run eval every N optimizer steps. 0 = only at end of each epoch.")
    g.add_argument("--enable-gen-eval", action="store_true",
                   help="Enable generated-quality evaluation during training (default: disabled).")
    g.add_argument("--gen-eval-every-epochs", type=int, default=100,
                   help="Every N epochs, sample on val split and compute 10 ConTSG metrics. 0 disables.")
    g.add_argument("--gen-eval-max-batches", type=int, default=0,
                   help="Limit generation-eval to first N val batches (0 means full val set).")
    g.add_argument(
        "--metrics-aligner-ckpt",
        "--metrics-aligner-v6-ckpt",
        dest="metrics_aligner_ckpt",
        type=Path,
        default=None,
        help=(
            "Path to aligner checkpoint used for generation metric embeddings. "
            "Must match TextToTimeSeriesAlignerV6 shape; defaults to disabled when unset."
        ),
    )
    g.add_argument("--metrics-k", type=int, default=5)
    g.add_argument("--metrics-mdd-bins", type=int, default=32)
    g.add_argument("--metrics-max-samples", type=int, default=2000)
    g.add_argument("--metrics-seed", type=int, default=123)
    g.add_argument("--check-aligner-retrieval", action="store_true",
                   help="Deprecated in v6 script: sanity-check is run automatically when "
                        "--text-encoder=aligner-v6 and val split is enabled.")
    g.add_argument("--check-aligner-max-batches", type=int, default=0,
                   help="Limit retrieval sanity-check to first N val batches on rank 0 "
                        "(0 means full val loader).")

    # ---- I/O ----
    g = p.add_argument_group("io")
    g.add_argument("--log-every", type=int, default=10)
    g.add_argument("--save", type=Path, default=Path("./denoiser_ckpt.pt"))
    g.add_argument("--save-best", type=Path, default=Path("./denoiser_best.pt"))
    g.add_argument("--save-every-epochs", type=int, default=100,
                   help="Save a periodic checkpoint every N epochs (0 to disable). "
                        "Path: <save stem>_epoch{N}<save suffix>.")

    # ---- wandb ----
    g = p.add_argument_group("wandb")
    g.add_argument("--wandb-project", default="text-ts-diffusion")
    g.add_argument("--wandb-entity", default=None)
    g.add_argument("--wandb-run-name", default=None)
    g.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    g.add_argument("--wandb-tags", nargs="+", default=[])
    g.add_argument("--wandb-notes", default=None)

    return p.parse_args()


# ================================================================== builders ===
def build_text_encoder(args: argparse.Namespace, device: torch.device, hf_token: str | None) -> nn.Module:
    if args.text_encoder == "frozen-gemma":
        return FrozenGemmaTextEncoder(
            model_id=args.llm_id,
            dtype=DTYPES[args.llm_dtype],
            device=device,
            hf_token=hf_token,
        )
    if args.text_encoder == "tunable-gemma":
        return TunableGemmaTextEncoder(
            model_id=args.llm_id,
            dtype=DTYPES[args.llm_dtype],
            device=device,
            proj_dim=args.tunable_proj_dim,
            hf_token=hf_token,
        )
    if args.text_encoder == "aligner-v6":
        if args.aligner_ckpt is None:
            raise ValueError("--aligner-ckpt is required when --text-encoder=aligner-v6.")
        return AlignerV6TextEncoder(
            ckpt_path=args.aligner_ckpt,
            device=device,
            hf_token=hf_token,
        )
    if args.text_encoder == "frozen-longclip":
        return FrozenLongCLIPTextEncoder(
            model_id=args.longclip_model_id,
            dtype=DTYPES[args.llm_dtype],
            device=device,
            hf_token=hf_token,
            output_type=args.longclip_output_type,
        )
    raise ValueError(f"unknown --text-encoder {args.text_encoder!r}")


def build_denoiser(args: argparse.Namespace, text_embed_dim: int, seq_len: int, device: torch.device) -> nn.Module:
    if args.model_type == "bridge":
        ns = SimpleNamespace(
            text_embed_dim=text_embed_dim,
            seq_len=seq_len,
            in_channels=args.in_channels,
            latent_dim=args.bridge_latent_dim,
            num_latents=args.bridge_num_latents,
            latent_unit=args.bridge_latent_unit,
            prototype_dim=args.bridge_prototype_dim,
            model_channels=args.bridge_model_channels,
            num_res_blocks=args.bridge_num_res_blocks,
            attention_resolutions=args.bridge_attention_resolutions,
            channel_mult=args.bridge_channel_mult,
            dropout=args.bridge_dropout,
            num_heads=args.bridge_num_heads,
            num_head_channels=args.bridge_num_head_channels,
            transformer_depth=args.bridge_transformer_depth,
            fusion_type=args.bridge_fusion_type,
            use_cfg=args.bridge_use_cfg,
            cond_drop_prob=args.bridge_cond_drop_prob,
            cfg_scale=args.bridge_cfg_scale,
            diffusion_steps=args.bridge_diffusion_steps,
            beta_start=args.bridge_beta_start,
            beta_end=args.bridge_beta_end,
            schedule=args.bridge_schedule,
            sampler=args.bridge_sampler,
            ddim_steps=args.bridge_ddim_steps,
            eta=args.bridge_eta,
            clip_denoised=args.bridge_clip_denoised,
            ema_decay1=args.ema_decay1,
            ema_decay2=args.ema_decay2,
        )
        return BridgeDenoiser(ns).to(device)

    if args.model_type == "verbalts":
        ns = SimpleNamespace(
            text_embed_dim=text_embed_dim,
            seq_len=seq_len,
            in_channels=args.in_channels,
            channels=args.verbalts_channels,
            multipatch_num=args.verbalts_multipatch_num,
            base_patch=args.verbalts_base_patch,
            L_patch_len=args.verbalts_L_patch_len,
            layers=args.verbalts_layers,
            nheads=args.verbalts_nheads,
            diffusion_embedding_dim=args.verbalts_diff_embed_dim,
            var_emb=args.verbalts_var_emb,
            time_emb=args.verbalts_time_emb,
            attention_mask_type=args.verbalts_attention_mask_type,
            condition_type=args.verbalts_condition_type,
            num_steps=args.verbalts_num_steps,
            beta_start=args.verbalts_beta_start,
            beta_end=args.verbalts_beta_end,
            schedule=args.verbalts_schedule,
            text_projector_stages=args.verbalts_text_projector_stages,
            textemb_hidden_dim=args.verbalts_textemb_hidden_dim,
            text_emb_dim=args.verbalts_text_emb_dim,
            noise_scale=args.noise_scale,
            ema_decay1=args.ema_decay1,
            ema_decay2=args.ema_decay2,
        )
        return VerbalTSDenoiser(ns).to(device)

    if args.model_type == "verbalts_jit":
        ns = SimpleNamespace(
            text_embed_dim=text_embed_dim,
            seq_len=seq_len,
            in_channels=args.in_channels,
            channels=args.verbalts_channels,
            multipatch_num=args.verbalts_multipatch_num,
            base_patch=args.verbalts_base_patch,
            L_patch_len=args.verbalts_L_patch_len,
            layers=args.verbalts_layers,
            nheads=args.verbalts_nheads,
            diffusion_embedding_dim=args.verbalts_diff_embed_dim,
            var_emb=args.verbalts_var_emb,
            time_emb=args.verbalts_time_emb,
            attention_mask_type=args.verbalts_attention_mask_type,
            condition_type=args.verbalts_condition_type,
            textemb_hidden_dim=args.verbalts_textemb_hidden_dim,
            text_emb_dim=args.verbalts_text_emb_dim,
            # JiT-style flow-matching args (instead of DDPM/DDIM)
            P_mean=args.p_mean,
            P_std=args.p_std,
            t_eps=args.t_eps,
            noise_scale=args.noise_scale,
            sampling_method=args.sampling_method,
            num_sampling_steps=args.num_sampling_steps,
            ema_decay1=args.ema_decay1,
            ema_decay2=args.ema_decay2,
        )
        return VerbalTSJiTDenoiser(ns).to(device)

    ns = SimpleNamespace(
        text_embed_dim=text_embed_dim,
        model_type=args.model_type,
        img_size=args.img_size,
        delay=args.delay,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        depth=args.depth,
        num_heads=args.num_heads,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        bottleneck_dim=args.bottleneck_dim,
        num_register_tokens=args.num_register_tokens,
        seq_len=seq_len,
        P_mean=args.p_mean,
        P_std=args.p_std,
        t_eps=args.t_eps,
        noise_scale=args.noise_scale,
        ema_decay1=args.ema_decay1,
        ema_decay2=args.ema_decay2,
        sampling_method=args.sampling_method,
        num_sampling_steps=args.num_sampling_steps,
        interval_min=args.interval_min,
        interval_max=args.interval_max,
    )
    return Denoiser(ns).to(device)


class TextConditioner(nn.Module):
    """Normalize/adapt cached text embeddings before denoiser conditioning."""

    def __init__(self, in_dim: int, norm_mode: str, adapter_hidden_dim: int, adapter_dropout: float):
        super().__init__()
        if norm_mode not in TEXT_NORM_MODES:
            raise ValueError(f"unknown text norm mode {norm_mode!r}; expected one of {TEXT_NORM_MODES}")
        self.norm = nn.Identity() if norm_mode == "none" else nn.LayerNorm(in_dim)
        self.adapter = None
        if adapter_hidden_dim > 0:
            self.adapter = nn.Sequential(
                nn.Linear(in_dim, adapter_hidden_dim),
                nn.GELU(),
                nn.Dropout(adapter_dropout),
                nn.Linear(adapter_hidden_dim, in_dim),
            )
            # Start near identity for stability.
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)
        # Learnable "no-condition" token used when a sample's condition is dropped
        # at train time. Zero-init so initial uncond behavior matches LN(0)=0.
        # Whether this is trainable is decided by the caller (frozen by default;
        # unfrozen when --cond-drop-prob > 0).
        self.null_embed = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x: torch.Tensor, drop_mask: torch.Tensor | None = None) -> torch.Tensor:
        if drop_mask is not None:
            # drop_mask: BoolTensor[B] (True -> replace this sample's cond with null).
            # Supports both 2D (B, D) (pooled text embed) and 3D (B, T, D)
            # (token-sequence text embed, e.g. frozen-longclip output_type=all).
            null = self.null_embed.to(dtype=x.dtype).expand_as(x)
            if x.dim() == 2:
                mask_shape = (-1, 1)
            elif x.dim() == 3:
                mask_shape = (-1, 1, 1)
            else:
                raise ValueError(f"text_embeds must be 2D or 3D, got {x.dim()}D shape {tuple(x.shape)}")
            x = torch.where(drop_mask.view(*mask_shape), null, x)
        y = self.norm(x)
        if self.adapter is None:
            return y
        return y + self.adapter(y)


class DenoiserWithTextConditioner(nn.Module):
    """Wrap denoiser so text conditioning is consistently applied."""

    def __init__(self, denoiser: Denoiser, conditioner: TextConditioner):
        super().__init__()
        self.denoiser = denoiser
        self.conditioner = conditioner

    def forward(self, x_ts: torch.Tensor, text_embeds: torch.Tensor,
                drop_mask: torch.Tensor | None = None):
        return self.denoiser(x_ts, self.conditioner(text_embeds, drop_mask=drop_mask))

    @torch.no_grad()
    def generate(self, text_embeds: torch.Tensor):
        return self.denoiser.generate(self.conditioner(text_embeds))

    @torch.no_grad()
    def _call_with_ema(self, which: str, fn):
        params = [p for p in self.denoiser.parameters()]
        if which == "ema1":
            ema_params = list(self.denoiser.ema_params1)
        elif which == "ema2":
            ema_params = list(self.denoiser.ema_params2)
        else:
            raise ValueError(f"unknown ema branch {which!r}")
        if len(params) != len(ema_params):
            raise RuntimeError(
                f"EMA size mismatch: params={len(params)} ema={len(ema_params)}"
            )
        backups = [p.detach().clone() for p in params]
        try:
            for p, e in zip(params, ema_params):
                p.data.copy_(e.data)
            return fn()
        finally:
            for p, b in zip(params, backups):
                p.data.copy_(b.data)

    @torch.no_grad()
    def forward_ema(self, x_ts: torch.Tensor, text_embeds: torch.Tensor, which: str = "ema2"):
        return self._call_with_ema(which, lambda: self.forward(x_ts, text_embeds))

    @torch.no_grad()
    def generate_ema(self, text_embeds: torch.Tensor, which: str = "ema2"):
        return self._call_with_ema(which, lambda: self.generate(text_embeds))

    @torch.no_grad()
    def update_ema(self):
        self.denoiser.update_ema()


# =================================================================== helpers ===
def series_to_xts(series: torch.Tensor) -> torch.Tensor:
    """Dataset gives (B, n_variates=1, L); DelayEmbedder.ts_to_img wants
    (batch, length, features). Permute once here."""
    return series.permute(0, 2, 1).contiguous()


def upsample_for_aligner(series_bvl: torch.Tensor, min_length: int) -> torch.Tensor:
    """Ensure series length is at least `min_length` before feeding aligner."""
    if min_length <= 0:
        return series_bvl
    cur_len = int(series_bvl.size(-1))
    if cur_len >= min_length:
        return series_bvl
    return F.interpolate(series_bvl, size=min_length, mode="linear", align_corners=False)


@torch.no_grad()
def evaluate(model: nn.Module, text_encoder: nn.Module, loader, device: torch.device,
             seed: int, is_distributed: bool, eval_mode: str = "ema2",
             text_cache: "IndexedCachedTextEmbeddingDataset | None" = None) -> dict[str, float]:
    """Mean diffusion loss over `loader`.

    The diffusion loss is stochastic in `t` and the noise sample, so we reseed
    the global RNG before iterating to make val numbers reproducible across
    epochs. Under DDP every rank runs the same loop and the sums are reduced
    across ranks (each rank holds a disjoint shard via DistributedSampler), so
    the reported loss is the global mean.

    When ``text_cache`` is provided, the loader is expected to yield
    ``(captions, idx, series)`` and the text embedding is fetched via
    ``text_cache.lookup(idx)`` (GPU-resident, indexed). Otherwise the loader
    yields either ``(captions, text_emb, series)`` (CPU cache) or
    ``(captions, series)`` (no cache, on-the-fly encode)."""
    was_training = model.training
    model.eval()
    txt_was_training = getattr(text_encoder, "training", False)
    if hasattr(text_encoder, "eval"):
        text_encoder.eval()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    total = torch.zeros(1, device=device)
    n_seen = torch.zeros(1, device=device)
    for batch in loader:
        if text_cache is not None:
            _captions, idx, series = batch
            text_cond = text_cache.lookup(idx.to(device))
        elif len(batch) == 3:
            _captions, text_emb, series = batch
            text_cond = text_emb.to(device)
        else:
            captions, series = batch
            text_cond = text_encoder.encode(captions).to(device)
        x_ts = series_to_xts(series).to(device)
        if eval_mode == "online":
            loss = model(x_ts, text_cond)
        else:
            loss = model.forward_ema(x_ts, text_cond, which=eval_mode)
        b = x_ts.size(0)
        total += loss.detach() * b
        n_seen += b

    if is_distributed:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_seen, op=dist.ReduceOp.SUM)

    if was_training:
        model.train()
    if txt_was_training and hasattr(text_encoder, "train"):
        text_encoder.train()

    n = max(n_seen.item(), 1.0)
    return {"loss": (total / n).item()}


@torch.no_grad()
def evaluate_aligner_retrieval(
    aligner_model: nn.Module,
    loader,
    device: torch.device,
    aligner_min_length: int,
    max_batches: int = 0,
) -> dict[str, float]:
    """Run aligner sanity-check (retrieval top-1 + mean paired cosine)."""
    total = 0
    correct_t2s = 0
    correct_s2t = 0
    cosine_sum = 0.0
    for bi, (captions, series) in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        series = series.to(device)
        series = upsample_for_aligner(series, aligner_min_length)
        out = aligner_model(captions, series, gather=False)
        b = len(captions)
        correct_t2s += int((out["acc_t2s"] * b).item())
        correct_s2t += int((out["acc_s2t"] * b).item())
        cosine_sum += float(out["cosine"].mean().item()) * b
        total += b
    if total == 0:
        return {"acc_t2s": 0.0, "acc_s2t": 0.0, "mean_cosine": 0.0}
    return {
        "acc_t2s": correct_t2s / total,
        "acc_s2t": correct_s2t / total,
        "mean_cosine": cosine_sum / total,
    }


def save_ckpt(path: Path, inner: nn.Module, args: argparse.Namespace,
              text_encoder_kind: str, text_embed_dim: int, seq_len: int,
              normalization: dict, split_info: dict,
              extra: dict | None = None) -> int:
    """Save the full denoiser state + args + shape info + normalization + split.

    `normalization` records how training inputs were standardized so a sampler
    can invert the scaling later. For 'per-sample' there's nothing global to
    save, but the mode itself tells the sampler the outputs live on a
    per-series-z-scored manifold.

    `split_info` records the exact train/val indices into the *base* dataset
    (after max_rows truncation) so an eval script can reconstruct the same
    val subset without needing identical CLI args.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: param.detach().cpu() for name, param in inner.state_dict().items()}
    payload = {
        "denoiser_state": state,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "text_encoder_kind": text_encoder_kind,
        "text_embed_dim": text_embed_dim,
        "seq_len": seq_len,
        "normalization": normalization,
        "split_info": split_info,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return len(state)


# ====================================================================== main ===
def main() -> None:
    args = parse_args()

    rank, local_rank, world_size, is_distributed = setup_ddp()
    is_main = rank == 0

    # Different seed per rank so dataloader workers / model dropout differ
    # across ranks while staying deterministic in --seed.
    torch.manual_seed(args.seed + rank)

    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)

    # ---------- wandb (rank 0 only) ----------
    wandb_config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    wandb_config["world_size"] = world_size
    wandb_config["global_batch_size"] = args.batch_size * world_size
    if is_main:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            tags=args.wandb_tags,
            notes=args.wandb_notes,
            config=wandb_config,
        )
    else:
        run = None

    # ---------- dataset ----------
    split_seed = args.seed if args.val_split_seed is None else args.val_split_seed
    norm_global_min = None
    norm_global_max = None
    if args.dataset_backend == "tsfragment":
        base_ds = TSFragmentDataset(
            source=args.source,
            length=args.length,
            min_length=0,
            max_rows=args.max_rows,
            cache_dir=args.cache_dir,
        )
        # Match v5 behavior: normalize first on base dataset, then split.
        ds = NormalizedTSDataset(base_ds, mode=args.normalize)
        seq_len = ds.upsampled_length
        if args.normalize == "global":
            norm_global_min = ds.global_min
            norm_global_max = ds.global_max

        n_total = len(ds)
        n_val = int(round(n_total * args.val_ratio)) if args.val_ratio > 0 else 0
        n_val = min(n_val, n_total - 1)
        n_train = n_total - n_val
        if n_val > 0:
            train_ds, val_ds = random_split(
                ds, [n_train, n_val],
                generator=torch.Generator().manual_seed(split_seed),
            )
            train_indices = list(train_ds.indices)
            val_indices = list(val_ds.indices)
            val_base = Subset(base_ds, val_indices)
        else:
            train_ds, val_ds = ds, None
            train_indices = list(range(n_total))
            val_indices = []
            val_base = None
    else:
        train_base = VerbalTSDataset(
            dataset_name=args.verbalts_name,
            split=args.verbalts_split,
            min_length=args.min_length,
            max_rows=args.max_rows,
            data_root=args.verbalts_root,
        )
        val_base = None
        if args.verbalts_val_split.lower() != "none":
            val_base = VerbalTSDataset(
                dataset_name=args.verbalts_name,
                split=args.verbalts_val_split,
                min_length=args.min_length,
                max_rows=args.max_rows,
                data_root=args.verbalts_root,
            )

        train_ds = NormalizedTSDataset(train_base, mode=args.normalize)
        val_ds = NormalizedTSDataset(val_base, mode=args.normalize) if val_base is not None else None
        if args.normalize == "global":
            norm_global_min = train_ds.global_min
            norm_global_max = train_ds.global_max
            if val_ds is not None:
                val_ds.global_min = norm_global_min
                val_ds.global_max = norm_global_max
        seq_len = train_ds.upsampled_length

        n_train = len(train_ds)
        n_val = 0 if val_ds is None else len(val_ds)
        n_total = n_train + n_val
        train_indices = list(range(n_train))
        val_indices = list(range(n_val))
    if is_main:
        if args.normalize == "global":
            print(f"normalize: global(train)  min={norm_global_min:.4f}  max={norm_global_max:.4f}")
        else:
            print(f"normalize: {args.normalize}")

    if args.model_type == "jit2d":
        if args.delay is None:
            args.delay = args.img_size
        if args.delay <= 0:
            raise ValueError(f"--delay must be positive, got {args.delay}")
        coverage = args.delay * (args.img_size - 1) + args.img_size
        if coverage < seq_len:
            raise ValueError(
                f"delay*(img_size-1)+img_size ({coverage}) < seq_len ({seq_len}); "
                f"the delay embedding cannot cover the series. "
                f"Increase --img-size, or increase --delay (currently {args.delay})."
            )
    elif args.model_type == "jit1d":
        if seq_len % args.patch_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by patch_size ({args.patch_size}) "
                f"for jit1d (BottleneckPatchEmbed1D uses Conv1d with stride=patch_size)."
            )
    elif args.model_type == "verbalts":
        # VerbalTS's TsPatchEmbedding internally ReplicationPad2d-pads to a
        # multiple of each scale's patch length, so we don't require seq_len
        # to be divisible by any of (base_patch * L_patch_len ** i). The
        # jit-only args (img_size/delay/patch_size/p_mean/p_std/t_eps/
        # interval_min/interval_max/sampling_method/num_sampling_steps) are
        # ignored. `noise_scale` is used only at the start of generate();
        # leave at 1.0 for DDPM-consistent sampling.
        if args.noise_scale != 1.0 and is_main:
            print(
                f"[warn] --model-type=verbalts uses DDPM sampling; --noise-scale={args.noise_scale} "
                f"will scale the initial x_T away from the trained N(0, I) distribution."
            )
        if is_main:
            print(
                f"[verbalts] DDPM training (num_steps={args.verbalts_num_steps}, "
                f"schedule={args.verbalts_schedule}). Loss values are noise-MSE and "
                f"NOT comparable to jit2d/jit1d v-MSE losses."
            )
    elif args.model_type == "bridge":
        # BRIDGE: DomainUnifiedPrototyper does two stride-2 convs then flattens, so
        # seq_len must be divisible by 4; the UNet downsamples by 2 once per
        # channel_mult level (len-1 times), so seq_len must be divisible by
        # 2**(len(channel_mult)-1). Most jit/diffusion args are ignored; the
        # --bridge-* args drive the architecture and DDPM schedule.
        unet_down = 2 ** (len(args.bridge_channel_mult) - 1)
        if seq_len % 4 != 0:
            raise ValueError(
                f"--model-type=bridge requires seq_len % 4 == 0 (DomainUnifiedPrototyper "
                f"downsamples by 4); got seq_len={seq_len}."
            )
        if seq_len % unet_down != 0:
            raise ValueError(
                f"--model-type=bridge requires seq_len % {unet_down} == 0 for "
                f"--bridge-channel-mult={args.bridge_channel_mult} (UNet downsamples by "
                f"2**(len-1)); got seq_len={seq_len}. Shorten --bridge-channel-mult."
            )
        if is_main:
            print(
                f"[bridge] DDPM training (diffusion_steps={args.bridge_diffusion_steps}, "
                f"schedule={args.bridge_schedule}, sampler={args.bridge_sampler}, "
                f"cfg_scale={args.bridge_cfg_scale}). Loss is noise-MSE and NOT comparable "
                f"to jit2d/jit1d v-MSE losses."
            )
    else:  # verbalts_jit
        # Same backbone-side constraints as `verbalts` (no seq_len divisibility
        # needed). DDPM-only args (num_steps/beta_*/schedule/text_projector_stages)
        # are ignored; the JiT-style flow-matching args (p_mean/p_std/t_eps/
        # sampling_method/num_sampling_steps) are read. Loss is v-MSE, directly
        # comparable to jit2d/jit1d.
        if is_main:
            print(
                f"[verbalts_jit] Linear flow-matching on the VerbalTS backbone "
                f"(p_mean={args.p_mean}, p_std={args.p_std}, sampling={args.sampling_method}, "
                f"steps={args.num_sampling_steps}). v-MSE loss; comparable to jit2d/jit1d."
            )

    # 3D text embeddings (frozen-longclip --longclip-output-type=all) only make
    # sense for the verbalts / verbalts_jit denoisers; jit2d/jit1d's text_embedder
    # and bridge's ConditioningMLP expect a pooled (B, D) embedding.
    if (
        args.text_encoder == "frozen-longclip"
        and args.longclip_output_type == "all"
        and args.model_type not in ("verbalts", "verbalts_jit")
    ):
        raise ValueError(
            "--longclip-output-type=all returns per-token (B, T, D) embeddings, which only "
            f"--model-type=verbalts / verbalts_jit can consume; got --model-type={args.model_type!r}. "
            "Use --longclip-output-type=cls for jit2d/jit1d/bridge, or switch --model-type."
        )

    # ---------- text encoder (+ optional text embedding cache) ----------
    hf_token = os.environ.get(args.hf_token_env) or None
    text_encoder = build_text_encoder(args, device=device, hf_token=hf_token)
    text_embed_dim = text_encoder.out_dim
    if is_main:
        print(f"text encoder: {args.text_encoder}  (out_dim={text_embed_dim})")
    use_cached_text_emb = args.text_encoder != "tunable-gemma"
    # Auto-switch to GPU-resident indexed caching when the encoder returns
    # unpooled token sequences (frozen-longclip output_type=all). For 3D
    # CLIP-L token sequences (T=248, D=768), a CPU-tensor cache + DataLoader
    # pickling per sample explodes RAM; the indexed path stores everything
    # on the GPU in bfloat16 and only ships int indices through workers.
    use_indexed_text_cache = (
        use_cached_text_emb
        and args.text_encoder == "frozen-longclip"
        and args.longclip_output_type == "all"
    )
    # When the indexed path is active we keep a handle here so the train /
    # eval loops can call .lookup(idx) at step time. Stays None otherwise.
    text_cache_train: IndexedCachedTextEmbeddingDataset | None = None
    text_cache_val: IndexedCachedTextEmbeddingDataset | None = None
    if use_cached_text_emb:
        cache_cls = IndexedCachedTextEmbeddingDataset if use_indexed_text_cache else CachedTextEmbeddingDataset
        if is_main and use_indexed_text_cache:
            print("[text-cache] using GPU-resident indexed cache (bfloat16) for unpooled CLIP tokens.")
        if args.dataset_backend == "tsfragment":
            # Match v5 behavior: cache text embeddings once on the full normalized
            # dataset, then split the cached dataset by precomputed indices.
            cached_ds = cache_cls(
                base=ds,
                text_encoder=text_encoder,
                device=device,
                batch_size=args.text_cache_batch_size,
                num_workers=args.num_workers,
                is_main=is_main,
            )
            if n_val > 0:
                train_ds = Subset(cached_ds, train_indices)
                val_ds = Subset(cached_ds, val_indices)
            else:
                train_ds, val_ds = cached_ds, None
            if use_indexed_text_cache:
                text_cache_train = cached_ds
                text_cache_val = cached_ds  # same bank: Subset just remaps indices
        else:
            train_ds = cache_cls(
                base=train_ds,
                text_encoder=text_encoder,
                device=device,
                batch_size=args.text_cache_batch_size,
                num_workers=args.num_workers,
                is_main=is_main,
            )
            if use_indexed_text_cache:
                text_cache_train = train_ds
            if val_ds is not None:
                val_ds = cache_cls(
                    base=val_ds,
                    text_encoder=text_encoder,
                    device=device,
                    batch_size=args.text_cache_batch_size,
                    num_workers=args.num_workers,
                    is_main=is_main,
                )
                if use_indexed_text_cache:
                    text_cache_val = val_ds
    elif is_main:
        print("[text-cache] disabled for tunable-gemma; encoding on-the-fly with gradients.")

    # collate function depends on cache mode
    if use_indexed_text_cache:
        train_collate_fn = collate_cached_indexed
    elif use_cached_text_emb:
        train_collate_fn = collate_cached
    else:
        train_collate_fn = collate

    if is_main:
        if args.dataset_backend == "tsfragment":
            split_desc = f"{args.source} / L={args.length}"
        else:
            split_desc = f"{args.verbalts_name} train={args.verbalts_split} val={args.verbalts_val_split if val_ds is not None else 'none'}"
        print(f"split: {split_desc}  (diffusion seq_len={seq_len})")
        print(f"aligner eval min_length={args.min_length}")
        print(f"size : {n_total} total ({n_train} train / {n_val} val)")
        print(f"world_size={world_size}  per-rank batch={args.batch_size}  "
              f"global batch={args.batch_size * world_size}")

    if is_distributed:
        sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True, seed=args.seed,
        )
        loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler,
            collate_fn=train_collate_fn, num_workers=args.num_workers,
            drop_last=True, pin_memory=True,
        )
    else:
        sampler = None
        loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=train_collate_fn, num_workers=args.num_workers, drop_last=True,
        )

    val_loader = None
    metric_loader = None
    if val_ds is not None:
        if is_distributed:
            val_sampler = DistributedSampler(
                val_ds, num_replicas=world_size, rank=rank,
                shuffle=False, drop_last=False,
            )
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size, sampler=val_sampler,
                collate_fn=train_collate_fn, num_workers=args.num_workers,
                drop_last=False, pin_memory=True,
            )
        else:
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False,
                collate_fn=train_collate_fn, num_workers=args.num_workers, drop_last=False,
            )
        if is_main:
            # IMPORTANT: generation metrics must use raw (unnormalized) time series.
            if args.dataset_backend == "tsfragment":
                raw_val_ds = Subset(base_ds, val_indices)
            else:
                raw_val_ds = val_base
            metric_loader = DataLoader(
                raw_val_ds, batch_size=args.batch_size, shuffle=False,
                collate_fn=collate, num_workers=args.num_workers, drop_last=False,
            )

    metric_embedder = None
    if args.enable_gen_eval:
        if args.metrics_aligner_ckpt is not None:
            if is_main:
                print(f"[metrics] loading aligner embedder from {args.metrics_aligner_ckpt}")
            metric_embedder = AlignerV6MetricEmbedder(
                ckpt_path=args.metrics_aligner_ckpt,
                device=device,
                hf_token=hf_token,
            )
        elif is_main:
            print("[metrics] --metrics-aligner-ckpt not set; generation metrics will be skipped.")
    elif is_main:
        print("[metrics] --enable-gen-eval not set; skipping aligner load for generation metrics.")

    # ---------- mandatory aligner sanity-check(s) on val split ----------
    sanity_loader = None
    if val_ds is not None and is_main:
        # IMPORTANT: aligner retrieval should run on raw (unnormalized) series.
        # Rebuild the val subset on top of base datasets instead of `val_ds`
        # (which may be normalized for diffusion training).
        if args.dataset_backend == "tsfragment":
            sanity_val_ds = Subset(base_ds, val_indices)
        else:
            sanity_val_ds = val_base
        sanity_loader = DataLoader(
            sanity_val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=args.num_workers,
            drop_last=False,
        )

    # 1) Sanity-check the aligner used as text encoder.
    if args.text_encoder == "aligner-v6" and val_ds is not None:
        if is_main:
            print("[sanity:text-encoder] running aligner retrieval on val split...")
            aligner_retrieval_model = text_encoder.build_retrieval_model(device=device)
            retrieval = evaluate_aligner_retrieval(
                aligner_retrieval_model,
                sanity_loader,
                device=device,
                aligner_min_length=args.min_length,
                max_batches=args.check_aligner_max_batches,
            )
            print(
                "[sanity:text-encoder] "
                f"mean_cos={retrieval['mean_cosine']:.4f}  "
                f"acc_t2s={retrieval['acc_t2s']:.4f}  "
                f"acc_s2t={retrieval['acc_s2t']:.4f}"
            )
            if run is not None:
                wandb.log(
                    {
                        "sanity/text_encoder/mean_cosine": retrieval["mean_cosine"],
                        "sanity/text_encoder/acc_t2s": retrieval["acc_t2s"],
                        "sanity/text_encoder/acc_s2t": retrieval["acc_s2t"],
                    },
                    step=0,
                )
            del aligner_retrieval_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if is_distributed:
            dist.barrier()

    # 2) Sanity-check the separate aligner used by generation metrics.
    if args.enable_gen_eval and args.metrics_aligner_ckpt is not None and val_ds is not None:
        if is_main:
            print(
                "[sanity:metrics-encoder] running aligner retrieval on val split "
                f"for metrics ckpt: {args.metrics_aligner_ckpt}"
            )
            metrics_retrieval_embedder = AlignerV6TextEncoder(
                ckpt_path=args.metrics_aligner_ckpt,
                device=device,
                hf_token=hf_token,
            )
            metrics_retrieval_model = metrics_retrieval_embedder.build_retrieval_model(device=device)
            retrieval = evaluate_aligner_retrieval(
                metrics_retrieval_model,
                sanity_loader,
                device=device,
                aligner_min_length=args.min_length,
                max_batches=args.check_aligner_max_batches,
            )
            print(
                "[sanity:metrics-encoder] "
                f"mean_cos={retrieval['mean_cosine']:.4f}  "
                f"acc_t2s={retrieval['acc_t2s']:.4f}  "
                f"acc_s2t={retrieval['acc_s2t']:.4f}"
            )
            if run is not None:
                wandb.log(
                    {
                        "sanity/metrics_encoder/mean_cosine": retrieval["mean_cosine"],
                        "sanity/metrics_encoder/acc_t2s": retrieval["acc_t2s"],
                        "sanity/metrics_encoder/acc_s2t": retrieval["acc_s2t"],
                    },
                    step=0,
                )
            del metrics_retrieval_model
            del metrics_retrieval_embedder
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if is_distributed:
            dist.barrier()

    # ---------- denoiser + text conditioner ----------
    base_denoiser = build_denoiser(args, text_embed_dim=text_embed_dim, seq_len=seq_len, device=device)
    text_conditioner = TextConditioner(
        in_dim=text_embed_dim,
        norm_mode=args.text_cond_norm,
        adapter_hidden_dim=args.text_adapter_hidden_dim,
        adapter_dropout=args.text_adapter_dropout,
    ).to(device)
    for p in text_conditioner.parameters():
        p.requires_grad_(False)
    # null_embed is the only conditioner param we actually want to learn when
    # condition-drop is enabled. Leaving it frozen when cond_drop_prob=0 keeps
    # it invisible to DDP (find_unused_parameters=False) and to the optimizer.
    if args.cond_drop_prob > 0.0:
        text_conditioner.null_embed.requires_grad_(True)
    inner = DenoiserWithTextConditioner(base_denoiser, text_conditioner).to(device)

    if is_distributed:
        # The EMA buffers are registered on `inner` and updated via
        # `inner.update_ema()` post-step, so DDP only needs to sync gradients
        # of the actual trainable parameters.
        model = DDP(inner, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
        inner_for_state = model.module
    else:
        model = inner
        inner_for_state = inner

    # BRIDGE needs an example time series to extract domain prototypes at
    # generation time. The original Bridge evaluator grabs a batch of real
    # training series for this; we mirror that by caching the first train batch
    # as the denoiser's example pool. (During training, Bridge instead uses each
    # target series as its own prototype source, handled inside BridgeDenoiser.)
    # The loader yields (..., series) under every collate mode, so series is [-1].
    if args.model_type == "bridge":
        _ex_batch = next(iter(loader))
        _ex_series = _ex_batch[-1]  # (B, 1, L), already normalized
        inner_for_state.denoiser.set_example_pool(series_to_xts(_ex_series))  # (B, L, C)
        if is_main:
            print(f"[bridge] example pool set from first train batch: "
                  f"{tuple(inner_for_state.denoiser._example_pool.shape)} (N, L, C)")

    if is_main:
        n_train_p = sum(p.numel() for p in inner.parameters() if p.requires_grad)
        n_total_p = sum(p.numel() for p in inner.parameters())
        n_cond_p = sum(p.numel() for p in text_conditioner.parameters() if p.requires_grad)
        print(f"denoiser+text-cond params: {n_train_p:,} trainable / {n_total_p:,} total")
        print(f"text conditioner trainable params: {n_cond_p:,}")
        if args.text_encoder == "tunable-gemma":
            n_txt = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
            print(f"tunable gemma trainable params: {n_txt:,}")

    if args.text_encoder == "tunable-gemma":
        gemma_lr = args.lr if args.tunable_gemma_lr is None else float(args.tunable_gemma_lr)
        denoiser_params = [p for p in inner.parameters() if p.requires_grad]
        gemma_params = [p for p in text_encoder.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            [
                {"params": denoiser_params, "lr": args.lr},
                {"params": gemma_params, "lr": gemma_lr},
            ],
            weight_decay=args.weight_decay,
        )
        if is_main:
            print(f"optimizer lr: denoiser={args.lr:.2e}, tunable_gemma={gemma_lr:.2e}")
    else:
        trainable_params = [p for p in inner.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Captured once and threaded into every checkpoint so sampling code knows
    # how (if at all) to invert the [-1, 1] rescaling on the model's outputs.
    normalization = {
        "scheme": "minmax_-1_1",
        "mode": args.normalize,
        "global_min": norm_global_min,
        "global_max": norm_global_max,
    }

    def denormalize_for_aligner(x_bvl: torch.Tensor) -> torch.Tensor:
        """Map model-space series back to raw series scale for aligner metrics."""
        if args.normalize == "none":
            return x_bvl
        if args.normalize == "global":
            if norm_global_min is None or norm_global_max is None:
                raise RuntimeError("global normalization selected but global_min/global_max missing.")
            mn = float(norm_global_min)
            mx = float(norm_global_max)
            return ((x_bvl + 1.0) * 0.5) * (mx - mn) + mn
        raise RuntimeError(
            "aligner-based generation evaluation requires invertible normalization; "
            "mode='per-sample' cannot be inverted. Use --normalize global or none."
        )

    # Persist the exact split so eval doesn't need to re-derive it from CLI args.
    split_info = {
        "split_seed": split_seed,
        "val_ratio": args.val_ratio,
        "n_total": n_total,
        "n_train": n_train,
        "n_val": n_val,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "dataset_backend": args.dataset_backend,
        "source": args.source,
        "length": args.length,
        "verbalts_name": args.verbalts_name,
        "verbalts_split": args.verbalts_split,
        "verbalts_val_split": args.verbalts_val_split,
        "max_rows": args.max_rows,
        "min_length": args.min_length,
    }

    # ---------- train ----------
    best_val_loss = float("inf")

    def run_validation(step_now: int, epoch_now: int) -> None:
        """Eval on every rank (so DDP stays in lockstep through the next
        training step), then log + maybe save on rank 0."""
        nonlocal best_val_loss
        if val_loader is None:
            return
        modes = ("online", "ema1", "ema2")
        all_metrics = {}
        for m in modes:
            all_metrics[m] = evaluate(
                inner_for_state, text_encoder, val_loader, device,
                seed=args.val_seed, is_distributed=is_distributed, eval_mode=m,
                text_cache=text_cache_val,
            )
        if not is_main:
            return
        payload = {}
        for m in modes:
            payload.update({f"val/{m}/{k}": v for k, v in all_metrics[m].items()})
        payload["val/epoch"] = epoch_now
        wandb.log(payload, step=step_now)
        print(
            f"  [val] epoch {epoch_now} step {step_now:>5} "
            f"online={all_metrics['online']['loss']:.4f} "
            f"ema1={all_metrics['ema1']['loss']:.4f} "
            f"ema2={all_metrics['ema2']['loss']:.4f}"
        )
        if all_metrics["ema2"]["loss"] < best_val_loss:
            best_val_loss = all_metrics["ema2"]["loss"]
            n_tensors = save_ckpt(
                args.save_best, inner_for_state, args,
                text_encoder_kind=args.text_encoder,
                text_embed_dim=text_embed_dim, seq_len=seq_len,
                normalization=normalization,
                split_info=split_info,
                extra={"val_loss_ema2": best_val_loss, "step": step_now, "epoch": epoch_now},
            )
            wandb.log({"val/best_loss_ema2": best_val_loss}, step=step_now)
            print(f"        → new best val/loss {best_val_loss:.4f}  "
                  f"saved {n_tensors} tensors → {args.save_best}")

    def run_generation_metrics(step_now: int, epoch_now: int) -> None:
        if not args.enable_gen_eval:
            return
        if args.gen_eval_every_epochs <= 0:
            return
        if (epoch_now + 1) % args.gen_eval_every_epochs != 0:
            return
        if (not is_main) or metric_loader is None or metric_embedder is None:
            if is_distributed:
                dist.barrier()
            return

        seed_all(args.metrics_seed + epoch_now)
        was_training = model.training
        model.eval()

        modes = ("online", "ema1", "ema2")
        mode_metrics = {}
        for m in modes:
            acc = ConTSGMetricsAccumulator(
                embedder=metric_embedder,
                cfg=MetricConfig(
                    k=args.metrics_k,
                    mdd_bins=args.metrics_mdd_bins,
                    max_samples=args.metrics_max_samples,
                    seed=args.metrics_seed + epoch_now,
                ),
            )
            with torch.no_grad():
                for bi, (captions, series) in enumerate(metric_loader):
                    if args.gen_eval_max_batches > 0 and bi >= args.gen_eval_max_batches:
                        break
                    real_bvl = series.to(device).float()
                    text_emb = text_encoder.encode(captions).to(device)
                    if m == "online":
                        gen_blv = inner_for_state.generate(text_emb.to(device)).float()
                    else:
                        gen_blv = inner_for_state.generate_ema(text_emb.to(device), which=m).float()
                    gen_bvl = gen_blv.permute(0, 2, 1).contiguous()
                    gen_bvl_for_aligner = denormalize_for_aligner(gen_bvl)
                    real_bvl_for_aligner = upsample_for_aligner(real_bvl, args.min_length)
                    gen_bvl_for_aligner = upsample_for_aligner(gen_bvl_for_aligner, args.min_length)
                    acc.update(captions=captions, real_series_bvl=real_bvl_for_aligner, gen_series_bvl=gen_bvl_for_aligner)
            mode_metrics[m] = acc.compute()

        payload = {}
        for m in modes:
            payload.update({f"gen_eval/{m}/{k}": v for k, v in mode_metrics[m].items()})
        payload["gen_eval/epoch"] = epoch_now
        wandb.log(payload, step=step_now)
        print(
            f"  [gen-eval] epoch {epoch_now}  "
            f"online_fid={mode_metrics['online']['fidelity/fid']:.4f}  "
            f"ema1_fid={mode_metrics['ema1']['fidelity/fid']:.4f}  "
            f"ema2_fid={mode_metrics['ema2']['fidelity/fid']:.4f}"
        )
        if was_training:
            model.train()
        if is_distributed:
            dist.barrier()

    model.train()
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if text_cache_train is not None:
                _captions, idx, series = batch
                text_cond = text_cache_train.lookup(idx.to(device))
            elif len(batch) == 3:
                _captions, text_emb, series = batch
                text_cond = text_emb.to(device)
            else:
                captions, series = batch
                text_cond = text_encoder.encode(captions).to(device)
            x_ts = series_to_xts(series).to(device)
            if args.cond_drop_prob > 0.0:
                drop_mask = (torch.rand(text_cond.size(0), device=device)
                             < args.cond_drop_prob)
            else:
                drop_mask = None
            loss = model(x_ts, text_cond, drop_mask=drop_mask)

            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                clip_params = [p for p in inner.parameters() if p.requires_grad]
                if args.text_encoder == "tunable-gemma":
                    clip_params += [p for p in text_encoder.parameters() if p.requires_grad]
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    clip_params,
                    args.grad_clip,
                )
            else:
                grad_norm = torch.tensor(float("nan"))
            opt.step()
            inner_for_state.update_ema()
            step += 1

            if is_main:
                loss_v = loss.item()
                wandb.log({
                    "train/loss": loss_v,
                    "train/grad_norm": float(grad_norm),
                    "train/lr": opt.param_groups[0]["lr"],
                    "train/epoch": epoch,
                }, step=step)
                if step % args.log_every == 0:
                    dt = time.time() - t0
                    print(f"epoch {epoch}  step {step:>5}  loss={loss_v:.4f}  "
                          f"gn={float(grad_norm):.3f}  ({dt:.1f}s)")

            if args.eval_every_steps > 0 and step % args.eval_every_steps == 0:
                run_validation(step, epoch)

        # End-of-epoch eval — every rank participates (all_reduce is collective).
        run_validation(step, epoch)
        run_generation_metrics(step, epoch)

        if is_main and args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0:
            ckpt_path = args.save.with_name(
                f"{args.save.stem}_epoch{epoch + 1}{args.save.suffix}"
            )
            n_tensors = save_ckpt(
                ckpt_path, inner_for_state, args,
                text_encoder_kind=args.text_encoder,
                text_embed_dim=text_embed_dim, seq_len=seq_len,
                normalization=normalization,
                split_info=split_info,
                extra={"step": step, "epoch": epoch},
            )
            print(f"  [ckpt] epoch {epoch + 1}  saved {n_tensors} tensors → {ckpt_path}")

    # ---------- final save (rank 0 only) ----------
    if is_main:
        n_tensors = save_ckpt(
            args.save, inner_for_state, args,
            text_encoder_kind=args.text_encoder,
            text_embed_dim=text_embed_dim, seq_len=seq_len,
            normalization=normalization,
            split_info=split_info,
            extra={"final_step": step, "final_epoch": args.epochs - 1},
        )
        print(f"saved final state ({n_tensors} tensors) → {args.save.resolve()}")
        if val_loader is not None:
            print(f"best val/loss = {best_val_loss:.4f}  → {args.save_best.resolve()}")

        if run is not None and args.wandb_mode != "disabled":
            if args.save.exists():
                art = wandb.Artifact(name=f"denoiser-last-{run.id}", type="model")
                art.add_file(str(args.save))
                run.log_artifact(art)
            if args.save_best.exists():
                art = wandb.Artifact(name=f"denoiser-best-{run.id}", type="model")
                art.add_file(str(args.save_best))
                run.log_artifact(art)
        wandb.finish()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
