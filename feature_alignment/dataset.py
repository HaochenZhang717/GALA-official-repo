"""
TSFragmentDataset

Loads ONE (source, fragment_length) split of WinfredGe/TSFragment-600K as
(caption, series) pairs for training the text↔series aligner.

Layout of each CSV in the dataset (one per split):
    SampleID, SampleNumID, TimeInterval, Text, TextEmbedding, OT
where `Text` is the caption and `OT` is a stringified list of floats whose
length equals `TimeInterval`.

Native fragment lengths are 24 / 48 / 96 — too short for Chronos-2's patch
tokenizer to give a useful embedding, so anything shorter than `min_length`
is linearly upsampled along the time axis.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset

REPO_ID = "WinfredGe/TSFragment-600K"

SOURCES = (
    "ETTh1",
    "ETTh1s",
    "ETTm1",
    "airquality",
    "electricity",
    "exchangerate",
    "traffic",
)
LENGTHS = (24, 48, 96)
VERBALTS_DATASETS = ("synthetic_u", "synthetic_m")
VERBALTS_SPLITS = ("train", "valid", "test")


def _parse_floats(s) -> np.ndarray:
    if isinstance(s, (list, tuple, np.ndarray)):
        return np.asarray(s, dtype=np.float32)
    return np.asarray(ast.literal_eval(str(s)), dtype=np.float32)


class TSFragmentDataset(Dataset):
    """One (source, length) split of TSFragment-600K.

    __getitem__ returns:
        caption: str
        series:  FloatTensor of shape (1, L) — single variate, ready to be
                 stacked into (B, n_variates=1, history_length) for Chronos.
    """

    def __init__(
        self,
        source: str = "ETTh1",
        length: int = 24,
        min_length: int = 64,
        max_rows: int | None = None,
        cache_dir: str | Path = "./hf_cache",
    ):
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}, expected one of {SOURCES}")
        if length not in LENGTHS:
            raise ValueError(f"unknown length {length!r}, expected one of {LENGTHS}")

        fname = f"embedding_cleaned_{source}_{length}.csv"
        local = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=fname,
            cache_dir=str(cache_dir),
        )

        df = pd.read_csv(local, usecols=["Text", "OT"], nrows=max_rows)
        df = df.dropna(subset=["Text", "OT"]).reset_index(drop=True)

        self.captions: list[str] = df["Text"].astype(str).tolist()

        series_np = np.stack([_parse_floats(s) for s in df["OT"]])  # (N, L)
        series_t = torch.from_numpy(series_np).float()

        if series_t.size(1) < min_length:
            # F.interpolate(mode='linear') needs (N, C, L).
            series_t = F.interpolate(
                series_t.unsqueeze(1),
                size=min_length,
                mode="linear",
                align_corners=False,
            ).squeeze(1)

        self.series = series_t  # (N, L')
        self.source = source
        self.length = length
        self.upsampled_length = self.series.size(1)

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        return self.captions[idx], self.series[idx].unsqueeze(0)  # (1, L)


class VerbalTSDataset(Dataset):
    """VerbalTS synthetic dataset loader for aligner training.

    Supports:
      - synthetic_u: univariate series, native shape (N, 128, 1)
      - synthetic_m: multivariate series, native shape (N, 128, 2)

    __getitem__ returns:
        caption: str
        series:  FloatTensor of shape (V, L), ready for collate -> (B, V, L)
    """

    def __init__(
        self,
        dataset_name: str = "synthetic_u",
        split: str = "train",
        min_length: int = 64,
        max_rows: int | None = None,
        data_root: str | Path = "./data/VerbalTSDatasets",
    ):
        if dataset_name not in VERBALTS_DATASETS:
            raise ValueError(
                f"unknown dataset_name {dataset_name!r}, expected one of {VERBALTS_DATASETS}"
            )
        if split not in VERBALTS_SPLITS:
            raise ValueError(f"unknown split {split!r}, expected one of {VERBALTS_SPLITS}")

        root = Path(data_root) / dataset_name
        ts_path = root / f"{split}_ts.npy"
        cap_path = root / f"{split}_text_caps.npy"
        if not ts_path.exists():
            raise FileNotFoundError(f"missing time-series file: {ts_path}")
        if not cap_path.exists():
            raise FileNotFoundError(f"missing caption file: {cap_path}")

        series_np = np.load(ts_path, allow_pickle=True)
        captions_np = np.load(cap_path, allow_pickle=True)

        if max_rows is not None:
            series_np = series_np[:max_rows]
            captions_np = captions_np[:max_rows]

        if len(series_np) != len(captions_np):
            raise ValueError(
                f"mismatched sample count: series={len(series_np)} captions={len(captions_np)}"
            )

        # Expected format is (N, L, V). Also support (N, L) as a fallback.
        if series_np.ndim == 2:
            series_np = series_np[:, :, None]
        if series_np.ndim != 3:
            raise ValueError(f"unexpected series shape {series_np.shape}, expected 2D or 3D array")

        series_t = torch.from_numpy(series_np).float().permute(0, 2, 1).contiguous()  # (N, V, L)

        if series_t.size(-1) < min_length:
            series_t = F.interpolate(
                series_t,
                size=min_length,
                mode="linear",
                align_corners=False,
            )

        self.series = series_t
        self.captions = [str(c[0] if isinstance(c, (list, tuple, np.ndarray)) else c) for c in captions_np]
        self.dataset_name = dataset_name
        self.split = split
        self.n_variates = int(self.series.size(1))
        self.upsampled_length = int(self.series.size(2))

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        return self.captions[idx], self.series[idx]  # (V, L)


def collate(batch):
    """DataLoader collate: keep captions as a list of str, stack series."""
    captions = [c for c, _ in batch]
    series = torch.stack([s for _, s in batch], dim=0)  # (B, 1, L)
    return captions, series


if __name__ == "__main__":
    ds = TSFragmentDataset(source="ETTh1", length=24, max_rows=None)
    print(f"split: {ds.source} / L={ds.length}  →  upsampled to L={ds.upsampled_length}")
    print(f"size : {len(ds)}")
    cap, ser = ds[0]
    print(f"caption[0]: {cap[:120]}{'...' if len(cap) > 120 else ''}")
    print(f"series[0] : shape={tuple(ser.shape)}  "
          f"min={ser.min():.3f}  max={ser.max():.3f}  mean={ser.mean():.3f}")
