#!/usr/bin/env python
"""Quantify morphological (shape) diversity of each TSFragment-600K source.

Motivation
----------
GALA's margin over the strongest baseline is smallest on `traffic`, and every
caption-adherence loss in the paper sits there. The hypothesis is that traffic
fragments are morphologically homogeneous: they are essentially one periodic
template repeated with different phase and amplitude, so there is little
caption-distinguishable shape left for an alignment to exploit.

To test that, every fragment is per-series z-normalized first, which removes
level and amplitude and leaves only the *shape*. We then ask how many degrees of
freedom that shape population actually has:

  pca_top2      fraction of shape variance captured by the first 2 PCs
  n_pc_90       number of PCs needed to reach 90% of the shape variance
  eff_rank      participation ratio (sum L)^2 / sum L^2 of the PCA spectrum,
                a continuous "effective number of shape dimensions"
  silhouette    KMeans(k) separation of the shape clusters
  inertia_per_n KMeans within-cluster scatter per fragment
  mean_abs_corr mean |Pearson r| over random fragment pairs

Low n_pc_90 / eff_rank / inertia and high pca_top2 all mean "few distinct
shapes". mean_abs_corr is reported because a phase-shifted periodic family has
near-zero *signed* mean correlation while still being low-dimensional, so the
signed statistic alone would be misleading.

Usage (conda env moment_ts):
    python analyze_shape_diversity.py
    python analyze_shape_diversity.py --n-per 3000 --k 8
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

REPO_ID = "WinfredGe/TSFragment-600K"
SOURCES = ["ETTh1s", "ETTm1", "electricity", "traffic"]
LENGTHS = [24, 48, 96]


def load_shapes(source: str, length: int, n: int, cache_dir: Path, rng) -> np.ndarray:
    """Sample `n` fragments spread over the whole file (not just the head)."""
    path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
                           filename=f"embedding_cleaned_{source}_{length}.csv",
                           cache_dir=str(cache_dir))
    per_chunk = max(1, n // 12)
    picked = []
    for chunk in pd.read_csv(path, usecols=["OT"], chunksize=20000):
        col = chunk["OT"].dropna()
        if len(col) == 0:
            continue
        take = min(per_chunk, len(col))
        idx = rng.choice(len(col), size=take, replace=False)
        picked.extend(col.iloc[i] for i in idx)
    s = np.stack([np.asarray(ast.literal_eval(str(x)), dtype=np.float64)
                  for x in picked])
    if len(s) > n:
        s = s[rng.choice(len(s), size=n, replace=False)]
    return s


def znorm(x: np.ndarray) -> np.ndarray:
    """Per-series z-normalization: keep shape, drop level and amplitude."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / np.maximum(sd, 1e-8)


def mean_abs_corr(z: np.ndarray, rng, m: int = 600) -> float:
    # A perfectly flat fragment z-normalizes to all zeros and would make
    # corrcoef divide by zero; drop those before sampling.
    z = z[z.std(axis=1) > 1e-6]
    if len(z) < 2:
        return float("nan")
    sub = z[rng.choice(len(z), size=min(m, len(z)), replace=False)]
    c = np.corrcoef(sub)
    iu = np.triu_indices_from(c, k=1)
    return float(np.abs(c[iu]).mean())


def analyze(z: np.ndarray, k: int, rng) -> dict:
    evr = PCA().fit(z).explained_variance_ratio_
    cum = np.cumsum(evr)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(z)
    labels = km.labels_
    sil = (float(silhouette_score(z, labels))
           if len(np.unique(labels)) > 1 else float("nan"))
    return dict(
        n=int(len(z)),
        pca_top1=float(evr[0]),
        pca_top2=float(evr[:2].sum()),
        n_pc_90=int(np.searchsorted(cum, 0.90) + 1),
        eff_rank=float(evr.sum() ** 2 / (evr ** 2).sum()),
        silhouette=sil,
        inertia_per_n=float(km.inertia_ / len(z)),
        mean_abs_corr=mean_abs_corr(z, rng),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=2000)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", type=Path, default=Path("./hf_cache"))
    ap.add_argument("--out", type=Path, default=Path("shape_diversity.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    res: dict[str, dict] = {}
    cache: dict[str, np.ndarray] = {}
    hdr = (f"{'source':13s}{'L':>4s}{'PCAtop2':>9s}{'n_pc_90':>9s}"
           f"{'eff_rank':>10s}{'silhou.':>9s}{'inert/n':>9s}{'|corr|':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for s in SOURCES:
        for L in LENGTHS:
            z = znorm(load_shapes(s, L, args.n_per, args.cache_dir, rng))
            r = analyze(z, args.k, rng)
            res[f"{s}_{L}"] = r
            cache[f"{s}_{L}"] = z.astype(np.float32)
            print(f"{s:13s}{L:>4d}{r['pca_top2']:9.3f}{r['n_pc_90']:9d}"
                  f"{r['eff_rank']:10.2f}{r['silhouette']:9.3f}"
                  f"{r['inertia_per_n']:9.2f}{r['mean_abs_corr']:8.3f}")
    args.out.write_text(json.dumps(res, indent=2))
    np.savez_compressed(args.out.with_suffix(".npz"), **cache)
    print(f"\nwrote {args.out} and {args.out.with_suffix('.npz')}")
    print("Fewer PCs / lower eff_rank / lower inertia / higher PCAtop2 "
          "=> fewer distinct shapes.")


if __name__ == "__main__":
    main()
