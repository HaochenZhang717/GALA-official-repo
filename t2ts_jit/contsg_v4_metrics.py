from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


@dataclass
class MetricConfig:
    k: int = 5
    mdd_bins: int = 32
    acd_max_lag: int = 50
    # <=0 means no subsampling (use all accumulated validation samples).
    max_samples: int = 0
    seed: int = 0


class AlignerV4MetricEmbedder:
    """Load a full TextToTimeSeriesAlignerV4 model for metric embedding."""

    def __init__(self, ckpt_path: Path, device: torch.device, hf_token: str | None = None):
        from aligner_v4 import TextToTimeSeriesAlignerV4

        payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        args = payload["args"]

        self.model = TextToTimeSeriesAlignerV4(
            llm_id=args["llm_id"],
            chronos_model_id=args["chronos_model_id"],
            proj_dim=int(args["proj_dim"]),
            llm_dtype=getattr(torch, args.get("llm_dtype", "bfloat16")),
            device=device,
            init_temperature=float(args["init_temperature"]),
            min_temperature=float(args["min_temperature"]),
            hf_token=hf_token,
        )
        self.model.load_state_dict(payload["trainable_state"], strict=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = torch.device(device)

    @torch.no_grad()
    def encode_text(self, captions: Sequence[str]) -> torch.Tensor:
        return self.model.encode_text(list(captions))

    @torch.no_grad()
    def encode_series(self, series_bvl: torch.Tensor) -> torch.Tensor:
        return self.model.encode_time_series(series_bvl)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def _cov(x: np.ndarray) -> np.ndarray:
    if x.shape[0] <= 1:
        d = x.shape[1]
        return np.eye(d, dtype=np.float64) * 1e-6
    c = np.cov(x, rowvar=False).astype(np.float64)
    c += np.eye(c.shape[0], dtype=np.float64) * 1e-6
    return c


def _sqrtm_psd(a: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(a)
    vals = np.clip(vals, a_min=0.0, a_max=None)
    return (vecs * np.sqrt(vals)[None, :]) @ vecs.T


def _trace_sqrt_product_psd(cov1: np.ndarray, cov2: np.ndarray) -> float:
    """Stable trace(sqrt(cov1 @ cov2)) for PSD covariances.

    Use the symmetric form:
      trace(sqrt(cov1 @ cov2)) = trace(sqrt(sqrt(cov1) @ cov2 @ sqrt(cov1)))
    where the inside matrix is symmetric PSD up to numerical noise.
    """
    sqrt_cov1 = _sqrtm_psd(cov1)
    mid = sqrt_cov1 @ cov2 @ sqrt_cov1
    # Numerical symmetrization for stability.
    mid = 0.5 * (mid + mid.T)
    vals = np.linalg.eigvalsh(mid)
    vals = np.clip(vals, a_min=0.0, a_max=None)
    return float(np.sum(np.sqrt(vals)))


def _frechet_distance(mu1: np.ndarray, cov1: np.ndarray, mu2: np.ndarray, cov2: np.ndarray) -> float:
    """Numerically stable Fréchet distance between two Gaussians.

    The exact formula is:
      ||mu1 - mu2||^2 + Tr(cov1 + cov2 - 2 * sqrt(cov1 @ cov2))
    """
    diff = mu1 - mu2
    tr_covmean = _trace_sqrt_product_psd(cov1, cov2)
    val = float(diff.dot(diff) + np.trace(cov1) + np.trace(cov2) - 2.0 * tr_covmean)
    # Guard against tiny negative values caused by floating-point noise.
    if val < 0.0 and val > -1e-8:
        return 0.0
    return max(val, 0.0)


def _subsample_indices(n: int, max_n: int, seed: int) -> np.ndarray:
    if max_n <= 0 or n <= max_n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def _pairwise_l2_sq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    d = x2 + y2 - 2.0 * (x @ y.T)
    return np.maximum(d, 0.0)


def _prdc_precision_recall(real: np.ndarray, gen: np.ndarray, k: int) -> tuple[float, float]:
    n_r = real.shape[0]
    n_g = gen.shape[0]
    if n_r < 2 or n_g < 2:
        return float("nan"), float("nan")

    k_r = min(max(k, 1), n_r - 1)
    k_g = min(max(k, 1), n_g - 1)

    d_rr = _pairwise_l2_sq(real, real)
    np.fill_diagonal(d_rr, np.inf)
    r_radius = np.partition(d_rr, kth=k_r - 1, axis=1)[:, k_r - 1]

    d_gg = _pairwise_l2_sq(gen, gen)
    np.fill_diagonal(d_gg, np.inf)
    g_radius = np.partition(d_gg, kth=k_g - 1, axis=1)[:, k_g - 1]

    d_gr = _pairwise_l2_sq(gen, real)  # query=gen, ref=real
    precision = float((d_gr <= r_radius[None, :]).any(axis=1).mean())

    d_rg = d_gr.T
    recall = float((d_rg <= g_radius[None, :]).any(axis=1).mean())
    return precision, recall


def _acf_profile(arr: np.ndarray, max_lag: int) -> np.ndarray:
    # arr: (N, V, L)
    n, v, l = arr.shape
    kmax = min(max_lag, l - 1)
    out = np.zeros((v, kmax + 1), dtype=np.float64)
    for vi in range(v):
        vals = []
        x_all = arr[:, vi, :]
        mu = x_all.mean(axis=1, keepdims=True)
        xc = x_all - mu
        denom = np.sum(xc * xc, axis=1) + 1e-8
        for k in range(kmax + 1):
            if k == 0:
                num = np.sum(xc * xc, axis=1)
            else:
                num = np.sum(xc[:, :-k] * xc[:, k:], axis=1)
            vals.append((num / denom).mean())
        out[vi] = np.asarray(vals)
    return out


class ConTSGMetricsAccumulator:
    def __init__(self, embedder: AlignerV4MetricEmbedder, cfg: MetricConfig):
        self.embedder = embedder
        self.cfg = cfg
        self.real_series: list[np.ndarray] = []
        self.gen_series: list[np.ndarray] = []
        self.real_emb: list[np.ndarray] = []
        self.gen_emb: list[np.ndarray] = []
        self.text_emb: list[np.ndarray] = []

    @torch.no_grad()
    def update(self, captions: Sequence[str], real_series_bvl: torch.Tensor, gen_series_bvl: torch.Tensor) -> None:
        real_series_bvl = real_series_bvl.to(self.embedder.device).float()
        gen_series_bvl = gen_series_bvl.to(self.embedder.device).float()

        r_emb = self.embedder.encode_series(real_series_bvl)
        g_emb = self.embedder.encode_series(gen_series_bvl)
        t_emb = self.embedder.encode_text(captions)

        self.real_series.append(_to_numpy(real_series_bvl))
        self.gen_series.append(_to_numpy(gen_series_bvl))
        self.real_emb.append(_to_numpy(r_emb))
        self.gen_emb.append(_to_numpy(g_emb))
        self.text_emb.append(_to_numpy(t_emb))

    def _concat(self):
        xr = np.concatenate(self.real_series, axis=0)
        xg = np.concatenate(self.gen_series, axis=0)
        zr = np.concatenate(self.real_emb, axis=0).astype(np.float64)
        zg = np.concatenate(self.gen_emb, axis=0).astype(np.float64)
        zt = np.concatenate(self.text_emb, axis=0).astype(np.float64)
        return xr, xg, zr, zg, zt

    def compute(self) -> dict[str, float]:
        xr, xg, zr, zg, zt = self._concat()

        n = min(len(zr), len(zg))
        idx = _subsample_indices(n, self.cfg.max_samples, self.cfg.seed)
        zr = zr[idx]
        zg = zg[idx]
        zt = zt[idx]
        xr = xr[idx]
        xg = xg[idx]

        mu_r, mu_g = zr.mean(axis=0), zg.mean(axis=0)
        cov_r, cov_g = _cov(zr), _cov(zg)
        fid = _frechet_distance(mu_r, cov_r, mu_g, cov_g)

        pr_p, pr_r = _prdc_precision_recall(zr, zg, self.cfg.k)

        # MDD over flattened values per variate with fixed bins from real range
        v = xr.shape[1]
        mdd_list = []
        for vi in range(v):
            rv = xr[:, vi, :].reshape(-1)
            gv = xg[:, vi, :].reshape(-1)
            lo, hi = float(rv.min()), float(rv.max())
            if hi - lo < 1e-8:
                mdd_list.append(0.0)
                continue
            hist_r, edges = np.histogram(rv, bins=self.cfg.mdd_bins, range=(lo, hi), density=False)
            hist_g, _ = np.histogram(np.clip(gv, lo, hi), bins=edges, density=False)
            pr_hist = hist_r / max(hist_r.sum(), 1)
            pg_hist = hist_g / max(hist_g.sum(), 1)
            mdd_list.append(float(np.abs(pr_hist - pg_hist).mean()))
        mdd = float(np.mean(mdd_list))

        acf_r = _acf_profile(xr, self.cfg.acd_max_lag)
        acf_g = _acf_profile(xg, self.cfg.acd_max_lag)
        acd = float(np.sqrt(np.sum((acf_r - acf_g) ** 2)))

        # SD/KD: per-variate pooled over all samples/time
        sd_vals = []
        kd_vals = []
        for vi in range(v):
            rv = xr[:, vi, :].reshape(-1)
            gv = xg[:, vi, :].reshape(-1)
            r_mu, g_mu = rv.mean(), gv.mean()
            r_std, g_std = rv.std() + 1e-8, gv.std() + 1e-8
            r_skew = np.mean(((rv - r_mu) / r_std) ** 3)
            g_skew = np.mean(((gv - g_mu) / g_std) ** 3)
            r_kurt = np.mean(((rv - r_mu) / r_std) ** 4)
            g_kurt = np.mean(((gv - g_mu) / g_std) ** 4)
            sd_vals.append(abs(r_skew - g_skew))
            kd_vals.append(abs(r_kurt - g_kurt))
        sd = float(np.mean(sd_vals))
        kd = float(np.mean(kd_vals))

        # CTTP score
        zt_n = zt / (np.linalg.norm(zt, axis=1, keepdims=True) + 1e-12)
        zg_n = zg / (np.linalg.norm(zg, axis=1, keepdims=True) + 1e-12)
        cttp = float(np.sum(zt_n * zg_n, axis=1).mean())

        # JFTSD + joint PRDC
        jr = np.concatenate([zr, zt], axis=1)
        jg = np.concatenate([zg, zt], axis=1)
        j_mu_r, j_mu_g = jr.mean(axis=0), jg.mean(axis=0)
        j_cov_r, j_cov_g = _cov(jr), _cov(jg)
        jftsd = _frechet_distance(j_mu_r, j_cov_r, j_mu_g, j_cov_g)

        jpr_p, jpr_r = _prdc_precision_recall(jr, jg, self.cfg.k)

        return {
            "fidelity/fid": fid,
            "fidelity/prdc_precision": pr_p,
            "fidelity/prdc_recall": pr_r,
            "fidelity/mdd": mdd,
            "fidelity/acd": acd,
            "fidelity/sd": sd,
            "fidelity/kd": kd,
            "adherence/cttp": cttp,
            "adherence/jftsd": jftsd,
            "adherence/joint_prdc_precision": jpr_p,
            "adherence/joint_prdc_recall": jpr_r,
            "meta/n_eval_samples": float(len(idx)),
        }


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
