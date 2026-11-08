"""One-rotation stitching of two frozen embedding spaces."""
from __future__ import annotations

import json
import time

import numpy as np

from .metrics import retrieval_metrics


def _normalize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)


def procrustes(x: np.ndarray, y: np.ndarray, coupling: np.ndarray | None = None) -> np.ndarray:
    """Orthogonal R minimising ||x R - y|| (or its coupling-weighted version, x and y unpaired)."""
    m = x.T @ (y if coupling is None else coupling @ y)
    u, _, vt = np.linalg.svd(m)
    return u @ vt


def _pca(data: np.ndarray, kmax: int) -> tuple[np.ndarray, np.ndarray]:
    data = _normalize(data)
    mean = data.mean(0)
    _, _, vt = np.linalg.svd(data - mean, full_matrices=False)
    return mean, vt[:kmax]


class Stitch:
    """Centre + PCA each side (fitted on unpaired pools), then one orthogonal rotation x -> y.

    x and y are any two frozen encoders' outputs (e.g. image and text). Only the pairs passed to
    `fit` carry correspondence; pools are optional unpaired corpora used for centring/PCA.
    """

    def __init__(self, k="auto", candidates=(16, 32, 64, 128, 256), folds: int = 5, seed: int = 0, meta=None):
        self.k = k
        self.candidates = tuple(candidates)
        self.folds = folds
        self.seed = seed
        self.meta = dict(meta or {})
        self.R: np.ndarray | None = None
        self.cv_scores: dict[int, float] = {}

    def _fit_pca(self, x_pool, y_pool, kmax):
        kmax = min(kmax, x_pool.shape[1], y_pool.shape[1], len(x_pool), len(y_pool))
        self.mx, self.cx = _pca(x_pool, kmax)
        self.my, self.cy = _pca(y_pool, kmax)
        return kmax

    def _proj(self, z, mean, comps, k):
        return _normalize((_normalize(z) - mean) @ comps[:k].T)

    def project_x(self, x, k=None):
        return self._proj(x, self.mx, self.cx, k or self.k_)

    def project_y(self, y, k=None):
        return self._proj(y, self.my, self.cy, k or self.k_)

    def fit(self, x, y, x_pool=None, y_pool=None) -> "Stitch":
        x, y = np.asarray(x), np.asarray(y)
        if len(x) != len(y):
            raise ValueError("x and y must be row-aligned pairs")
        t0 = time.perf_counter()
        x_pool = x if x_pool is None else np.asarray(x_pool)
        y_pool = y if y_pool is None else np.asarray(y_pool)
        kmax = self._fit_pca(x_pool, y_pool, max(self.candidates) if self.k == "auto" else int(self.k))
        if self.k == "auto":
            self.k_ = self._select_k(x, y, kmax)
        else:
            self.k_ = min(int(self.k), kmax)
        self.R = procrustes(self.project_x(x), self.project_y(y))
        self.fit_seconds = time.perf_counter() - t0
        self.meta.update(n_pairs=len(x), k=self.k_)
        return self

    def _select_k(self, x, y, kmax):
        n = len(x)
        ks = [c for c in self.candidates if c <= 0.8 * n and c <= kmax] or [min(min(self.candidates), kmax)]
        rng = np.random.default_rng(self.seed)
        folds = np.array_split(rng.permutation(n), self.folds)
        for k in ks:
            px, py = self.project_x(x, k), self.project_y(y, k)
            scores = []
            for f in folds:
                tr = np.setdiff1d(np.arange(n), f)
                r = procrustes(px[tr], py[tr])
                m = retrieval_metrics(px[f] @ r, py[f], np.arange(len(f)))
                scores.append((m["t2i_R@1"] + m["t2i_R@5"]) / 2)
            self.cv_scores[k] = float(np.mean(scores))
        return max(self.cv_scores, key=self.cv_scores.get)

    def transform_x(self, x):
        return self.project_x(x) @ self.R

    def transform_y(self, y):
        return self.project_y(y)

    def save(self, path):
        meta = dict(self.meta, k=self.k_, cv_scores={str(k): v for k, v in self.cv_scores.items()})
        np.savez(path, R=self.R, mx=self.mx, cx=self.cx[: self.k_], my=self.my, cy=self.cy[: self.k_],
                 meta=np.array(json.dumps(meta)))

    @classmethod
    def load(cls, path) -> "Stitch":
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        s = cls(k=meta["k"], meta=meta)
        s.k_, s.R, s.mx, s.cx, s.my, s.cy = meta["k"], d["R"], d["mx"], d["cx"], d["my"], d["cy"]
        s.cv_scores = {int(k): v for k, v in meta.get("cv_scores", {}).items()}
        return s
