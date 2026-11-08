"""Pair-free stitching: Gromov-Wasserstein initialisation + Wasserstein-Procrustes refinement."""
from __future__ import annotations

import time

import numpy as np

from .align import Stitch, procrustes


def fit_unpaired(x, y, k: int = 64, n: int = 1000, iters: int = 30, seed: int = 0, x_pool=None, y_pool=None) -> Stitch:
    """Fit a rotation from two unpaired sets (no correspondence assumed). Requires `pot`."""
    import ot

    t0 = time.perf_counter()
    x, y = np.asarray(x), np.asarray(y)
    s = Stitch(k=k, seed=seed)
    s.k_ = min(k, s._fit_pca(x if x_pool is None else x_pool, y if y_pool is None else y_pool, k))
    rng = np.random.default_rng(seed)
    xs = s.project_x(x[rng.choice(len(x), min(n, len(x)), replace=False)])
    ys = s.project_y(y[rng.choice(len(y), min(n, len(y)), replace=False)])
    p, q = ot.unif(len(xs)), ot.unif(len(ys))
    coupling = ot.gromov.gromov_wasserstein(1 - xs @ xs.T, 1 - ys @ ys.T, p, q, "square_loss")
    r = procrustes(xs, ys, coupling)
    for _ in range(iters):
        coupling = ot.emd(p, q, 1 - (xs @ r) @ ys.T)
        r_new = procrustes(xs, ys, coupling)
        if np.allclose(r_new, r, atol=1e-6):
            break
        r = r_new
    s.R = r
    s.fit_seconds = time.perf_counter() - t0
    s.meta.update(n_pairs=0, k=s.k_, method="unpaired-gw-wp")
    return s
