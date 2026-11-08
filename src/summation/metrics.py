from __future__ import annotations

import numpy as np


def _normalize(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)


def retrieval_metrics(img: np.ndarray, txt: np.ndarray, txt2img: np.ndarray, ks=(1, 5, 10)) -> dict:
    """Cross-modal retrieval recalls. txt2img[j] is the index of the image caption j belongs to."""
    img, txt = _normalize(img), _normalize(txt)
    txt2img = np.asarray(txt2img)
    sim = txt @ img.T
    correct = sim[np.arange(len(txt)), txt2img]
    t2i_rank = (sim > correct[:, None]).sum(1)
    sim_t = sim.T
    i2t_rank = np.full(len(img), np.inf)
    for i in range(len(img)):
        own = np.flatnonzero(txt2img == i)
        if len(own):
            best = sim_t[i, own].max()
            i2t_rank[i] = (sim_t[i] > best).sum()
    out = {}
    for k in ks:
        out[f"t2i_R@{k}"] = float((t2i_rank < k).mean())
    for k in ks:
        out[f"i2t_R@{k}"] = float((i2t_rank < k).mean())
    out["mR"] = float(np.mean([out[f"t2i_R@{k}"] for k in ks] + [out[f"i2t_R@{k}"] for k in ks]))
    return out


def mutual_knn(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    """Mutual k-NN alignment (Huh et al. 2024): mean overlap of k-NN sets of paired rows."""
    a, b = _normalize(a), _normalize(b)

    def knn(x):
        s = x @ x.T
        np.fill_diagonal(s, -np.inf)
        return np.argpartition(-s, k, axis=1)[:, :k]

    na, nb = knn(a), knn(b)
    return float(np.mean([len(np.intersect1d(na[i], nb[i])) / k for i in range(len(a))]))
