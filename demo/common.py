"""Pre-registered evaluation suite shared by the demos."""
from __future__ import annotations

import numpy as np

from summation import Stitch, mutual_knn, retrieval_metrics
from summation.unpaired import fit_unpaired

SEEDS = (0, 1, 2)


def first_caption_idx(owner):
    return np.searchsorted(owner, np.arange(owner.max() + 1))


def stitched_runs(load, img, txt, own_tr, own_te, ns):
    xtr, ytr = load(img, "train", "image"), load(txt, "train", "text")
    xte, yte = load(img, "test", "image"), load(txt, "test", "text")
    starts, counts = first_caption_idx(own_tr), np.bincount(own_tr)
    out = {"mutual_10nn_native": mutual_knn(xte, yte[first_caption_idx(own_te)], 10)}
    for n in ns:
        n_eff = min(n, len(xtr))
        runs = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(xtr), n_eff, replace=False)
            cap = starts[idx] + rng.integers(0, counts[idx])
            s = Stitch(seed=seed).fit(xtr[idx], ytr[cap], x_pool=xtr, y_pool=ytr)
            m = retrieval_metrics(s.transform_x(xte), s.transform_y(yte), own_te)
            m.update(k=s.k_, fit_seconds=s.fit_seconds, seed=seed, n_pairs=n_eff)
            runs.append(m)
        out[str(n)] = runs
        print(img, txt, n_eff, round(np.mean([r["mR"] for r in runs]), 4), [r["k"] for r in runs], flush=True)
    return out


def evaluate_suite(load, own_tr, own_te, baselines, stitch_pairs, ns, exploratory_clip=None):
    first_te = first_caption_idx(own_te)
    res = {"baselines": {}, "stitched": {}, "zero_pair": {}, "exploratory": {},
           "n_test_images": int(own_te.max() + 1), "n_test_texts": int(len(own_te))}
    for name in baselines:
        img, txt = load(name, "test", "image"), load(name, "test", "text")
        m = retrieval_metrics(img, txt, own_te)
        m["mutual_10nn"] = mutual_knn(img, txt[first_te], 10)
        res["baselines"][name] = m
        print(name, round(m["mR"], 4), flush=True)
    for label, (i, t) in stitch_pairs.items():
        res["stitched"][label] = stitched_runs(load, i, t, own_tr, own_te, ns)

    i, t = stitch_pairs["primary"]
    xtr, ytr = load(i, "train", "image"), load(t, "train", "text")
    xte, yte = load(i, "test", "image"), load(t, "test", "text")
    half = len(np.bincount(own_tr)) // 2
    runs = []
    for seed in SEEDS:
        s = fit_unpaired(xtr[:half], ytr[own_tr >= half], k=64, n=1000, seed=seed, x_pool=xtr, y_pool=ytr)
        m = retrieval_metrics(s.transform_x(xte), s.transform_y(yte), own_te)
        m.update(k=s.k_, fit_seconds=s.fit_seconds, seed=seed)
        runs.append(m)
        print("zero-pair", seed, round(m["t2i_R@10"], 4), round(m["mR"], 4), flush=True)
    res["zero_pair"]["primary"] = runs
    if exploratory_clip:
        res["exploratory"][f"{exploratory_clip}+rotation"] = stitched_runs(
            load, exploratory_clip, exploratory_clip, own_tr, own_te, ns=(max(ns),))
    return res
