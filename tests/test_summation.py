import json
import subprocess
import sys

import numpy as np
import pytest

from summation import Stitch, mutual_knn, procrustes, retrieval_metrics


def _ortho(d, rng):
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return q


def _two_views(n, d=48, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, d)) * np.linspace(3, 0.3, d)
    x = z @ _ortho(d, rng) + noise * rng.normal(size=(n, d))
    y = z @ _ortho(d, rng) + noise * rng.normal(size=(n, d))
    return x, y


def test_procrustes_recovers_rotation():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 16))
    r = _ortho(16, rng)
    assert np.allclose(procrustes(x, x @ r), r, atol=1e-8)


def test_metrics_identity_and_mutual_knn():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(50, 8))
    m = retrieval_metrics(a, a, np.arange(50))
    assert m["t2i_R@1"] == 1.0 and m["i2t_R@1"] == 1.0 and m["mR"] == 1.0
    assert mutual_knn(a, a, 5) == 1.0
    assert mutual_knn(a, rng.normal(size=(50, 8)), 5) < 0.3


def test_i2t_counts_any_own_caption():
    img = np.eye(3)
    txt = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]], float)
    owner = np.array([0, 0, 1, 1])
    m = retrieval_metrics(img, txt, owner, ks=(1,))
    assert m["i2t_R@1"] == pytest.approx(2 / 3)


def test_stitch_few_pairs_generalises():
    x, y = _two_views(1300)
    s = Stitch(seed=0).fit(x[:100], y[:100], x_pool=x[:1000], y_pool=y[:1000])
    m = retrieval_metrics(s.transform_x(x[1000:]), s.transform_y(y[1000:]), np.arange(300))
    assert m["t2i_R@1"] > 0.5
    assert s.k_ in s.cv_scores


def test_save_load_roundtrip(tmp_path):
    x, y = _two_views(300)
    s = Stitch(k=16).fit(x, y)
    s.save(tmp_path / "m.npz")
    t = Stitch.load(tmp_path / "m.npz")
    assert np.allclose(s.transform_x(x), t.transform_x(x)) and np.allclose(s.transform_y(y), t.transform_y(y))


def test_unpaired_returns_rotation():
    pytest.importorskip("ot")
    from summation.unpaired import fit_unpaired

    x, y = _two_views(400, d=16)
    s = fit_unpaired(x, y[::-1], k=8, n=200)
    assert np.allclose(s.R @ s.R.T, np.eye(8), atol=1e-8)


def test_cli_fit_eval(tmp_path):
    x, y = _two_views(600)
    for name, a in [("x", x[:300]), ("y", y[:300]), ("xt", x[300:]), ("yt", y[300:])]:
        np.save(tmp_path / f"{name}.npy", a)
    run = lambda *a: subprocess.run([sys.executable, "-m", "summation.cli", *a], check=True, capture_output=True, text=True).stdout
    fit = json.loads(run("fit", "--x", str(tmp_path / "x.npy"), "--y", str(tmp_path / "y.npy"), "--out", str(tmp_path / "m.npz")))
    assert fit["pairs"] == 300
    ev = json.loads(run("eval", "--map", str(tmp_path / "m.npz"), "--x", str(tmp_path / "xt.npy"), "--y", str(tmp_path / "yt.npy")))
    assert ev["t2i_R@1"] > 0.5
