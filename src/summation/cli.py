"""summation CLI: embed, fit, search, eval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _sidecar(npy: str | Path) -> Path:
    return Path(npy).with_suffix(".json")


def _read_sidecar(npy) -> dict:
    p = _sidecar(npy)
    return json.loads(p.read_text()) if p.exists() else {}


def _list_images(src: str) -> list[str]:
    p = Path(src)
    if p.is_dir():
        return sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXT)
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def cmd_embed(a):
    from .encoders import load_encoder

    enc = load_encoder(a.encoder)
    if a.images:
        ids, modality = _list_images(a.images), "image"
    else:
        ids, modality = [t for t in Path(a.texts).read_text().splitlines() if t.strip()], "text"
    emb = enc.encode(ids, modality)
    np.save(a.out, emb.astype(np.float32))
    _sidecar(a.out).write_text(json.dumps({"encoder": a.encoder, "modality": modality, "ids": ids}))
    print(f"{len(ids)} {modality}s -> {a.out} {emb.shape}")


def cmd_fit(a):
    from .align import Stitch

    x, y = np.load(a.x), np.load(a.y)
    sx, sy = _read_sidecar(a.x), _read_sidecar(a.y)
    meta = {
        "x_encoder": a.x_encoder or sx.get("encoder"), "x_modality": sx.get("modality"),
        "y_encoder": a.y_encoder or sy.get("encoder"), "y_modality": sy.get("modality"),
    }
    xp = np.load(a.x_pool) if a.x_pool else None
    yp = np.load(a.y_pool) if a.y_pool else None
    if a.unpaired:
        from .unpaired import fit_unpaired

        s = fit_unpaired(x, y, k=int(a.k) if a.k != "auto" else 64, n=a.n, x_pool=xp, y_pool=yp)
    else:
        s = Stitch(k=a.k if a.k == "auto" else int(a.k)).fit(x, y, xp, yp)
    s.meta.update(meta)
    s.save(a.out)
    print(json.dumps({"k": s.k_, "pairs": s.meta["n_pairs"], "fit_seconds": round(s.fit_seconds, 3),
                      "cv_scores": s.cv_scores, "out": a.out}))


def cmd_search(a):
    from .align import Stitch
    from .encoders import load_encoder

    s = Stitch.load(a.map)
    index = np.load(a.index)
    side = a.index_side or _infer_side(s.meta, _read_sidecar(a.index).get("encoder"))
    qside = "y" if side == "x" else "x"
    spec, modality = s.meta.get(f"{qside}_encoder"), s.meta.get(f"{qside}_modality") or "text"
    if not spec:
        raise SystemExit(f"map has no {qside}_encoder; pass --x-encoder/--y-encoder at fit time")
    q = load_encoder(spec).encode([a.query], modality)
    tq = s.transform_x(q) if qside == "x" else s.transform_y(q)
    ti = s.transform_x(index) if side == "x" else s.transform_y(index)
    scores = (ti @ tq.T).ravel()
    ids = _read_sidecar(a.index).get("ids") or [str(i) for i in range(len(index))]
    for rank, i in enumerate(np.argsort(-scores)[: a.k], 1):
        print(f"{rank}\t{scores[i]:.4f}\t{ids[i]}")


def _infer_side(meta, index_encoder):
    if index_encoder and index_encoder == meta.get("y_encoder") and index_encoder != meta.get("x_encoder"):
        return "y"
    return "x"


def cmd_eval(a):
    from .align import Stitch
    from .metrics import mutual_knn, retrieval_metrics

    s = Stitch.load(a.map)
    x, y = np.load(a.x), np.load(a.y)
    y2x = np.load(a.y2x) if a.y2x else np.arange(len(y))
    out = retrieval_metrics(s.transform_x(x), s.transform_y(y), y2x)
    if len(x) == len(y):
        out["mutual_10nn_native"] = mutual_knn(x, y, 10)
    print(json.dumps(out, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(prog="summation", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed", help="embed images or texts with a frozen encoder")
    e.add_argument("--encoder", required=True, help="hf-image:<id> | st:<id> | clip:<arch>:<pretrained>")
    g = e.add_mutually_exclusive_group(required=True)
    g.add_argument("--images", help="directory or file with one image path per line")
    g.add_argument("--texts", help="file with one text per line")
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_embed)

    f = sub.add_parser("fit", help="fit one rotation x -> y")
    f.add_argument("--x", required=True, help=".npy, row-aligned with --y unless --unpaired")
    f.add_argument("--y", required=True)
    f.add_argument("--x-pool", help="optional unpaired x corpus for centring/PCA")
    f.add_argument("--y-pool")
    f.add_argument("--k", default="auto")
    f.add_argument("--unpaired", action="store_true", help="no pairs: GW + Wasserstein-Procrustes")
    f.add_argument("--n", type=int, default=1000, help="points per side for --unpaired")
    f.add_argument("--x-encoder")
    f.add_argument("--y-encoder")
    f.add_argument("--out", required=True)
    f.set_defaults(func=cmd_fit)

    s = sub.add_parser("search", help="query an embedded index from the other modality")
    s.add_argument("--map", required=True)
    s.add_argument("--index", required=True, help=".npy from `embed`")
    s.add_argument("--index-side", choices=["x", "y"])
    s.add_argument("--query", required=True, help="text, or an image path")
    s.add_argument("-k", type=int, default=5)
    s.set_defaults(func=cmd_search)

    v = sub.add_parser("eval", help="retrieval metrics on held-out pairs")
    v.add_argument("--map", required=True)
    v.add_argument("--x", required=True)
    v.add_argument("--y", required=True)
    v.add_argument("--y2x", help=".npy int array: row of x each y belongs to (default: aligned)")
    v.set_defaults(func=cmd_eval)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
