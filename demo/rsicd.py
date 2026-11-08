"""RSICD demo: embed with frozen encoders, then run the pre-registered evaluation (PREREG.md).

  python demo/rsicd.py extract --only images|texts   # cache embeddings in demo/cache
  python demo/rsicd.py evaluate                       # -> demo/results.json
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CACHE = HERE / "cache"
DATA = Path.home() / "data" / "rsicd"
NS = (100, 300, 1000)

STITCH_IMAGE = {"dinov2-base": "hf-image:facebook/dinov2-base", "dinov2-small": "hf-image:facebook/dinov2-small"}
STITCH_TEXT = {"mpnet": "st:sentence-transformers/all-mpnet-base-v2", "minilm": "st:sentence-transformers/all-MiniLM-L6-v2"}
STITCH_PAIRS = {"primary": ("dinov2-base", "mpnet"), "secondary": ("dinov2-small", "minilm")}
CLIPS = {"clip-b32": "clip:ViT-B-32-quickgelu:openai", "clip-l14": "clip:ViT-L-14-quickgelu:openai",
         "siglip-b16": "clip:ViT-B-16-SigLIP:webli", "remoteclip-b32": None}
TRAIN_TOO = {"dinov2-base", "dinov2-small", "mpnet", "minilm", "clip-b32"}


def remoteclip_spec():
    from huggingface_hub import hf_hub_download

    return "clip:ViT-B-32:" + hf_hub_download("chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt")


def load_split(split):
    import pyarrow.parquet as pq

    t = pq.read_table(DATA / f"{split}.parquet").to_pydict()
    caps = [c for cs in t["captions"] for c in cs]
    owner = np.repeat(np.arange(len(t["captions"])), [len(cs) for cs in t["captions"]])
    return t["image"], caps, owner


def images_of(raw):
    from PIL import Image

    return [Image.open(io.BytesIO(r["bytes"])).convert("RGB") for r in raw]


def cache_path(name, split, modality):
    return CACHE / f"{name}_{split}_{modality}.npy"


def extract(only, names=None):
    import torch

    from summation.encoders import load_encoder

    torch.set_num_threads(int(__import__("os").environ.get("THREADS", "8")))
    CACHE.mkdir(exist_ok=True)
    jobs = []
    if only in (None, "images"):
        jobs += [(n, s, "image") for n, s in STITCH_IMAGE.items()] + [(n, s, "image") for n, s in CLIPS.items()]
    if only in (None, "texts"):
        jobs += [(n, s, "text") for n, s in STITCH_TEXT.items()] + [(n, s, "text") for n, s in CLIPS.items()]
    if names:
        jobs = [j for j in jobs if j[0] in names]
    splits = {}
    for name, spec, modality in jobs:
        spec = spec or remoteclip_spec()
        enc = None
        for split in ("test", "train") if name in TRAIN_TOO else ("test",):
            out = cache_path(name, split, modality)
            if out.exists():
                continue
            if split not in splits:
                raw, caps, owner = load_split(split)
                splits[split] = (raw, caps, owner)
                np.save(CACHE / f"owner_{split}.npy", owner)
            raw, caps, _ = splits[split]
            enc = enc or load_encoder(spec)
            t0 = time.perf_counter()
            items = images_of(raw) if modality == "image" else caps
            emb = enc.encode(items, modality)
            np.save(out, emb.astype(np.float32))
            print(f"{name} {split} {modality} {emb.shape} {time.perf_counter() - t0:.0f}s", flush=True)


def load(name, split, modality):
    return np.load(cache_path(name, split, modality))


def evaluate():
    from common import evaluate_suite

    own_te, own_tr = np.load(CACHE / "owner_test.npy"), np.load(CACHE / "owner_train.npy")
    res = evaluate_suite(load, own_tr, own_te, list(CLIPS), STITCH_PAIRS, NS, exploratory_clip="clip-b32")
    (HERE / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["extract", "evaluate"])
    p.add_argument("--only", choices=["images", "texts"])
    p.add_argument("--names", nargs="*", help="restrict extraction to these encoder names")
    a = p.parse_args()
    extract(a.only, a.names) if a.cmd == "extract" else evaluate()
