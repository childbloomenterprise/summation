"""DROID robot-frame demo (PREREG_ROBOT.md).

  python demo/droid.py frames     # decode middle exterior_1_left frame per annotated episode
  python demo/droid.py extract    # embed frames + instructions
  python demo/droid.py evaluate   # -> demo/results_droid.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CACHE = HERE / "cache"
DATA = Path.home() / "data" / "droid"
CAM = "videos/observation.images.exterior_1_left/"
NS = (100, 300, 1000)
IMAGE = {"dinov2-base": "hf-image:facebook/dinov2-base", "dinov2-small": "hf-image:facebook/dinov2-small"}
TEXT = {"mpnet": "st:sentence-transformers/all-mpnet-base-v2", "minilm": "st:sentence-transformers/all-MiniLM-L6-v2"}
CLIPS = {"clip-b32": "clip:ViT-B-32-quickgelu:openai", "clip-l14": "clip:ViT-L-14-quickgelu:openai",
         "siglip-b16": "clip:ViT-B-16-SigLIP:webli"}
TRAIN_TOO = {"dinov2-base", "dinov2-small", "mpnet", "minilm", "clip-b32"}
PAIRS = {"primary": ("dinov2-base", "mpnet"), "secondary": ("dinov2-small", "minilm")}


def episodes():
    import pyarrow.parquet as pq

    meta = pq.read_table(DATA / "meta" / "ep000.parquet",
                         columns=["episode_index", "data/file_index", CAM + "chunk_index", CAM + "file_index",
                                  CAM + "from_timestamp", CAM + "to_timestamp"]).to_pydict()
    instr = {}
    for f in range(3):
        t = pq.read_table(DATA / "data" / f"file-{f:03d}.parquet",
                          columns=["episode_index", "frame_index", "language_instruction",
                                   "language_instruction_2", "language_instruction_3"]).to_pydict()
        for i, fr, *ls in zip(t["episode_index"], t["frame_index"], t["language_instruction"],
                              t["language_instruction_2"], t["language_instruction_3"]):
            if fr == 0:
                instr[i] = [s.strip() for s in ls if s and s.strip() and s.strip() != "No action"]
    eps = []
    for j, ep in enumerate(meta["episode_index"]):
        d = meta["data/file_index"][j]
        if d <= 2 and instr.get(ep):
            eps.append(dict(ep=ep, split="test" if d == 2 else "train", texts=instr[ep],
                            video=(meta[CAM + "chunk_index"][j], meta[CAM + "file_index"][j]),
                            t=(meta[CAM + "from_timestamp"][j] + meta[CAM + "to_timestamp"][j]) / 2))
    return eps


def frames():
    import av

    eps = episodes()
    CACHE.mkdir(exist_ok=True)
    out = {}
    by_video = {}
    for e in eps:
        by_video.setdefault(e["video"], []).append(e)
    for (ch, f), es in sorted(by_video.items()):
        with av.open(str(DATA / "video" / f"{ch:03d}_file-{f:03d}.mp4")) as c:
            s = c.streams.video[0]
            for e in sorted(es, key=lambda e: e["t"]):
                c.seek(int(e["t"] / s.time_base), stream=s, backward=True)
                for fr in c.decode(s):
                    if fr.time is not None and fr.time >= e["t"] - 1e-3:
                        out[e["ep"]] = fr.to_ndarray(format="rgb24")
                        break
        print("video", ch, f, len(es), flush=True)
    for split in ("train", "test"):
        es = [e for e in eps if e["split"] == split and e["ep"] in out]
        np.save(CACHE / f"droid_frames_{split}.npy", np.stack([out[e["ep"]] for e in es]))
        (CACHE / f"droid_meta_{split}.json").write_text(json.dumps([dict(ep=e["ep"], texts=e["texts"]) for e in es]))
        np.save(CACHE / f"droid_owner_{split}.npy", np.repeat(np.arange(len(es)), [len(e["texts"]) for e in es]))
        print(split, len(es), "episodes", sum(len(e["texts"]) for e in es), "texts")


def cache_path(name, split, modality):
    return CACHE / f"droid_{name}_{split}_{modality}.npy"


def load(name, split, modality):
    return np.load(cache_path(name, split, modality))


def extract(names=None):
    import torch
    from PIL import Image

    from summation.encoders import load_encoder

    torch.set_num_threads(int(os.environ.get("THREADS", "8")))
    jobs = [(n, s, "image") for n, s in {**IMAGE, **CLIPS}.items()] + [(n, s, "text") for n, s in {**TEXT, **CLIPS}.items()]
    for name, spec, modality in jobs:
        if names and name not in names:
            continue
        enc = None
        for split in ("test", "train") if name in TRAIN_TOO else ("test",):
            out = cache_path(name, split, modality)
            if out.exists():
                continue
            enc = enc or load_encoder(spec)
            t0 = time.perf_counter()
            if modality == "image":
                items = [Image.fromarray(a) for a in np.load(CACHE / f"droid_frames_{split}.npy")]
            else:
                items = [t for e in json.loads((CACHE / f"droid_meta_{split}.json").read_text()) for t in e["texts"]]
            emb = enc.encode(items, modality)
            np.save(out, emb.astype(np.float32))
            print(name, split, modality, emb.shape, f"{time.perf_counter() - t0:.0f}s", flush=True)


def evaluate():
    from common import evaluate_suite

    own_tr, own_te = np.load(CACHE / "droid_owner_train.npy"), np.load(CACHE / "droid_owner_test.npy")
    res = evaluate_suite(load, own_tr, own_te, list(CLIPS), PAIRS, NS, exploratory_clip="clip-b32")
    (HERE / "results_droid.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["frames", "extract", "evaluate"])
    p.add_argument("--names", nargs="*")
    a = p.parse_args()
    {"frames": frames, "extract": lambda: extract(a.names), "evaluate": evaluate}[a.cmd]()
