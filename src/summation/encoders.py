"""Frozen encoders addressed by a spec string.

  hf-image:<model_id>        HF vision backbone, pooled/CLS output (e.g. hf-image:facebook/dinov2-base)
  st:<model_id>              sentence-transformers text model (e.g. st:sentence-transformers/all-mpnet-base-v2)
  clip:<arch>:<pretrained>   open_clip image+text model; pretrained may be a tag or a checkpoint path/URL
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


class Encoder:
    spec: str
    modalities: tuple[str, ...]

    def encode_images(self, images: Sequence, batch_size: int = 32) -> np.ndarray:
        raise ValueError(f"{self.spec} does not encode images")

    def encode_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        raise ValueError(f"{self.spec} does not encode texts")

    def encode(self, items: Sequence, modality: str) -> np.ndarray:
        return self.encode_images(items) if modality == "image" else self.encode_texts(items)


def _open_image(x):
    from PIL import Image

    if isinstance(x, (str, Path)):
        return Image.open(x).convert("RGB")
    return x.convert("RGB")


class HFImageEncoder(Encoder):
    modalities = ("image",)

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.spec = f"hf-image:{model_id}"
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval()

    def encode_images(self, images, batch_size=32):
        out = []
        for i in range(0, len(images), batch_size):
            batch = [_open_image(x) for x in images[i : i + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt")
            with self.torch.inference_mode():
                o = self.model(**inputs)
            pooled = getattr(o, "pooler_output", None)
            if pooled is None:
                pooled = o.last_hidden_state[:, 0]
            out.append(pooled.float().numpy())
        return np.concatenate(out)


class STEncoder(Encoder):
    modalities = ("text",)

    def __init__(self, model_id: str):
        from sentence_transformers import SentenceTransformer

        self.spec = f"st:{model_id}"
        self.model = SentenceTransformer(model_id, device="cpu")

    def encode_texts(self, texts, batch_size=64):
        return self.model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True).astype(np.float32)


class OpenCLIPEncoder(Encoder):
    modalities = ("image", "text")

    def __init__(self, arch: str, pretrained: str):
        import open_clip
        import torch

        self.spec = f"clip:{arch}:{pretrained}"
        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(arch)

    def encode_images(self, images, batch_size=32):
        out = []
        for i in range(0, len(images), batch_size):
            x = self.torch.stack([self.preprocess(_open_image(im)) for im in images[i : i + batch_size]])
            with self.torch.inference_mode():
                out.append(self.model.encode_image(x).float().numpy())
        return np.concatenate(out)

    def encode_texts(self, texts, batch_size=64):
        out = []
        texts = list(texts)
        for i in range(0, len(texts), batch_size):
            tok = self.tokenizer(texts[i : i + batch_size])
            with self.torch.inference_mode():
                out.append(self.model.encode_text(tok).float().numpy())
        return np.concatenate(out)


def load_encoder(spec: str) -> Encoder:
    kind, _, rest = spec.partition(":")
    if kind == "hf-image":
        return HFImageEncoder(rest)
    if kind == "st":
        return STEncoder(rest)
    if kind == "clip":
        arch, _, pretrained = rest.partition(":")
        return OpenCLIPEncoder(arch, pretrained)
    raise ValueError(f"unknown encoder spec {spec!r}; expected hf-image:, st: or clip:")


def modality_of(spec: str, side_hint: str | None = None) -> str:
    kind = spec.partition(":")[0]
    if kind == "hf-image":
        return "image"
    if kind == "st":
        return "text"
    if side_hint is None:
        raise ValueError(f"{spec} encodes both modalities; specify which")
    return side_hint
