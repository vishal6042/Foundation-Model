"""Frozen text encoders for entity descriptions and captions.

`HashingTextEncoder` is dependency-free and deterministic (used for tests and quick ablations).
`SentenceTransformerEncoder` is the real one (DomusFM used all-MiniLM-L6-v2).
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class TextEncoder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingTextEncoder:
    """Bag of word + character-trigram features hashed into `dim` buckets, L2-normalised.

    Shares features between related strings ("kitchen motion" vs "kitchen door") but has no world
    knowledge (stove ~ oven). Swap for SentenceTransformerEncoder for real experiments.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self._cache: dict[str, np.ndarray] = {}

    def _bucket(self, feat: str) -> tuple[int, float]:
        h = hashlib.md5(feat.encode()).digest()
        return int.from_bytes(h[:4], "little") % self.dim, 1.0 if h[4] & 1 else -1.0

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = re.findall(r"[a-z0-9]+", text.lower())
        feats = [f"w:{w}" for w in words]
        for w in words:
            padded = f"#{w}#"
            feats += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
        for f in feats:
            idx, sign = self._bucket(f)
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            if t not in self._cache:
                self._cache[t] = self._encode_one(t)
            out.append(self._cache[t])
        return np.stack(out) if out else np.zeros((0, self.dim), dtype=np.float32)


class SentenceTransformerEncoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dependency: pip install .[text]

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            embs = self._model.encode(missing, normalize_embeddings=True, convert_to_numpy=True)
            self._cache.update(zip(missing, embs.astype(np.float32)))
        return np.stack([self._cache[t] for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


def build_text_encoder(cfg: dict) -> TextEncoder:
    kind = cfg.get("kind", "hashing")
    if kind == "hashing":
        return HashingTextEncoder(cfg.get("dim", 256))
    if kind == "sentence_transformer":
        return SentenceTransformerEncoder(cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
    raise ValueError(f"unknown text encoder: {kind}")
