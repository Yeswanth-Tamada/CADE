"""Text embeddings of playbooks and incidents (paper Section 3: a sentence-transformer model).

:class:`SentenceTransformerEmbedder` is the paper's choice (``all-MiniLM-L6-v2`` by default). It
needs the optional ``sentence-transformers`` package and downloads the model on first use. When
that is not possible, :func:`build_embedder` falls back to :class:`HashingEmbedder`, a
deterministic bag-of-words feature-hashing embedder that needs only NumPy; the fallback is
logged and reported by ``/health``. Both return L2-normalised vectors, so the inner product is the
cosine similarity.
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from app.core.logging import get_logger, log_event

logger = get_logger("library")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class Embedder(ABC):
    name: str
    dimension: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """``(len(texts), dimension)`` float32 array of unit-length rows."""


class HashingEmbedder(Embedder):
    """Deterministic feature hashing of unigrams and bigrams (signed, sublinear counts)."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.name = f"hashing-{dimension}"

    def _features(self, text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[str, int] = {}
            for feature in self._features(text):
                counts[feature] = counts.get(feature, 0) + 1
            for feature, count in counts.items():
                digest = hashlib.sha256(feature.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, index] += sign * (1.0 + np.log(count))
        return _normalise(matrix)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency

        self._model = SentenceTransformer(model_name, device="cpu")
        # sentence-transformers 5.x renamed the accessor; support both
        dimension = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dimension = int(dimension())
        self.name = f"st:{model_name}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32).reshape(len(texts), self.dimension)


def build_embedder(kind: str, model_name: str) -> tuple[Embedder, Optional[str]]:
    """``(embedder, fallback_reason)``; the reason is None when the requested embedder loaded."""
    if kind == "sentence_transformers":
        try:
            return SentenceTransformerEmbedder(model_name), None
        except Exception as exc:  # noqa: BLE001 - missing package, no network for the first download, ...
            reason = f"sentence-transformers unavailable ({type(exc).__name__}); using the hashing embedder"
            log_event(logger, "embedder fallback", level=logging.WARNING, operation="startup", stage="library",
                      outcome="fallback", error_category=type(exc).__name__)  # fmt: skip
            return HashingEmbedder(), reason
    return HashingEmbedder(), None
