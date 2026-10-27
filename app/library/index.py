"""Vector indexes over library embeddings (paper Section 3: a FAISS store).

:class:`FaissIndex` (exact inner-product search, ``IndexFlatIP``) needs the optional
``faiss-cpu`` package; :class:`NumpyIndex` is an exact NumPy equivalent used as the fallback and
in tests. The index is rebuilt from the database at startup, so it never has to be persisted.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from app.core.logging import get_logger, log_event

logger = get_logger("library")


class VectorIndex(ABC):
    name: str

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._lock = threading.Lock()

    @abstractmethod
    def upsert(self, entry_id: str, vector: np.ndarray) -> None: ...

    @abstractmethod
    def remove(self, entry_id: str) -> None: ...

    @abstractmethod
    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        """``[(entry_id, inner product)]`` best first."""

    @abstractmethod
    def __len__(self) -> int: ...


class NumpyIndex(VectorIndex):
    name = "numpy"

    def __init__(self, dimension: int) -> None:
        super().__init__(dimension)
        self._ids: list[str] = []
        self._matrix = np.zeros((0, dimension), dtype=np.float32)

    def upsert(self, entry_id: str, vector: np.ndarray) -> None:
        with self._lock:
            row = np.asarray(vector, dtype=np.float32).reshape(1, self.dimension)
            if entry_id in self._ids:
                self._matrix[self._ids.index(entry_id)] = row
            else:
                self._ids.append(entry_id)
                self._matrix = np.vstack([self._matrix, row])

    def remove(self, entry_id: str) -> None:
        with self._lock:
            if entry_id in self._ids:
                index = self._ids.index(entry_id)
                self._ids.pop(index)
                self._matrix = np.delete(self._matrix, index, axis=0)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        with self._lock:
            if not self._ids:
                return []
            scores = self._matrix @ np.asarray(vector, dtype=np.float32).reshape(self.dimension)
            order = np.argsort(-scores, kind="stable")[:k]
            return [(self._ids[i], float(scores[i])) for i in order]

    def __len__(self) -> int:
        return len(self._ids)


class FaissIndex(VectorIndex):
    name = "faiss"

    def __init__(self, dimension: int) -> None:
        import faiss  # optional dependency

        super().__init__(dimension)
        self._faiss = faiss
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        self._to_int: dict[str, int] = {}
        self._to_str: dict[int, str] = {}
        self._next = 1

    def upsert(self, entry_id: str, vector: np.ndarray) -> None:
        with self._lock:
            if entry_id in self._to_int:
                self._index.remove_ids(np.array([self._to_int[entry_id]], dtype=np.int64))
                number = self._to_int[entry_id]
            else:
                number, self._next = self._next, self._next + 1
                self._to_int[entry_id], self._to_str[number] = number, entry_id
            row = np.asarray(vector, dtype=np.float32).reshape(1, self.dimension)
            self._index.add_with_ids(row, np.array([number], dtype=np.int64))

    def remove(self, entry_id: str) -> None:
        with self._lock:
            number = self._to_int.pop(entry_id, None)
            if number is not None:
                self._to_str.pop(number, None)
                self._index.remove_ids(np.array([number], dtype=np.int64))

    def search(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        with self._lock:
            if self._index.ntotal == 0:
                return []
            query = np.asarray(vector, dtype=np.float32).reshape(1, self.dimension)
            scores, numbers = self._index.search(query, min(k, int(self._index.ntotal)))
            return [(self._to_str[int(n)], float(s)) for s, n in zip(scores[0], numbers[0]) if int(n) in self._to_str]

    def __len__(self) -> int:
        return int(self._index.ntotal)


def build_index(kind: str, dimension: int) -> tuple[VectorIndex, Optional[str]]:
    """``(index, fallback_reason)``."""
    if kind == "faiss":
        try:
            return FaissIndex(dimension), None
        except ImportError:
            log_event(logger, "vector index fallback", level=logging.WARNING, operation="startup", stage="library",
                      outcome="fallback", error_category="ImportError")  # fmt: skip
            return NumpyIndex(dimension), "faiss-cpu is not installed; using the NumPy index"
    return NumpyIndex(dimension), None
