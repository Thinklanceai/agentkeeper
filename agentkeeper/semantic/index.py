"""Vector index for semantic recall.

A vector index maps `fact_id -> embedding` and supports cosine
similarity search. Two implementations are available:

- `InMemoryVectorIndex` (default): pure Python, brute-force cosine,
  no external dependency. Sufficient up to ~10k facts per agent.
- `SqliteVecIndex`: persisted via the `sqlite-vec` extension when
  available. Survives restarts and scales further. Falls back to
  the in-memory index if `sqlite-vec` is not installed.

Embeddings are stored as L2-normalised float lists; cosine similarity
reduces to a dot product. Results are returned as `(fact_id, score)`
tuples sorted by score descending.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .._fastmath import batch_dot
from .._fastmath import dot as _dot  # noqa: F401  (re-exported for compression.contradiction)


class VectorIndex(ABC):
    """Abstract vector index."""

    @abstractmethod
    def upsert(self, fact_id: str, vector: Sequence[float]) -> None:
        """Insert or update a vector for a fact."""

    @abstractmethod
    def delete(self, fact_id: str) -> None:
        """Remove a vector. No-op if missing."""

    @abstractmethod
    def search(
        self,
        query: Sequence[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Return up to `top_k` `(fact_id, score)` pairs, sorted desc.

        Scores below `min_score` are filtered out.
        """

    @abstractmethod
    def size(self) -> int:
        """Number of vectors in the index."""


class InMemoryVectorIndex(VectorIndex):
    """Simple dict-backed cosine index.

    Vectors are expected to be L2-normalised — cosine similarity then
    reduces to a dot product, which is what we compute. The
    `SemanticRecaller` ensures normalisation upstream.
    """

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._vectors: dict[str, list[float]] = {}

    def upsert(self, fact_id: str, vector: Sequence[float]) -> None:
        if len(vector) != self._dimension:
            raise ValueError(
                f"Vector dim {len(vector)} != index dim {self._dimension}"
            )
        self._vectors[fact_id] = list(vector)

    def delete(self, fact_id: str) -> None:
        self._vectors.pop(fact_id, None)

    def search(
        self,
        query: Sequence[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[str, float]]:
        if top_k <= 0 or not self._vectors:
            return []
        fact_ids = list(self._vectors.keys())
        matrix = [self._vectors[fid] for fid in fact_ids]
        scores = batch_dot(query, matrix)
        scored = [
            (fid, score)
            for fid, score in zip(fact_ids, scores, strict=True)
            if score >= min_score
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def size(self) -> int:
        return len(self._vectors)

    def clear(self) -> None:
        self._vectors.clear()
