"""Deterministic mock embedding provider.

Produces stable, low-dimensional vectors derived from a hash of each
token. Vectors are L2-normalised so cosine similarity is well defined.

Useful for tests and offline development. Two semantically similar
sentences will get *different* vectors (no real semantic structure),
but two *identical* strings always get the same vector, which is
enough for round-trip and storage tests.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from .base import EmbeddingProvider

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


class MockEmbeddingProvider(EmbeddingProvider):
    """Hash-bag deterministic embeddings. Test-only."""

    def __init__(self, dimension: int = 64, model_name: str = "mock-v1") -> None:
        if dimension < 4:
            raise ValueError("dimension must be >= 4")
        self._dimension = dimension
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_single(text) for text in texts]

    def _embed_single(self, text: str) -> list[float]:
        vec = [0.0] * self._dimension
        tokens = _TOKEN_PATTERN.findall(text.lower())
        if not tokens:
            tokens = [text.strip().lower() or "_empty_"]

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            for i, byte in enumerate(digest):
                idx = i % self._dimension
                # Map byte (0..255) to (-1, 1) range
                vec[idx] += (byte / 127.5) - 1.0

        # L2-normalise
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]
