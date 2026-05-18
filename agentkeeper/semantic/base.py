"""Embedding provider contract.

A provider knows how to turn a list of strings into a list of dense
vectors of fixed dimension. Implementations are interchangeable:
sentence-transformers (default, local), OpenAI (API), Voyage AI (API),
or a deterministic mock for tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingProvider(ABC):
    """Minimal contract for any embedding backend."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension produced by this provider."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the model used, for diagnostics and storage."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: Non-empty iterable of strings.

        Returns:
            One vector per text, in the same order. Each vector has
            length `self.dimension`.
        """

    def embed_one(self, text: str) -> list[float]:
        """Convenience: embed a single text."""
        return self.embed([text])[0]
