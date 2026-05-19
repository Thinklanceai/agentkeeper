"""Local sentence-transformers embedding provider.

Loads a small, high-quality model on first use. Default model is
`all-MiniLM-L6-v2` (384 dims, ~80MB), the de-facto standard for
local embeddings: fast on CPU, good quality, zero API cost,
no lock-in.

Install via the [semantic] extra:
    pip install 'agentkeeper[semantic]'
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import EmbeddingProvider


class SentenceTransformerProvider(EmbeddingProvider):
    """Local embeddings via sentence-transformers."""

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install 'agentkeeper[semantic]'"
            ) from exc

        self._name = model_name or self.DEFAULT_MODEL
        self._model: Any = SentenceTransformer(self._name)

    @property
    def dimension(self) -> int:
        # SentenceTransformer exposes get_sentence_embedding_dimension()
        return int(self._model.get_sentence_embedding_dimension())

    @property
    def model_name(self) -> str:
        return self._name

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        # convert_to_numpy=False keeps it list-of-lists, normalized for cosine
        result = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=False,
            show_progress_bar=False,
        )
        return [list(map(float, vec)) for vec in result]
