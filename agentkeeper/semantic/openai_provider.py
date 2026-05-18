"""OpenAI embedding provider (cloud, opt-in)."""

from __future__ import annotations

from collections.abc import Sequence

from .base import EmbeddingProvider

_OPENAI_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings via the official SDK."""

    DEFAULT_MODEL = "text-embedding-3-small"

    def __init__(self, api_key: str, model_name: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "openai is required for OpenAI embeddings. "
                "Install with: pip install 'agentkeeper[openai]'"
            ) from exc

        self._client = OpenAI(api_key=api_key)
        self._name = model_name or self.DEFAULT_MODEL
        self._dim = _OPENAI_DIMENSIONS.get(self._name, 1536)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._name

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self._name, input=list(texts)
        )
        return [list(d.embedding) for d in response.data]
