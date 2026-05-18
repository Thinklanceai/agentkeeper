"""Semantic recall: embeddings, vector indexing, meaning-based retrieval."""

from .base import EmbeddingProvider
from .index import InMemoryVectorIndex, VectorIndex
from .mock import MockEmbeddingProvider
from .recaller import SemanticRecaller

__all__ = [
    "EmbeddingProvider",
    "InMemoryVectorIndex",
    "MockEmbeddingProvider",
    "SemanticRecaller",
    "VectorIndex",
]
