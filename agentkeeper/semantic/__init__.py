"""Semantic recall: embeddings, vector indexing, meaning-based retrieval."""

from .base import EmbeddingProvider
from .factory import make_vector_index
from .index import InMemoryVectorIndex, VectorIndex
from .mock import MockEmbeddingProvider
from .recaller import SemanticRecaller
from .sqlite_vec_index import SqliteVecIndex
from .sqlite_vec_index import is_available as sqlite_vec_available

__all__ = [
    "EmbeddingProvider",
    "InMemoryVectorIndex",
    "MockEmbeddingProvider",
    "SemanticRecaller",
    "SqliteVecIndex",
    "VectorIndex",
    "make_vector_index",
    "sqlite_vec_available",
]
