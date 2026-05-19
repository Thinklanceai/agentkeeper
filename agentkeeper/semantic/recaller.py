"""Semantic recall: meaning-based fact retrieval.

The `SemanticRecaller` is the user-facing object behind
`agent.recall(query, top_k=5)`. It owns:

- an embedding provider (sentence-transformers by default)
- a vector index keyed by fact_id (in-memory or sqlite-vec)

Indexing is lazy: facts are embedded on first recall, then cached on
the Fact's `metadata['embedding_hash']` to skip re-embedding identical
content. Embeddings themselves live in the index, not on the Fact —
this keeps the CSO JSON small.

Index selection is automatic via `make_vector_index()`:

- `AGENTKEEPER_VECTOR_INDEX=auto` (default): sqlite-vec when installed,
  in-memory otherwise.
- `AGENTKEEPER_VECTOR_INDEX=sqlite_vec`: force persistence.
- `AGENTKEEPER_VECTOR_INDEX=in_memory`: force in-memory.

With sqlite-vec, the index survives restarts. The recaller still
re-validates content hashes against `_indexed_hashes` so that a
content edit triggers a re-embed automatically.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from .base import EmbeddingProvider
from .factory import make_vector_index
from .index import VectorIndex

if TYPE_CHECKING:
    from ..cso.types import CognitiveStateObject, Fact


def _content_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


class SemanticRecaller:
    """Embedding-based recall over an agent's CSO."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        cso: CognitiveStateObject,
        index: VectorIndex | None = None,
    ) -> None:
        self._provider = provider
        self._cso = cso
        # Explicit index wins (tests, custom backends). Otherwise the
        # factory consults env vars and availability.
        self._index = index or make_vector_index(
            agent_id=cso.agent_id,
            dimension=provider.dimension,
        )
        self._indexed_hashes: dict[str, str] = {}  # fact_id -> content_hash

    @property
    def provider(self) -> EmbeddingProvider:
        return self._provider

    @property
    def index_size(self) -> int:
        return self._index.size()

    # --- indexing ---------------------------------------------------

    def index_all(self, batch_size: int = 32) -> int:
        """Embed and index every fact not yet indexed (or whose content
        has changed). Returns the number of facts indexed in this call.
        """
        to_embed: list[tuple[str, str]] = []  # (fact_id, content)

        for fact in self._cso.memory_facts:
            current_hash = _content_hash(fact.content)
            previous_hash = self._indexed_hashes.get(fact.id)
            if previous_hash == current_hash:
                continue
            to_embed.append((fact.id, fact.content))

        if not to_embed:
            return 0

        indexed = 0
        for start in range(0, len(to_embed), batch_size):
            batch = to_embed[start : start + batch_size]
            texts = [content for _, content in batch]
            vectors = self._provider.embed(texts)
            for (fact_id, content), vector in zip(batch, vectors, strict=True):
                self._index.upsert(fact_id, vector)
                self._indexed_hashes[fact_id] = _content_hash(content)
                indexed += 1

        return indexed

    def remove(self, fact_id: str) -> None:
        self._index.delete(fact_id)
        self._indexed_hashes.pop(fact_id, None)

    # --- recall -----------------------------------------------------

    def recall(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[Fact, float]]:
        """Return up to `top_k` facts most semantically similar to `query`.

        Pairs are sorted by descending cosine similarity. Pairs whose
        score falls below `min_score` are filtered out.
        """
        if not query.strip() or top_k <= 0:
            return []

        # Ensure all facts are indexed (cheap if already done).
        self.index_all()

        query_vec = self._provider.embed_one(query)
        hits = self._index.search(query_vec, top_k=top_k, min_score=min_score)

        # Resolve fact_id -> Fact
        by_id = {f.id: f for f in self._cso.memory_facts}
        results: list[tuple[Fact, float]] = []
        for fact_id, score in hits:
            fact = by_id.get(fact_id)
            if fact is None:
                # Fact was deleted between indexing and recall — drop it
                self._index.delete(fact_id)
                continue
            results.append((fact, score))
        return results
