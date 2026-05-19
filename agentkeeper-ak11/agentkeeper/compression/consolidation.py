"""Semantic consolidation — collapse near-duplicate facts.

Over time agents accumulate redundant facts: paraphrases of the same
information, slightly-updated values, duplicate notes. The consolidation
step:

1. Finds clusters of facts whose embeddings are within a similarity
   threshold of each other (default cosine 0.85).
2. Keeps the canonical fact of each cluster (most important + most
   recent) and either:
   - drops the rest (algorithmic mode), or
   - replaces the cluster with an LLM-synthesised summary (LLM mode,
     opt-in).

Identity-bound facts (importance >= 0.9) are protected: clusters that
contain a critical fact keep the critical fact as canonical, and only
the lower-importance siblings are dropped.

The LLM synthesiser is **optional**. Without it, consolidation is
purely algorithmic and free. With it, the user's own provider is used
(no hidden third-party calls).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..cso.types import Fact
from ..semantic.base import EmbeddingProvider
from ..semantic.index import _dot


@dataclass
class ConsolidationConfig:
    similarity_threshold: float = 0.85
    # Only consolidate facts from these tiers. Identity-injected items
    # (handled by the CRE separately) are not in `memory_facts`.
    target_tiers: tuple[str, ...] = ("semantic", "episodic", "working")


@dataclass
class ConsolidationResult:
    clusters_found: int = 0
    facts_removed: int = 0
    facts_synthesised: int = 0
    removed_ids: list[str] = field(default_factory=list)
    added_ids: list[str] = field(default_factory=list)


# A synthesiser turns a list of cluster facts into a single consolidated
# content string. The default returns the canonical fact's content
# unchanged (no LLM call needed).
Synthesiser = Callable[[list[Fact]], str]


def _canonical_content(cluster: list[Fact]) -> str:
    """Default synthesiser: pick the most important and most recent fact."""
    canonical = max(
        cluster,
        key=lambda f: (f.importance, f.last_accessed_at),
    )
    return canonical.content


def consolidate(
    facts: list[Fact],
    provider: EmbeddingProvider,
    config: ConsolidationConfig | None = None,
    synthesiser: Synthesiser | None = None,
) -> ConsolidationResult:
    """Cluster near-duplicate facts and consolidate them in place.

    The `facts` list is **mutated**: cluster members beyond the canonical
    one are removed, and the canonical fact's content may be updated by
    the synthesiser. Critical facts are kept as-is — only their
    non-critical clones are dropped.

    Returns a `ConsolidationResult` summarising what happened.
    """
    config = config or ConsolidationConfig()
    synthesiser = synthesiser or _canonical_content
    result = ConsolidationResult()

    # Only consider facts in target tiers AND not protected (principles,
    # hard constraints). Protected facts are exempt from any compression.
    targets = [
        f for f in facts
        if f.tier.value in config.target_tiers and not f.protected
    ]
    if len(targets) < 2:
        return result

    # Embed all targets in one batch
    vectors = provider.embed([f.content for f in targets])
    fact_vectors: dict[str, list[float]] = {
        f.id: vec for f, vec in zip(targets, vectors, strict=True)
    }

    # Greedy clustering: walk facts, group each into the first cluster
    # whose centroid is within the threshold; else start a new cluster.
    clusters: list[list[Fact]] = []
    centroids: list[list[float]] = []

    for fact in targets:
        vec = fact_vectors[fact.id]
        assigned = False
        for i, centroid in enumerate(centroids):
            if _dot(vec, centroid) >= config.similarity_threshold:
                clusters[i].append(fact)
                # update centroid as running mean (re-normalise lazily)
                cluster_vecs = [fact_vectors[m.id] for m in clusters[i]]
                centroids[i] = _mean_vector(cluster_vecs)
                assigned = True
                break
        if not assigned:
            clusters.append([fact])
            centroids.append(vec)

    # Process non-singleton clusters
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        result.clusters_found += 1

        # Pick canonical: highest importance, then most recently accessed
        canonical = max(
            cluster,
            key=lambda f: (f.importance, f.last_accessed_at),
        )

        # Generate consolidated content
        new_content = synthesiser(cluster)
        if new_content != canonical.content:
            canonical.content = new_content
            # token count is invalidated; CRE recomputes on next pass
            canonical.token_count = 0
            result.facts_synthesised += 1

        # Remove other cluster members from `facts`
        ids_to_drop = {f.id for f in cluster if f.id != canonical.id}
        if ids_to_drop:
            # Filter in place
            survivors = [f for f in facts if f.id not in ids_to_drop]
            facts.clear()
            facts.extend(survivors)
            result.removed_ids.extend(ids_to_drop)
            result.facts_removed += len(ids_to_drop)

    return result


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    """Element-wise mean. Used as a cluster centroid update."""
    if not vectors:
        return []
    n = len(vectors)
    dim = len(vectors[0])
    out = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            out[i] += v[i]
    return [x / n for x in out]
