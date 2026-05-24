"""Contradiction detection and arbitration.

When two facts are semantically very similar but their *value* contradicts
each other, the agent's reasoning becomes inconsistent. AgentKeeper
detects such contradictions and arbitrates between them.

Detection heuristic:
1. The two facts have high embedding similarity (> 0.75) — they talk
   about the same subject.
2. Yet one contains a value/state and another contains a *different*
   value for the same subject. Detected via:
   - Key-value form: `key: A` vs `key: B` with the same key.
   - Polarity opposition: presence of negation markers in one and not
     the other (`is X` vs `is not X`, `agreed` vs `refused`, etc.).

Arbitration rule (deterministic, no LLM):
1. Critical wins over non-critical.
2. If equal importance: most recently accessed wins.
3. If still tied: most recently created wins.

The loser is **flagged** (importance reduced, metadata annotated) rather
than deleted. The user can inspect via `agent.contradictions()`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .._fastmath import dot as _dot
from ..cso.types import Fact
from ..semantic.base import EmbeddingProvider


@dataclass
class ContradictionConfig:
    # Similarity threshold above which two facts are considered "about
    # the same subject" and worth checking for contradiction.
    similarity_threshold: float = 0.75
    # When a contradiction is resolved, the loser's importance is
    # multiplied by this factor (kept around but de-prioritised).
    loser_importance_factor: float = 0.3
    target_tiers: tuple[str, ...] = ("semantic", "episodic")


@dataclass
class Contradiction:
    winner_id: str
    loser_id: str
    reason: str
    similarity: float


@dataclass
class ContradictionResult:
    pairs_found: int = 0
    contradictions: list[Contradiction] = field(default_factory=list)


_NEGATION_PATTERNS = [
    re.compile(r"\bnot\b", re.IGNORECASE),
    re.compile(r"\bno\b", re.IGNORECASE),
    re.compile(r"\bnever\b", re.IGNORECASE),
    re.compile(r"\brefused?\b", re.IGNORECASE),
    re.compile(r"\bcancell?ed\b", re.IGNORECASE),
    re.compile(r"\brejected?\b", re.IGNORECASE),
    re.compile(r"\bn'?est\s+pas\b", re.IGNORECASE),
    re.compile(r"\bjamais\b", re.IGNORECASE),
    re.compile(r"\brefus", re.IGNORECASE),
    re.compile(r"\bannul", re.IGNORECASE),
]


def _has_negation(text: str) -> bool:
    return any(p.search(text) for p in _NEGATION_PATTERNS)


def _kv_key(text: str) -> str | None:
    """Return the key part of a `key: value` fact, or None."""
    if ":" not in text:
        return None
    key, _, _ = text.partition(":")
    return key.strip().lower()


def _kv_value(text: str) -> str | None:
    if ":" not in text:
        return None
    _, _, value = text.partition(":")
    return value.strip().lower()


def _polarity_opposite(a: str, b: str) -> bool:
    return _has_negation(a) != _has_negation(b)


def _values_differ(a: str, b: str) -> bool:
    """Detect whether two key-value facts have the same key but
    different values."""
    key_a = _kv_key(a)
    key_b = _kv_key(b)
    if not key_a or not key_b or key_a != key_b:
        return False
    val_a = _kv_value(a) or ""
    val_b = _kv_value(b) or ""
    return val_a != val_b and val_a != "" and val_b != ""


def _detect_contradiction(a: Fact, b: Fact, similarity: float) -> str | None:
    """Return a textual reason if `a` and `b` contradict, else None."""
    if _values_differ(a.content, b.content):
        return "key-value-divergence"
    if _polarity_opposite(a.content, b.content):
        return "polarity-opposition"
    return None


def _winner_loser(a: Fact, b: Fact) -> tuple[Fact, Fact]:
    """Apply the arbitration rule. Returns (winner, loser)."""

    def score(f: Fact) -> tuple[float, str, str]:
        # Higher is better
        return (f.importance, f.last_accessed_at, f.created_at)

    if score(a) >= score(b):
        return a, b
    return b, a


def detect_and_resolve(
    facts: list[Fact],
    provider: EmbeddingProvider,
    config: ContradictionConfig | None = None,
    now: datetime | None = None,
) -> ContradictionResult:
    """Detect contradictions among `facts` and resolve them in place.

    The losers' importance is reduced and their metadata annotated;
    they are not deleted. The user can re-promote them manually if
    needed.
    """
    config = config or ContradictionConfig()
    result = ContradictionResult()
    now = now or datetime.now(timezone.utc)

    # Only target facts in eligible tiers AND not protected. Identity-level
    # facts (principles, hard constraints) are exempt from arbitration —
    # they define the agent and cannot be overruled by ordinary memory.
    targets = [
        f for f in facts
        if f.tier.value in config.target_tiers and not f.protected
    ]
    if len(targets) < 2:
        return result

    vectors = provider.embed([f.content for f in targets])
    n = len(targets)

    # Already-resolved losers are skipped on further iterations to avoid
    # cascading.
    already_lost: set[str] = set()

    # Vectorised prefilter: contradictions only occur between highly
    # similar facts (similarity >= threshold). For each row i we compute
    # similarities against all j > i in one batched dot product, then
    # only run the (cheaper but still non-trivial) contradiction check on
    # candidate pairs that clear the threshold. We compute per-row rather
    # than materialising the full n*n matrix, to bound memory at large n
    # (a 10k*10k float matrix would be ~800MB).
    from .._fastmath import HAS_NUMPY

    mat = None
    if HAS_NUMPY and n > 1:
        import numpy as _np

        mat = _np.asarray(vectors, dtype=float)

    for i in range(n):
        if targets[i].id in already_lost:
            continue
        if mat is not None:
            # Similarities of row i against every other row, vectorised.
            row = (mat @ mat[i])
            candidates = [
                j
                for j in range(i + 1, n)
                if row[j] >= config.similarity_threshold
            ]
            row_scores = row
        else:
            candidates = [
                j
                for j in range(i + 1, n)
                if _dot(vectors[i], vectors[j]) >= config.similarity_threshold
            ]
            row_scores = None
        for j in candidates:
            if targets[j].id in already_lost:
                continue
            similarity = (
                float(row_scores[j])
                if row_scores is not None
                else _dot(vectors[i], vectors[j])
            )
            reason = _detect_contradiction(targets[i], targets[j], similarity)
            if reason is None:
                continue

            result.pairs_found += 1
            winner, loser = _winner_loser(targets[i], targets[j])
            loser.importance = max(
                0.05, loser.importance * config.loser_importance_factor
            )
            loser.metadata = dict(loser.metadata)
            loser.metadata.update(
                {
                    "contradicted_by": winner.id,
                    "contradiction_reason": reason,
                    "contradiction_detected_at": now.isoformat(),
                }
            )
            result.contradictions.append(
                Contradiction(
                    winner_id=winner.id,
                    loser_id=loser.id,
                    reason=reason,
                    similarity=similarity,
                )
            )
            already_lost.add(loser.id)

    return result
