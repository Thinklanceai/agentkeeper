"""Relational triples — the atomic unit of the graph memory layer.

A `Triple` represents a directed relation between two entities, with
optional metadata. Examples:

    Triple("Acme", "owns", "Globex")
    Triple("Alice", "works_at", "Acme", confidence=0.9)
    Triple("Project Phoenix", "uses_provider", "Anthropic")

Triples are stored separately from `Fact`s and have their own retention
policy (TTL, protected flag). Unlike facts, triples are intentionally
structured — they are made for traversal queries, not for prose recall.

Why a separate type instead of overloading Fact? Because facts are
prose ("Alice works at Acme") and triples are structure
(Alice -[works_at]-> Acme). Mixing the two would hurt both:
- Compression would over-merge prose facts that happen to share a
  subject ("Alice is happy" / "Alice is tired" both about Alice).
- Graph traversal would need to NL-parse facts at query time.

By keeping them separate, the agent can hold both:
- "Acme is owned by Globex, a Belgian holding" (prose Fact)
- ("Acme", "owned_by", "Globex"), ("Globex", "located_in", "BE")
  (structured Triples for graph queries)

The user can extract triples from a Fact via `agent.link(...)` whenever
the relation matters for navigation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Triple:
    """A directed relation between two entities.

    Attributes:
        id: Stable UUID v4 identifier.
        subject: The entity from which the relation originates.
        predicate: The relation type (e.g. "owns", "works_at",
            "located_in"). Free-form string — no schema enforced.
        object: The entity at which the relation points.
        confidence: Float 0-1. Defaults to 1.0 (asserted, not inferred).
        protected: When True, the triple survives every compression
            and is exempt from `purge_expired` unless the caller passes
            `include_protected=True`.
        created_at: ISO-8601 UTC timestamp.
        last_accessed_at: Updated on traversal hits.
        access_count: Incremented on traversal hits.
        expires_at: Optional ISO-8601 timestamp. Triples with a past
            expiration are removed by `purge_expired` and the
            compression pipeline (same as Facts).
        metadata: Free-form dict for user extensions (source fact id,
            provenance, etc.).
    """

    id: str
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    protected: bool = False
    created_at: str = field(default_factory=_utcnow_iso)
    last_accessed_at: str = field(default_factory=_utcnow_iso)
    access_count: int = 0
    expires_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create(
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 1.0,
        protected: bool = False,
        ttl: timedelta | str | int | float | None = None,
        expires_at: str | datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Triple:
        """Build a Triple with smart defaults.

        Mirrors `Fact.create()` for consistency: explicit `expires_at`
        wins over `ttl`, both fall back to None (no expiration).

        Args:
            subject, predicate, object: The relation, all required.
            confidence: 0-1, clipped.
            protected: Survives compression and default purges.
            ttl: Relative TTL, accepts timedelta, '30d', etc.
            expires_at: Absolute expiry timestamp (overrides ttl).
            metadata: Free-form dict.
        """
        if not subject or not predicate or not object:
            from ..errors import ConfigurationError

            raise ConfigurationError(
                "Triple requires non-empty subject, predicate, and object"
            )

        confidence = max(0.0, min(1.0, confidence))

        expires_iso: str | None = None
        if expires_at is not None:
            if isinstance(expires_at, datetime):
                expires_iso = expires_at.isoformat()
            elif isinstance(expires_at, str):
                expires_iso = expires_at
        elif ttl is not None:
            from ..retention.ttl import compute_expires_at

            expires_iso = compute_expires_at(ttl)

        return Triple(
            id=str(uuid.uuid4()),
            subject=subject.strip(),
            predicate=predicate.strip(),
            object=object.strip(),
            confidence=confidence,
            protected=protected,
            expires_at=expires_iso,
            metadata=dict(metadata) if metadata else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "protected": self.protected,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "expires_at": self.expires_at,
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Triple:
        return Triple(
            id=data["id"],
            subject=data["subject"],
            predicate=data["predicate"],
            object=data["object"],
            confidence=float(data.get("confidence", 1.0)),
            protected=bool(data.get("protected", False)),
            created_at=data.get("created_at", _utcnow_iso()),
            last_accessed_at=data.get("last_accessed_at", _utcnow_iso()),
            access_count=int(data.get("access_count", 0)),
            expires_at=data.get("expires_at"),
            metadata=dict(data.get("metadata", {}) or {}),
        )

    def __repr__(self) -> str:
        return (
            f"Triple({self.subject!r} -[{self.predicate}]-> {self.object!r}, "
            f"conf={self.confidence:.2f})"
        )
