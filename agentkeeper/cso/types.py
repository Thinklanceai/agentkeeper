"""Cognitive State Object — the foundational data model of AgentKeeper.

A CSO represents the full cognitive state of an agent at a given moment:
its identity, its memory facts (organised in tiers), and metadata
required to reconstruct that state across model switches and restarts.

Backward compatibility:
- `Fact.create(content, critical=True)` still works exactly as before.
  Internally, `critical=True` maps to `tier=semantic, importance=0.95`.
- `cso.add_fact(content, critical=True)` likewise.
- Old serialised CSOs (v0.1 schema) deserialise transparently via
  `from_dict`, which fills in defaults for the new fields.

Higher-level behaviour (selection, reconstruction, compression) lives
in `agentkeeper.cre`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .identity import AgentIdentity
from .inference import infer_importance, infer_tier
from .tiers import MemoryTier, is_valid_tier


def _utcnow_iso() -> str:
    """ISO-8601 timestamp in UTC, RFC 3339 compatible."""
    return datetime.now(timezone.utc).isoformat()


def _normalise_tier(tier: MemoryTier | str | None) -> MemoryTier | None:
    if tier is None:
        return None
    if isinstance(tier, MemoryTier):
        return tier
    if isinstance(tier, str):
        if not is_valid_tier(tier):
            raise ValueError(
                f"Unknown tier: {tier!r}. "
                f"Valid: {[t.value for t in MemoryTier]}"
            )
        return MemoryTier(tier)
    raise TypeError(f"tier must be MemoryTier, str, or None; got {type(tier)}")


@dataclass
class Fact:
    """A single unit of knowledge held by an agent.

    Attributes:
        id: Stable identifier (UUID v4).
        content: The textual content of the fact.
        tier: Memory tier (working, episodic, semantic, archival).
        importance: Float 0.0 - 1.0. Higher = harder to evict.
                    The CRE force-includes facts with importance >= 0.9
                    (called "critical" in the legacy API).
        token_count: Approximate token count. Computed lazily by the CRE.
        created_at: ISO-8601 UTC timestamp of creation.
        last_accessed_at: Updated whenever this fact is retrieved.
        access_count: Number of times this fact has been retrieved.
        when: Optional event timestamp (only meaningful for `episodic`
              facts; the moment the event occurred, not when it was stored).
        metadata: Free-form dict for user extensions.
    """

    id: str
    content: str
    tier: MemoryTier = MemoryTier.SEMANTIC
    importance: float = 0.5
    token_count: int = 0
    created_at: str = field(default_factory=_utcnow_iso)
    last_accessed_at: str = field(default_factory=_utcnow_iso)
    access_count: int = 0
    when: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- legacy compatibility ---------------------------------------

    @property
    def critical(self) -> bool:
        """Compatibility with v0.1: a 'critical' fact is one with
        importance >= 0.9. The CRE force-includes such facts regardless
        of token budget."""
        return self.importance >= 0.9

    # --- factories --------------------------------------------------

    @staticmethod
    def create(
        content: str,
        critical: bool | None = None,
        tier: MemoryTier | str | None = None,
        importance: float | None = None,
        when: str | datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Fact:
        """Build a Fact with smart defaults.

        Resolution order:
        1. If `tier` is given explicitly, use it; otherwise infer from content.
        2. If `importance` is given, use it; otherwise infer from content+tier.
        3. If `critical=True` is passed (legacy), promote importance to >= 0.95.
        """
        resolved_tier = _normalise_tier(tier) or infer_tier(content)

        if importance is None:
            resolved_importance = infer_importance(content, resolved_tier)
        else:
            resolved_importance = max(0.0, min(1.0, importance))

        if critical is True and resolved_importance < 0.95:
            resolved_importance = 0.95
        elif critical is False and importance is None:
            # legacy non-critical default
            resolved_importance = min(resolved_importance, 0.5)

        when_iso: str | None = None
        if isinstance(when, datetime):
            when_iso = when.isoformat()
        elif isinstance(when, str):
            when_iso = when

        return Fact(
            id=str(uuid.uuid4()),
            content=content,
            tier=resolved_tier,
            importance=resolved_importance,
            when=when_iso,
            metadata=dict(metadata) if metadata else {},
        )

    # --- serialisation ----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "tier": self.tier.value,
            "importance": self.importance,
            "token_count": self.token_count,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "when": self.when,
            "metadata": dict(self.metadata),
            # legacy field, useful for v0.1 readers
            "critical": self.critical,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Fact:
        # Legacy migration: v0.1 facts only have id/content/critical/token_count.
        legacy_critical = bool(data.get("critical", False))
        tier_raw = data.get("tier")
        tier = MemoryTier.SEMANTIC if tier_raw is None else MemoryTier(tier_raw)

        importance = data.get("importance")
        if importance is None:
            importance = 0.95 if legacy_critical else 0.5

        return Fact(
            id=data["id"],
            content=data["content"],
            tier=tier,
            importance=float(importance),
            token_count=int(data.get("token_count", 0)),
            created_at=data.get("created_at", _utcnow_iso()),
            last_accessed_at=data.get("last_accessed_at", _utcnow_iso()),
            access_count=int(data.get("access_count", 0)),
            when=data.get("when"),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class CognitiveStateObject:
    """Full cognitive state of an agent.

    This object is the unit of persistence: saving an agent means
    serialising its CSO; loading an agent means deserialising it.

    Attributes:
        agent_id: Stable agent identifier.
        identity: Persistent self-model (name, role, principles).
        memory_facts: All facts the agent has ever stored.
        created_at, updated_at: ISO-8601 UTC timestamps.
    """

    agent_id: str
    memory_facts: list[Fact] = field(default_factory=list)
    identity: AgentIdentity = field(default_factory=AgentIdentity)
    created_at: str = field(default_factory=_utcnow_iso)
    updated_at: str = field(default_factory=_utcnow_iso)

    @staticmethod
    def create(agent_id: str | None = None) -> CognitiveStateObject:
        return CognitiveStateObject(agent_id=agent_id or str(uuid.uuid4()))

    # --- mutation ----------------------------------------------------

    def add_fact(
        self,
        content: str,
        critical: bool | None = None,
        tier: MemoryTier | str | None = None,
        importance: float | None = None,
        when: str | datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Fact:
        """Add a fact. Backward compatible with the v0.1 signature."""
        fact = Fact.create(
            content,
            critical=critical,
            tier=tier,
            importance=importance,
            when=when,
            metadata=metadata,
        )
        self.memory_facts.append(fact)
        self.updated_at = _utcnow_iso()
        return fact

    def set_identity(self, identity: AgentIdentity) -> None:
        self.identity = identity
        self.updated_at = _utcnow_iso()

    # --- selectors ---------------------------------------------------

    def critical_facts(self) -> list[Fact]:
        return [f for f in self.memory_facts if f.critical]

    def facts_by_tier(self, tier: MemoryTier | str) -> list[Fact]:
        normalised = _normalise_tier(tier)
        return [f for f in self.memory_facts if f.tier == normalised]

    # --- serialisation ----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "identity": self.identity.to_dict(),
            "memory_facts": [f.to_dict() for f in self.memory_facts],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "schema_version": 2,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> CognitiveStateObject:
        identity_data = data.get("identity", {}) or {}
        cso = CognitiveStateObject(
            agent_id=data["agent_id"],
            created_at=data.get("created_at", _utcnow_iso()),
            updated_at=data.get("updated_at", _utcnow_iso()),
            identity=AgentIdentity.from_dict(identity_data),
        )
        cso.memory_facts = [Fact.from_dict(f) for f in data.get("memory_facts", [])]
        return cso
