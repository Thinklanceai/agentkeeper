"""Cognitive State Object — the foundational data model of AgentKeeper.

A CSO represents the full cognitive state of an agent at a given moment:
its memory facts, its identity (in later versions), and metadata required
to reconstruct that state across model switches and restarts.

This module is intentionally minimal. Higher-level behaviour (selection,
reconstruction, compression) lives in `agentkeeper.cre`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utcnow_iso() -> str:
    """ISO-8601 timestamp in UTC, RFC 3339 compatible."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Fact:
    """A single unit of knowledge held by an agent.

    Attributes:
        id: Stable identifier (UUID v4).
        content: The textual content of the fact.
        critical: Whether this fact must always be reconstructed
                  (force-included regardless of token budget).
        token_count: Approximate token count. Computed lazily by the CRE.
    """

    id: str
    content: str
    critical: bool = False
    token_count: int = 0

    @staticmethod
    def create(content: str, critical: bool = False) -> Fact:
        return Fact(
            id=str(uuid.uuid4()),
            content=content,
            critical=critical,
        )


@dataclass
class CognitiveStateObject:
    """Full cognitive state of an agent.

    This object is the unit of persistence: saving an agent means
    serialising its CSO; loading an agent means deserialising it.
    """

    agent_id: str
    memory_facts: list[Fact] = field(default_factory=list)
    created_at: str = field(default_factory=_utcnow_iso)
    updated_at: str = field(default_factory=_utcnow_iso)

    @staticmethod
    def create(agent_id: str | None = None) -> CognitiveStateObject:
        return CognitiveStateObject(agent_id=agent_id or str(uuid.uuid4()))

    def add_fact(self, content: str, critical: bool = False) -> Fact:
        fact = Fact.create(content, critical)
        self.memory_facts.append(fact)
        self.updated_at = _utcnow_iso()
        return fact

    def critical_facts(self) -> list[Fact]:
        return [f for f in self.memory_facts if f.critical]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "memory_facts": [
                {
                    "id": f.id,
                    "content": f.content,
                    "critical": f.critical,
                    "token_count": f.token_count,
                }
                for f in self.memory_facts
            ],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> CognitiveStateObject:
        cso = CognitiveStateObject(
            agent_id=data["agent_id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )
        cso.memory_facts = [
            Fact(
                id=f["id"],
                content=f["content"],
                critical=f["critical"],
                token_count=f.get("token_count", 0),
            )
            for f in data["memory_facts"]
        ]
        return cso
