"""Storage backend contract.

Every storage backend implements four operations:

- `save(cso)`     : persist a CognitiveStateObject (insert or update)
- `load(agent_id)` : return the stored CSO, or None if missing
- `delete(agent_id)` : remove an agent's CSO; idempotent
- `list_agent_ids()` : enumerate stored agent identifiers

Implementations choose their storage substrate (SQLite, encrypted
SQLite, Postgres, S3, etc.) but must honour the contract exactly.
This lets the agent layer stay backend-agnostic.

Note: this is a *narrow* contract on purpose. AgentKeeper does not
provide cross-backend transactions, distributed locks, or replication
primitives. Backends that need those features should expose them via
their own concrete API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..cso.types import CognitiveStateObject


class BaseStorage(ABC):
    """Abstract storage backend for `CognitiveStateObject` persistence."""

    @abstractmethod
    def save(self, cso: CognitiveStateObject) -> None:
        """Persist a CSO. Upserts on `agent_id`."""

    @abstractmethod
    def load(self, agent_id: str) -> CognitiveStateObject | None:
        """Return the CSO for `agent_id`, or None if not present."""

    @abstractmethod
    def delete(self, agent_id: str) -> None:
        """Remove the CSO for `agent_id`. Idempotent: no-op if missing."""

    @abstractmethod
    def list_agent_ids(self) -> list[str]:
        """Return a list of agent IDs currently stored."""

    # --- ergonomics ---------------------------------------------------

    def exists(self, agent_id: str) -> bool:
        """Convenience predicate. Backends may override for efficiency."""
        return agent_id in self.list_agent_ids()
