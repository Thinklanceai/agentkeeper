"""Postgres storage backend — *stub* for v1.1.

This module declares the `PostgresStorage` class so the public API and
documentation can reference it, but it does **not** ship a working
implementation in v1.1. The full implementation lands in v1.2 once
the SQLAlchemy/asyncpg story is settled.

The stub is here on purpose: announcing the backend without it would
look like vapourware; hiding it entirely would force a breaking import
later. A clean `NotImplementedError` with a roadmap pointer is the
honest middle ground.

If you need Postgres today, two options:

1. Implement `BaseStorage` directly with your driver of choice. The
   contract is four methods, well below 100 lines. Pin to your version
   of AgentKeeper to avoid surprises.
2. Track the milestone at
   https://github.com/Thinklanceai/agentkeeper/milestones/v1.2
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseStorage

if TYPE_CHECKING:
    from ..cso.types import CognitiveStateObject


class PostgresStorage(BaseStorage):
    """Postgres storage backend. **Not implemented in v1.1.**

    Raises:
        NotImplementedError: every method raises until v1.2.

    Args:
        dsn: Postgres connection string (e.g.
            ``postgresql://user:pass@host:5432/dbname``).
        schema: optional schema name; defaults to ``"public"``.
    """

    def __init__(self, dsn: str, schema: str = "public") -> None:
        self.dsn = dsn
        self.schema = schema
        raise NotImplementedError(
            "PostgresStorage is scheduled for v1.2. For v1.1, use "
            "SQLiteStorage or EncryptedSQLiteStorage, or implement "
            "BaseStorage directly with your Postgres driver of choice. "
            "See https://github.com/Thinklanceai/agentkeeper for details."
        )

    def save(self, cso: CognitiveStateObject) -> None:  # pragma: no cover
        raise NotImplementedError

    def load(self, agent_id: str) -> CognitiveStateObject | None:  # pragma: no cover
        raise NotImplementedError

    def delete(self, agent_id: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def list_agent_ids(self) -> list[str]:  # pragma: no cover
        raise NotImplementedError
