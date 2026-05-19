"""SQLite-based persistence for Cognitive State Objects.

This module persists CSOs to a local SQLite database. It is intentionally
the simplest possible backend: zero infrastructure, no external services,
fully self-contained.

In future versions, alternative backends (filesystem, Postgres, encrypted
storage) will plug in via the same interface.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..cso.types import CognitiveStateObject

DEFAULT_DB_PATH = "agentkeeper.db"


class Storage:
    """Persist CSOs to a local SQLite database.

    The database path can be overridden via constructor argument or via
    the AGENTKEEPER_DB environment variable. The schema is created on
    first connection.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        env_path = os.getenv("AGENTKEEPER_DB")
        chosen = db_path or env_path or DEFAULT_DB_PATH
        self.db_path = str(chosen)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id   TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
            conn.commit()

    def save(self, cso: CognitiveStateObject) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO agents
                    (agent_id, state_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    cso.agent_id,
                    json.dumps(cso.to_dict()),
                    cso.created_at,
                    cso.updated_at,
                ),
            )
            conn.commit()

    def load(self, agent_id: str) -> CognitiveStateObject | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM agents WHERE agent_id = ?",
                (agent_id,),
            ).fetchone()
            if row is None:
                return None
            return CognitiveStateObject.from_dict(json.loads(row[0]))

    def delete(self, agent_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM agents WHERE agent_id = ?", (agent_id,))
            conn.commit()

    def list_agent_ids(self) -> list[str]:
        """Return all agent IDs stored in this database."""
        with self._connect() as conn:
            rows = conn.execute("SELECT agent_id FROM agents").fetchall()
            return [row[0] for row in rows]
