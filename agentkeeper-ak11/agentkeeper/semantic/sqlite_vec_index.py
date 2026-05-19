"""Persistent vector index backed by the `sqlite-vec` SQLite extension.

`sqlite-vec` ships an in-process vector search extension for SQLite
that scales well beyond pure-Python brute force while remaining
zero-infrastructure (single file on disk, no daemon, no client).

This index implements the same `VectorIndex` contract as
`InMemoryVectorIndex`, so callers don't need to know which backend
they're talking to. The recaller transparently uses whichever index
is selected at construction time.

The extension is *optional*. Install via:

    pip install 'agentkeeper[vec]'

When the package is missing, `is_available()` returns False and the
recaller falls back to `InMemoryVectorIndex` automatically.

Design choices:

- **Vector blobs**: vectors stored as little-endian float32 BLOBs in
  a `vec0` virtual table. sqlite-vec internally manages the index.
- **Cosine via dot product**: vectors are expected to be L2-normalised
  upstream (the `SemanticRecaller` does this). sqlite-vec exposes
  `vec_distance_cosine`; we use `1 - cosine_distance` as the score so
  the contract matches the in-memory backend.
- **Same DB file as storage**: by default the index lives in the same
  SQLite file as the agent storage (`AGENTKEEPER_DB`), in a dedicated
  table per agent. One file, one process, no orchestration.
- **Connection ownership**: the index owns its own short-lived
  connection. It does not share the long-lived connection used by
  the storage layer — fewer locking surprises during concurrent
  recall+save.

Per-agent isolation: each agent gets its own virtual table named
`vec_idx_<sanitised_agent_id>` to avoid scanning unrelated vectors.
"""

from __future__ import annotations

import os
import re
import sqlite3
import struct
from collections.abc import Sequence

from .index import VectorIndex


def is_available() -> bool:
    """Return True when the `sqlite-vec` extension can be loaded.

    Checks both that the `sqlite_vec` Python package is importable and
    that SQLite was built with `enable_load_extension`. Both can fail
    independently (the second is the trickier one on macOS).
    """
    try:
        import sqlite_vec  # noqa: F401
    except ImportError:
        return False
    # Probe by trying to open a throwaway in-memory connection.
    try:
        conn = sqlite3.connect(":memory:")
        try:
            conn.enable_load_extension(True)
        except (sqlite3.NotSupportedError, AttributeError):
            return False
        finally:
            conn.close()
    except Exception:
        return False
    return True


_TABLE_NAME_RE = re.compile(r"[^a-zA-Z0-9_]")


def _sanitise_agent_id(agent_id: str) -> str:
    """Produce a safe identifier fragment for SQL table names.

    Allows only `[a-zA-Z0-9_]`; everything else becomes `_`. Truncates
    to 60 chars to leave room for the `vec_idx_` prefix.
    """
    sanitised = _TABLE_NAME_RE.sub("_", agent_id)[:60]
    return sanitised or "default"


def _pack_vector(vec: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


class SqliteVecIndex(VectorIndex):
    """Persistent vector index using the sqlite-vec extension.

    Each agent has its own `vec0` virtual table so recall queries are
    naturally scoped without an extra WHERE clause.

    Args:
        agent_id: Used to compute the table name.
        dimension: Embedding dimensionality. Must match what the
            embedding provider produces.
        db_path: Path to the SQLite file. Defaults to env var
            `AGENTKEEPER_DB` or `agentkeeper.db`.
    """

    def __init__(
        self,
        agent_id: str,
        dimension: int,
        db_path: str | None = None,
    ) -> None:
        if not is_available():
            raise RuntimeError(
                "sqlite-vec is not available. "
                "Install with: pip install 'agentkeeper[vec]'"
            )
        import sqlite_vec

        self._dimension = dimension
        self._agent_id = agent_id
        self._table = f"vec_idx_{_sanitise_agent_id(agent_id)}"
        self._db_path = db_path or os.getenv(
            "AGENTKEEPER_DB", "agentkeeper.db"
        )

        self._conn = sqlite3.connect(self._db_path)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        # Create the per-agent virtual table if missing. The exact
        # column syntax is `vector float[N]`; sqlite-vec parses it.
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._table} "
            f"USING vec0(fact_id TEXT PRIMARY KEY, vector float[{dimension}])"
        )
        self._conn.commit()

    # --- VectorIndex contract --------------------------------------

    def upsert(self, fact_id: str, vector: Sequence[float]) -> None:
        if len(vector) != self._dimension:
            raise ValueError(
                f"Vector dim {len(vector)} != index dim {self._dimension}"
            )
        blob = _pack_vector(vector)
        # sqlite-vec's vec0 table supports UPSERT via DELETE + INSERT.
        # A single REPLACE-style UPSERT isn't allowed on virtual tables
        # in all SQLite versions; deletion is cheap because it's
        # primary-key indexed.
        with self._conn:
            self._conn.execute(
                f"DELETE FROM {self._table} WHERE fact_id = ?", (fact_id,)
            )
            self._conn.execute(
                f"INSERT INTO {self._table}(fact_id, vector) VALUES (?, ?)",
                (fact_id, blob),
            )

    def delete(self, fact_id: str) -> None:
        with self._conn:
            self._conn.execute(
                f"DELETE FROM {self._table} WHERE fact_id = ?", (fact_id,)
            )

    def search(
        self,
        query: Sequence[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[str, float]]:
        if top_k <= 0:
            return []
        if len(query) != self._dimension:
            raise ValueError(
                f"Query dim {len(query)} != index dim {self._dimension}"
            )
        blob = _pack_vector(query)
        # `MATCH` + `k` is the recommended sqlite-vec idiom; `distance`
        # is the cosine distance (lower = more similar). Score = 1 - d.
        rows = self._conn.execute(
            f"SELECT fact_id, distance FROM {self._table} "
            "WHERE vector MATCH ? AND k = ? "
            "ORDER BY distance ASC",
            (blob, top_k),
        ).fetchall()

        results: list[tuple[str, float]] = []
        for fact_id, distance in rows:
            score = 1.0 - float(distance)
            if score >= min_score:
                results.append((fact_id, score))
        return results

    def size(self) -> int:
        row = self._conn.execute(
            f"SELECT COUNT(*) FROM {self._table}"
        ).fetchone()
        return int(row[0]) if row else 0

    def clear(self) -> None:
        """Drop and recreate the table. Useful after dimension change."""
        with self._conn:
            self._conn.execute(f"DROP TABLE IF EXISTS {self._table}")
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._table} "
                f"USING vec0(fact_id TEXT PRIMARY KEY, "
                f"vector float[{self._dimension}])"
            )

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        import contextlib

        with contextlib.suppress(Exception):
            self._conn.close()

    def __del__(self) -> None:  # pragma: no cover
        self.close()
