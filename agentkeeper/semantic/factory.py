"""Vector index factory with auto-detection.

The recaller asks for "a VectorIndex for this agent" and gets the best
available one without knowing the details:

- `AGENTKEEPER_VECTOR_INDEX=sqlite_vec`: force sqlite-vec; raise if
  unavailable.
- `AGENTKEEPER_VECTOR_INDEX=in_memory`: force the in-memory index.
- `AGENTKEEPER_VECTOR_INDEX=auto` (default): use sqlite-vec when
  available, otherwise fall back to the in-memory index.

This keeps the recaller's constructor signature simple: pass the
agent's id and the embedding dimension, get an index back.
"""

from __future__ import annotations

import os

from ..logging import get_logger
from .index import InMemoryVectorIndex, VectorIndex
from .sqlite_vec_index import SqliteVecIndex, is_available

_log = get_logger(__name__)


def make_vector_index(
    agent_id: str,
    dimension: int,
    backend: str | None = None,
    db_path: str | None = None,
) -> VectorIndex:
    """Return the appropriate VectorIndex for this agent.

    Args:
        agent_id: Scopes the persistent index when sqlite-vec is used.
        dimension: Embedding dimensionality.
        backend: Override env var. One of `"auto"`, `"in_memory"`,
            `"sqlite_vec"`. Default reads `AGENTKEEPER_VECTOR_INDEX`.
        db_path: SQLite path for the persistent index. Defaults to
            `AGENTKEEPER_DB` env var, then `agentkeeper.db`.

    Raises:
        RuntimeError: when `sqlite_vec` is requested explicitly but the
            extension isn't installed/loadable.
    """
    chosen = (
        backend
        or os.getenv("AGENTKEEPER_VECTOR_INDEX", "auto").strip().lower()
    )

    if chosen == "in_memory":
        return InMemoryVectorIndex(dimension=dimension)

    if chosen == "sqlite_vec":
        if not is_available():
            raise RuntimeError(
                "AGENTKEEPER_VECTOR_INDEX=sqlite_vec but sqlite-vec is "
                "not installed or not loadable. Install with "
                "'pip install agentkeeper[vec]'."
            )
        return SqliteVecIndex(
            agent_id=agent_id,
            dimension=dimension,
            db_path=db_path,
        )

    # auto
    if is_available():
        _log.debug(
            "Using SqliteVecIndex for agent %s (dim=%d)", agent_id, dimension
        )
        return SqliteVecIndex(
            agent_id=agent_id,
            dimension=dimension,
            db_path=db_path,
        )
    _log.debug(
        "sqlite-vec unavailable; falling back to InMemoryVectorIndex "
        "for agent %s (dim=%d)",
        agent_id,
        dimension,
    )
    return InMemoryVectorIndex(dimension=dimension)
