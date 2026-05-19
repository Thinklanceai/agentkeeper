"""Storage backend factory.

Resolves a `BaseStorage` instance based on the `AGENTKEEPER_STORAGE`
environment variable or an explicit argument:

- ``sqlite`` (default): `SQLiteStorage` — local file, no encryption.
- ``encrypted_sqlite``: `EncryptedSQLiteStorage` — same file, payload
  encrypted at rest. Requires ``AGENTKEEPER_ENCRYPTION_KEY``.
- ``postgres``: `PostgresStorage` — raises `NotImplementedError` in
  v1.1 (scheduled for v1.2).

Backward compatibility: callers that don't set the env var get the
exact same `SQLiteStorage` behaviour as in v1.0.
"""

from __future__ import annotations

import os

from ..errors import ConfigurationError
from ..logging import get_logger
from .base import BaseStorage
from .encrypted_sqlite import EncryptedSQLiteStorage
from .postgres import PostgresStorage
from .sqlite_store import SQLiteStorage

_log = get_logger(__name__)


def make_storage(backend: str | None = None) -> BaseStorage:
    """Return the storage backend configured for this process.

    Args:
        backend: One of ``"sqlite"``, ``"encrypted_sqlite"``,
            ``"postgres"``. Defaults to the ``AGENTKEEPER_STORAGE`` env
            var, then to ``"sqlite"``.

    Raises:
        ConfigurationError: for unknown backends or missing required
            configuration (e.g. encryption key, Postgres DSN).
    """
    chosen = (
        backend
        or os.getenv("AGENTKEEPER_STORAGE", "sqlite").strip().lower()
    )

    if chosen in ("sqlite", ""):
        return SQLiteStorage()

    if chosen == "encrypted_sqlite":
        return EncryptedSQLiteStorage()

    if chosen == "postgres":
        dsn = os.getenv("AGENTKEEPER_POSTGRES_DSN")
        if not dsn:
            raise ConfigurationError(
                "AGENTKEEPER_STORAGE=postgres requires "
                "AGENTKEEPER_POSTGRES_DSN to be set."
            )
        return PostgresStorage(dsn=dsn)

    raise ConfigurationError(
        f"Unknown AGENTKEEPER_STORAGE backend: {chosen!r}. "
        "Valid: 'sqlite', 'encrypted_sqlite', 'postgres'."
    )
