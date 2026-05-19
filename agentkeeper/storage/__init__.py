"""Pluggable storage backends.

Public surface:

- `BaseStorage`              : the contract every backend implements
- `SQLiteStorage`            : default, zero-dep local file
- `Storage`                  : backward-compat alias for `SQLiteStorage`
- `EncryptedSQLiteStorage`   : at-rest encryption via Fernet
- `PostgresStorage`          : stub, full impl in v1.2
- `make_storage()`           : env-var-driven factory
"""

from .base import BaseStorage
from .encrypted_sqlite import EncryptedSQLiteStorage
from .factory import make_storage
from .postgres import PostgresStorage
from .sqlite_store import SQLiteStorage, Storage

__all__ = [
    "BaseStorage",
    "EncryptedSQLiteStorage",
    "PostgresStorage",
    "SQLiteStorage",
    "Storage",
    "make_storage",
]
