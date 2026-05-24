"""Encrypted SQLite storage backend.

Encrypts the JSON payload at rest using Fernet (AES-128-CBC + HMAC-SHA256
as defined by the `cryptography` package). Keys are 32 url-safe base64
bytes (44 chars). Generate one with:

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

…and pass it via the constructor or the `AGENTKEEPER_ENCRYPTION_KEY`
env var.

Mixed-mode tolerance: when loading, the backend detects whether a row
is already encrypted (Fernet payloads start with a known prefix). Plain
JSON rows from a previous `SQLiteStorage` deployment are read and
upgraded on next save. This makes migration progressive without
downtime.

Caveats:
- This encrypts the *payload column only*. Table/column structure,
  agent IDs, and timestamps are visible in the SQLite file. If you
  need full database encryption, mount the SQLite file inside a
  filesystem-level encrypted volume (LUKS, FileVault, etc.) or use a
  Postgres backend with TDE.
- Losing the key means losing every encrypted agent permanently.
  AgentKeeper has no key recovery mechanism — that is the *point* of
  encryption at rest.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from ..cso.types import CognitiveStateObject
from ..errors import ConfigurationError
from .sqlite_store import SQLiteStorage

if TYPE_CHECKING:
    from cryptography.fernet import Fernet

# Magic prefix every Fernet token starts with after base64 decoding
# (version byte 0x80). The encrypted *string* therefore starts with
# "gAAAAA". This is a robust runtime detector that does not require
# us to deserialise the payload first.
_FERNET_PREFIX = "gAAAAA"


def _require_cryptography() -> type[Fernet]:
    """Import `cryptography.fernet.Fernet` lazily with a helpful error."""
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ConfigurationError(
            "EncryptedSQLiteStorage requires the 'cryptography' package. "
            "Install with: pip install 'agentkeeper[encrypted]'"
        ) from exc
    return Fernet


class EncryptedSQLiteStorage(SQLiteStorage):
    """SQLite storage with payload-at-rest encryption.

    Args:
        db_path: SQLite database path. Defaults to `AGENTKEEPER_DB` env var.
        encryption_key: Fernet key (44-char url-safe base64). If omitted,
            read from `AGENTKEEPER_ENCRYPTION_KEY`.

    Raises:
        ConfigurationError: if no key is provided and the env var is unset.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        encryption_key: str | bytes | None = None,
    ) -> None:
        fernet_cls = _require_cryptography()

        key: str | bytes | None = encryption_key or os.getenv(
            "AGENTKEEPER_ENCRYPTION_KEY"
        )
        if not key:
            raise ConfigurationError(
                "EncryptedSQLiteStorage requires an encryption key. "
                "Pass `encryption_key=...` or set AGENTKEEPER_ENCRYPTION_KEY."
            )
        if isinstance(key, str):
            key = key.encode("utf-8")

        try:
            self._fernet: Fernet = fernet_cls(key)
        except Exception as exc:
            raise ConfigurationError(
                "Invalid Fernet encryption key. Generate a valid one with:\n"
                "  python -c \"from cryptography.fernet import Fernet; "
                'print(Fernet.generate_key().decode())"'
            ) from exc

        # init schema after fernet is set so _decode_payload can run if needed
        super().__init__(db_path=db_path)

    # --- override payload hooks --------------------------------------

    def _encode_payload(self, cso: CognitiveStateObject) -> str:
        plaintext = json.dumps(cso.to_dict()).encode("utf-8")
        return self._fernet.encrypt(plaintext).decode("utf-8")

    def _decode_payload(self, payload: str) -> CognitiveStateObject:
        # Detect legacy plain-JSON rows so a deployment can migrate
        # progressively. After load + save the row is re-encrypted.
        if not payload.startswith(_FERNET_PREFIX):
            return CognitiveStateObject.from_dict(json.loads(payload))
        try:
            plaintext = self._fernet.decrypt(payload.encode("utf-8"))
        except Exception as exc:
            raise ConfigurationError(
                "Failed to decrypt agent payload. Either the encryption "
                "key is wrong, or the stored data is corrupted."
            ) from exc
        return CognitiveStateObject.from_dict(json.loads(plaintext.decode("utf-8")))

    # --- key management ----------------------------------------------

    @staticmethod
    def generate_key() -> str:
        """Generate a fresh Fernet key as a url-safe base64 string.

        Convenience wrapper so users don't need to import cryptography
        directly. The returned key can be passed to the constructor or
        stored in AGENTKEEPER_ENCRYPTION_KEY.
        """
        fernet_cls = _require_cryptography()
        return fernet_cls.generate_key().decode("utf-8")
