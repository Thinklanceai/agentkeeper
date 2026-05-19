"""Tests for the storage factory, ABC, and backward-compat alias."""

from __future__ import annotations

from pathlib import Path

import pytest

import agentkeeper
from agentkeeper import (
    BaseStorage,
    EncryptedSQLiteStorage,
    SQLiteStorage,
    Storage,
    make_storage,
)
from agentkeeper.errors import ConfigurationError


@pytest.fixture(autouse=True)
def _reset_module_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agentkeeper, "_storage", None)


class TestBackwardCompat:
    def test_storage_is_sqlite_storage(self) -> None:
        # The legacy `Storage` symbol must still exist and be a
        # SQLiteStorage to keep v1.0 code working unchanged.
        assert Storage is SQLiteStorage

    def test_sqlite_storage_implements_base(self) -> None:
        assert issubclass(SQLiteStorage, BaseStorage)


class TestFactoryDefault:
    def test_default_returns_sqlite(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("AGENTKEEPER_STORAGE", raising=False)
        monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak.db"))
        storage = make_storage()
        assert isinstance(storage, SQLiteStorage)

    def test_explicit_sqlite(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak.db"))
        storage = make_storage(backend="sqlite")
        assert isinstance(storage, SQLiteStorage)

    def test_unknown_backend_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("AGENTKEEPER_STORAGE", "weird")
        with pytest.raises(ConfigurationError):
            make_storage()


class TestEncryptedStorageRequiresKey:
    def test_missing_key_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak.db"))
        monkeypatch.delenv("AGENTKEEPER_ENCRYPTION_KEY", raising=False)
        with pytest.raises(ConfigurationError):
            EncryptedSQLiteStorage()

    def test_invalid_key_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak.db"))
        with pytest.raises(ConfigurationError):
            EncryptedSQLiteStorage(encryption_key="not-a-real-key")


class TestEncryptedStorageRoundTrip:
    def test_save_then_load(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak.db"))
        key = EncryptedSQLiteStorage.generate_key()
        storage = EncryptedSQLiteStorage(encryption_key=key)

        # Use the public API to create + save an agent
        monkeypatch.setattr(agentkeeper, "_storage", storage)
        agent = agentkeeper.create(agent_id="encr-1", provider="mock")
        agent.fact("budget: 50k EUR")
        agent.principle("never share PII")
        agent.save()

        loaded = agentkeeper.load("encr-1", provider="mock")
        assert any("budget: 50k EUR" in f.content for f in loaded.facts)
        assert any(f.protected for f in loaded.facts)

    def test_payload_actually_encrypted_at_rest(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The on-disk JSON column must not contain plaintext content."""
        import sqlite3

        db = tmp_path / "ak.db"
        monkeypatch.setenv("AGENTKEEPER_DB", str(db))
        key = EncryptedSQLiteStorage.generate_key()
        storage = EncryptedSQLiteStorage(encryption_key=key)

        monkeypatch.setattr(agentkeeper, "_storage", storage)
        agent = agentkeeper.create(agent_id="encr-2", provider="mock")
        agent.fact("UNIQUE_PLAINTEXT_MARKER_42")
        agent.save()

        # Read raw row — content must be opaque
        conn = sqlite3.connect(str(db))
        try:
            row = conn.execute(
                "SELECT state_json FROM agents WHERE agent_id=?",
                ("encr-2",),
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        assert "UNIQUE_PLAINTEXT_MARKER_42" not in row[0]
        # Fernet payloads start with "gAAAAA" (version byte 0x80, base64)
        assert row[0].startswith("gAAAAA")

    def test_wrong_key_cannot_decrypt(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db_path = str(tmp_path / "ak.db")
        monkeypatch.setenv("AGENTKEEPER_DB", db_path)

        key_a = EncryptedSQLiteStorage.generate_key()
        storage_a = EncryptedSQLiteStorage(encryption_key=key_a)
        monkeypatch.setattr(agentkeeper, "_storage", storage_a)
        agent = agentkeeper.create(agent_id="encr-3", provider="mock")
        agent.fact("secret")
        agent.save()

        # New storage with a different key
        key_b = EncryptedSQLiteStorage.generate_key()
        storage_b = EncryptedSQLiteStorage(encryption_key=key_b)
        with pytest.raises(ConfigurationError):
            storage_b.load("encr-3")


class TestMixedModeMigration:
    def test_plaintext_rows_load_into_encrypted_storage(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Legacy plain-JSON rows must remain readable after switching."""
        db = str(tmp_path / "ak.db")
        monkeypatch.setenv("AGENTKEEPER_DB", db)

        # Step 1: write with plain SQLiteStorage
        plain = SQLiteStorage(db_path=db)
        monkeypatch.setattr(agentkeeper, "_storage", plain)
        agent = agentkeeper.create(agent_id="legacy-row", provider="mock")
        agent.fact("legacy data")
        agent.save()

        # Step 2: re-open with encrypted storage (same DB file)
        key = EncryptedSQLiteStorage.generate_key()
        encrypted = EncryptedSQLiteStorage(
            db_path=db, encryption_key=key
        )
        loaded = encrypted.load("legacy-row")
        assert loaded is not None
        assert any("legacy data" in f.content for f in loaded.memory_facts)


class TestPostgresStub:
    def test_construction_raises(self) -> None:
        from agentkeeper.storage import PostgresStorage

        with pytest.raises(NotImplementedError):
            PostgresStorage(dsn="postgresql://localhost/x")

    def test_factory_requires_dsn(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("AGENTKEEPER_STORAGE", "postgres")
        monkeypatch.delenv("AGENTKEEPER_POSTGRES_DSN", raising=False)
        with pytest.raises(ConfigurationError):
            make_storage()
