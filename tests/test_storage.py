"""Tests for the SQLite storage backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentkeeper.cso.types import CognitiveStateObject
from agentkeeper.storage.sqlite_store import Storage


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    db = tmp_path / "test.db"
    return Storage(db_path=str(db))


class TestStorage:
    def test_save_then_load_roundtrip(self, storage: Storage) -> None:
        cso = CognitiveStateObject.create(agent_id="a1")
        cso.add_fact("budget: 50k", critical=True)
        cso.add_fact("note")
        storage.save(cso)

        loaded = storage.load("a1")
        assert loaded is not None
        assert loaded.agent_id == "a1"
        assert len(loaded.memory_facts) == 2
        assert loaded.memory_facts[0].critical is True

    def test_load_missing_returns_none(self, storage: Storage) -> None:
        assert storage.load("does-not-exist") is None

    def test_save_overwrites_existing(self, storage: Storage) -> None:
        cso = CognitiveStateObject.create(agent_id="a1")
        cso.add_fact("v1")
        storage.save(cso)

        cso.add_fact("v2")
        storage.save(cso)

        loaded = storage.load("a1")
        assert loaded is not None
        assert len(loaded.memory_facts) == 2

    def test_delete_removes_agent(self, storage: Storage) -> None:
        cso = CognitiveStateObject.create(agent_id="a1")
        storage.save(cso)
        assert storage.load("a1") is not None

        storage.delete("a1")
        assert storage.load("a1") is None

    def test_delete_missing_is_noop(self, storage: Storage) -> None:
        # should not raise
        storage.delete("never-existed")

    def test_list_agent_ids(self, storage: Storage) -> None:
        for aid in ["a", "b", "c"]:
            storage.save(CognitiveStateObject.create(agent_id=aid))

        ids = storage.list_agent_ids()
        assert set(ids) == {"a", "b", "c"}

    def test_list_empty_storage(self, storage: Storage) -> None:
        assert storage.list_agent_ids() == []


class TestStoragePathResolution:
    def test_explicit_path_wins_over_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_db = tmp_path / "env.db"
        explicit_db = tmp_path / "explicit.db"
        monkeypatch.setenv("AGENTKEEPER_DB", str(env_db))

        s = Storage(db_path=str(explicit_db))
        s.save(CognitiveStateObject.create(agent_id="x"))

        assert explicit_db.exists()
        assert not env_db.exists()

    def test_env_var_is_picked_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_db = tmp_path / "from-env.db"
        monkeypatch.setenv("AGENTKEEPER_DB", str(env_db))

        s = Storage()
        s.save(CognitiveStateObject.create(agent_id="x"))

        assert env_db.exists()
