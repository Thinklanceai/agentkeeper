"""Tests for the vector index factory and sqlite-vec backend.

The sqlite-vec tests are skipped automatically when the extension is
not installed/loadable in this environment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentkeeper.semantic.factory import make_vector_index
from agentkeeper.semantic.index import InMemoryVectorIndex
from agentkeeper.semantic.sqlite_vec_index import (
    SqliteVecIndex,
    _sanitise_agent_id,
    is_available,
)

# Skip the whole module if sqlite-vec is not installed.
pytestmark_vec = pytest.mark.skipif(
    not is_available(),
    reason="sqlite-vec extension not available in this environment",
)


class TestSanitiseAgentId:
    def test_alnum_passthrough(self) -> None:
        assert _sanitise_agent_id("agent_42") == "agent_42"

    def test_special_chars_replaced(self) -> None:
        assert _sanitise_agent_id("a-b@c.d") == "a_b_c_d"

    def test_long_id_truncated(self) -> None:
        long = "x" * 200
        assert len(_sanitise_agent_id(long)) <= 60

    def test_empty_falls_back(self) -> None:
        assert _sanitise_agent_id("@@@") == "_" * 3 or _sanitise_agent_id("@@@") != ""


class TestFactoryRespectsEnv:
    def test_in_memory_forced(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("AGENTKEEPER_VECTOR_INDEX", "in_memory")
        monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak.db"))
        idx = make_vector_index(agent_id="a", dimension=4)
        assert isinstance(idx, InMemoryVectorIndex)

    def test_sqlite_vec_missing_raises_when_forced(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if is_available():
            pytest.skip("sqlite-vec is available; cannot test the forced-missing path")
        monkeypatch.setenv("AGENTKEEPER_VECTOR_INDEX", "sqlite_vec")
        monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak.db"))
        with pytest.raises(RuntimeError):
            make_vector_index(agent_id="a", dimension=4)

    def test_auto_falls_back_to_in_memory_when_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if is_available():
            pytest.skip("sqlite-vec is available; auto would pick it")
        monkeypatch.delenv("AGENTKEEPER_VECTOR_INDEX", raising=False)
        monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak.db"))
        idx = make_vector_index(agent_id="a", dimension=4)
        assert isinstance(idx, InMemoryVectorIndex)


@pytestmark_vec
class TestSqliteVecIndexBasics:
    def test_create_and_size(self, tmp_path: Path) -> None:
        idx = SqliteVecIndex(
            agent_id="a", dimension=4, db_path=str(tmp_path / "ak.db")
        )
        try:
            assert idx.size() == 0
        finally:
            idx.close()

    def test_upsert_then_search(self, tmp_path: Path) -> None:
        idx = SqliteVecIndex(
            agent_id="a", dimension=3, db_path=str(tmp_path / "ak.db")
        )
        try:
            idx.upsert("f1", [1.0, 0.0, 0.0])
            idx.upsert("f2", [0.0, 1.0, 0.0])
            idx.upsert("f3", [0.0, 0.0, 1.0])
            results = idx.search([1.0, 0.0, 0.0], top_k=2)
            assert len(results) >= 1
            assert results[0][0] == "f1"
            # cosine score for identical vector ≈ 1.0
            assert results[0][1] >= 0.99
        finally:
            idx.close()

    def test_delete(self, tmp_path: Path) -> None:
        idx = SqliteVecIndex(
            agent_id="a", dimension=3, db_path=str(tmp_path / "ak.db")
        )
        try:
            idx.upsert("f1", [1.0, 0.0, 0.0])
            assert idx.size() == 1
            idx.delete("f1")
            assert idx.size() == 0
            # idempotent
            idx.delete("nonexistent")
        finally:
            idx.close()

    def test_dimension_mismatch_rejected(self, tmp_path: Path) -> None:
        idx = SqliteVecIndex(
            agent_id="a", dimension=4, db_path=str(tmp_path / "ak.db")
        )
        try:
            with pytest.raises(ValueError):
                idx.upsert("f1", [1.0, 0.0, 0.0])
        finally:
            idx.close()

    def test_persists_across_index_instances(self, tmp_path: Path) -> None:
        db = str(tmp_path / "ak.db")
        idx1 = SqliteVecIndex(agent_id="persisted", dimension=3, db_path=db)
        idx1.upsert("f1", [1.0, 0.0, 0.0])
        idx1.upsert("f2", [0.0, 1.0, 0.0])
        idx1.close()

        # Re-open: same data must be there
        idx2 = SqliteVecIndex(agent_id="persisted", dimension=3, db_path=db)
        try:
            assert idx2.size() == 2
            results = idx2.search([1.0, 0.0, 0.0], top_k=1)
            assert results[0][0] == "f1"
        finally:
            idx2.close()

    def test_per_agent_isolation(self, tmp_path: Path) -> None:
        db = str(tmp_path / "ak.db")
        a = SqliteVecIndex(agent_id="alpha", dimension=3, db_path=db)
        b = SqliteVecIndex(agent_id="beta", dimension=3, db_path=db)
        try:
            a.upsert("f1", [1.0, 0.0, 0.0])
            b.upsert("f2", [0.0, 1.0, 0.0])
            assert a.size() == 1
            assert b.size() == 1
            # Search in a should not see f2
            results = a.search([0.0, 1.0, 0.0], top_k=5)
            assert all(fid != "f2" for fid, _ in results)
        finally:
            a.close()
            b.close()

    def test_clear_resets_index(self, tmp_path: Path) -> None:
        idx = SqliteVecIndex(
            agent_id="a", dimension=3, db_path=str(tmp_path / "ak.db")
        )
        try:
            idx.upsert("f1", [1.0, 0.0, 0.0])
            idx.upsert("f2", [0.0, 1.0, 0.0])
            idx.clear()
            assert idx.size() == 0
        finally:
            idx.close()


@pytestmark_vec
class TestRecallerWithPersistentIndex:
    def test_recall_survives_recaller_recreate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The hash cache is per-recaller, but the vector index is persistent.

        A second recaller created over the same agent's persistent index
        sees its data; it just won't know which content hashes correspond
        to which fact ids until `index_all()` runs once more (which
        re-embeds and re-upserts — idempotent in the persistent index).
        """
        import agentkeeper

        monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak.db"))
        monkeypatch.setenv("AGENTKEEPER_VECTOR_INDEX", "sqlite_vec")
        monkeypatch.setenv("AGENTKEEPER_EMBEDDING_PROVIDER", "mock")
        monkeypatch.setattr(agentkeeper, "_storage", None)

        agent = agentkeeper.create(agent_id="persist1", provider="mock")
        agent.fact("budget: 50k EUR")
        agent.fact("client: Acme")
        hits = agent.recall("budget", top_k=2)
        assert len(hits) > 0
        agent.save()

        # Wipe in-process state, simulate a restart
        agent2 = agentkeeper.load("persist1", provider="mock")
        hits2 = agent2.recall("budget", top_k=2)
        assert len(hits2) > 0
