"""Tests for the Agent graph-memory surface."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import agentkeeper


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak.db"))
    monkeypatch.setenv("AGENTKEEPER_EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("AGENTKEEPER_VECTOR_INDEX", "in_memory")
    monkeypatch.setattr(agentkeeper, "_storage", None)


class TestAgentLink:
    def test_link_basic(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.link("Acme", "owns", "Globex")
        assert len(agent.triples) == 1
        assert agent.triples[0].subject == "Acme"

    def test_link_chains(self) -> None:
        agent = (
            agentkeeper.create(agent_id="a", provider="mock")
            .link("Acme", "owns", "Globex")
            .link("Alice", "works_at", "Acme")
            .fact("budget: 50k")
        )
        assert len(agent.triples) == 2
        assert len(agent.facts) == 1

    def test_link_with_confidence(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.link("Alice", "works_at", "Acme", confidence=0.7)
        assert agent.triples[0].confidence == 0.7

    def test_link_with_ttl(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.link("X", "p", "Y", ttl="1h")
        assert agent.triples[0].expires_at is not None


class TestAgentUnlink:
    def test_unlink_exact(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.link("A", "p", "B")
        agent.link("A", "q", "C")
        removed = agent.unlink(subject="A", predicate="p", object="B")
        assert removed == 1
        assert len(agent.triples) == 1
        assert agent.triples[0].predicate == "q"

    def test_unlink_by_subject(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.link("A", "p", "B")
        agent.link("A", "q", "C")
        agent.link("X", "p", "Y")
        removed = agent.unlink(subject="A")
        assert removed == 2
        assert len(agent.triples) == 1

    def test_unlink_protected_preserved(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.link("A", "p", "B", protected=True)
        agent.link("A", "p", "C")
        removed = agent.unlink(subject="A")
        assert removed == 1  # only the unprotected one
        assert len(agent.triples) == 1


class TestFindRelated:
    def test_real_use_case(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.link("Alice", "works_at", "Acme")
        agent.link("Acme", "owned_by", "Globex")
        agent.link("Globex", "located_in", "BE")
        agent.link("Bob", "works_at", "Acme")

        # Who/what is connected to Acme in 1 hop?
        result = agent.find_related("Acme", max_hops=1, direction="both")
        # Alice, Bob (via in), Globex (via out) — all at distance 1
        assert "Alice" in result
        assert "Bob" in result
        assert "Globex" in result
        assert result["Acme"] == 0

    def test_two_hop(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.link("Alice", "works_at", "Acme")
        agent.link("Acme", "owned_by", "Globex")
        # From Alice, 2 hops out should reach Globex
        result = agent.find_related("Alice", max_hops=2, direction="out")
        assert result.get("Globex") == 2


class TestPersistence:
    def test_triples_survive_save_load(self) -> None:
        agent = agentkeeper.create(agent_id="persist-graph", provider="mock")
        agent.link("Acme", "owns", "Globex", confidence=0.8)
        agent.link("Alice", "works_at", "Acme")
        agent.set_identity(name="Aria")
        agent.fact("budget: 50k")
        agent.save()

        loaded = agentkeeper.load("persist-graph", provider="mock")
        assert len(loaded.triples) == 2
        assert loaded.identity.name == "Aria"
        assert len(loaded.facts) == 1

        # Graph traversal works on the loaded agent
        result = loaded.find_related("Globex", max_hops=2, direction="in")
        assert "Acme" in result
        assert "Alice" in result


class TestPurgeExpired:
    def test_purges_expired_triples(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.link("A", "p", "B")
        agent.link("C", "p", "D", ttl="1h")
        # Force the second to be expired
        agent.triples[1].expires_at = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat()
        purged = agent.purge_expired()
        assert purged == 1
        assert len(agent.triples) == 1
        assert agent.triples[0].subject == "A"

    def test_protected_triples_preserved(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.link("A", "p", "B", protected=True)
        agent.triples[0].expires_at = (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).isoformat()
        assert agent.purge_expired() == 0


class TestGDPR:
    def test_export_includes_triples(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.link("Acme", "owns", "Globex")
        agent.fact("budget: 50k")
        export = agent.gdpr_export()
        assert "triples" in export
        assert len(export["triples"]) == 1

    def test_purge_removes_triples_by_default(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.link("A", "p", "B")
        agent.link("X", "p", "Y")
        agent.fact("ordinary")
        removed = agent.gdpr_purge()
        # 1 fact + 2 triples = 3
        assert removed == 3
        assert len(agent.triples) == 0
        assert len(agent.facts) == 0

    def test_purge_facts_only_option(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.link("A", "p", "B")
        agent.fact("ordinary")
        removed = agent.gdpr_purge(include_triples=False)
        assert removed == 1  # only the fact
        assert len(agent.triples) == 1


class TestHealth:
    def test_health_reports_graph(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.link("A", "p", "B")
        agent.link("A", "q", "C")
        h = agent.health()
        assert h["graph"]["triples"] == 2
        assert h["graph"]["entities"] == 3


class TestAsyncMirror:
    def test_async_link_and_find(self) -> None:
        agent = agentkeeper.create_async(agent_id="a", provider="mock")
        agent.link("Acme", "owns", "Globex")
        result = agent.find_related("Acme", max_hops=1)
        assert "Globex" in result

    def test_async_persistence(self) -> None:
        agent = agentkeeper.create_async(agent_id="ap", provider="mock")
        agent.link("X", "knows", "Y")
        agent.save()
        loaded = agentkeeper.load_async("ap", provider="mock")
        assert len(loaded.triples) == 1
