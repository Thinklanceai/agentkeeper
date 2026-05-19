"""Tests for the AK-3 Agent.recall() API."""

from __future__ import annotations

from pathlib import Path

import pytest

import agentkeeper
from agentkeeper import MockEmbeddingProvider


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak-test.db"))
    monkeypatch.setenv("AGENTKEEPER_EMBEDDING_PROVIDER", "mock")
    monkeypatch.setattr(agentkeeper, "_storage", None)


class TestAgentRecall:
    def test_recall_empty_agent(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        assert agent.recall("anything") == []

    def test_recall_returns_top_k(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.remember("budget: 50000 EUR")
        agent.remember("client: Acme Corporation")
        agent.remember("deploy target: AWS")
        results = agent.recall("budget", top_k=2)
        assert len(results) <= 2

    def test_recall_exact_match_returns_target_first(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.remember("client: Acme Corporation")
        agent.remember("budget: 50000 EUR")
        agent.remember("technology stack: Python")
        results = agent.recall("client: Acme Corporation", top_k=1)
        assert len(results) == 1
        fact, score = results[0]
        assert "Acme" in fact.content
        assert score > 0.99

    def test_recall_respects_min_score(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.remember("client: Acme")
        # With min_score very high, almost nothing matches
        results = agent.recall("entirely unrelated", top_k=5, min_score=0.99)
        assert len(results) <= 1

    def test_recall_after_forget(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.remember("budget: 50k EUR")
        target_id = agent.last_fact().id  # type: ignore[union-attr]
        agent.recall("budget")  # triggers indexing
        agent.forget(target_id)
        results = agent.recall("budget", top_k=5)
        assert all(f.id != target_id for f, _ in results)


class TestEmbeddingProviderOverride:
    def test_set_embedding_provider_swaps_recaller(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.remember("budget: 50k")
        agent.recall("anything")  # creates initial recaller

        agent.set_embedding_provider(MockEmbeddingProvider(dimension=128))
        agent.recall("anything")  # should rebuild with new dimension
        # No exception, no dimension mismatch

    def test_create_accepts_embedding_provider(self) -> None:
        agent = agentkeeper.create(
            agent_id="a",
            provider="mock",
            embedding_provider=MockEmbeddingProvider(dimension=32),
        )
        agent.remember("x")
        results = agent.recall("x", top_k=1)
        assert len(results) == 1


class TestSemanticBoostInAsk:
    def test_ask_does_not_require_recaller(self) -> None:
        # When no recall() has been triggered, ask still works without
        # embeddings (baseline reconstruction).
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.remember("budget: 50k EUR", critical=True)
        response = agent.ask("status?")
        assert "budget: 50k EUR" in response

    def test_ask_uses_recaller_after_explicit_setup(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.remember("budget: 50k EUR")
        agent.remember("favourite colour: blue")
        # Trigger recaller initialisation
        agent.recall("anything")
        # Now ask — it should not crash and should include some facts
        response = agent.ask("What is the budget?")
        assert "budget: 50k EUR" in response
