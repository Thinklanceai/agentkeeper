"""Tests for the AK-5 identity hardening guarantees.

These tests assert the core promise of cognitive continuity:
an agent's identity (name, role, principles, constraints, and any
protected fact) survives every form of compression, persistence,
and provider switch.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import agentkeeper
from agentkeeper.compression.contradiction import ContradictionConfig
from agentkeeper.compression.pipeline import CompressionConfig
from agentkeeper.cso.types import Fact


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak-test.db"))
    monkeypatch.setenv("AGENTKEEPER_EMBEDDING_PROVIDER", "mock")
    monkeypatch.setattr(agentkeeper, "_storage", None)


def _utc(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


class TestProtectedFlag:
    def test_fact_protected_default_false(self) -> None:
        f = Fact.create("hello")
        assert f.protected is False

    def test_fact_protected_forces_high_importance(self) -> None:
        f = Fact.create("never share PII", protected=True, importance=0.2)
        assert f.protected is True
        assert f.importance >= 0.95

    def test_principle_helper_sets_protected(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.principle("never share PII")
        last = agent.last_fact()
        assert last is not None
        assert last.protected is True
        assert last.importance >= 0.95

    def test_fact_helper_does_not_set_protected(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.fact("budget: 50k")
        last = agent.last_fact()
        assert last is not None
        assert last.protected is False


class TestProtectedSurvivesDecay:
    def test_protected_fact_never_decays(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.principle("never share PII")
        principle = agent.last_fact()
        assert principle is not None
        # Age it heavily
        principle.last_accessed_at = _utc(days_ago=10_000)
        agent.compress()
        assert principle.importance >= 0.95

    def test_critical_non_protected_also_immortal(self) -> None:
        """Backward compat: criticals without protected still don't decay."""
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.fact("budget: 50k EUR", importance=0.95)
        critical_fact = agent.last_fact()
        assert critical_fact is not None
        critical_fact.last_accessed_at = _utc(days_ago=10_000)
        agent.compress()
        assert critical_fact.importance >= 0.95


class TestProtectedSurvivesConsolidation:
    def test_protected_fact_never_merged_even_with_duplicates(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        # Two identical principles + one ordinary duplicate
        agent.principle("never share PII")
        agent.principle("never share PII")
        agent.fact("budget: 50k")
        agent.fact("budget: 50k")
        before_protected = sum(1 for f in agent.facts if f.protected)
        before_total = len(agent.facts)
        agent.compress()
        after_protected = sum(1 for f in agent.facts if f.protected)
        # Both protected facts remain — duplicates of principles are NOT merged.
        assert after_protected == before_protected
        # Only the non-protected duplicate was merged.
        assert len(agent.facts) < before_total


class TestProtectedSurvivesContradiction:
    def test_protected_fact_never_flagged(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.principle("never share PII without consent")
        # Add a fact that would normally contradict via polarity
        agent.fact("always share PII without consent")
        config = CompressionConfig(
            run_decay=False,
            run_consolidation=False,
            contradiction=ContradictionConfig(similarity_threshold=0.3),
        )
        agent.compress(config=config)
        # The principle is untouched
        principle = next(f for f in agent.facts if f.protected)
        assert "contradicted_by" not in principle.metadata
        assert principle.importance >= 0.95


class TestIdentitySurvivesEverything:
    def test_identity_after_save_load(self) -> None:
        agent = agentkeeper.create(agent_id="hard-id-1", provider="mock")
        agent.set_identity(
            name="Aria",
            role="EU broker copilot",
            principles=["never share PII", "always disclose conflicts"],
            constraints=["EU data only"],
        )
        agent.save()
        loaded = agentkeeper.load("hard-id-1", provider="mock")
        assert loaded.identity.name == "Aria"
        assert loaded.identity.role == "EU broker copilot"
        assert len(loaded.identity.principles) == 2
        assert len(loaded.identity.constraints) == 1

    def test_identity_after_compression(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.set_identity(
            name="Aria",
            role="copilot",
            principles=["P1", "P2"],
        )
        for i in range(20):
            agent.remember(f"note number {i}")
        agent.compress()
        assert agent.identity.name == "Aria"
        assert agent.identity.role == "copilot"
        assert agent.identity.principles == ["P1", "P2"]

    def test_identity_after_provider_switch(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.set_identity(name="Aria", role="copilot")
        agent.switch_provider("anthropic")
        agent.switch_provider("openai")
        agent.switch_provider("gemini")
        agent.switch_provider("mock")
        assert agent.identity.name == "Aria"
        assert agent.identity.role == "copilot"

    def test_identity_after_100_compressions(self) -> None:
        """Iterated compression should never erode identity."""
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.set_identity(
            name="Aria",
            role="copilot",
            principles=["never share PII"],
            constraints=["EU only"],
        )
        agent.principle("always confirm in writing")

        for i in range(100):
            agent.remember(f"transient note {i}")
            agent.compress()

        assert agent.identity.name == "Aria"
        assert agent.identity.role == "copilot"
        assert "never share PII" in agent.identity.principles
        assert "EU only" in agent.identity.constraints
        principles_remaining = sum(1 for f in agent.facts if f.protected)
        assert principles_remaining >= 1


class TestSetIdentityModes:
    def test_replace_is_default(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.set_identity(name="A", principles=["P1"])
        agent.set_identity(name="B", principles=["P2"])
        assert agent.identity.name == "B"
        assert agent.identity.principles == ["P2"]

    def test_merge_appends_new_principles(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.set_identity(name="Aria", principles=["P1"], constraints=["C1"])
        agent.set_identity(
            principles=["P2", "P3"],
            constraints=["C2"],
            merge=True,
        )
        assert agent.identity.name == "Aria"  # preserved
        assert agent.identity.principles == ["P1", "P2", "P3"]
        assert agent.identity.constraints == ["C1", "C2"]

    def test_merge_deduplicates(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.set_identity(principles=["P1", "P2"])
        agent.set_identity(principles=["P2", "P3"], merge=True)
        assert agent.identity.principles == ["P1", "P2", "P3"]


class TestIdentityAudit:
    def test_audit_reports_empty_identity(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        audit = agent.identity_audit()
        assert audit["identity"]["is_empty"] is True
        assert audit["protected_facts"]["count"] == 0

    def test_audit_reports_populated_identity(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.set_identity(
            name="Aria",
            role="copilot",
            principles=["P1", "P2"],
            constraints=["C1"],
        )
        agent.principle("X")
        agent.principle("Y")
        audit = agent.identity_audit()
        assert audit["identity"]["is_empty"] is False
        assert audit["identity"]["principles_count"] == 2
        assert audit["identity"]["constraints_count"] == 1
        assert audit["identity"]["token_cost"] > 0
        assert audit["protected_facts"]["count"] == 2
        assert "X" in audit["protected_facts"]["contents"]


class TestProtectedRoundTrip:
    def test_protected_flag_persists(self) -> None:
        agent = agentkeeper.create(agent_id="rt", provider="mock")
        agent.principle("never share PII")
        agent.save()
        loaded = agentkeeper.load("rt", provider="mock")
        assert loaded.facts[0].protected is True
