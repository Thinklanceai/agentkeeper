"""Tests for the AK-9 typed memory helpers and agent.health()."""

from __future__ import annotations

from pathlib import Path

import pytest

import agentkeeper
from agentkeeper import FactType


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak-test.db"))
    monkeypatch.setenv("AGENTKEEPER_EMBEDDING_PROVIDER", "mock")
    monkeypatch.setattr(agentkeeper, "_storage", None)


class TestTypedHelpers:
    def test_decision_sets_correct_type(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.decision("use Anthropic for production")
        last = agent.last_fact()
        assert last is not None
        assert last.fact_type == FactType.DECISION

    def test_preference_sets_correct_type(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.preference("prefer concise responses")
        last = agent.last_fact()
        assert last is not None
        assert last.fact_type == FactType.PREFERENCE

    def test_constraint_sets_correct_type(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.constraint("token budget 8000")
        last = agent.last_fact()
        assert last is not None
        assert last.fact_type == FactType.CONSTRAINT

    def test_relationship_sets_correct_type(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.relationship("Acme is owned by Globex")
        last = agent.last_fact()
        assert last is not None
        assert last.fact_type == FactType.RELATIONSHIP

    def test_task_state_sets_correct_type_and_tier(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.task_state("step 3 of 5 complete")
        last = agent.last_fact()
        assert last is not None
        assert last.fact_type == FactType.TASK_STATE
        assert last.tier.value == "working"

    def test_transient_sets_correct_type_and_tier(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.transient("intermediate value 42")
        last = agent.last_fact()
        assert last is not None
        assert last.fact_type == FactType.TRANSIENT
        assert last.tier.value == "working"

    def test_principle_marks_identity_type(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.principle("never share PII")
        last = agent.last_fact()
        assert last is not None
        assert last.fact_type == FactType.IDENTITY
        assert last.protected is True

    def test_event_marks_event_type(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.event("contract signed", when="2026-05-15")
        last = agent.last_fact()
        assert last is not None
        assert last.fact_type == FactType.EVENT

    def test_fact_keeps_generic_type(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.fact("budget: 50k EUR")
        last = agent.last_fact()
        assert last is not None
        assert last.fact_type == FactType.FACT


class TestFluent:
    def test_chain_typed_helpers(self) -> None:
        agent = (
            agentkeeper.create(agent_id="a", provider="mock")
            .decision("d1")
            .preference("p1")
            .constraint("c1")
        )
        types = {f.fact_type for f in agent.facts}
        assert FactType.DECISION in types
        assert FactType.PREFERENCE in types
        assert FactType.CONSTRAINT in types


class TestHealth:
    def test_health_empty_agent(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        h = agent.health()
        assert h["total_facts"] == 0
        assert h["critical_facts"] == 0
        assert h["protected_facts"] == 0
        assert h["contradicted_facts"] == 0
        assert h["identity"]["present"] is False

    def test_health_reports_type_distribution(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.decision("d1").decision("d2").preference("p1").transient("t1")
        h = agent.health()
        dist = h["fact_type_distribution"]
        assert dist["decision"] == 2
        assert dist["preference"] == 1
        assert dist["transient"] == 1

    def test_health_reports_tier_distribution(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.fact("x").event("y").task_state("z")
        h = agent.health()
        tiers = h["tier_distribution"]
        assert tiers["semantic"] >= 1
        assert tiers["episodic"] >= 1
        assert tiers["working"] >= 1

    def test_health_reports_identity(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.set_identity(
            name="Aria",
            role="copilot",
            principles=["p1"],
            constraints=["c1"],
        )
        h = agent.health()
        assert h["identity"]["present"] is True
        assert h["identity"]["name"] == "Aria"
        assert h["identity"]["principles_count"] == 1
        assert h["identity"]["constraints_count"] == 1

    def test_health_counts_protected_facts(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.principle("p")
        agent.principle("q")
        agent.fact("ordinary")
        h = agent.health()
        assert h["protected_facts"] == 2

    def test_health_reports_importance_stats(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.fact("low", importance=0.3)
        agent.fact("high", importance=0.9)
        h = agent.health()
        assert h["importance"]["max"] >= 0.9
        assert 0.0 < h["importance"]["mean"] < 1.0


class TestFactTypeRoundTrip:
    def test_fact_type_survives_save_load(self) -> None:
        agent = agentkeeper.create(agent_id="rt", provider="mock")
        agent.decision("d1")
        agent.preference("p1")
        agent.save()
        loaded = agentkeeper.load("rt", provider="mock")
        types = {f.fact_type for f in loaded.facts}
        assert FactType.DECISION in types
        assert FactType.PREFERENCE in types

    def test_legacy_fact_loads_as_fact_type(self) -> None:
        from agentkeeper.cso.types import CognitiveStateObject

        # Simulate a v1.0 serialised CSO (no fact_type field)
        legacy = {
            "agent_id": "legacy",
            "identity": {},
            "memory_facts": [
                {
                    "id": "f1",
                    "content": "legacy fact",
                    "tier": "semantic",
                    "importance": 0.5,
                    "critical": False,
                    "token_count": 3,
                    "created_at": "2026-04-01T00:00:00+00:00",
                    "last_accessed_at": "2026-04-01T00:00:00+00:00",
                    "access_count": 0,
                    "when": None,
                    "metadata": {},
                }
            ],
            "created_at": "2026-04-01T00:00:00+00:00",
            "updated_at": "2026-04-01T00:00:00+00:00",
        }
        cso = CognitiveStateObject.from_dict(legacy)
        assert cso.memory_facts[0].fact_type == FactType.FACT
