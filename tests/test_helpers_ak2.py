"""Tests for the AK-2 cognitive helpers and identity injection."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

import agentkeeper
from agentkeeper import MemoryTier


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak-test.db"))
    monkeypatch.setattr(agentkeeper, "_storage", None)


class TestFactHelper:
    def test_fact_creates_semantic_fact(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.fact("budget: 50k EUR")
        added = agent.last_fact()
        assert added is not None
        assert added.tier == MemoryTier.SEMANTIC
        assert added.importance == 0.7

    def test_fact_accepts_custom_importance(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.fact("budget: 50k", importance=0.95)
        last = agent.last_fact()
        assert last is not None
        assert last.importance == 0.95
        assert last.critical is True


class TestEventHelper:
    def test_event_creates_episodic_fact(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.event("client signed contract")
        last = agent.last_fact()
        assert last is not None
        assert last.tier == MemoryTier.EPISODIC

    def test_event_stores_when(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        dt = datetime(2026, 5, 15, 14, 0, 0)
        agent.event("contract signed", when=dt)
        last = agent.last_fact()
        assert last is not None
        assert last.when is not None
        assert "2026-05-15" in last.when


class TestPrincipleHelper:
    def test_principle_creates_very_high_importance(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.principle("never share PII without consent")
        last = agent.last_fact()
        assert last is not None
        assert last.importance >= 0.9
        assert last.critical is True


class TestRememberInference:
    def test_temporal_content_inferred_as_episodic(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.remember("yesterday the deployment failed")
        last = agent.last_fact()
        assert last is not None
        assert last.tier == MemoryTier.EPISODIC

    def test_stable_content_inferred_as_semantic(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.remember("budget: 50000 EUR")
        last = agent.last_fact()
        assert last is not None
        assert last.tier == MemoryTier.SEMANTIC

    def test_explicit_tier_overrides_inference(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.remember("yesterday it rained", tier="semantic")
        last = agent.last_fact()
        assert last is not None
        assert last.tier == MemoryTier.SEMANTIC

    def test_invalid_tier_raises(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        with pytest.raises(ValueError, match="Unknown tier"):
            agent.remember("test", tier="not-a-tier")


class TestSetIdentity:
    def test_set_identity_stores_name_and_role(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.set_identity(name="Aria", role="EU broker copilot")
        assert agent.identity.name == "Aria"
        assert agent.identity.role == "EU broker copilot"

    def test_identity_appears_in_prompt(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.set_identity(name="Aria", role="copilot")
        agent.remember("budget: 50k", critical=True)
        response = agent.ask("status?")
        # Mock adapter echoes the system prompt → identity must be in it
        assert "Aria" in response
        assert "copilot" in response

    def test_identity_survives_save_load(self) -> None:
        agent = agentkeeper.create(agent_id="identity-test", provider="mock")
        agent.set_identity(
            name="Aria",
            role="EU broker copilot",
            principles=["never share PII"],
            constraints=["EU data only"],
        )
        agent.save()

        loaded = agentkeeper.load("identity-test", provider="mock")
        assert loaded.identity.name == "Aria"
        assert loaded.identity.role == "EU broker copilot"
        assert "never share PII" in loaded.identity.principles
        assert "EU data only" in loaded.identity.constraints

    def test_identity_token_cost_reflected_in_stats(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        before = agent.stats()
        assert before["identity_present"] is False
        assert before["identity_token_cost"] == 0

        agent.set_identity(name="Aria", role="EU broker copilot")
        after = agent.stats()
        assert after["identity_present"] is True
        assert after["identity_token_cost"] > 0


class TestStatsExtended:
    def test_stats_include_tier_breakdown(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.fact("budget: 50k", importance=0.95)
        agent.event("client refused offer A")
        agent.principle("never share PII")
        stats = agent.stats()
        breakdown = stats["tier_breakdown"]
        assert breakdown["semantic"] >= 2  # fact + principle
        assert breakdown["episodic"] >= 1  # event
