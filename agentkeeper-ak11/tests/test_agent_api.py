"""Tests for the public top-level API: agentkeeper.create / load / Agent."""

from __future__ import annotations

from pathlib import Path

import pytest

import agentkeeper
from agentkeeper import MockAdapter


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test gets its own AgentKeeper database, never polluting cwd."""
    monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak-test.db"))
    # Reset the module-level cached storage so it picks up the env var.
    monkeypatch.setattr(agentkeeper, "_storage", None)


class TestCreate:
    def test_creates_agent_with_explicit_id(self) -> None:
        agent = agentkeeper.create(agent_id="alice", provider="mock")
        assert agent.id == "alice"
        assert agent.default_provider == "mock"
        assert agent.facts == []

    def test_creates_agent_with_auto_id(self) -> None:
        agent = agentkeeper.create(provider="mock")
        assert agent.id  # non-empty uuid
        assert len(agent.id) == 36

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            agentkeeper.create(provider="not-a-real-provider")


class TestRememberAndAsk:
    def test_remember_appends_fact(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.remember("budget: 50k", critical=True)
        assert len(agent.facts) == 1
        assert agent.facts[0].critical is True

    def test_remember_is_fluent(self) -> None:
        agent = (
            agentkeeper.create(agent_id="fluent", provider="mock")
            .remember("a", critical=True)
            .remember("b")
            .remember("c")
        )
        assert len(agent.facts) == 3

    def test_ask_includes_memory_in_prompt(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.remember("budget: 50k EUR", critical=True)
        response = agent.ask("What is the budget?")
        assert "budget: 50k EUR" in response

    def test_ask_with_explicit_provider(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="anthropic")
        agent.remember("x", critical=True)
        # Even though default is anthropic, we override to mock.
        response = agent.ask("what?", provider="mock")
        assert "x" in response

    def test_forget_removes_fact(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.remember("a")
        agent.remember("b")
        fact_id = agent.facts[0].id
        agent.forget(fact_id)
        assert len(agent.facts) == 1
        assert agent.facts[0].content == "b"

    def test_forget_unknown_id_is_noop(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.remember("a")
        agent.forget("not-a-real-id")
        assert len(agent.facts) == 1


class TestSwitchProvider:
    def test_switch_changes_default(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.switch_provider("anthropic")
        assert agent.default_provider == "anthropic"

    def test_switch_preserves_facts(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.remember("survive me", critical=True)
        agent.switch_provider("openai")
        assert len(agent.facts) == 1

    def test_switch_unknown_provider_raises(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        with pytest.raises(ValueError):
            agent.switch_provider("fake-provider")


class TestPersistence:
    def test_save_then_load(self) -> None:
        agent = agentkeeper.create(agent_id="persisted", provider="mock")
        agent.remember("budget: 50k", critical=True)
        agent.remember("note")
        agent.save()

        loaded = agentkeeper.load("persisted", provider="mock")
        assert loaded.id == "persisted"
        assert len(loaded.facts) == 2

    def test_load_missing_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            agentkeeper.load("never-saved", provider="mock")

    def test_delete_removes_agent(self) -> None:
        agent = agentkeeper.create(agent_id="ephemeral", provider="mock")
        agent.save()
        assert "ephemeral" in agentkeeper.list_agents()
        agentkeeper.delete("ephemeral")
        assert "ephemeral" not in agentkeeper.list_agents()

    def test_list_agents_empty_by_default(self) -> None:
        assert agentkeeper.list_agents() == []

    def test_list_agents_returns_all(self) -> None:
        for aid in ["a", "b", "c"]:
            agentkeeper.create(agent_id=aid, provider="mock").save()
        assert set(agentkeeper.list_agents()) == {"a", "b", "c"}


class TestStats:
    def test_stats_returns_dict(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.remember("c1", critical=True)
        agent.remember("nc1")
        stats = agent.stats()
        assert stats["total_facts"] == 2
        assert stats["critical_total"] == 1


class TestAdapterCaching:
    def test_adapter_is_cached(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.remember("x")
        agent.ask("q1")
        adapter1 = agent._adapter_cache["mock"]
        agent.ask("q2")
        adapter2 = agent._adapter_cache["mock"]
        assert adapter1 is adapter2  # same instance reused

    def test_isinstance_mockadapter(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.remember("x")
        agent.ask("q")
        assert isinstance(agent._adapter_cache["mock"], MockAdapter)


class TestRepr:
    def test_repr_includes_id_and_count(self) -> None:
        agent = agentkeeper.create(agent_id="reprtest", provider="mock")
        agent.remember("a").remember("b")
        r = repr(agent)
        assert "reprtest" in r
        # Either "facts=2" or any way the count is exposed
        assert "facts=2" in r or "2 facts" in r


class TestAgentDataclassAccess:
    def test_facts_is_a_list(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        assert isinstance(agent.facts, list)
