"""Tests for LangChain and CrewAI framework integration stubs.

These tests do not require LangChain or CrewAI to be installed — the
integrations are deliberately framework-free at the Python level.
They just produce strings that frameworks can consume.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import agentkeeper
from agentkeeper.integrations.crewai import (
    CrewAICognitiveBackstory,
    crewai_backstory,
)
from agentkeeper.integrations.langchain import (
    LangChainCognitiveProvider,
    langchain_system_prompt,
)


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak.db"))
    monkeypatch.setenv("AGENTKEEPER_EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("AGENTKEEPER_VECTOR_INDEX", "in_memory")
    monkeypatch.setattr(agentkeeper, "_storage", None)


def _seed_agent(agent_id: str = "a") -> agentkeeper.Agent:
    agent = agentkeeper.create(agent_id=agent_id, provider="mock")
    agent.set_identity(
        name="Aria",
        role="EU broker copilot",
        principles=["never share PII"],
    )
    agent.fact("budget: 50k EUR")
    agent.decision("use Anthropic for production")
    return agent


class TestLangChainHelpers:
    def test_function_returns_non_empty_string(self) -> None:
        agent = _seed_agent()
        prompt = langchain_system_prompt(agent, message="What's the budget?")
        assert isinstance(prompt, str)
        assert len(prompt) > 50
        # Identity should be present somewhere
        assert "Aria" in prompt

    def test_function_respects_model_override(self) -> None:
        agent = _seed_agent()
        claude_prompt = langchain_system_prompt(
            agent, message="?", model="claude-sonnet-4-5-20250929"
        )
        gpt_prompt = langchain_system_prompt(
            agent, message="?", model="gpt-4o"
        )
        # Claude format = XML, GPT format = sections — distinct strings
        assert "<agent_identity>" in claude_prompt
        assert "<agent_identity>" not in gpt_prompt

    def test_callable_provider(self) -> None:
        agent = _seed_agent()
        provider = LangChainCognitiveProvider(agent, model="gpt-4o")
        p1 = provider("question one")
        p2 = provider("question two")
        assert isinstance(p1, str) and isinstance(p2, str)
        assert "Aria" in p1
        assert "Aria" in p2


class TestCrewAIHelpers:
    def test_function_returns_non_empty_string(self) -> None:
        agent = _seed_agent()
        backstory = crewai_backstory(agent, task="Summarise Q3 results")
        assert isinstance(backstory, str)
        assert len(backstory) > 50
        assert "Aria" in backstory

    def test_callable_backstory(self) -> None:
        agent = _seed_agent()
        backstory = CrewAICognitiveBackstory(agent, model="gpt-4o")
        b1 = backstory("task one")
        b2 = backstory("task two")
        assert isinstance(b1, str)
        assert isinstance(b2, str)


class TestNoExternalDeps:
    def test_langchain_module_importable_without_langchain(self) -> None:
        # If LangChain were a hard dep, importing this module would fail
        # in environments without it. Our integration is dep-free.
        import agentkeeper.integrations.langchain  # noqa: F401

    def test_crewai_module_importable_without_crewai(self) -> None:
        import agentkeeper.integrations.crewai  # noqa: F401
