"""Tests for the AK-7 async agent API."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import agentkeeper
from agentkeeper import (
    AgentNotFoundError,
    UnknownProviderError,
    create_async,
    load_async,
)


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak-test.db"))
    monkeypatch.setenv("AGENTKEEPER_EMBEDDING_PROVIDER", "mock")
    monkeypatch.setattr(agentkeeper, "_storage", None)
    # Reset async storage too
    import agentkeeper.async_agent as aa

    monkeypatch.setattr(aa, "_async_storage", None)


class TestAsyncCreate:
    def test_create_async_returns_async_agent(self) -> None:
        agent = create_async(agent_id="a", provider="mock")
        assert agent.id == "a"
        assert agent.default_provider == "mock"

    def test_create_async_unknown_provider(self) -> None:
        with pytest.raises(UnknownProviderError):
            create_async(provider="fake")


class TestAsyncAsk:
    @pytest.mark.asyncio
    async def test_ask_returns_response(self) -> None:
        agent = create_async(agent_id="a", provider="mock")
        agent.remember("budget: 50k EUR", critical=True)
        response = await agent.ask("What is the budget?")
        assert "budget: 50k EUR" in response

    @pytest.mark.asyncio
    async def test_ask_unknown_provider(self) -> None:
        agent = create_async(agent_id="a", provider="mock")
        with pytest.raises(UnknownProviderError):
            await agent.ask("q", provider="fake-provider")

    @pytest.mark.asyncio
    async def test_parallel_asks(self) -> None:
        """Multiple parallel asks should all return without blocking."""
        agent = create_async(agent_id="a", provider="mock")
        agent.fact("x", importance=0.95)
        results = await asyncio.gather(
            agent.ask("q1"),
            agent.ask("q2"),
            agent.ask("q3"),
        )
        assert all("x" in r for r in results)


class TestAsyncRecall:
    @pytest.mark.asyncio
    async def test_recall_returns_pairs(self) -> None:
        agent = create_async(agent_id="a", provider="mock")
        agent.remember("budget: 50k EUR")
        agent.remember("client: Acme")
        results = await agent.recall("budget", top_k=2)
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_parallel_recalls(self) -> None:
        agent = create_async(agent_id="a", provider="mock")
        for i in range(10):
            agent.fact(f"fact number {i}")
        results = await asyncio.gather(
            agent.recall("fact 1"),
            agent.recall("fact 5"),
            agent.recall("fact 9"),
        )
        assert all(len(r) > 0 for r in results)


class TestAsyncCompress:
    @pytest.mark.asyncio
    async def test_compress_returns_report(self) -> None:
        agent = create_async(agent_id="a", provider="mock")
        agent.fact("budget: 50k", importance=0.5)
        agent.fact("budget: 50k", importance=0.5)
        report = await agent.compress()
        assert report.facts_before >= report.facts_after

    @pytest.mark.asyncio
    async def test_use_llm_now_supported(self) -> None:
        """AK-13: AsyncAgent.compress(use_llm=True) works via the
        agent's async adapter. With the MockAdapter we just confirm
        the call completes and returns a report — no provider drama."""
        agent = create_async(agent_id="a", provider="mock")
        agent.fact("budget: 50k EUR")
        agent.fact("budget: 50k EUR")  # duplicate to trigger consolidation
        report = await agent.compress(use_llm=True)
        assert report.facts_before >= report.facts_after


class TestAsyncPersistence:
    def test_save_then_load_async(self) -> None:
        agent = create_async(agent_id="persist", provider="mock")
        agent.set_identity(name="Aria", principles=["never share PII"])
        agent.remember("budget: 50k EUR", critical=True)
        agent.save()

        loaded = load_async("persist", provider="mock")
        assert loaded.identity.name == "Aria"
        assert any("budget: 50k EUR" in f.content for f in loaded.facts)

    def test_load_missing_raises(self) -> None:
        with pytest.raises(AgentNotFoundError):
            load_async("never-saved", provider="mock")

    def test_sync_and_async_share_storage(self) -> None:
        # Save via sync API, load via async API
        agent = agentkeeper.create(agent_id="shared", provider="mock")
        agent.remember("hello", critical=True)
        agent.save()

        loaded = load_async("shared", provider="mock")
        assert any("hello" in f.content for f in loaded.facts)


class TestAsyncHelpers:
    @pytest.mark.asyncio
    async def test_principle_protected(self) -> None:
        agent = create_async(agent_id="a", provider="mock")
        agent.principle("never share PII")
        last = agent.last_fact()
        assert last is not None
        assert last.protected is True

    @pytest.mark.asyncio
    async def test_event_episodic(self) -> None:
        agent = create_async(agent_id="a", provider="mock")
        agent.event("contract signed", when="2026-05-15")
        last = agent.last_fact()
        assert last is not None
        assert last.tier.value == "episodic"
        assert last.when is not None
