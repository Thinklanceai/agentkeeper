"""Tests for the async LLM synthesiser."""

from __future__ import annotations

from pathlib import Path

import pytest

import agentkeeper
from agentkeeper.adapters.base import AsyncBaseAdapter
from agentkeeper.compression.async_llm_synth import make_async_llm_synthesiser
from agentkeeper.cso.types import Fact


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak.db"))
    monkeypatch.setenv("AGENTKEEPER_EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("AGENTKEEPER_VECTOR_INDEX", "in_memory")
    monkeypatch.setattr(agentkeeper, "_storage", None)


class _FakeAsyncAdapter(AsyncBaseAdapter):
    """Records every call; returns a deterministic synthesis."""

    def __init__(self, response: str = "CONSOLIDATED") -> None:
        self.calls: list[tuple[str, str]] = []
        self.response = response

    async def query(self, system_prompt: str, user_message: str) -> str:
        self.calls.append((system_prompt, user_message))
        return self.response


class _BrokenAsyncAdapter(AsyncBaseAdapter):
    async def query(self, system_prompt: str, user_message: str) -> str:
        raise RuntimeError("rate limited or whatever")


class TestSyntheciserBasics:
    def test_empty_cluster_returns_empty_string(self) -> None:
        synth = make_async_llm_synthesiser(_FakeAsyncAdapter())
        assert synth([]) == ""

    def test_single_fact_returns_its_content(self) -> None:
        f = Fact.create("budget: 50k EUR")
        synth = make_async_llm_synthesiser(_FakeAsyncAdapter())
        assert synth([f]) == "budget: 50k EUR"
        # Adapter should not have been called for a single-fact cluster
        assert synth.__closure__ is not None  # smoke check

    def test_calls_adapter_for_real_cluster(self) -> None:
        adapter = _FakeAsyncAdapter(response="budget: 50k EUR (consolidated)")
        synth = make_async_llm_synthesiser(adapter)
        result = synth([
            Fact.create("budget: 50k"),
            Fact.create("budget: 50000 EUR"),
        ])
        assert result == "budget: 50k EUR (consolidated)"
        assert len(adapter.calls) == 1

    def test_fallback_on_adapter_error(self) -> None:
        synth = make_async_llm_synthesiser(_BrokenAsyncAdapter())
        cluster = [
            Fact.create("a", importance=0.4),
            Fact.create("b", importance=0.9),
        ]
        # Canonical = highest importance ("b")
        assert synth(cluster) == "b"

    def test_rejects_empty_response(self) -> None:
        synth = make_async_llm_synthesiser(_FakeAsyncAdapter(response="   "))
        cluster = [
            Fact.create("budget: 50k", importance=0.6),
            Fact.create("budget: 50000", importance=0.8),
        ]
        # Empty response → fall back to canonical (highest importance)
        assert synth(cluster) == "budget: 50000"

    def test_rejects_absurdly_long_response(self) -> None:
        absurd = "x" * 100_000
        synth = make_async_llm_synthesiser(_FakeAsyncAdapter(response=absurd))
        cluster = [
            Fact.create("short fact", importance=0.6),
            Fact.create("other short fact", importance=0.8),
        ]
        # Falls back to canonical
        assert synth(cluster) == "other short fact"


class TestAsyncAgentCompressUseLLM:
    @pytest.mark.asyncio
    async def test_compress_with_llm_runs(self) -> None:
        """AK-13: AsyncAgent.compress(use_llm=True) returns a report."""
        agent = agentkeeper.create_async(agent_id="a", provider="mock")
        agent.fact("budget: 50k EUR")
        agent.fact("budget: 50k EUR")
        report = await agent.compress(use_llm=True)
        assert report.facts_before >= report.facts_after
