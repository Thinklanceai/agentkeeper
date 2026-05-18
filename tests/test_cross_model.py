"""Tests for cross-model context reconstruction via the CRE."""

from __future__ import annotations

from pathlib import Path

import pytest

import agentkeeper
from agentkeeper.benchmark.cross_model import run_cross_model_benchmark
from agentkeeper.translation.profiles import PromptFormat, get_profile


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak-test.db"))
    monkeypatch.setenv("AGENTKEEPER_EMBEDDING_PROVIDER", "mock")
    monkeypatch.setattr(agentkeeper, "_storage", None)


class TestCREUsesProfile:
    def test_anthropic_prompt_is_xml(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="anthropic")
        agent.set_identity(name="Aria")
        agent.principle("never share PII")
        agent.fact("budget: 50k EUR", importance=0.95)
        # Switch to mock for execution but ask for the anthropic-format prompt
        # by going through the CRE directly.
        from agentkeeper.cre.engine import CognitiveReconstructionEngine

        cre = CognitiveReconstructionEngine(agent._cso)
        prompt = cre.build_context_prompt("claude-sonnet-4-5-20250929", "Q?")
        assert "<agent_identity>" in prompt
        assert "<memory>" in prompt

    def test_openai_prompt_is_sections(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="openai")
        agent.set_identity(name="Aria")
        agent.fact("budget: 50k", importance=0.95)
        from agentkeeper.cre.engine import CognitiveReconstructionEngine

        cre = CognitiveReconstructionEngine(agent._cso)
        prompt = cre.build_context_prompt("gpt-4o", "Q?")
        assert "AGENT IDENTITY" in prompt
        assert "CURRENT TASK" in prompt

    def test_gemini_prompt_is_narrative(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="gemini")
        agent.set_identity(name="Aria", role="copilot")
        from agentkeeper.cre.engine import CognitiveReconstructionEngine

        cre = CognitiveReconstructionEngine(agent._cso)
        prompt = cre.build_context_prompt("gemini-1.5-pro", "Q?")
        assert "You are Aria" in prompt

    def test_ollama_prompt_is_minimal(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="ollama")
        agent.set_identity(name="Aria")
        agent.principle("never share PII")
        from agentkeeper.cre.engine import CognitiveReconstructionEngine

        cre = CognitiveReconstructionEngine(agent._cso)
        prompt = cre.build_context_prompt("llama3", "Q?")
        # Minimal format keeps it tight
        assert "AGENT IDENTITY" not in prompt
        assert "<agent_identity>" not in prompt


class TestProviderResolution:
    def test_claude_model_resolves_to_anthropic(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        from agentkeeper.cre.engine import CognitiveReconstructionEngine

        cre = CognitiveReconstructionEngine(agent._cso)
        assert cre._profile_for("claude-sonnet-4-5-20250929").format == PromptFormat.XML

    def test_gpt_model_resolves_to_openai(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        from agentkeeper.cre.engine import CognitiveReconstructionEngine

        cre = CognitiveReconstructionEngine(agent._cso)
        assert cre._profile_for("gpt-4o").format == PromptFormat.SECTIONS

    def test_gemini_model_resolves_to_gemini(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        from agentkeeper.cre.engine import CognitiveReconstructionEngine

        cre = CognitiveReconstructionEngine(agent._cso)
        assert cre._profile_for("gemini-2.0-flash").format == PromptFormat.NARRATIVE

    def test_unknown_model_falls_back_safely(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        from agentkeeper.cre.engine import CognitiveReconstructionEngine

        cre = CognitiveReconstructionEngine(agent._cso)
        # Should not crash; falls back to openai-style
        prompt = cre.build_context_prompt("some-weird-llm-9000", "Q?")
        assert "Q?" in prompt


class TestProfileBudgets:
    def test_unknown_model_uses_profile_budget(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        from agentkeeper.cre.engine import CognitiveReconstructionEngine

        cre = CognitiveReconstructionEngine(agent._cso)
        # Unknown model "claude-something-new" → anthropic profile budget
        budget = cre._budget_for("claude-something-new", None)
        assert budget == get_profile("anthropic").effective_context_tokens

    def test_explicit_budget_wins(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        from agentkeeper.cre.engine import CognitiveReconstructionEngine

        cre = CognitiveReconstructionEngine(agent._cso)
        assert cre._budget_for("claude-something-new", 9999) == 9999


class TestCrossModelBenchmark:
    def test_runs_for_all_default_providers(self) -> None:
        report = run_cross_model_benchmark()
        # 4 providers (anthropic, openai, gemini, ollama) expected
        assert len(report.results) == 4
        provider_names = {r.provider for r in report.results}
        assert {"anthropic", "openai", "gemini", "ollama"}.issubset(provider_names)

    def test_each_result_reports_format(self) -> None:
        report = run_cross_model_benchmark()
        formats = {r.provider: r.format for r in report.results}
        assert formats["anthropic"] == "xml"
        assert formats["openai"] == "sections"
        assert formats["gemini"] == "narrative"
        assert formats["ollama"] == "minimal"

    def test_recovery_rate_in_unit_range(self) -> None:
        report = run_cross_model_benchmark()
        for r in report.results:
            assert 0.0 <= r.recovery_rate <= 1.0

    def test_summary_renders_table(self) -> None:
        report = run_cross_model_benchmark()
        out = report.summary()
        assert "PROVIDER" in out
        assert "FORMAT" in out
