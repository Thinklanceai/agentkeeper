"""Tests for the Cognitive Reconstruction Engine."""

from __future__ import annotations

import pytest

from agentkeeper.cre.engine import CognitiveReconstructionEngine, estimate_tokens
from agentkeeper.cso.types import CognitiveStateObject


class TestEstimateTokens:
    def test_empty_string_returns_at_least_one(self) -> None:
        assert estimate_tokens("") == 1

    def test_short_string(self) -> None:
        assert estimate_tokens("ab") == 1

    def test_approximation_rule(self) -> None:
        # Rule of thumb: ~1 token per 4 chars
        assert estimate_tokens("a" * 40) == 10
        assert estimate_tokens("a" * 100) == 25


class TestPrioritize:
    def test_empty_cso_returns_empty(self) -> None:
        cso = CognitiveStateObject.create(agent_id="empty")
        cre = CognitiveReconstructionEngine(cso)
        assert cre.prioritize("gpt-4-turbo") == []

    def test_single_fact_within_budget(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        cso.add_fact("budget: 50k")
        cre = CognitiveReconstructionEngine(cso)
        selected = cre.prioritize("gpt-4-turbo")
        assert len(selected) == 1

    def test_critical_facts_come_first(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        cso.add_fact("non critical 1")
        cso.add_fact("CRITICAL FACT", critical=True)
        cso.add_fact("non critical 2")
        cre = CognitiveReconstructionEngine(cso)
        selected = cre.prioritize("gpt-4-turbo", max_tokens=1000)
        assert selected[0].critical is True
        assert selected[0].content == "CRITICAL FACT"

    def test_tight_budget_forces_critical_inclusion(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        # Many non-critical facts that fill the budget
        for i in range(20):
            cso.add_fact(f"non critical fact number {i} with some content")
        # One critical fact added last
        cso.add_fact("CRITICAL: must survive eviction", critical=True)

        cre = CognitiveReconstructionEngine(cso)
        # Use a small budget that cannot fit everything
        selected = cre.prioritize("gpt-4-turbo", max_tokens=40)

        critical_in_selection = [f for f in selected if f.critical]
        assert len(critical_in_selection) == 1
        assert critical_in_selection[0].content == "CRITICAL: must survive eviction"

    def test_uses_default_budget_when_model_unknown(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        for i in range(5):
            cso.add_fact(f"fact {i}")
        cre = CognitiveReconstructionEngine(cso)
        selected = cre.prioritize("unknown-model-9000")
        # default budget is large enough to fit all 5 facts
        assert len(selected) == 5

    def test_explicit_budget_overrides_model_limit(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        for _ in range(20):
            cso.add_fact("a" * 100)  # ~25 tokens each
        cre = CognitiveReconstructionEngine(cso)
        # Tight explicit budget
        selected = cre.prioritize("gpt-4-turbo", max_tokens=50)
        # We expect at most 2 facts (each ~25 tokens)
        assert len(selected) <= 2

    def test_critical_eviction_displaces_largest_non_critical(self) -> None:
        # Goal: force the eviction path inside prioritize().
        # We seed many small non-criticals AND one big non-critical that should be evicted
        # so the critical can fit.
        cso = CognitiveStateObject.create(agent_id="a")
        cso.add_fact("BIG NON-CRITICAL " + "x" * 200)  # ~50 tokens
        for i in range(10):
            cso.add_fact(f"nc{i}")  # tiny
        cso.add_fact("CRITICAL MUST SURVIVE THE EVICTION " * 3, critical=True)

        cre = CognitiveReconstructionEngine(cso)
        # Budget that forces eviction of the big non-critical for the critical to fit
        selected = cre.prioritize("gpt-4-turbo", max_tokens=80)
        critical = [f for f in selected if f.critical]
        assert len(critical) == 1
        # The big non-critical should have been evicted
        evicted_correctly = not any(
            "BIG NON-CRITICAL" in f.content for f in selected
        )
        assert evicted_correctly

    def test_critical_dropped_when_no_non_critical_to_evict(self) -> None:
        # Two criticals, both bigger than the budget — the second one should be dropped
        # because there is nothing non-critical to evict.
        cso = CognitiveStateObject.create(agent_id="a")
        cso.add_fact("first critical " + "x" * 200, critical=True)
        cso.add_fact("second critical " + "y" * 200, critical=True)
        cre = CognitiveReconstructionEngine(cso)
        selected = cre.prioritize("gpt-4-turbo", max_tokens=60)
        # Only one critical should fit; the other is gracefully dropped
        assert len(selected) <= 1


class TestBuildContextPrompt:
    def test_empty_memory_renders_task(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        cre = CognitiveReconstructionEngine(cso)
        prompt = cre.build_context_prompt("gpt-4-turbo", "Hello?")
        # The new prompt structure mentions an empty memory section explicitly.
        assert "Hello?" in prompt
        assert "MEMORY" in prompt

    def test_includes_facts_in_prompt(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        cso.add_fact("budget: 50k", critical=True)
        cre = CognitiveReconstructionEngine(cso)
        prompt = cre.build_context_prompt("gpt-4-turbo", "What is the budget?")
        assert "budget: 50k" in prompt
        # Critical facts are marked with a star
        assert "★" in prompt
        assert "What is the budget?" in prompt


class TestReconstructionStats:
    def test_stats_structure(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        cso.add_fact("c1", critical=True)
        cso.add_fact("nc1")
        cre = CognitiveReconstructionEngine(cso)
        stats = cre.reconstruction_stats("gpt-4-turbo")
        # Legacy v0.1 keys must still be present
        legacy_keys = {
            "total_facts",
            "selected_facts",
            "critical_total",
            "critical_selected",
            "critical_recovery_rate",
            "tokens_used",
            "token_budget",
        }
        assert legacy_keys.issubset(set(stats.keys()))

    def test_critical_recovery_rate_is_one_when_all_fit(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        cso.add_fact("c1", critical=True)
        cso.add_fact("c2", critical=True)
        cre = CognitiveReconstructionEngine(cso)
        stats = cre.reconstruction_stats("gpt-4-turbo")
        assert stats["critical_recovery_rate"] == 1.0

    def test_critical_recovery_rate_is_zero_when_no_critical(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        cso.add_fact("nc1")
        cre = CognitiveReconstructionEngine(cso)
        stats = cre.reconstruction_stats("gpt-4-turbo")
        assert stats["critical_recovery_rate"] == 0


@pytest.mark.parametrize(
    "model",
    ["gpt-4", "gpt-4-turbo", "gpt-4o", "claude-sonnet-4-5-20250929", "gemini-1.5-pro"],
)
def test_known_models_have_budgets(model: str) -> None:
    assert model in CognitiveReconstructionEngine.MODEL_TOKEN_LIMITS
