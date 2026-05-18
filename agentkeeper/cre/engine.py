"""Cognitive Reconstruction Engine.

The CRE is the core of AgentKeeper. Given a CSO (the agent's cognitive
state), a target model and a token budget, it selects the optimal subset
of facts to inject and builds the system prompt that reconstructs the
agent's cognitive context for that model.

This module is deterministic and side-effect free. All non-determinism
(LLM calls, embeddings) lives in higher layers.
"""

from __future__ import annotations

from typing import Any

from ..cso.types import CognitiveStateObject, Fact


def estimate_tokens(text: str) -> int:
    """Approximate token count without a tokenizer dependency.

    Rule of thumb: ~1 token per 4 characters on average across major
    English-trained models (OpenAI, Anthropic). Slightly conservative
    for code-heavy or non-English content. Good enough for budget
    decisions; precise tokenization happens at the provider boundary.
    """
    return max(1, len(text) // 4)


class CognitiveReconstructionEngine:
    """Reconstructs agent cognitive state for a target model.

    The engine does NOT inject all facts blindly. It selects, prioritises,
    and formats them based on:

    1. Critical flag (force-included).
    2. Token budget for the target model.
    3. Token efficiency (shorter facts first when budget is tight).
    """

    # Conservative budgets — leave room for the user task prompt itself.
    # Updated to match 2026 model lineup. Unknown models fall back to DEFAULT.
    MODEL_TOKEN_LIMITS: dict[str, int] = {
        # OpenAI
        "gpt-4": 6_000,
        "gpt-4-turbo": 8_000,
        "gpt-4o": 8_000,
        "gpt-4o-mini": 6_000,
        # Anthropic
        "claude-3-haiku": 4_000,
        "claude-3-5-sonnet-20241022": 8_000,
        "claude-sonnet-4-5-20250929": 12_000,
        "claude-opus-4": 16_000,
        # Google
        "gemini-1.5-pro": 16_000,
        "gemini-2.0-flash": 8_000,
        # Local
        "llama3": 4_000,
        "llama3.1": 6_000,
    }

    DEFAULT_TOKEN_LIMIT = 4_000

    def __init__(self, cso: CognitiveStateObject) -> None:
        self.cso = cso
        self._count_tokens_for_all_facts()

    def _count_tokens_for_all_facts(self) -> None:
        for fact in self.cso.memory_facts:
            if fact.token_count == 0:
                fact.token_count = estimate_tokens(fact.content)

    def _budget_for(self, target_model: str, max_tokens: int | None) -> int:
        if max_tokens is not None:
            return max_tokens
        return self.MODEL_TOKEN_LIMITS.get(target_model, self.DEFAULT_TOKEN_LIMIT)

    def prioritize(
        self, target_model: str, max_tokens: int | None = None
    ) -> list[Fact]:
        """Return the optimal subset of facts for a given model and budget.

        Selection rules:
        - Critical facts come first; they are force-included even when the
          budget is tight (by evicting the largest non-critical facts).
        - Non-critical facts are ordered by token efficiency (smaller first)
          to maximise fact count per token spent.
        """
        budget = self._budget_for(target_model, max_tokens)

        # Critical first (group 0), then non-critical (group 1).
        # Within each group: shortest facts first.
        sorted_facts = sorted(
            self.cso.memory_facts,
            key=lambda f: (0 if f.critical else 1, f.token_count),
        )

        selected: list[Fact] = []
        used_tokens = 0

        for fact in sorted_facts:
            if used_tokens + fact.token_count <= budget:
                selected.append(fact)
                used_tokens += fact.token_count
                continue

            if fact.critical:
                # Force-include critical facts by evicting the largest
                # non-critical fact already selected, until it fits.
                while used_tokens + fact.token_count > budget:
                    largest_non_critical_idx = self._find_largest_non_critical(
                        selected
                    )
                    if largest_non_critical_idx is None:
                        # No non-critical to evict. Drop this critical
                        # rather than over-budgeting. Caller can detect
                        # this via reconstruction_stats().
                        break
                    evicted = selected.pop(largest_non_critical_idx)
                    used_tokens -= evicted.token_count

                if used_tokens + fact.token_count <= budget:
                    selected.append(fact)
                    used_tokens += fact.token_count

        return selected

    @staticmethod
    def _find_largest_non_critical(facts: list[Fact]) -> int | None:
        largest_idx = None
        largest_tokens = -1
        for i, f in enumerate(facts):
            if not f.critical and f.token_count > largest_tokens:
                largest_tokens = f.token_count
                largest_idx = i
        return largest_idx

    def build_context_prompt(
        self,
        target_model: str,
        task: str,
        max_tokens: int | None = None,
    ) -> str:
        """Build the system prompt that reconstructs the agent's context."""
        facts = self.prioritize(target_model, max_tokens)

        if not facts:
            return f"Task: {task}"

        facts_text = "\n".join(
            f"- {'[CRITICAL] ' if f.critical else ''}{f.content}" for f in facts
        )

        return (
            "You are a persistent AI agent. Your memory from previous sessions:\n\n"
            f"{facts_text}\n\n"
            f"Current task: {task}\n\n"
            "Use your memory to maintain continuity. "
            "Do not ask for information you already have."
        )

    def reconstruction_stats(
        self, target_model: str, max_tokens: int | None = None
    ) -> dict[str, Any]:
        """Return diagnostic stats about reconstruction for a target model."""
        selected = self.prioritize(target_model, max_tokens)
        total_facts = len(self.cso.memory_facts)
        critical_total = len(self.cso.critical_facts())
        critical_selected = sum(1 for f in selected if f.critical)
        tokens_used = sum(f.token_count for f in selected)

        return {
            "total_facts": total_facts,
            "selected_facts": len(selected),
            "critical_total": critical_total,
            "critical_selected": critical_selected,
            "critical_recovery_rate": (
                round(critical_selected / critical_total, 3)
                if critical_total
                else 0.0
            ),
            "tokens_used": tokens_used,
            "token_budget": self._budget_for(target_model, max_tokens),
        }
