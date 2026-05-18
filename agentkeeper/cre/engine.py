"""Cognitive Reconstruction Engine.

The CRE is the core of AgentKeeper. Given a CSO (the agent's cognitive
state), a target model and a token budget, it reconstructs an optimal
system prompt that re-instates the agent's cognitive context for that
model.

Reconstruction principles (in order of priority):

1. **Identity is sacred.** The agent's name, role, principles, and hard
   constraints are always injected first, regardless of token budget.
   Identity bytes are reserved before any fact is considered.

2. **Importance is currency.** Facts are ranked by importance (0.0-1.0).
   Facts with importance >= 0.9 (legacy "critical") are force-included,
   evicting lower-importance facts if necessary.

3. **Tiers shape narration.** Within the same importance bucket, tiers
   are presented in a meaningful order: semantic → episodic → working
   → archival. This produces a context that reads naturally for the LLM:
   stable facts first, events second, ephemera last.

4. **Token estimates are conservative.** We use a 4-chars-per-token rule
   to avoid hitting hard provider limits. Precise tokenisation happens
   at the provider boundary.
"""

from __future__ import annotations

from typing import Any

from ..cso.tiers import MemoryTier
from ..cso.types import CognitiveStateObject, Fact


def estimate_tokens(text: str) -> int:
    """Approximate token count without a tokenizer dependency.

    Rule of thumb: ~1 token per 4 characters on average across major
    English-trained models (OpenAI, Anthropic). Slightly conservative
    for code-heavy or non-English content. Good enough for budget
    decisions; precise tokenisation happens at the provider boundary.
    """
    return max(1, len(text) // 4)


# Order in which tiers appear in the reconstructed context.
# Semantic first because stable structured facts give the LLM
# the strongest grounding signal.
_TIER_RENDER_ORDER: list[MemoryTier] = [
    MemoryTier.SEMANTIC,
    MemoryTier.EPISODIC,
    MemoryTier.WORKING,
    MemoryTier.ARCHIVAL,
]


class CognitiveReconstructionEngine:
    """Reconstructs agent cognitive state for a target model.

    The engine does NOT inject all facts blindly. It selects, prioritises,
    and formats them based on:

    1. Identity injection (always included, fixed cost).
    2. Importance ranking (high importance first).
    3. Token budget for the target model.
    4. Token efficiency (smaller facts first within ties).
    """

    # Conservative budgets — leave room for the user task prompt itself.
    # Unknown models fall back to DEFAULT.
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

    # Threshold above which a fact is force-included. Matches the legacy
    # `critical` flag semantics.
    CRITICAL_THRESHOLD = 0.9

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

    # --- public selection ------------------------------------------

    def prioritize(
        self, target_model: str, max_tokens: int | None = None
    ) -> list[Fact]:
        """Return the optimal subset of facts for a given model and budget.

        Identity (if non-empty) is rendered separately by
        `build_context_prompt`; its byte cost is reserved here so we
        never overrun by including the identity later.

        Selection rules:
        - Facts with importance >= CRITICAL_THRESHOLD are force-included
          (may evict lower-importance facts already selected).
        - Non-critical facts are ordered by importance desc, then by
          token efficiency (smaller first to maximise fact count).
        """
        full_budget = self._budget_for(target_model, max_tokens)
        identity_cost = self._identity_token_cost()
        budget = max(0, full_budget - identity_cost)

        # Group 0 = critical, Group 1 = non-critical.
        # Within each group: importance desc, then token_count asc.
        sorted_facts = sorted(
            self.cso.memory_facts,
            key=lambda f: (
                0 if f.importance >= self.CRITICAL_THRESHOLD else 1,
                -f.importance,
                f.token_count,
            ),
        )

        selected: list[Fact] = []
        used_tokens = 0

        for fact in sorted_facts:
            if used_tokens + fact.token_count <= budget:
                selected.append(fact)
                used_tokens += fact.token_count
                continue

            if fact.importance >= self.CRITICAL_THRESHOLD:
                # Force-include critical facts by evicting the largest
                # non-critical fact already selected, until it fits.
                while used_tokens + fact.token_count > budget:
                    evict_idx = self._find_largest_non_critical(selected)
                    if evict_idx is None:
                        break
                    evicted = selected.pop(evict_idx)
                    used_tokens -= evicted.token_count

                if used_tokens + fact.token_count <= budget:
                    selected.append(fact)
                    used_tokens += fact.token_count

        return selected

    def _find_largest_non_critical(self, facts: list[Fact]) -> int | None:
        largest_idx = None
        largest_tokens = -1
        for i, f in enumerate(facts):
            if (
                f.importance < self.CRITICAL_THRESHOLD
                and f.token_count > largest_tokens
            ):
                largest_tokens = f.token_count
                largest_idx = i
        return largest_idx

    def _identity_token_cost(self) -> int:
        rendered = self.cso.identity.render_for_prompt()
        if not rendered:
            return 0
        return estimate_tokens(rendered)

    # --- public rendering ------------------------------------------

    def build_context_prompt(
        self,
        target_model: str,
        task: str,
        max_tokens: int | None = None,
    ) -> str:
        """Build the system prompt that reconstructs the agent's context."""
        identity_block = self.cso.identity.render_for_prompt()
        facts = self.prioritize(target_model, max_tokens)

        sections: list[str] = []

        if identity_block:
            sections.append(identity_block)

        if facts:
            sections.append("MEMORY (reconstructed from prior sessions):")
            sections.append(self._render_facts_by_tier(facts))
        else:
            sections.append("MEMORY: (empty)")

        sections.append(f"CURRENT TASK: {task}")
        sections.append(
            "Use your identity and memory to maintain continuity. "
            "Do not ask for information you already have."
        )

        return "\n\n".join(sections)

    def _render_facts_by_tier(self, facts: list[Fact]) -> str:
        """Render facts grouped by tier in a stable, scannable order."""
        groups: dict[MemoryTier, list[Fact]] = {t: [] for t in _TIER_RENDER_ORDER}
        for f in facts:
            groups.setdefault(f.tier, []).append(f)

        lines: list[str] = []
        for tier in _TIER_RENDER_ORDER:
            tier_facts = groups.get(tier, [])
            if not tier_facts:
                continue
            lines.append(f"  [{tier.value}]")
            for f in tier_facts:
                marker = " ★" if f.importance >= self.CRITICAL_THRESHOLD else ""
                if tier == MemoryTier.EPISODIC and f.when:
                    lines.append(f"    -{marker} ({f.when}) {f.content}")
                else:
                    lines.append(f"    -{marker} {f.content}")
        return "\n".join(lines)

    # --- diagnostics -----------------------------------------------

    def reconstruction_stats(
        self, target_model: str, max_tokens: int | None = None
    ) -> dict[str, Any]:
        """Return diagnostic stats about reconstruction for a target model."""
        selected = self.prioritize(target_model, max_tokens)
        total_facts = len(self.cso.memory_facts)
        critical_total = len(
            [f for f in self.cso.memory_facts if f.importance >= self.CRITICAL_THRESHOLD]
        )
        critical_selected = len(
            [f for f in selected if f.importance >= self.CRITICAL_THRESHOLD]
        )
        tokens_used = sum(f.token_count for f in selected)
        full_budget = self._budget_for(target_model, max_tokens)
        identity_cost = self._identity_token_cost()

        tier_breakdown: dict[str, int] = {t.value: 0 for t in MemoryTier}
        for f in selected:
            tier_breakdown[f.tier.value] += 1

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
            "token_budget": full_budget,
            "identity_token_cost": identity_cost,
            "identity_present": not self.cso.identity.is_empty(),
            "tier_breakdown": tier_breakdown,
        }
