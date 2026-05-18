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

3. **Semantic boost (optional).** When a query is provided AND a
   semantic recaller is attached, fact importance is boosted by
   relevance to the query. This biases reconstruction toward facts
   that actually matter for the task at hand, instead of injecting
   memory blindly.

4. **Tiers shape narration.** Within the same importance bucket, tiers
   are presented in a meaningful order: semantic → episodic → working
   → archival.

5. **Token estimates are conservative.** 4-chars-per-token rule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..cso.tiers import MemoryTier
from ..cso.types import CognitiveStateObject, Fact
from ..translation.profiles import CognitiveProfile, get_profile
from ..translation.renderers import render as render_for_profile

if TYPE_CHECKING:
    from ..semantic.recaller import SemanticRecaller


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

    # Maximum semantic boost added to a fact's effective importance.
    # Tuned so a perfectly-matching fact (score=1.0) gets a meaningful
    # bump without surpassing the critical threshold by itself —
    # critical facts always win.
    SEMANTIC_BOOST_MAX = 0.3

    def __init__(
        self,
        cso: CognitiveStateObject,
        semantic_recaller: SemanticRecaller | None = None,
    ) -> None:
        self.cso = cso
        self._recaller = semantic_recaller
        self._count_tokens_for_all_facts()

    def _count_tokens_for_all_facts(self) -> None:
        for fact in self.cso.memory_facts:
            if fact.token_count == 0:
                fact.token_count = estimate_tokens(fact.content)

    def _resolve_provider(self, target_model: str) -> str:
        """Infer the logical provider from a model or provider name."""
        m = target_model.lower()
        if m.startswith("claude") or m == "anthropic":
            return "anthropic"
        if m.startswith(("gpt", "o1", "o3", "o4")) or m == "openai":
            return "openai"
        if m.startswith("gemini") or m == "gemini" or m == "google":
            return "gemini"
        if m.startswith(("llama", "mistral", "qwen", "phi")) or m == "ollama":
            return "ollama"
        if m == "mock":
            return "mock"
        # Last-ditch: treat the input itself as a provider name.
        return target_model

    def _profile_for(self, target_model: str) -> CognitiveProfile:
        return get_profile(self._resolve_provider(target_model))

    def _budget_for(self, target_model: str, max_tokens: int | None) -> int:
        if max_tokens is not None:
            return max_tokens
        # Prefer the per-model exact budget when known; otherwise fall back
        # to the provider's cognitive profile.
        if target_model in self.MODEL_TOKEN_LIMITS:
            return self.MODEL_TOKEN_LIMITS[target_model]
        profile = self._profile_for(target_model)
        return profile.effective_context_tokens

    # --- public selection ------------------------------------------

    def prioritize(
        self,
        target_model: str,
        max_tokens: int | None = None,
        query: str | None = None,
    ) -> list[Fact]:
        """Return the optimal subset of facts for a given model and budget.

        Identity (if non-empty) is rendered separately by
        `build_context_prompt`; its byte cost is reserved here so we
        never overrun by including the identity later.

        Selection rules:
        - Facts with importance >= CRITICAL_THRESHOLD are force-included
          (may evict lower-importance facts already selected).
        - Non-critical facts are ordered by *effective* importance desc.
          Effective importance = importance + semantic_boost(query, fact).
          Semantic boost is zero unless a query and a recaller are both
          provided.
        - Ties broken by token efficiency (smaller first).
        """
        full_budget = self._budget_for(target_model, max_tokens)
        identity_cost = self._identity_token_cost()
        budget = max(0, full_budget - identity_cost)

        boosts = self._semantic_boosts(query)

        def effective_importance(f: Fact) -> float:
            return f.importance + boosts.get(f.id, 0.0)

        # Group 0 = critical (by raw importance), Group 1 = non-critical.
        # Within each group: effective importance desc, then token_count asc.
        sorted_facts = sorted(
            self.cso.memory_facts,
            key=lambda f: (
                0 if f.importance >= self.CRITICAL_THRESHOLD else 1,
                -effective_importance(f),
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

    def _semantic_boosts(self, query: str | None) -> dict[str, float]:
        """Return per-fact importance boosts based on semantic relevance.

        Returns an empty dict when no query or no recaller is available.
        """
        if not query or self._recaller is None:
            return {}
        try:
            # Pull a large top_k to score the whole short-listed set.
            results = self._recaller.recall(
                query, top_k=max(50, len(self.cso.memory_facts))
            )
        except Exception:
            # Embedding failures must never break reconstruction.
            return {}

        boosts: dict[str, float] = {}
        for fact, score in results:
            # cosine ∈ [-1, 1] → clamp to [0, 1] then scale
            clamped = max(0.0, min(1.0, score))
            boosts[fact.id] = clamped * self.SEMANTIC_BOOST_MAX
        return boosts

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
        """Build the system prompt that reconstructs the agent's context.

        The format is chosen by the target provider's CognitiveProfile:
        XML for Claude, sections for GPT-4 family, narrative for Gemini,
        minimal for Ollama/local models. When a semantic recaller is
        attached, the `task` doubles as the relevance query.
        """
        profile = self._profile_for(target_model)
        facts = self.prioritize(target_model, max_tokens, query=task)
        grouped: dict[MemoryTier, list[Fact]] = {t: [] for t in MemoryTier}
        for f in facts:
            grouped[f.tier].append(f)
        return render_for_profile(profile, self.cso.identity, grouped, task)

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
