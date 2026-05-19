"""AgentKeeper — Cognitive continuity infrastructure for AI agents.

Public API:

    import agentkeeper

    agent = agentkeeper.create(agent_id="my-agent")

    # Identity (optional but recommended; injected into every reconstruction)
    agent.set_identity(
        name="Aria",
        role="EU insurance broker copilot",
        principles=["never share PII without explicit consent"],
        constraints=["never recommend non-EU providers"],
    )

    # Magical (auto-routed to the right tier)
    agent.remember("budget: 50k EUR")               # → semantic
    agent.remember("client refused offer on March 15")  # → episodic
    agent.remember("never share PII without consent")    # → principle

    # Explicit (full control)
    agent.fact("client name: Acme Corporation", importance=0.95)
    agent.event("contract signed", when="2026-05-15T14:00:00+00:00")
    agent.principle("never recommend competitor products")

    response = agent.ask("What is the project budget?", provider="anthropic")

    agent.save()
    agent2 = agentkeeper.load("my-agent")

The library is vendor-agnostic and infrastructure-free: storage defaults
to local SQLite, no external services are required to get started. Real
provider calls require the corresponding API keys (OPENAI_API_KEY,
ANTHROPIC_API_KEY, GEMINI_API_KEY) or a running Ollama instance.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from .adapters.base import BaseAdapter, MockAdapter
from .async_agent import AsyncAgent, create_async, load_async
from .compression.llm_synth import make_llm_synthesiser
from .compression.pipeline import (
    CompressionConfig,
    CompressionReport,
)
from .compression.pipeline import (
    compress as _compress_pipeline,
)
from .cre.engine import CognitiveReconstructionEngine
from .cso.fact_types import FactType
from .cso.identity import AgentIdentity
from .cso.tiers import MemoryTier
from .cso.types import CognitiveStateObject, Fact
from .errors import (
    AgentKeeperError,
    AgentNotFoundError,
    CompressionError,
    ConfigurationError,
    EmbeddingError,
    ProviderError,
    RetriableProviderError,
    UnknownProviderError,
    UnknownTierError,
)
from .logging import get_logger
from .retention import MemoryPolicy, compute_expires_at, is_expired, parse_ttl
from .semantic.base import EmbeddingProvider
from .semantic.mock import MockEmbeddingProvider
from .semantic.recaller import SemanticRecaller
from .storage.sqlite_store import Storage
from .translation.profiles import (
    CognitiveProfile,
    PromptFormat,
    get_profile,
    known_providers,
    register_profile,
)

__version__ = "1.1.0-dev"

_log = get_logger(__name__)


# --- adapter factories (lazy imports) -------------------------------


def _make_openai_adapter() -> BaseAdapter:
    from .adapters.openai import OpenAIAdapter

    return OpenAIAdapter(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_MODEL", "gpt-4-turbo"),
    )


def _make_anthropic_adapter() -> BaseAdapter:
    from .adapters.anthropic import AnthropicAdapter

    return AnthropicAdapter(
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
    )


def _make_gemini_adapter() -> BaseAdapter:
    from .adapters.gemini import GeminiAdapter

    return GeminiAdapter(
        api_key=os.getenv("GEMINI_API_KEY", ""),
        model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
    )


def _make_ollama_adapter() -> BaseAdapter:
    from .adapters.ollama import OllamaAdapter

    return OllamaAdapter(
        model=os.getenv("OLLAMA_MODEL", "llama3"),
        host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )


def _make_mock_adapter() -> BaseAdapter:
    return MockAdapter()


_PROVIDER_FACTORIES: dict[str, Any] = {
    "openai": _make_openai_adapter,
    "anthropic": _make_anthropic_adapter,
    "gemini": _make_gemini_adapter,
    "ollama": _make_ollama_adapter,
    "mock": _make_mock_adapter,
}


def _resolve_default_embedding_provider() -> EmbeddingProvider:
    """Pick the best available embedding provider for this environment.

    Resolution order:
    1. AGENTKEEPER_EMBEDDING_PROVIDER env var ('sentence-transformers',
       'openai', 'mock').
    2. sentence-transformers if installed (recommended default).
    3. OpenAI if OPENAI_API_KEY is set.
    4. Mock (zero-dependency fallback for tests and offline use).
    """
    explicit = os.getenv("AGENTKEEPER_EMBEDDING_PROVIDER", "").strip().lower()

    if explicit in ("sentence-transformers", "st", "local"):
        from .semantic.sentence_transformers_provider import (
            SentenceTransformerProvider,
        )
        return SentenceTransformerProvider(
            model_name=os.getenv("AGENTKEEPER_EMBEDDING_MODEL") or None
        )

    if explicit == "openai":
        from .semantic.openai_provider import OpenAIEmbeddingProvider
        return OpenAIEmbeddingProvider(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model_name=os.getenv("AGENTKEEPER_EMBEDDING_MODEL") or None,
        )

    if explicit == "mock":
        return MockEmbeddingProvider()

    # Auto-detect: prefer sentence-transformers if importable
    try:
        from .semantic.sentence_transformers_provider import (
            SentenceTransformerProvider,
        )
        return SentenceTransformerProvider()
    except ImportError:
        pass

    if os.getenv("OPENAI_API_KEY"):
        from .semantic.openai_provider import OpenAIEmbeddingProvider
        return OpenAIEmbeddingProvider(api_key=os.environ["OPENAI_API_KEY"])

    return MockEmbeddingProvider()


# A single shared Storage instance per process. The DB path may be
# overridden via the AGENTKEEPER_DB environment variable.
_storage: Storage | None = None


def _get_storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = Storage()
    return _storage


class Agent:
    """A persistent agent with cross-model cognitive continuity.

    The Agent is a thin facade over a CognitiveStateObject. It exposes
    a fluent, side-effect-light API for identity, memory, asking,
    switching providers, and persisting state.
    """

    def __init__(
        self,
        cso: CognitiveStateObject,
        default_provider: str = "anthropic",
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._cso = cso
        self._default_provider = default_provider
        self._adapter_cache: dict[str, BaseAdapter] = {}
        self._last_fact: Fact | None = None
        self._embedding_provider = embedding_provider
        self._recaller: SemanticRecaller | None = None

    # --- identity ----------------------------------------------------

    @property
    def id(self) -> str:
        return self._cso.agent_id

    @property
    def facts(self) -> list[Fact]:
        return self._cso.memory_facts

    @property
    def identity(self) -> AgentIdentity:
        return self._cso.identity

    @property
    def default_provider(self) -> str:
        return self._default_provider

    def set_identity(
        self,
        name: str = "",
        role: str = "",
        principles: list[str] | None = None,
        constraints: list[str] | None = None,
        merge: bool = False,
    ) -> Agent:
        """Define the agent's persistent self-model.

        Identity is injected into every reconstructed context regardless
        of token budget. It survives compression, model switches and
        restarts.

        Args:
            name, role: Replace if provided, ignored if empty (in merge mode).
            principles: Behavioural commitments.
            constraints: Hard limits.
            merge: When True, principles and constraints are appended
                to the existing identity (deduplicated). When False
                (default), the entire identity is replaced.
        """
        if merge:
            existing = self._cso.identity
            new_name = name or existing.name
            new_role = role or existing.role
            new_principles = list(existing.principles)
            for p in principles or []:
                if p not in new_principles:
                    new_principles.append(p)
            new_constraints = list(existing.constraints)
            for c in constraints or []:
                if c not in new_constraints:
                    new_constraints.append(c)
        else:
            new_name = name
            new_role = role
            new_principles = list(principles or [])
            new_constraints = list(constraints or [])

        self._cso.set_identity(
            AgentIdentity(
                name=new_name,
                role=new_role,
                principles=new_principles,
                constraints=new_constraints,
            )
        )
        return self

    def identity_audit(self) -> dict[str, Any]:
        """Return a diagnostic snapshot of the agent's identity layer.

        Useful for verifying that compression, save/load and provider
        switches have not eroded the agent's self-model.
        """
        identity = self._cso.identity
        protected_facts = [f for f in self._cso.memory_facts if f.protected]
        return {
            "identity": {
                "name": identity.name,
                "role": identity.role,
                "principles_count": len(identity.principles),
                "constraints_count": len(identity.constraints),
                "is_empty": identity.is_empty(),
                "token_cost": (
                    CognitiveReconstructionEngine(self._cso)._identity_token_cost()
                ),
            },
            "protected_facts": {
                "count": len(protected_facts),
                "contents": [f.content for f in protected_facts],
            },
        }

    # --- memory: magical entry point --------------------------------

    def remember(
        self,
        content: str,
        critical: bool | None = None,
        tier: MemoryTier | str | None = None,
        importance: float | None = None,
        when: str | datetime | None = None,
        metadata: dict[str, Any] | None = None,
        ttl: Any = None,
    ) -> Agent:
        """Add a fact to the agent's memory and return self for chaining.

        If `tier` and `importance` are omitted, AgentKeeper infers them
        from the content. Pass them explicitly to override.

        Pass `ttl="30d"` (or `timedelta(days=30)`, or an ISO `"P30D"`)
        to expire this fact automatically. Combined with a
        `MemoryPolicy` it gives full GDPR-style retention control.

        After calling `remember`, the resolved fact is available via
        `agent.last_fact()`.
        """
        self._last_fact = self._cso.add_fact(
            content,
            critical=critical,
            tier=tier,
            importance=importance,
            when=when,
            metadata=metadata,
            ttl=ttl,
        )
        return self

    def last_fact(self) -> Fact | None:
        """Return the last fact added via remember/fact/event/principle."""
        return self._last_fact

    # --- memory: explicit helpers -----------------------------------

    def fact(self, content: str, importance: float = 0.7) -> Agent:
        """Add a generic semantic fact (e.g. 'budget: 50k EUR').

        Use the typed helpers (`decision`, `preference`, `constraint`,
        `relationship`, `task_state`, `transient`) when you want the
        compression pipeline to apply the appropriate retention policy.
        """
        self._last_fact = self._cso.add_fact(
            content,
            tier=MemoryTier.SEMANTIC,
            importance=importance,
            fact_type=FactType.FACT,
        )
        return self

    def event(
        self,
        content: str,
        when: str | datetime | None = None,
        importance: float = 0.5,
    ) -> Agent:
        """Add an episodic event with an optional timestamp."""
        self._last_fact = self._cso.add_fact(
            content,
            tier=MemoryTier.EPISODIC,
            importance=importance,
            when=when,
            fact_type=FactType.EVENT,
        )
        return self

    def principle(self, content: str) -> Agent:
        """Add a behavioural principle (high importance, protected from compression).

        Principles are core to the agent's identity and survive every
        form of compression (decay, consolidation, contradiction). They
        are always injected into reconstructed context.
        """
        self._last_fact = self._cso.add_fact(
            content,
            tier=MemoryTier.SEMANTIC,
            importance=0.95,
            protected=True,
            fact_type=FactType.IDENTITY,
        )
        return self

    # --- typed memory classes (AK-9) --------------------------------

    def decision(self, content: str, importance: float = 0.8) -> Agent:
        """Record a decision the agent has made.

        Decisions decay 5× slower than ordinary facts. They are NOT
        protected — they can be superseded by newer contradicting
        decisions via the contradiction arbitration pass.
        """
        self._last_fact = self._cso.add_fact(
            content,
            tier=MemoryTier.SEMANTIC,
            importance=importance,
            fact_type=FactType.DECISION,
        )
        return self

    def preference(self, content: str, importance: float = 0.6) -> Agent:
        """Record a soft preference (favourite style, tone, defaults).

        Preferences decay 2× slower than ordinary facts.
        """
        self._last_fact = self._cso.add_fact(
            content,
            tier=MemoryTier.SEMANTIC,
            importance=importance,
            fact_type=FactType.PREFERENCE,
        )
        return self

    def constraint(self, content: str, importance: float = 0.85) -> Agent:
        """Record a situational hard limit (token budget, region, SLA).

        Distinct from `AgentIdentity.constraints` which are immutable
        identity rules. These are environmental and can change over time.
        Constraints decay 5× slower than ordinary facts.
        """
        self._last_fact = self._cso.add_fact(
            content,
            tier=MemoryTier.SEMANTIC,
            importance=importance,
            fact_type=FactType.CONSTRAINT,
        )
        return self

    def relationship(self, content: str, importance: float = 0.7) -> Agent:
        """Record a relational fact (X works at Y, A is the parent of B)."""
        self._last_fact = self._cso.add_fact(
            content,
            tier=MemoryTier.SEMANTIC,
            importance=importance,
            fact_type=FactType.RELATIONSHIP,
        )
        return self

    def task_state(self, content: str, importance: float = 0.5) -> Agent:
        """Record current task progress.

        Task-state facts decay 2× faster than ordinary facts because
        they are most relevant while the task is active.
        """
        self._last_fact = self._cso.add_fact(
            content,
            tier=MemoryTier.WORKING,
            importance=importance,
            fact_type=FactType.TASK_STATE,
        )
        return self

    def transient(self, content: str, importance: float = 0.3) -> Agent:
        """Record an ephemeral working-memory item.

        Transient facts decay 5× faster than ordinary facts. Useful for
        intermediate computations or short-lived context.
        """
        self._last_fact = self._cso.add_fact(
            content,
            tier=MemoryTier.WORKING,
            importance=importance,
            fact_type=FactType.TRANSIENT,
        )
        return self

    def forget(self, fact_id: str) -> Agent:
        """Remove a fact by ID. No-op if the ID does not exist."""
        self._cso.memory_facts = [
            f for f in self._cso.memory_facts if f.id != fact_id
        ]
        if self._recaller is not None:
            self._recaller.remove(fact_id)
        return self

    # --- semantic recall --------------------------------------------

    def set_embedding_provider(self, provider: EmbeddingProvider) -> Agent:
        """Override the default embedding provider for this agent.

        Triggers a rebuild of the index on next recall.
        """
        self._embedding_provider = provider
        self._recaller = None
        return self

    def recall(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[Fact, float]]:
        """Find facts most semantically similar to `query`.

        Returns up to `top_k` `(Fact, score)` pairs sorted by cosine
        similarity descending. Pairs below `min_score` are filtered.
        """
        recaller = self._get_recaller()
        return recaller.recall(query, top_k=top_k, min_score=min_score)

    def _get_recaller(self) -> SemanticRecaller:
        if self._recaller is None:
            if self._embedding_provider is None:
                self._embedding_provider = _resolve_default_embedding_provider()
            self._recaller = SemanticRecaller(
                self._embedding_provider, self._cso
            )
        return self._recaller

    # --- cognitive compression --------------------------------------

    def compress(
        self,
        config: CompressionConfig | None = None,
        use_llm: bool = False,
        llm_provider: str | None = None,
    ) -> CompressionReport:
        """Run the cognitive compression pipeline against this agent.

        Args:
            config: Override individual pass toggles and thresholds.
                If None, defaults run all three passes (decay,
                consolidation, contradiction).
            use_llm: When True, use the agent's LLM provider to
                synthesise consolidated facts (richer summaries but
                slower and consumes tokens). When False (default),
                consolidation keeps canonical facts unchanged.
            llm_provider: Provider to use for synthesis. Defaults to
                this agent's default provider.

        Returns:
            A `CompressionReport` describing what changed. The agent's
            facts are mutated in place; reflect to disk via `save()`.
        """
        # Make sure an embedding provider exists — semantic passes need it.
        recaller = self._get_recaller()
        synth = None
        if use_llm:
            provider_name = llm_provider or self._default_provider
            adapter = self._get_adapter(provider_name)
            synth = make_llm_synthesiser(adapter)
        report = _compress_pipeline(
            self._cso,
            embedding_provider=recaller.provider,
            config=config,
            synthesiser=synth,
        )
        # Recaller's index may now be stale (facts removed / content changed).
        # Force a rebuild on next recall.
        self._recaller = None
        return report

    def contradictions(self) -> list[Fact]:
        """Return facts that have been flagged as contradicted.

        These facts carry a `contradicted_by` entry in their metadata.
        They are kept (with reduced importance) so the user can inspect
        and decide whether to re-promote or delete them.
        """
        return [
            f for f in self._cso.memory_facts
            if "contradicted_by" in f.metadata
        ]

    # --- interaction -------------------------------------------------

    def ask(
        self,
        question: str,
        provider: str | None = None,
        token_budget: int = 4_000,
    ) -> str:
        """Ask the agent a question.

        The cognitive context is reconstructed for the target provider
        under the given token budget, with semantic relevance boosting
        when a recaller is available.
        """
        chosen = provider or self._default_provider
        adapter = self._get_adapter(chosen)
        cre = CognitiveReconstructionEngine(
            self._cso, semantic_recaller=self._maybe_recaller()
        )
        prompt = cre.build_context_prompt(
            chosen, question, max_tokens=token_budget
        )
        return adapter.query(prompt, question)

    def switch_provider(self, provider: str) -> Agent:
        """Change the default provider. Memory survives the switch."""
        if provider not in _PROVIDER_FACTORIES:
            raise UnknownProviderError(
                f"Unknown provider: {provider}. "
                f"Available: {sorted(_PROVIDER_FACTORIES)}"
            )
        self._default_provider = provider
        return self

    # --- persistence -------------------------------------------------

    def save(self) -> Agent:
        """Persist the agent's CSO to local storage."""
        _get_storage().save(self._cso)
        return self

    # --- diagnostics -------------------------------------------------

    def stats(
        self,
        provider: str | None = None,
        token_budget: int = 4_000,
    ) -> dict[str, Any]:
        """Return reconstruction stats for a given provider and budget."""
        chosen = provider or self._default_provider
        cre = CognitiveReconstructionEngine(self._cso)
        return cre.reconstruction_stats(chosen, max_tokens=token_budget)

    def health(self) -> dict[str, Any]:
        """Return a cognitive observability snapshot.

        Designed for production monitoring of long-lived agents. The
        report covers:

        - Memory volume and tier distribution.
        - Fact-type distribution (decisions, preferences, constraints,
          relationships, transient, ...).
        - Importance histogram (mean, max, count above critical threshold).
        - Stale memory ratio (facts not accessed in 30+ days).
        - Contradiction count and protected-fact count.
        - Identity presence.

        Returns:
            A dict suitable for logging, Prometheus export, or feeding
            into a dashboard.
        """
        from datetime import datetime, timezone

        from .compression.decay import days_since_last_access

        now = datetime.now(timezone.utc)
        facts = self._cso.memory_facts

        tier_distribution: dict[str, int] = {t.value: 0 for t in MemoryTier}
        type_distribution: dict[str, int] = {t.value: 0 for t in FactType}
        importances: list[float] = []
        stale_count = 0
        contradicted_count = 0
        protected_count = 0
        critical_count = 0

        for f in facts:
            tier_distribution[f.tier.value] += 1
            type_distribution[f.fact_type.value] += 1
            importances.append(f.importance)
            if f.importance >= 0.9:
                critical_count += 1
            if f.protected:
                protected_count += 1
            if "contradicted_by" in f.metadata:
                contradicted_count += 1
            try:
                if days_since_last_access(f, now=now) > 30.0:
                    stale_count += 1
            except Exception:
                # Malformed timestamps don't crash the health report.
                pass

        total = len(facts)
        mean_importance = (
            sum(importances) / total if total else 0.0
        )
        max_importance = max(importances) if importances else 0.0
        stale_ratio = (stale_count / total) if total else 0.0

        identity = self._cso.identity

        return {
            "total_facts": total,
            "critical_facts": critical_count,
            "protected_facts": protected_count,
            "contradicted_facts": contradicted_count,
            "stale_facts": stale_count,
            "stale_ratio": round(stale_ratio, 3),
            "importance": {
                "mean": round(mean_importance, 3),
                "max": round(max_importance, 3),
            },
            "tier_distribution": tier_distribution,
            "fact_type_distribution": type_distribution,
            "identity": {
                "present": not identity.is_empty(),
                "name": identity.name,
                "role": identity.role,
                "principles_count": len(identity.principles),
                "constraints_count": len(identity.constraints),
            },
        }

    # --- retention + GDPR (AK-10) -----------------------------------

    def set_memory_policy(self, policy: Any) -> Agent:
        """Attach a `MemoryPolicy` to this agent.

        Once set, every `remember`/`fact`/`decision`/... call that omits
        an explicit `ttl` consults the policy. Protected facts are
        exempt (the policy's `respect_protected` flag, default True).
        """
        self._cso.set_memory_policy(policy)
        return self

    def memory_policy(self) -> Any:
        """Return the currently-attached `MemoryPolicy`, or None."""
        return getattr(self._cso, "memory_policy", None)

    def purge_expired(self) -> int:
        """Remove facts whose `expires_at` is in the past.

        Returns the number of facts purged. Protected facts are
        always preserved. Also updates the semantic index if one is
        attached.
        """
        from datetime import datetime, timezone

        from .retention.ttl import is_expired

        now = datetime.now(timezone.utc)
        purged_ids: list[str] = []
        kept: list[Fact] = []
        for f in self._cso.memory_facts:
            if f.protected:
                kept.append(f)
                continue
            if f.expires_at and is_expired(f.expires_at, now=now):
                purged_ids.append(f.id)
                continue
            kept.append(f)
        if purged_ids:
            self._cso.memory_facts = kept
            if self._recaller is not None:
                for fid in purged_ids:
                    self._recaller.remove(fid)
        return len(purged_ids)

    def gdpr_export(self) -> dict[str, Any]:
        """Return a JSON-serialisable export of every fact this agent
        holds.

        Useful for fulfilling GDPR Article 20 (right to data portability).
        Includes the full Fact payload (content, type, tier, importance,
        timestamps, metadata) plus the agent identity. Sensitive data
        masking is the caller's responsibility — this method exports
        whatever the agent stored verbatim.
        """
        identity = self._cso.identity
        from datetime import datetime, timezone

        return {
            "schema_version": "1.1",
            "agent_id": self._cso.agent_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "identity": {
                "name": identity.name,
                "role": identity.role,
                "principles": list(identity.principles),
                "constraints": list(identity.constraints),
            },
            "facts": [f.to_dict() for f in self._cso.memory_facts],
        }

    def gdpr_purge(
        self,
        predicate: Any = None,
        *,
        include_protected: bool = False,
    ) -> int:
        """Bulk-delete facts matching `predicate`.

        Useful for fulfilling GDPR Article 17 (right to erasure).
        `predicate` is a callable `(Fact) -> bool`. If `None`, ALL
        non-protected facts are purged — pass `include_protected=True`
        to also purge identity-level facts.

        Returns the number of facts removed.
        """
        purged_ids: list[str] = []
        kept: list[Fact] = []
        for f in self._cso.memory_facts:
            should_purge = predicate(f) if predicate is not None else True
            if f.protected and not include_protected:
                kept.append(f)
                continue
            if should_purge:
                purged_ids.append(f.id)
                continue
            kept.append(f)
        if purged_ids:
            self._cso.memory_facts = kept
            if self._recaller is not None:
                for fid in purged_ids:
                    self._recaller.remove(fid)
        return len(purged_ids)

    # --- internals ---------------------------------------------------

    def _get_adapter(self, provider: str) -> BaseAdapter:
        if provider not in _PROVIDER_FACTORIES:
            raise UnknownProviderError(f"Unknown provider: {provider}")
        if provider not in self._adapter_cache:
            self._adapter_cache[provider] = _PROVIDER_FACTORIES[provider]()
        return self._adapter_cache[provider]

    def _maybe_recaller(self) -> SemanticRecaller | None:
        """Return the recaller if one was explicitly configured.

        Unlike _get_recaller, this never triggers default provider
        resolution — `ask()` works without embeddings unless the user
        opts in via recall() or set_embedding_provider().
        """
        return self._recaller

    def __repr__(self) -> str:
        identity = self._cso.identity
        identity_repr = f", identity={identity.name!r}" if identity.name else ""
        return (
            f"Agent(id={self.id!r}, "
            f"facts={len(self.facts)}, "
            f"provider={self._default_provider!r}"
            f"{identity_repr})"
        )


# --- top-level convenience API --------------------------------------


def create(
    agent_id: str | None = None,
    provider: str = "anthropic",
    embedding_provider: EmbeddingProvider | None = None,
) -> Agent:
    """Create a new agent with a fresh cognitive state."""
    if provider not in _PROVIDER_FACTORIES:
        raise UnknownProviderError(
            f"Unknown provider: {provider}. "
            f"Available: {sorted(_PROVIDER_FACTORIES)}"
        )
    cso = CognitiveStateObject.create(agent_id=agent_id)
    return Agent(
        cso,
        default_provider=provider,
        embedding_provider=embedding_provider,
    )


def load(
    agent_id: str,
    provider: str = "anthropic",
    embedding_provider: EmbeddingProvider | None = None,
) -> Agent:
    """Load an existing agent from storage."""
    cso = _get_storage().load(agent_id)
    if cso is None:
        raise AgentNotFoundError(f"Agent {agent_id!r} not found in storage.")
    return Agent(
        cso,
        default_provider=provider,
        embedding_provider=embedding_provider,
    )


def delete(agent_id: str) -> None:
    """Permanently delete an agent from storage."""
    _get_storage().delete(agent_id)


def list_agents() -> list[str]:
    """List the IDs of all agents currently in storage."""
    return _get_storage().list_agent_ids()


__all__ = [
    "Agent",
    "AgentIdentity",
    "AgentKeeperError",
    "AgentNotFoundError",
    "AsyncAgent",
    "BaseAdapter",
    "CognitiveProfile",
    "CognitiveReconstructionEngine",
    "CognitiveStateObject",
    "CompressionConfig",
    "CompressionError",
    "CompressionReport",
    "ConfigurationError",
    "EmbeddingError",
    "EmbeddingProvider",
    "Fact",
    "FactType",
    "MemoryPolicy",
    "MemoryTier",
    "MockAdapter",
    "MockEmbeddingProvider",
    "PromptFormat",
    "ProviderError",
    "RetriableProviderError",
    "SemanticRecaller",
    "Storage",
    "UnknownProviderError",
    "UnknownTierError",
    "__version__",
    "compute_expires_at",
    "create",
    "create_async",
    "delete",
    "get_logger",
    "get_profile",
    "is_expired",
    "known_providers",
    "list_agents",
    "load",
    "load_async",
    "make_llm_synthesiser",
    "parse_ttl",
    "register_profile",
]
