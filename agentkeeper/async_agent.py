"""Async-native agent facade.

`AsyncAgent` mirrors the public `Agent` API but exposes `async`
counterparts for I/O-bound methods: `ask`, `recall`, `compress`. CPU-bound
operations (remember, fact, event, principle, set_identity, save, delete)
are kept sync — they touch only local state or SQLite.

`AsyncAgent` and `Agent` interoperate freely: they both wrap a
`CognitiveStateObject` and share storage, so an agent saved by one can
be loaded by the other.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any

from .adapters.base import AsyncBaseAdapter, AsyncMockAdapter
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
from .errors import AgentNotFoundError, UnknownProviderError
from .logging import get_logger
from .semantic.base import EmbeddingProvider
from .semantic.recaller import SemanticRecaller
from .storage.sqlite_store import Storage

_log = get_logger(__name__)


# --- async adapter factories ----------------------------------------


def _make_async_openai() -> AsyncBaseAdapter:
    from .adapters.openai import AsyncOpenAIAdapter

    return AsyncOpenAIAdapter(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_MODEL", "gpt-4-turbo"),
    )


def _make_async_anthropic() -> AsyncBaseAdapter:
    from .adapters.anthropic import AsyncAnthropicAdapter

    return AsyncAnthropicAdapter(
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
    )


def _make_async_mock() -> AsyncBaseAdapter:
    return AsyncMockAdapter()


_ASYNC_PROVIDER_FACTORIES: dict[str, Any] = {
    "openai": _make_async_openai,
    "anthropic": _make_async_anthropic,
    "mock": _make_async_mock,
}


# --- shared storage helper ------------------------------------------


_async_storage: Storage | None = None


def _async_get_storage() -> Storage:
    global _async_storage
    if _async_storage is None:
        _async_storage = Storage()
    return _async_storage


# --- AsyncAgent -----------------------------------------------------


class AsyncAgent:
    """Async facade over a CognitiveStateObject.

    The memory-mutation surface is identical to `Agent` (sync), since
    those operations are local. I/O-bound methods (`ask`, `recall`,
    `compress`) are async-native.
    """

    def __init__(
        self,
        cso: CognitiveStateObject,
        default_provider: str = "anthropic",
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._cso = cso
        self._default_provider = default_provider
        self._adapter_cache: dict[str, AsyncBaseAdapter] = {}
        self._last_fact: Fact | None = None
        self._embedding_provider = embedding_provider
        self._recaller: SemanticRecaller | None = None

    # --- identity ---------------------------------------------------

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
    ) -> AsyncAgent:
        """Same semantics as `Agent.set_identity`."""
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

    # --- memory mutation (sync — local state only) -----------------

    def remember(
        self,
        content: str,
        critical: bool | None = None,
        tier: MemoryTier | str | None = None,
        importance: float | None = None,
        when: str | datetime | None = None,
        metadata: dict[str, Any] | None = None,
        ttl: Any = None,
    ) -> AsyncAgent:
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

    def fact(self, content: str, importance: float = 0.7) -> AsyncAgent:
        self._last_fact = self._cso.add_fact(
            content, tier=MemoryTier.SEMANTIC, importance=importance
        )
        return self

    def event(
        self,
        content: str,
        when: str | datetime | None = None,
        importance: float = 0.5,
    ) -> AsyncAgent:
        self._last_fact = self._cso.add_fact(
            content,
            tier=MemoryTier.EPISODIC,
            importance=importance,
            when=when,
        )
        return self

    def principle(self, content: str) -> AsyncAgent:
        self._last_fact = self._cso.add_fact(
            content,
            tier=MemoryTier.SEMANTIC,
            importance=0.95,
            protected=True,
            fact_type=FactType.IDENTITY,
        )
        return self

    # --- typed memory classes (AK-9) -------------------------------

    def decision(self, content: str, importance: float = 0.8) -> AsyncAgent:
        """Record a decision (5× slower decay than ordinary facts)."""
        self._last_fact = self._cso.add_fact(
            content,
            tier=MemoryTier.SEMANTIC,
            importance=importance,
            fact_type=FactType.DECISION,
        )
        return self

    def preference(self, content: str, importance: float = 0.6) -> AsyncAgent:
        """Record a soft preference."""
        self._last_fact = self._cso.add_fact(
            content,
            tier=MemoryTier.SEMANTIC,
            importance=importance,
            fact_type=FactType.PREFERENCE,
        )
        return self

    def constraint(self, content: str, importance: float = 0.85) -> AsyncAgent:
        """Record a situational hard limit."""
        self._last_fact = self._cso.add_fact(
            content,
            tier=MemoryTier.SEMANTIC,
            importance=importance,
            fact_type=FactType.CONSTRAINT,
        )
        return self

    def relationship(self, content: str, importance: float = 0.7) -> AsyncAgent:
        """Record a relational fact."""
        self._last_fact = self._cso.add_fact(
            content,
            tier=MemoryTier.SEMANTIC,
            importance=importance,
            fact_type=FactType.RELATIONSHIP,
        )
        return self

    def task_state(self, content: str, importance: float = 0.5) -> AsyncAgent:
        """Record current task progress."""
        self._last_fact = self._cso.add_fact(
            content,
            tier=MemoryTier.WORKING,
            importance=importance,
            fact_type=FactType.TASK_STATE,
        )
        return self

    def transient(self, content: str, importance: float = 0.3) -> AsyncAgent:
        """Record an ephemeral working-memory item."""
        self._last_fact = self._cso.add_fact(
            content,
            tier=MemoryTier.WORKING,
            importance=importance,
            fact_type=FactType.TRANSIENT,
        )
        return self

    def last_fact(self) -> Fact | None:
        return self._last_fact

    def forget(self, fact_id: str) -> AsyncAgent:
        self._cso.memory_facts = [
            f for f in self._cso.memory_facts if f.id != fact_id
        ]
        if self._recaller is not None:
            self._recaller.remove(fact_id)
        return self

    # --- async interaction -----------------------------------------

    async def ask(
        self,
        question: str,
        provider: str | None = None,
        token_budget: int = 4_000,
    ) -> str:
        chosen = provider or self._default_provider
        adapter = self._get_adapter(chosen)
        cre = CognitiveReconstructionEngine(
            self._cso, semantic_recaller=self._maybe_recaller()
        )
        prompt = cre.build_context_prompt(
            chosen, question, max_tokens=token_budget
        )
        _log.debug("AsyncAgent.ask provider=%s tokens=%d", chosen, token_budget)
        return await adapter.query(prompt, question)

    async def recall(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[Fact, float]]:
        """Async wrapper around the (CPU-bound) semantic recaller.

        Recall is currently CPU-bound: the embedding provider does the
        heavy work. We run it in a thread executor so the event loop
        is not blocked when callers await many `recall` operations.
        """
        recaller = self._get_recaller()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: recaller.recall(query, top_k=top_k, min_score=min_score),
        )

    async def compress(
        self,
        config: CompressionConfig | None = None,
        use_llm: bool = False,
    ) -> CompressionReport:
        """Run cognitive compression. Sync-only LLM synth for now (AK-7
        ships async I/O; the LLM-backed synth still uses the sync
        adapter to keep behaviour consistent across both APIs)."""
        recaller = self._get_recaller()
        synth = None
        if use_llm:
            # LLM synth needs a sync adapter — caller must register one
            # via the sync API path. We don't auto-bridge here to keep
            # the surface narrow.
            raise NotImplementedError(
                "use_llm=True is not supported on AsyncAgent yet. "
                "Use the sync Agent.compress(use_llm=True) for now."
            )
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: _compress_pipeline(
                self._cso,
                embedding_provider=recaller.provider,
                config=config,
                synthesiser=synth,
            ),
        )

    def switch_provider(self, provider: str) -> AsyncAgent:
        if provider not in _ASYNC_PROVIDER_FACTORIES:
            raise UnknownProviderError(
                f"Unknown async provider: {provider}. "
                f"Available: {sorted(_ASYNC_PROVIDER_FACTORIES)}"
            )
        self._default_provider = provider
        return self

    # --- persistence (sync — local SQLite) -------------------------

    def save(self) -> AsyncAgent:
        _async_get_storage().save(self._cso)
        return self

    # --- diagnostics -----------------------------------------------

    def stats(
        self,
        provider: str | None = None,
        token_budget: int = 4_000,
    ) -> dict[str, Any]:
        chosen = provider or self._default_provider
        cre = CognitiveReconstructionEngine(self._cso)
        return cre.reconstruction_stats(chosen, max_tokens=token_budget)

    def contradictions(self) -> list[Fact]:
        return [
            f for f in self._cso.memory_facts
            if "contradicted_by" in f.metadata
        ]

    def health(self) -> dict[str, Any]:
        """Return a cognitive observability snapshot (same shape as Agent.health)."""
        from . import Agent

        # Build a sync Agent view over the same CSO so we reuse one
        # implementation. We don't keep a reference — purely for the
        # method call.
        return Agent(self._cso, self._default_provider).health()

    # --- retention + GDPR (AK-10) ----------------------------------

    def set_memory_policy(self, policy: Any) -> AsyncAgent:
        """Attach a `MemoryPolicy` to this agent."""
        self._cso.set_memory_policy(policy)
        return self

    def memory_policy(self) -> Any:
        """Return the currently-attached `MemoryPolicy`, or None."""
        return getattr(self._cso, "memory_policy", None)

    def purge_expired(self) -> int:
        """Remove facts whose `expires_at` is in the past."""
        from . import Agent

        # Reuse the sync Agent logic; touches only local state.
        agent = Agent(self._cso, self._default_provider)
        agent._recaller = self._recaller  # share index so it stays in sync
        return agent.purge_expired()

    def gdpr_export(self) -> dict[str, Any]:
        """Export every fact this agent holds, GDPR Article 20-style."""
        from . import Agent

        return Agent(self._cso, self._default_provider).gdpr_export()

    def gdpr_purge(
        self,
        predicate: Any = None,
        *,
        include_protected: bool = False,
    ) -> int:
        """Bulk-delete facts matching `predicate`, GDPR Article 17-style."""
        from . import Agent

        agent = Agent(self._cso, self._default_provider)
        agent._recaller = self._recaller
        return agent.gdpr_purge(predicate, include_protected=include_protected)

    # --- internals -------------------------------------------------

    def _get_adapter(self, provider: str) -> AsyncBaseAdapter:
        if provider not in _ASYNC_PROVIDER_FACTORIES:
            raise UnknownProviderError(
                f"Unknown async provider: {provider}. "
                f"Available: {sorted(_ASYNC_PROVIDER_FACTORIES)}"
            )
        if provider not in self._adapter_cache:
            self._adapter_cache[provider] = _ASYNC_PROVIDER_FACTORIES[
                provider
            ]()
        return self._adapter_cache[provider]

    def _get_recaller(self) -> SemanticRecaller:
        if self._recaller is None:
            if self._embedding_provider is None:
                # Reuse the sync resolver for consistency.
                from . import _resolve_default_embedding_provider

                self._embedding_provider = _resolve_default_embedding_provider()
            self._recaller = SemanticRecaller(
                self._embedding_provider, self._cso
            )
        return self._recaller

    def _maybe_recaller(self) -> SemanticRecaller | None:
        return self._recaller

    def __repr__(self) -> str:
        identity = self._cso.identity
        identity_repr = f", identity={identity.name!r}" if identity.name else ""
        return (
            f"AsyncAgent(id={self.id!r}, "
            f"facts={len(self.facts)}, "
            f"provider={self._default_provider!r}"
            f"{identity_repr})"
        )


# --- top-level async factories --------------------------------------


def create_async(
    agent_id: str | None = None,
    provider: str = "anthropic",
    embedding_provider: EmbeddingProvider | None = None,
) -> AsyncAgent:
    """Create a new async agent with a fresh cognitive state."""
    if provider not in _ASYNC_PROVIDER_FACTORIES:
        raise UnknownProviderError(
            f"Unknown async provider: {provider}. "
            f"Available: {sorted(_ASYNC_PROVIDER_FACTORIES)}"
        )
    cso = CognitiveStateObject.create(agent_id=agent_id)
    return AsyncAgent(
        cso,
        default_provider=provider,
        embedding_provider=embedding_provider,
    )


def load_async(
    agent_id: str,
    provider: str = "anthropic",
    embedding_provider: EmbeddingProvider | None = None,
) -> AsyncAgent:
    """Load an existing agent into an AsyncAgent facade."""
    cso = _async_get_storage().load(agent_id)
    if cso is None:
        raise AgentNotFoundError(
            f"Agent {agent_id!r} not found in storage."
        )
    return AsyncAgent(
        cso,
        default_provider=provider,
        embedding_provider=embedding_provider,
    )


# Suppress make_llm_synthesiser unused warning — imported for symmetry
# with the sync module's surface.
_ = make_llm_synthesiser
