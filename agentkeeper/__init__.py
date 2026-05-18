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

__version__ = "0.8.0-dev"  # bumped on each sprint; v1.0.0 ships at AK-8

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
    ) -> Agent:
        """Add a fact to the agent's memory and return self for chaining.

        If `tier` and `importance` are omitted, AgentKeeper infers them
        from the content. Pass them explicitly to override.

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
        )
        return self

    def last_fact(self) -> Fact | None:
        """Return the last fact added via remember/fact/event/principle."""
        return self._last_fact

    # --- memory: explicit helpers -----------------------------------

    def fact(self, content: str, importance: float = 0.7) -> Agent:
        """Add a stable, semantic fact (e.g. 'budget: 50k EUR')."""
        self._last_fact = self._cso.add_fact(
            content, tier=MemoryTier.SEMANTIC, importance=importance
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
    "create",
    "create_async",
    "delete",
    "get_logger",
    "get_profile",
    "known_providers",
    "list_agents",
    "load",
    "load_async",
    "make_llm_synthesiser",
    "register_profile",
]
