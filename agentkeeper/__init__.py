"""AgentKeeper — Cognitive continuity infrastructure for AI agents.

Public API:

    import agentkeeper

    agent = agentkeeper.create(agent_id="my-agent")
    agent.remember("budget: 50k EUR", critical=True)

    response = agent.ask("What is the project budget?", provider="anthropic")

    agent.save()
    agent2 = agentkeeper.load("my-agent")

The library is intentionally vendor-agnostic and infrastructure-free:
storage defaults to local SQLite, no external services are required to
get started. Real provider calls require the corresponding API keys
(OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY) or a running Ollama
instance for local models.
"""

from __future__ import annotations

import os
from typing import Any

from .adapters.base import BaseAdapter, MockAdapter
from .cre.engine import CognitiveReconstructionEngine
from .cso.types import CognitiveStateObject, Fact
from .storage.sqlite_store import Storage

__version__ = "0.2.0-dev"  # bumped on each sprint; v1.0.0 ships at AK-8


# Lazy adapter factories — avoid importing optional SDKs unless used.
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
    a fluent, side-effect-light API for remembering, asking, switching
    providers, and persisting state.
    """

    def __init__(
        self,
        cso: CognitiveStateObject,
        default_provider: str = "anthropic",
    ) -> None:
        self._cso = cso
        self._default_provider = default_provider
        # Adapter cache to avoid reconstructing clients per call.
        self._adapter_cache: dict[str, BaseAdapter] = {}

    # --- identity ----------------------------------------------------

    @property
    def id(self) -> str:
        return self._cso.agent_id

    @property
    def facts(self) -> list[Fact]:
        return self._cso.memory_facts

    @property
    def default_provider(self) -> str:
        return self._default_provider

    # --- memory ------------------------------------------------------

    def remember(self, content: str, critical: bool = False) -> Agent:
        """Add a fact to the agent's memory."""
        self._cso.add_fact(content, critical=critical)
        return self

    def forget(self, fact_id: str) -> Agent:
        """Remove a fact by ID. No-op if the ID does not exist."""
        self._cso.memory_facts = [
            f for f in self._cso.memory_facts if f.id != fact_id
        ]
        return self

    # --- interaction -------------------------------------------------

    def ask(
        self,
        question: str,
        provider: str | None = None,
        token_budget: int = 4_000,
    ) -> str:
        """Ask the agent a question.

        The cognitive context is reconstructed for the target provider
        under the given token budget, then injected as the system prompt.
        """
        chosen = provider or self._default_provider
        adapter = self._get_adapter(chosen)
        cre = CognitiveReconstructionEngine(self._cso)
        prompt = cre.build_context_prompt(
            chosen, question, max_tokens=token_budget
        )
        return adapter.query(prompt, question)

    def switch_provider(self, provider: str) -> Agent:
        """Change the default provider. Memory is preserved across the switch."""
        if provider not in _PROVIDER_FACTORIES:
            raise ValueError(
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
            raise ValueError(f"Unknown provider: {provider}")
        if provider not in self._adapter_cache:
            self._adapter_cache[provider] = _PROVIDER_FACTORIES[provider]()
        return self._adapter_cache[provider]

    def __repr__(self) -> str:
        return (
            f"Agent(id={self.id!r}, "
            f"facts={len(self.facts)}, "
            f"provider={self._default_provider!r})"
        )


# --- top-level convenience API --------------------------------------


def create(
    agent_id: str | None = None,
    provider: str = "anthropic",
) -> Agent:
    """Create a new agent with a fresh cognitive state."""
    if provider not in _PROVIDER_FACTORIES:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available: {sorted(_PROVIDER_FACTORIES)}"
        )
    cso = CognitiveStateObject.create(agent_id=agent_id)
    return Agent(cso, default_provider=provider)


def load(agent_id: str, provider: str = "anthropic") -> Agent:
    """Load an existing agent from storage."""
    cso = _get_storage().load(agent_id)
    if cso is None:
        raise ValueError(f"Agent {agent_id!r} not found in storage.")
    return Agent(cso, default_provider=provider)


def delete(agent_id: str) -> None:
    """Permanently delete an agent from storage."""
    _get_storage().delete(agent_id)


def list_agents() -> list[str]:
    """List the IDs of all agents currently in storage."""
    return _get_storage().list_agent_ids()


__all__ = [
    "Agent",
    "BaseAdapter",
    "CognitiveReconstructionEngine",
    "CognitiveStateObject",
    "Fact",
    "MockAdapter",
    "Storage",
    "__version__",
    "create",
    "delete",
    "list_agents",
    "load",
]
