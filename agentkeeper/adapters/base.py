"""Base adapter contract and synthetic Mock implementations.

The base contract is intentionally narrow: a provider knows how to take
a system prompt + user message and return a text response. Everything
else (prioritisation, reconstruction, fact extraction) lives outside.

Both sync and async contracts are exposed. Real provider implementations
typically support both via the corresponding SDK.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    """Minimal sync contract for any LLM provider."""

    @abstractmethod
    def query(self, system_prompt: str, user_message: str) -> str:
        """Synchronous text completion.

        Args:
            system_prompt: The reconstructed cognitive context.
            user_message: The user-facing task or question.

        Returns:
            The provider's textual response.
        """
        raise NotImplementedError


class AsyncBaseAdapter(ABC):
    """Minimal async contract for any LLM provider."""

    @abstractmethod
    async def query(self, system_prompt: str, user_message: str) -> str:
        """Asynchronous text completion."""
        raise NotImplementedError


class MockAdapter(BaseAdapter):
    """Deterministic sync adapter for testing and offline benchmarks.

    Returns a response that echoes the injected system prompt, so that
    fact-extraction logic can verify which facts were preserved by the
    reconstruction pipeline.
    """

    def __init__(self) -> None:
        self._last_system_prompt = ""

    def query(self, system_prompt: str, user_message: str) -> str:
        self._last_system_prompt = system_prompt
        return f"Based on my memory: {system_prompt}"


class AsyncMockAdapter(AsyncBaseAdapter):
    """Async counterpart of `MockAdapter` for tests."""

    def __init__(self) -> None:
        self._last_system_prompt = ""

    async def query(self, system_prompt: str, user_message: str) -> str:
        self._last_system_prompt = system_prompt
        return f"Based on my memory: {system_prompt}"
