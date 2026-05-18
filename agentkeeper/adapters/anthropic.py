"""Anthropic adapter (lazy-imported, sync + async)."""

from __future__ import annotations

from .base import AsyncBaseAdapter, BaseAdapter


class AnthropicAdapter(BaseAdapter):
    """Sync wrapper around the anthropic SDK."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'anthropic' package is required for AnthropicAdapter. "
                "Install with: pip install 'agentkeeper[anthropic]' "
                "or pip install anthropic"
            ) from exc
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def query(self, system_prompt: str, user_message: str) -> str:
        from ..errors import ProviderError, RetriableProviderError

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as exc:  # pragma: no cover - real SDK errors
            if _is_retriable(exc):
                raise RetriableProviderError("anthropic", str(exc)) from exc
            raise ProviderError("anthropic", str(exc)) from exc

        first_block = response.content[0]
        return getattr(first_block, "text", "") or ""


class AsyncAnthropicAdapter(AsyncBaseAdapter):
    """Async wrapper around the anthropic SDK (uses AsyncAnthropic)."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'anthropic' package is required for AsyncAnthropicAdapter. "
                "Install with: pip install 'agentkeeper[anthropic]'"
            ) from exc
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    async def query(self, system_prompt: str, user_message: str) -> str:
        from ..errors import ProviderError, RetriableProviderError

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as exc:  # pragma: no cover
            if _is_retriable(exc):
                raise RetriableProviderError("anthropic", str(exc)) from exc
            raise ProviderError("anthropic", str(exc)) from exc

        first_block = response.content[0]
        return getattr(first_block, "text", "") or ""


def _is_retriable(exc: BaseException) -> bool:
    """Best-effort detection of transient errors from the Anthropic SDK.

    We avoid importing anthropic.* error classes at module import time
    so the adapter remains lazy.
    """
    name = type(exc).__name__.lower()
    return any(
        token in name
        for token in (
            "rate",
            "timeout",
            "overload",
            "apiconnect",
            "internalserver",
        )
    )
