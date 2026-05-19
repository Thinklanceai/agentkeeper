"""OpenAI adapter (lazy-imported, sync + async)."""

from __future__ import annotations

from .base import AsyncBaseAdapter, BaseAdapter


class OpenAIAdapter(BaseAdapter):
    """Sync wrapper around the openai SDK."""

    def __init__(self, api_key: str, model: str = "gpt-4-turbo") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required for OpenAIAdapter. "
                "Install with: pip install 'agentkeeper[openai]' "
                "or pip install openai"
            ) from exc
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def query(self, system_prompt: str, user_message: str) -> str:
        from ..errors import ProviderError, RetriableProviderError

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=500,
            )
        except Exception as exc:  # pragma: no cover
            if _is_retriable(exc):
                raise RetriableProviderError("openai", str(exc)) from exc
            raise ProviderError("openai", str(exc)) from exc

        content = response.choices[0].message.content
        return content if content is not None else ""


class AsyncOpenAIAdapter(AsyncBaseAdapter):
    """Async wrapper around the openai SDK (uses AsyncOpenAI)."""

    def __init__(self, api_key: str, model: str = "gpt-4-turbo") -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required for AsyncOpenAIAdapter. "
                "Install with: pip install 'agentkeeper[openai]'"
            ) from exc
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def query(self, system_prompt: str, user_message: str) -> str:
        from ..errors import ProviderError, RetriableProviderError

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=500,
            )
        except Exception as exc:  # pragma: no cover
            if _is_retriable(exc):
                raise RetriableProviderError("openai", str(exc)) from exc
            raise ProviderError("openai", str(exc)) from exc

        content = response.choices[0].message.content
        return content if content is not None else ""


def _is_retriable(exc: BaseException) -> bool:
    """Best-effort detection of transient errors from the OpenAI SDK."""
    name = type(exc).__name__.lower()
    return any(
        token in name
        for token in (
            "rate",
            "timeout",
            "apiconnect",
            "internalserver",
            "serviceunavailable",
        )
    )
