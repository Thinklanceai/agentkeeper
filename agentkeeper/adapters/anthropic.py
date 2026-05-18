"""Anthropic adapter (lazy-imported)."""

from __future__ import annotations

from .base import BaseAdapter


class AnthropicAdapter(BaseAdapter):
    """Wrapper around the anthropic SDK."""

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
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        # response.content is a list of content blocks; text is in the first.
        first_block = response.content[0]
        return getattr(first_block, "text", "") or ""
