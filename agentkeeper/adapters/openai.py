"""OpenAI adapter (lazy-imported)."""

from __future__ import annotations

from .base import BaseAdapter


class OpenAIAdapter(BaseAdapter):
    """Wrapper around the openai SDK."""

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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=500,
        )
        content = response.choices[0].message.content
        return content if content is not None else ""
