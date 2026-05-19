"""Google Gemini adapter (lazy-imported)."""

from __future__ import annotations

from .base import BaseAdapter


class GeminiAdapter(BaseAdapter):
    """Wrapper around the google-generativeai SDK."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-pro") -> None:
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'google-generativeai' package is required for GeminiAdapter. "
                "Install with: pip install 'agentkeeper[gemini]'"
            ) from exc
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    def query(self, system_prompt: str, user_message: str) -> str:
        response = self._model.generate_content(
            f"{system_prompt}\n\n{user_message}"
        )
        return response.text or ""
