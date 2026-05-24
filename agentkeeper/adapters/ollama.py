"""Ollama adapter — local models via HTTP API.

Uses only the standard library; no extra dependencies. Requires a running
Ollama instance reachable at `host` (default http://localhost:11434).
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from .base import BaseAdapter


class OllamaAdapter(BaseAdapter):
    """Wrapper around a local Ollama server."""

    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def query(self, system_prompt: str, user_message: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
            }
        ).encode()

        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body: dict[str, Any] = json.loads(resp.read())

        message = body.get("message")
        if not isinstance(message, dict):
            return ""
        content = message.get("content", "")
        return content if isinstance(content, str) else str(content)
