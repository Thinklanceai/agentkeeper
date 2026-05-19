"""Async LLM synthesiser for cognitive consolidation.

The compression pipeline's consolidation pass accepts a synchronous
`Synthesiser = Callable[[list[Fact]], str]`. To use an async LLM
adapter for synthesis, we expose `make_async_llm_synthesiser`, which
returns a synchronous callable that runs an `AsyncBaseAdapter.query()`
behind the scenes.

Why a sync facade over async? Because compression itself is dispatched
to a thread executor by `AsyncAgent.compress()` (via
`loop.run_in_executor`). The synthesiser callable therefore runs on a
worker thread, *not* on the event loop. In that context, `asyncio.run`
creates a fresh loop for the duration of the call — exactly the right
shape. We do not block the caller's main event loop.

Graceful fallback: any failure (rate limit, timeout, malformed
response) returns the canonical fact's content unchanged. Same
contract as the sync `make_llm_synthesiser`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from ..adapters.base import AsyncBaseAdapter
from ..cso.types import Fact
from .llm_synth import _SYNTHESIS_INSTRUCTION


def make_async_llm_synthesiser(
    adapter: AsyncBaseAdapter,
    instruction: str | None = None,
) -> Callable[[list[Fact]], str]:
    """Return a synthesiser callable that uses an async `adapter`.

    The returned callable is synchronous (matches the `Synthesiser`
    contract used by the consolidation pass), but internally drives
    an `await adapter.query(...)` via a short-lived asyncio loop.

    Usage:
        from agentkeeper.compression.async_llm_synth import (
            make_async_llm_synthesiser,
        )

        synth = make_async_llm_synthesiser(my_async_anthropic_adapter)
        report = await async_agent.compress(use_llm=True)
    """
    system_prompt = instruction or _SYNTHESIS_INSTRUCTION

    def synthesise(cluster: list[Fact]) -> str:
        if not cluster:
            return ""
        if len(cluster) == 1:
            return cluster[0].content

        canonical = max(
            cluster,
            key=lambda f: (f.importance, f.last_accessed_at),
        )
        bullet_list = "\n".join(f"- {f.content}" for f in cluster)
        user_msg = (
            f"Facts to consolidate:\n{bullet_list}\n\n"
            "Output one replacement fact."
        )

        async def _call() -> str:
            return await adapter.query(system_prompt, user_msg)

        try:
            # We are on a worker thread (consolidation runs inside
            # loop.run_in_executor). asyncio.run is safe here: it
            # creates a new loop, runs the coroutine, and tears down.
            response = asyncio.run(_call()).strip()
            if not response or len(response) > max(
                256, len(canonical.content) * 4
            ):
                return canonical.content
            return response
        except Exception:
            # Any failure: keep the canonical fact unchanged.
            return canonical.content

    return synthesise
