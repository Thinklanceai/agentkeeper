"""LLM-backed synthesiser for cognitive consolidation.

When the algorithmic consolidator finds a cluster of near-duplicate
facts, it normally keeps the canonical one untouched. If the user
opts in by passing this synthesiser, it instead asks an LLM (the
agent's *own* provider — no hidden third-party calls) to write a
single consolidated fact that captures the essence of the cluster.

The synthesiser is **purely a callable** so it composes cleanly with
the rest of the pipeline. Failure modes (rate limits, transient
errors) fall back gracefully to the canonical content.
"""

from __future__ import annotations

from collections.abc import Callable

from ..adapters.base import BaseAdapter
from ..cso.types import Fact

_SYNTHESIS_INSTRUCTION = (
    "You are a memory consolidator for an AI agent. You will receive "
    "several near-duplicate facts that the agent has stored over time. "
    "Write ONE concise replacement fact that preserves every distinct "
    "piece of information across them. Keep the same factual form "
    "('key: value' if appropriate). Do not editorialise. Output only "
    "the consolidated fact, no preamble, no quotes."
)


def make_llm_synthesiser(
    adapter: BaseAdapter,
    instruction: str | None = None,
) -> Callable[[list[Fact]], str]:
    """Return a synthesiser callable that uses `adapter` to consolidate.

    Usage:
        from agentkeeper.compression.llm_synth import make_llm_synthesiser

        synth = make_llm_synthesiser(my_anthropic_adapter)
        agent.compress(synthesiser=synth)
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
        try:
            response = adapter.query(system_prompt, user_msg).strip()
            # Sanity check: reject empty or absurdly long outputs.
            if not response or len(response) > max(
                256, len(canonical.content) * 4
            ):
                return canonical.content
            return response
        except Exception:
            # Graceful fallback: any LLM failure keeps the canonical fact.
            return canonical.content

    return synthesise
