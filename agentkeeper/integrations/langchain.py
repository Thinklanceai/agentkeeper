"""LangChain integration for AgentKeeper.

LangChain agents call an LLM with a system prompt + a message history.
The cleanest insertion point for cognitive continuity is the **system
prompt**: each call, AgentKeeper reconstructs a fresh system prompt
that injects identity, principles, and the facts most relevant to the
incoming user message.

This module gives you two ways to wire that in:

1. ``langchain_system_prompt(agent, message, model)`` — a one-shot
   helper that returns the reconstructed system prompt as a string.
   Use it directly when building a ``ChatPromptTemplate``.

2. ``LangChainCognitiveProvider(agent, model)`` — a callable whose
   ``__call__`` accepts a runtime message and returns the system
   prompt. Plug it into your chain wherever you'd normally hardcode
   a system message.

We deliberately do **not** wrap ``BaseChatModel`` or ``Runnable``
directly: those interfaces churn faster than this library can keep up
with, and LangChain ships its own memory abstractions. Treat
AgentKeeper as the **reconstruction layer**, not as a replacement for
LangChain's chain composition.

Install LangChain separately:

    pip install langchain langchain-openai  # or your preferred provider

AgentKeeper has no hard dependency on LangChain; if it's not installed,
the helpers below still work — they just produce a plain string that
*any* framework can use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..cre.engine import CognitiveReconstructionEngine

if TYPE_CHECKING:
    from .. import Agent


def langchain_system_prompt(
    agent: Agent,
    message: str,
    model: str | None = None,
    token_budget: int | None = None,
) -> str:
    """Reconstruct a system prompt tailored for the next LangChain call.

    Args:
        agent: An AgentKeeper Agent whose cognitive state should be
            injected.
        message: The user message about to be sent. Drives semantic
            recall so the prompt focuses on relevant facts.
        model: Target model name (drives format selection — XML for
            Claude, sections for GPT-4, etc.). Defaults to the agent's
            current provider.
        token_budget: Override the reconstruction budget. Defaults to
            the per-provider profile.

    Returns:
        A ready-to-use system prompt string.
    """
    target = model or agent.default_provider
    cre = CognitiveReconstructionEngine(agent._cso)
    return cre.build_context_prompt(target, message, max_tokens=token_budget)


class LangChainCognitiveProvider:
    """A callable that produces a fresh system prompt per LangChain call.

    Usage::

        from langchain_core.prompts import ChatPromptTemplate
        from agentkeeper.integrations.langchain import (
            LangChainCognitiveProvider,
        )

        provider = LangChainCognitiveProvider(agent, model="gpt-4o")
        prompt = ChatPromptTemplate.from_messages([
            ("system", provider(message="What's our budget?")),
            ("human", "{input}"),
        ])
    """

    def __init__(
        self,
        agent: Agent,
        model: str | None = None,
        token_budget: int | None = None,
    ) -> None:
        self._agent = agent
        self._model = model
        self._token_budget = token_budget

    def __call__(self, message: str) -> str:
        return langchain_system_prompt(
            self._agent,
            message,
            model=self._model,
            token_budget=self._token_budget,
        )


__all__ = [
    "LangChainCognitiveProvider",
    "langchain_system_prompt",
]
