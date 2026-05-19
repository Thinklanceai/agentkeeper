"""CrewAI integration for AgentKeeper.

CrewAI agents are configured with a ``role``, ``goal``, and ``backstory``.
The backstory is essentially their system prompt. AgentKeeper provides
two helpers that fit naturally into this model:

1. ``crewai_backstory(agent, task, model)`` — returns a reconstructed
   backstory string. Use it when instantiating a ``crewai.Agent``::

       from crewai import Agent as CrewAgent
       from agentkeeper.integrations.crewai import crewai_backstory

       crew_agent = CrewAgent(
           role="Insurance broker copilot",
           goal="Help the human pick a policy",
           backstory=crewai_backstory(my_ak_agent, task="..."),
       )

2. ``CrewAICognitiveBackstory(agent, model)`` — a callable producing a
   fresh backstory per task. Useful when the CrewAI agent is reused
   across tasks and you want the cognitive state injected each time.

As with the LangChain integration, this module has no hard dependency
on CrewAI. Install it separately::

    pip install crewai

If CrewAI is not installed, the helpers still work — they produce a
plain string that any orchestration framework can consume.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..cre.engine import CognitiveReconstructionEngine

if TYPE_CHECKING:
    from .. import Agent


def crewai_backstory(
    agent: Agent,
    task: str,
    model: str | None = None,
    token_budget: int | None = None,
) -> str:
    """Reconstruct a CrewAI-friendly backstory for the given task.

    Args:
        agent: An AgentKeeper Agent.
        task: The CrewAI task description. Drives semantic recall.
        model: Target model name (defaults to the agent's current
            provider; drives the prompt format).
        token_budget: Reconstruction budget override.

    Returns:
        A backstory string ready to drop into ``crewai.Agent(backstory=...)``.
    """
    target = model or agent.default_provider
    cre = CognitiveReconstructionEngine(agent._cso)
    return cre.build_context_prompt(target, task, max_tokens=token_budget)


class CrewAICognitiveBackstory:
    """Callable that produces a fresh backstory per CrewAI task.

    Use when the same CrewAI agent runs multiple tasks and you want
    each task to see freshly-reconstructed cognitive state.

    Usage::

        from agentkeeper.integrations.crewai import (
            CrewAICognitiveBackstory,
        )

        backstory = CrewAICognitiveBackstory(my_ak_agent, model="gpt-4o")
        crew_agent.backstory = backstory(task="Summarise Q3 results")
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

    def __call__(self, task: str) -> str:
        return crewai_backstory(
            self._agent,
            task,
            model=self._model,
            token_budget=self._token_budget,
        )


__all__ = [
    "CrewAICognitiveBackstory",
    "crewai_backstory",
]
