"""Framework-agnostic adapter layer.

AgentKeeper is provider-agnostic *and* framework-agnostic. Use these
integrations to plug cognitive continuity into existing agent frameworks
without rewriting their orchestration.

Each integration exposes:

- ``wrap(agent_or_runtime)`` — adapt a framework-native object so that
  it draws context from AgentKeeper before each call.
- ``ContextProvider`` — a small callable that produces a reconstructed
  system prompt for the framework's native prompt slot.

Supported frameworks in v1.1:

- ``agentkeeper.integrations.langchain``
- ``agentkeeper.integrations.crewai``

Coming in v1.2:

- AutoGen
- OpenAI Agents SDK
- Semantic Kernel

The integrations are intentionally *thin*. AgentKeeper does not own
the orchestration — it owns the cognitive layer underneath. If you
need richer integration than these stubs provide, instantiate a
``CognitiveReconstructionEngine`` directly and feed its output to your
framework's prompt slot manually.
"""

__all__: list[str] = []
