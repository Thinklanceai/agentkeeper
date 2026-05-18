"""AgentKeeper demo — cognitive continuity infrastructure.

Demonstrates the full v1.0 narrative offline:

- Identity that survives every form of compression
- Memory hierarchy + semantic recall + cognitive compression
- Cross-model reconstruction: same agent state rendered differently
  for Claude (XML), GPT-4 (sections), Gemini (narrative), Ollama (minimal)
"""

from __future__ import annotations

import os


def main() -> None:
    os.environ.setdefault("AGENTKEEPER_EMBEDDING_PROVIDER", "mock")
    import agentkeeper
    from agentkeeper.benchmark.cross_model import run_cross_model_benchmark
    from agentkeeper.cre.engine import CognitiveReconstructionEngine

    print("=" * 72)
    print("AgentKeeper — cross-model cognitive translation")
    print("=" * 72)

    agent = agentkeeper.create(agent_id="aria-demo", provider="mock")
    agent.set_identity(
        name="Aria",
        role="EU insurance broker copilot",
        principles=["never share PII without explicit consent"],
        constraints=["EU data residency only"],
    )
    agent.principle("always confirm budget changes in writing")
    agent.fact("client name: Acme Corporation", importance=0.95)
    agent.fact("budget: 50000 EUR", importance=0.9)
    agent.event("contract signed", when="2026-05-15")

    cre = CognitiveReconstructionEngine(agent._cso)
    task = "What is the project status?"

    for label, model in (
        ("CLAUDE (XML)",     "claude-sonnet-4-5-20250929"),
        ("GPT-4 (SECTIONS)", "gpt-4o"),
        ("GEMINI (NARRATIVE)", "gemini-1.5-pro"),
        ("OLLAMA (MINIMAL)",   "llama3"),
    ):
        prompt = cre.build_context_prompt(model, task)
        print(f"\n--- {label} ---")
        print(prompt[:550])
        print("..." if len(prompt) > 550 else "")

    print("\n" + "=" * 72)
    print("Cross-model recovery benchmark (MockAdapter fallback)")
    print("=" * 72)
    report = run_cross_model_benchmark(task=task)
    print()
    print(report.summary())
    agentkeeper.delete("aria-demo")
    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
