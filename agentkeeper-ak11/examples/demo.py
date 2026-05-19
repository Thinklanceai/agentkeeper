"""AgentKeeper v1.0 — the 90-second tour.

Demonstrates every major capability in one runnable file, offline,
without any API key:

  1. Persistent identity
  2. Memory hierarchy + smart tier inference
  3. Semantic recall
  4. Cognitive compression (decay + consolidation + contradiction)
  5. Identity hardening (principles survive aggressive compression)
  6. Cross-model translation (XML / sections / narrative / minimal)
  7. Async API

Run:
    python examples/demo.py
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone


def _banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def demo_sync() -> None:
    import agentkeeper
    from agentkeeper.compression.contradiction import ContradictionConfig
    from agentkeeper.compression.pipeline import CompressionConfig
    from agentkeeper.cre.engine import CognitiveReconstructionEngine

    _banner("AgentKeeper v1.0 — cognitive continuity in action")

    agent = agentkeeper.create(agent_id="aria-demo", provider="mock")
    agent.set_identity(
        name="Aria",
        role="EU insurance broker copilot",
        principles=["never share PII without explicit consent"],
        constraints=["EU data residency only"],
    )
    agent.principle("always confirm budget changes in writing")
    agent.fact("client: Acme Corporation", importance=0.95)
    agent.fact("budget: 50000 EUR", importance=0.6)
    agent.fact("budget: 50000 EUR", importance=0.5)
    agent.fact("budget: 75000 EUR", importance=0.7)
    agent.event("contract signed", when="2026-05-15")
    agent.remember("favourite colour: blue")

    stale = next(f for f in agent.facts if "colour" in f.content)
    stale.last_accessed_at = (
        datetime.now(timezone.utc) - timedelta(days=60)
    ).isoformat()

    print(f"\nBefore compression: {len(agent.facts)} facts")
    for f in agent.facts:
        tag = "🛡" if f.protected else " "
        print(f"  {tag} [{f.tier.value:8s}] importance={f.importance:.2f} "
              f"{f.content}")

    config = CompressionConfig(
        contradiction=ContradictionConfig(similarity_threshold=0.3),
    )
    report = agent.compress(config=config)

    print(f"\nAfter compression: {len(agent.facts)} facts")
    for f in agent.facts:
        tag = "🛡" if f.protected else " "
        flag = " ⚠ contradicted" if "contradicted_by" in f.metadata else ""
        print(f"  {tag} [{f.tier.value:8s}] importance={f.importance:.2f} "
              f"{f.content}{flag}")

    print(f"\nCompression report: {report.to_dict()}")

    _banner("Cross-model translation — same state, four formats")
    cre = CognitiveReconstructionEngine(agent._cso)
    for label, model in (
        ("CLAUDE (XML)",        "claude-sonnet-4-5-20250929"),
        ("GPT-4 (SECTIONS)",    "gpt-4o"),
        ("GEMINI (NARRATIVE)",  "gemini-1.5-pro"),
        ("OLLAMA (MINIMAL)",    "llama3"),
    ):
        prompt = cre.build_context_prompt(model, "Status?")
        print(f"\n--- {label} ---")
        print(prompt[:380].rstrip())
        print("...")

    audit = agent.identity_audit()
    _banner("Identity audit — survives every form of compression")
    print(f"  name:                {audit['identity']['name']}")
    print(f"  role:                {audit['identity']['role']}")
    print(f"  principles_count:    {audit['identity']['principles_count']}")
    print(f"  constraints_count:   {audit['identity']['constraints_count']}")
    print(f"  protected_facts:     {audit['protected_facts']['count']}")

    agent.save()
    reloaded = agentkeeper.load("aria-demo", provider="mock")
    print(f"\n✓ Persisted + reloaded — identity intact: "
          f"{reloaded.identity.name!r}")
    agentkeeper.delete("aria-demo")


async def demo_async() -> None:
    import agentkeeper

    _banner("Async API — parallel asks")
    agent = agentkeeper.create_async(agent_id="aria-async", provider="mock")
    agent.set_identity(name="Aria", role="copilot")
    agent.fact("budget: 50000 EUR", importance=0.95)
    agent.fact("client: Acme", importance=0.95)

    questions = [
        "What is the budget?",
        "Who is the client?",
        "What's our project status?",
    ]
    responses = await asyncio.gather(*(agent.ask(q) for q in questions))
    for q, r in zip(questions, responses, strict=True):
        print(f"  Q: {q}")
        print(f"  A: {r[:90]}...")

    agent.save()
    agentkeeper.delete("aria-async")


def main() -> None:
    os.environ.setdefault("AGENTKEEPER_EMBEDDING_PROVIDER", "mock")
    demo_sync()
    asyncio.run(demo_async())
    _banner("Demo complete — pip install agentkeeper")


if __name__ == "__main__":
    main()
