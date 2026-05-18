"""AgentKeeper demo — production-grade async API.

Showcases AK-7 capabilities:
- AsyncAgent: full async API for ask / recall / compress
- 5 parallel asks vs sequential — same correctness, real concurrency
- Typed exception hierarchy
- Structured logging (one line per operation)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time


async def main() -> None:
    os.environ.setdefault("AGENTKEEPER_EMBEDDING_PROVIDER", "mock")
    # Surface AgentKeeper's structured logs.
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    import agentkeeper
    from agentkeeper import create_async

    print("=" * 72)
    print("AgentKeeper async demo")
    print("=" * 72)

    agent = create_async(agent_id="aria-async", provider="mock")
    agent.set_identity(
        name="Aria",
        role="EU insurance broker copilot",
        principles=["never share PII"],
    )
    agent.principle("always confirm budget changes in writing")
    agent.fact("client: Acme Corporation", importance=0.95)
    agent.fact("budget: 50000 EUR", importance=0.9)
    agent.event("contract signed", when="2026-05-15")

    print(f"\n✓ {agent}")

    # 5 parallel asks
    questions = [
        "What is the budget?",
        "Who is the client?",
        "When was the contract signed?",
        "What are my principles?",
        "Summarise the project.",
    ]

    print("\n--- 5 parallel asks ---")
    t0 = time.monotonic()
    responses = await asyncio.gather(*(agent.ask(q) for q in questions))
    dt = time.monotonic() - t0
    print(f"  completed in {dt * 1000:.1f}ms")
    for q, r in zip(questions, responses, strict=True):
        print(f"  Q: {q}")
        print(f"  A: {r[:80]}...")

    # Async recall + compress
    print("\n--- async recall + compress ---")
    hits = await agent.recall("Acme", top_k=2)
    print(f"  recall returned {len(hits)} hit(s)")
    report = await agent.compress()
    print(f"  compression report: facts {report.facts_before} → {report.facts_after}")

    agent.save()
    print(f"\n✓ Persisted. id={agent.id!r}")
    agentkeeper.delete("aria-async")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
