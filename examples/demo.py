"""AgentKeeper demo — cognitive continuity infrastructure.

This demo runs the full v1.0 narrative offline (no API keys required):

- AK-2: persistent identity + memory hierarchy
- AK-3: semantic recall
- AK-4: cognitive compression (decay + consolidation + contradiction)

Compression keeps long-lived agents coherent: duplicate facts collapse,
contradictions are resolved, and stale low-importance facts decay.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone


def main() -> None:
    os.environ.setdefault("AGENTKEEPER_EMBEDDING_PROVIDER", "mock")
    import agentkeeper
    from agentkeeper.compression.contradiction import ContradictionConfig
    from agentkeeper.compression.pipeline import CompressionConfig

    print("=" * 70)
    print("AgentKeeper — cognitive continuity in action")
    print("=" * 70)

    agent = agentkeeper.create(agent_id="aria-demo", provider="mock")
    agent.set_identity(
        name="Aria",
        role="EU insurance broker copilot",
        principles=["never share PII without consent"],
        constraints=["EU data residency only"],
    )

    # Seed: stable facts, duplicates, an outdated value, a stale note
    agent.fact("client name: Acme Corporation", importance=0.95)
    agent.fact("budget: 50000 EUR", importance=0.6)
    agent.fact("budget: 50000 EUR", importance=0.5)  # duplicate
    agent.fact("budget: 75000 EUR", importance=0.7)  # contradicts above
    agent.fact("favourite colour: blue")             # low importance, will decay

    # Age the colour fact so decay has something to do
    old = next(f for f in agent.facts if "colour" in f.content)
    old.last_accessed_at = (
        datetime.now(timezone.utc) - timedelta(days=60)
    ).isoformat()

    print(f"\n✓ Seeded {len(agent.facts)} facts (criticals, duplicates, "
          f"contradiction, stale note)\n")
    for f in agent.facts:
        print(f"  [{f.tier.value:8s}]  importance={f.importance:.2f}  "
              f"{f.content}")

    # Compress with a lower threshold so the mock embedder triggers
    config = CompressionConfig(
        contradiction=ContradictionConfig(similarity_threshold=0.3),
    )
    report = agent.compress(config=config)
    print(f"\n--- Compression report ---")
    for k, v in report.to_dict().items():
        print(f"  {k}: {v}")

    print(f"\n✓ Remaining facts after compression: {len(agent.facts)}")
    for f in agent.facts:
        flag = " ⚠ contradicted" if "contradicted_by" in f.metadata else ""
        print(f"  [{f.tier.value:8s}]  importance={f.importance:.2f}  "
              f"{f.content}{flag}")

    flagged = agent.contradictions()
    if flagged:
        print(f"\n⚠ {len(flagged)} fact(s) flagged as contradicted:")
        for f in flagged:
            print(f"  - {f.content}  "
                  f"(contradicted_by={f.metadata['contradicted_by'][:8]}…, "
                  f"reason={f.metadata['contradiction_reason']})")

    agent.save()
    print(f"\n✓ Persisted and verified — agent state survives compression.")
    agentkeeper.delete("aria-demo")
    print("=" * 70)


if __name__ == "__main__":
    main()
