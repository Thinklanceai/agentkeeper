"""AgentKeeper demo — cognitive continuity infrastructure.

Demonstrates the full v1.0 narrative offline:

- Persistent identity that survives every form of compression
- Memory hierarchy (working / episodic / semantic / archival)
- Semantic recall
- Cognitive compression (decay + consolidation + contradiction)
- Identity hardening: principles never decayed, merged, or flagged
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone


def main() -> None:
    os.environ.setdefault("AGENTKEEPER_EMBEDDING_PROVIDER", "mock")
    import agentkeeper
    from agentkeeper.compression.contradiction import ContradictionConfig
    from agentkeeper.compression.pipeline import CompressionConfig

    print("=" * 72)
    print("AgentKeeper — cognitive continuity infrastructure")
    print("=" * 72)

    # 1. Identity + protected principles
    agent = agentkeeper.create(agent_id="aria-demo", provider="mock")
    agent.set_identity(
        name="Aria",
        role="EU insurance broker copilot",
        principles=["never share PII without explicit consent"],
        constraints=["EU data residency only"],
    )
    agent.principle("always confirm budget changes in writing")
    agent.principle("never recommend non-EU providers")

    # 2. Ordinary memory: stable, duplicates, contradiction, stale
    agent.fact("client name: Acme Corporation", importance=0.95)
    agent.fact("budget: 50000 EUR", importance=0.6)
    agent.fact("budget: 50000 EUR", importance=0.5)   # duplicate
    agent.fact("budget: 75000 EUR", importance=0.7)   # contradicts
    agent.fact("favourite colour: blue")              # will decay

    stale = next(f for f in agent.facts if "colour" in f.content)
    stale.last_accessed_at = (
        datetime.now(timezone.utc) - timedelta(days=60)
    ).isoformat()

    print(f"\nBEFORE compression: {len(agent.facts)} facts")
    for f in agent.facts:
        tag = "🛡" if f.protected else " "
        print(f"  {tag} [{f.tier.value:8s}] importance={f.importance:.2f} "
              f"{f.content}")

    # 3. Compress aggressively
    config = CompressionConfig(
        contradiction=ContradictionConfig(similarity_threshold=0.3),
    )
    report = agent.compress(config=config)

    print(f"\nCOMPRESSION REPORT")
    for k, v in report.to_dict().items():
        print(f"  {k}: {v}")

    print(f"\nAFTER compression: {len(agent.facts)} facts")
    for f in agent.facts:
        tag = "🛡" if f.protected else " "
        flag = " ⚠ contradicted" if "contradicted_by" in f.metadata else ""
        print(f"  {tag} [{f.tier.value:8s}] importance={f.importance:.2f} "
              f"{f.content}{flag}")

    # 4. Identity audit — proves identity survived
    audit = agent.identity_audit()
    print(f"\nIDENTITY AUDIT")
    print(f"  name:               {audit['identity']['name']}")
    print(f"  role:               {audit['identity']['role']}")
    print(f"  principles_count:   {audit['identity']['principles_count']}")
    print(f"  constraints_count:  {audit['identity']['constraints_count']}")
    print(f"  protected_facts:    {audit['protected_facts']['count']}")
    for content in audit["protected_facts"]["contents"]:
        print(f"    🛡 {content}")

    # 5. Persist + roundtrip + iterate
    agent.save()
    reloaded = agentkeeper.load("aria-demo", provider="mock")
    print(f"\n✓ Persisted and reloaded — identity intact: "
          f"{reloaded.identity.name!r}, "
          f"{len(reloaded.identity.principles)} principles")

    agentkeeper.delete("aria-demo")
    print("=" * 72)


if __name__ == "__main__":
    main()
