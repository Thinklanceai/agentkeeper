"""AgentKeeper demo — cognitive continuity infrastructure.

Showcases AK-2 + AK-3 capabilities:
- Persistent identity (always injected)
- Smart memory routing (semantic / episodic / working / archival)
- Importance ranking with auto-inference
- Semantic recall — find facts by meaning, not keywords

Uses the MockAdapter and Mock embedding provider so it runs offline
without any API key.
"""

from __future__ import annotations

import os
from datetime import datetime


def main() -> None:
    # Use mock embeddings for offline demo. In production, leave this
    # unset to use sentence-transformers (recommended) or set it to
    # 'openai' for cloud embeddings.
    os.environ.setdefault("AGENTKEEPER_EMBEDDING_PROVIDER", "mock")

    import agentkeeper

    print("=" * 70)
    print("AgentKeeper demo — cognitive continuity infrastructure")
    print("=" * 70)

    # 1. Create an agent with a persistent identity
    agent = agentkeeper.create(agent_id="aria-001", provider="mock")
    agent.set_identity(
        name="Aria",
        role="EU insurance broker copilot",
        principles=[
            "never share PII without explicit consent",
            "always disclose conflicts of interest",
        ],
        constraints=[
            "EU data residency only",
            "never recommend non-EU providers",
        ],
    )
    print(f"\n✓ Created: {agent}")

    # 2. Smart-routed memory
    agent.remember("budget: 50000 EUR")
    agent.remember("client refused offer A yesterday")
    agent.remember("never accept gifts above 50 EUR")
    agent.fact("client name: Acme Corporation", importance=0.95)
    agent.event(
        "contract signed",
        when=datetime(2026, 5, 15, 14, 0, 0),
        importance=0.8,
    )
    agent.principle("always confirm budget changes in writing")
    print(f"✓ {len(agent.facts)} facts indexed across tiers")

    # 3. Semantic recall — meaning-based
    print("\nSemantic recall: 'money allocated to the project'")
    for fact, score in agent.recall("money allocated to the project", top_k=3):
        print(f"  {score:.3f}  [{fact.tier.value}]  {fact.content}")

    # 4. Save and reload
    agent.save()
    reloaded = agentkeeper.load("aria-001", provider="mock")
    print(f"\n✓ Reloaded: {reloaded}")
    print(f"  identity preserved: {reloaded.identity.name!r}")
    print(f"  principles: {len(reloaded.identity.principles)}, "
          f"constraints: {len(reloaded.identity.constraints)}")

    # 5. Ask the agent — context is reconstructed with identity + tiers
    response = reloaded.ask("What do we know about the Acme deal?")
    print(f"\nReconstructed system prompt (first 500 chars):")
    print(response[:500])
    print("...")

    agentkeeper.delete("aria-001")
    print("\n✓ Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
