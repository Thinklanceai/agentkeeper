"""Minimal AgentKeeper demo.

Shows the canonical lifecycle: create → remember → save → load → ask.
Uses the MockAdapter so it runs without any API key.

Run with:
    python examples/demo.py
"""

from __future__ import annotations

import agentkeeper


def main() -> None:
    print("=" * 60)
    print("AgentKeeper demo — cognitive continuity in 10 lines")
    print("=" * 60)

    # 1. Create an agent
    agent = agentkeeper.create(agent_id="demo-001", provider="mock")
    print(f"\n✓ Created: {agent}")

    # 2. Teach it some facts
    (
        agent.remember("project budget: 50000 EUR", critical=True)
        .remember("client: Acme Corporation", critical=True)
        .remember("tech stack: Python FastAPI", critical=True)
        .remember("favorite color: blue")
    )
    print(f"✓ Remembered {len(agent.facts)} facts "
          f"({len(agent._cso.critical_facts())} critical)")

    # 3. Reconstruction stats for the target model
    stats = agent.stats()
    print(f"\nReconstruction stats:")
    print(f"  critical recovery: {stats['critical_recovery_rate'] * 100:.0f}%")
    print(f"  tokens used:       {stats['tokens_used']} / {stats['token_budget']}")

    # 4. Persist
    agent.save()
    print(f"\n✓ Saved")

    # 5. Switch provider — memory survives
    agent.switch_provider("mock")  # for the demo we stay on mock
    response = agent.ask("What is our budget and which stack do we use?")
    print(f"\nResponse:\n{response[:300]}...")

    # 6. Reload from disk in a fresh process simulation
    reloaded = agentkeeper.load("demo-001", provider="mock")
    print(f"\n✓ Reloaded: {reloaded}")

    # 7. Cleanup
    agentkeeper.delete("demo-001")
    print(f"✓ Deleted")


if __name__ == "__main__":
    main()
