"""AgentKeeper demo — cognitive continuity in action.

This demo shows the AK-2 capabilities:
- Persistent identity (name, role, principles, constraints)
- Smart memory routing (semantic / episodic / working / archival)
- Importance ranking with auto-inferred defaults
- Cross-restart continuity

It uses the MockAdapter so it runs without any API key.
"""

from __future__ import annotations

from datetime import datetime

import agentkeeper


def main() -> None:
    print("=" * 70)
    print("AgentKeeper demo — cognitive continuity infrastructure")
    print("=" * 70)

    # 1. Create an agent and define its persistent identity
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

    # 2. Smart routing — let AgentKeeper infer the tier
    agent.remember("budget: 50000 EUR")              # → semantic
    agent.remember("client refused offer A yesterday")  # → episodic
    agent.remember("never accept gifts above 50 EUR")   # → principle (high importance)
    print(f"✓ Smart-routed 3 facts:")
    for f in agent.facts[-3:]:
        print(f"    [{f.tier.value:10s}] importance={f.importance:.2f}  {f.content}")

    # 3. Explicit helpers when you want full control
    agent.fact("client name: Acme Corporation", importance=0.95)
    agent.event(
        "contract signed",
        when=datetime(2026, 5, 15, 14, 0, 0),
        importance=0.8,
    )
    agent.principle("always confirm budget changes in writing")
    print(f"✓ Added 3 explicit facts via fact/event/principle helpers")

    # 4. Reconstruction stats for the target model
    stats = agent.stats()
    print()
    print("Reconstruction stats:")
    print(f"  identity present:   {stats['identity_present']} "
          f"(cost: {stats['identity_token_cost']} tokens)")
    print(f"  total facts:        {stats['total_facts']}")
    print(f"  selected facts:     {stats['selected_facts']}")
    print(f"  critical recovery:  {stats['critical_recovery_rate'] * 100:.0f}%")
    print(f"  tier breakdown:     {stats['tier_breakdown']}")
    print(f"  tokens used:        {stats['tokens_used']} / {stats['token_budget']}")

    # 5. Persist
    agent.save()
    print(f"\n✓ Saved")

    # 6. Ask a question — the mock adapter echoes the system prompt
    response = agent.ask("What do we know about the Acme deal?")
    print(f"\nReconstructed system prompt (first 600 chars):")
    print(response[:600])
    print("...")

    # 7. Reload — identity and tiers all survive
    reloaded = agentkeeper.load("aria-001", provider="mock")
    print(f"\n✓ Reloaded: {reloaded}")
    print(f"  identity preserved: name={reloaded.identity.name!r}, "
          f"role={reloaded.identity.role!r}")
    print(f"  principles preserved: {len(reloaded.identity.principles)}")
    print(f"  constraints preserved: {len(reloaded.identity.constraints)}")

    # 8. Cleanup
    agentkeeper.delete("aria-001")
    print(f"✓ Deleted")
    print("=" * 70)


if __name__ == "__main__":
    main()
