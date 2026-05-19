"""Memory classes — typed cognitive vocabulary for facts.

A `Fact` is not just text + importance. Different *kinds* of facts age
differently and play different roles in reconstruction. AgentKeeper
distinguishes:

- `decision`     : a choice the agent has made (use library X, switch
                   provider Y). Very long-lived; almost as resistant to
                   decay as principles.
- `preference`   : a soft inclination (favourite colour, preferred
                   communication style). Long-lived but updatable.
- `constraint`   : a hard limit that came from the environment, not
                   from identity (token budget, deployment region).
                   Distinct from `AgentIdentity.constraints` which are
                   *immutable* identity rules — these are situational.
- `relationship` : who is who and how do they relate
                   (client X is owned by team Y).
- `task_state`   : current progress on an ongoing task. Decays fast
                   once the task is done.
- `transient`    : ephemeral working-memory items (recent messages,
                   intermediate computations). Decays very fast.
- `identity`     : self-referential statements about the agent itself.
                   Internally promoted to protected/critical.
- `event`        : episodic — already first-class via the `episodic`
                   tier and the `agent.event()` helper. Kept here for
                   completeness so users can be explicit.
- `fact`         : the generic semantic statement. Default.

The `fact_type` is **orthogonal** to `tier`. A `decision` is normally
stored at the `semantic` tier (stable knowledge), but the type drives
*decay behaviour*: decisions decay 5× slower than transient notes.

Why decay multipliers and not separate decay configs? Because we already
have a clean `DecayConfig(half_life_days=...)`. Multiplying the half-life
per type composes cleanly without rewriting the pipeline.
"""

from __future__ import annotations

from enum import Enum


class FactType(str, Enum):
    """Cognitive class of a fact.

    Drives decay rate and reconstruction prioritisation. Independent of
    `MemoryTier` (working/episodic/semantic/archival).
    """

    DECISION = "decision"
    PREFERENCE = "preference"
    CONSTRAINT = "constraint"
    RELATIONSHIP = "relationship"
    TASK_STATE = "task_state"
    TRANSIENT = "transient"
    IDENTITY = "identity"
    EVENT = "event"
    FACT = "fact"


# Per-type multipliers applied to the base `half_life_days` from
# `DecayConfig`. >1 means slower decay (more durable), <1 means faster
# decay (more ephemeral). Tuned so that:
#   - decisions and constraints last ~5x longer than the base default
#   - transient items expire ~5x faster
#   - fact / event / relationship sit at the baseline
TYPE_DECAY_MULTIPLIER: dict[FactType, float] = {
    FactType.DECISION:     5.0,
    FactType.CONSTRAINT:   5.0,
    FactType.IDENTITY:     10.0,   # nearly immortal; protected facts use this
    FactType.PREFERENCE:   2.0,
    FactType.RELATIONSHIP: 1.5,
    FactType.FACT:         1.0,
    FactType.EVENT:        1.0,
    FactType.TASK_STATE:   0.5,
    FactType.TRANSIENT:    0.2,
}


def half_life_multiplier(fact_type: FactType) -> float:
    """Return the decay-rate multiplier for a fact type.

    Used by `compression.decay` to compute the effective half-life:

        effective_half_life = base_half_life_days * multiplier

    A `decision` with the default base of 30 days effectively decays
    over 150 days; a `transient` decays over 6 days.
    """
    return TYPE_DECAY_MULTIPLIER.get(fact_type, 1.0)


def is_valid_fact_type(value: str) -> bool:
    try:
        FactType(value)
        return True
    except ValueError:
        return False
