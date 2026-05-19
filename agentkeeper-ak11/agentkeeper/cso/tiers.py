"""Memory hierarchy primitives.

A cognitive system is more than a flat list of facts. AgentKeeper organises
memory into four tiers, inspired by classical cognitive psychology
(William James, Tulving, Baddeley):

- `working`   : immediate, short-lived context for the current task. High
                turnover. Auto-evicted when the working set grows beyond
                a small threshold.
- `episodic`  : time-anchored events. "On March 15, the client refused
                offer A." Preserves temporal ordering for narrative recall.
- `semantic`  : stable facts about the world. "Budget is 50k EUR." This
                is the default tier and matches the v0.1 behaviour.
- `archival`  : long-term, compressed knowledge. Reserved for v1.0
                cognitive compression (Sprint AK-4). At AK-2 this tier
                exists structurally but no fact is auto-routed to it yet.

Each tier has a default selection priority used by the CRE. Higher tier
priority means "consider these first when reconstructing context".
"""

from __future__ import annotations

from enum import Enum


class MemoryTier(str, Enum):
    """A cognitive memory tier."""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    ARCHIVAL = "archival"


# Default selection priority within the CRE.
# Higher = considered first during reconstruction.
# These can be overridden per-CRE instance in advanced setups.
TIER_PRIORITY: dict[MemoryTier, int] = {
    MemoryTier.SEMANTIC: 4,   # stable, structured, highest signal-to-noise
    MemoryTier.EPISODIC: 3,   # contextual, time-anchored
    MemoryTier.WORKING: 2,    # immediate but ephemeral
    MemoryTier.ARCHIVAL: 1,   # compressed, lower fidelity, last resort
}


def is_valid_tier(value: str) -> bool:
    """Return True if `value` is a known MemoryTier name."""
    try:
        MemoryTier(value)
        return True
    except ValueError:
        return False
