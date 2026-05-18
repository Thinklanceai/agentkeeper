"""Importance decay over time.

Cognitive systems forget. AgentKeeper models this with an exponential
decay applied to each fact's `importance` field based on how long it
has been since the fact was last accessed.

Critical facts (importance >= 0.9) and identity-level statements are
exempt from decay — they are the agent's stable self.

The decay is intentionally gentle and predictable: a half-life of
30 days for ordinary facts, infinite for criticals. Tunable via
`DecayConfig`.

Decay is a *pure function*: it takes a Fact and a moment in time, and
returns a new effective importance. Nothing is mutated unless the
caller (the compression pipeline) explicitly chooses to persist the
new value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from ..cso.types import Fact


@dataclass(frozen=True)
class DecayConfig:
    """Parameters controlling importance decay."""

    # Half-life in days for non-critical facts.
    half_life_days: float = 30.0
    # Facts with importance >= this value never decay.
    immortal_threshold: float = 0.9
    # Floor — decayed importance never goes below this.
    importance_floor: float = 0.05


DEFAULT_DECAY = DecayConfig()


def _parse_iso(timestamp: str) -> datetime:
    """Parse an ISO-8601 timestamp produced by the rest of the system."""
    dt = datetime.fromisoformat(timestamp)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def days_since_last_access(fact: Fact, now: datetime | None = None) -> float:
    """Return how many days have passed since the fact was last accessed."""
    now = now or datetime.now(timezone.utc)
    accessed = _parse_iso(fact.last_accessed_at)
    delta = now - accessed
    return delta.total_seconds() / 86400.0


def decayed_importance(
    fact: Fact,
    now: datetime | None = None,
    config: DecayConfig = DEFAULT_DECAY,
) -> float:
    """Compute the effective importance of a fact at time `now`.

    Pure function — does not mutate the fact. Protected facts and
    facts at or above the immortal threshold are returned unchanged.
    """
    if fact.protected or fact.importance >= config.immortal_threshold:
        return fact.importance

    days = max(0.0, days_since_last_access(fact, now=now))
    # Exponential decay: importance(t) = importance(0) * (1/2) ** (t / half_life)
    decay_factor = math.pow(0.5, days / config.half_life_days)
    decayed = fact.importance * decay_factor
    return max(config.importance_floor, decayed)


def apply_decay_in_place(
    facts: list[Fact],
    now: datetime | None = None,
    config: DecayConfig = DEFAULT_DECAY,
) -> int:
    """Mutate facts' `importance` to their decayed values.

    Returns the number of facts whose importance changed.
    Protected and critical facts are left untouched.
    """
    changed = 0
    for fact in facts:
        if fact.protected or fact.importance >= config.immortal_threshold:
            continue
        new_importance = decayed_importance(fact, now=now, config=config)
        if not math.isclose(new_importance, fact.importance, rel_tol=1e-6):
            fact.importance = new_importance
            changed += 1
    return changed
