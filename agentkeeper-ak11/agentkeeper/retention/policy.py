"""Memory retention policies.

A `MemoryPolicy` declares default TTLs for facts based on their type
or tier. Policies are applied at *write time* — when a fact is added
without an explicit `ttl`, the policy decides what to inject.

Example:

    policy = MemoryPolicy(
        default_ttl="90d",
        per_type={"transient": "1h", "task_state": "7d"},
        per_tier={"working": "1d"},
    )
    agent.set_memory_policy(policy)
    agent.transient("ephemeral note")  # auto-expires in 1 hour

Protected facts (principles, identity) are NEVER given a TTL by the
policy. They are the agent's stable self; expiring them would defeat
the purpose of identity hardening.

Precedence (most specific wins):
    explicit `ttl=` argument  >  per_type  >  per_tier  >  default_ttl
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from ..cso.fact_types import FactType
from ..cso.tiers import MemoryTier
from .ttl import parse_ttl


@dataclass
class MemoryPolicy:
    """Global retention defaults for an agent.

    Attributes:
        default_ttl: TTL applied to facts that match no more-specific
            rule. `None` means "no expiration by default".
        per_type: Override by FactType.value (e.g. `"transient": "1h"`).
        per_tier: Override by MemoryTier.value (e.g. `"working": "1d"`).
        respect_protected: When True (default), protected facts never
            receive a TTL even if a policy would normally apply. Strongly
            recommended.
    """

    default_ttl: timedelta | str | int | float | None = None
    per_type: dict[str, timedelta | str | int | float] = field(default_factory=dict)
    per_tier: dict[str, timedelta | str | int | float] = field(default_factory=dict)
    respect_protected: bool = True

    def __post_init__(self) -> None:
        # Validate (and normalise) every TTL eagerly so callers fail
        # fast on bad config rather than at the first `remember`.
        if self.default_ttl is not None:
            parse_ttl(self.default_ttl)
        for key, value in self.per_type.items():
            FactType(key)  # raises if unknown
            parse_ttl(value)
        for key, value in self.per_tier.items():
            MemoryTier(key)  # raises if unknown
            parse_ttl(value)

    def resolve(
        self,
        fact_type: FactType,
        tier: MemoryTier,
        protected: bool,
    ) -> timedelta | str | int | float | None:
        """Return the TTL to apply, or `None` for no expiration."""
        if protected and self.respect_protected:
            return None
        if fact_type.value in self.per_type:
            return self.per_type[fact_type.value]
        if tier.value in self.per_tier:
            return self.per_tier[tier.value]
        return self.default_ttl
