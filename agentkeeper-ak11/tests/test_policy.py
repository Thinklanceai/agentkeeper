"""Tests for MemoryPolicy precedence rules."""

from __future__ import annotations

import pytest

from agentkeeper.cso.fact_types import FactType
from agentkeeper.cso.tiers import MemoryTier
from agentkeeper.errors import ConfigurationError
from agentkeeper.retention.policy import MemoryPolicy


class TestPolicyConstruction:
    def test_empty_policy_valid(self) -> None:
        p = MemoryPolicy()
        assert p.default_ttl is None

    def test_unknown_fact_type_in_policy_rejected(self) -> None:
        with pytest.raises(ValueError):
            MemoryPolicy(per_type={"banana": "1h"})

    def test_unknown_tier_in_policy_rejected(self) -> None:
        with pytest.raises(ValueError):
            MemoryPolicy(per_tier={"banana": "1h"})

    def test_bad_ttl_in_policy_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            MemoryPolicy(default_ttl="never")


class TestPolicyPrecedence:
    def test_default_ttl_only(self) -> None:
        p = MemoryPolicy(default_ttl="30d")
        assert p.resolve(FactType.FACT, MemoryTier.SEMANTIC, False) == "30d"

    def test_per_type_overrides_default(self) -> None:
        p = MemoryPolicy(default_ttl="30d", per_type={"transient": "1h"})
        assert p.resolve(FactType.TRANSIENT, MemoryTier.WORKING, False) == "1h"
        assert p.resolve(FactType.FACT, MemoryTier.SEMANTIC, False) == "30d"

    def test_per_tier_overrides_default(self) -> None:
        p = MemoryPolicy(default_ttl="30d", per_tier={"working": "1d"})
        assert p.resolve(FactType.FACT, MemoryTier.WORKING, False) == "1d"
        assert p.resolve(FactType.FACT, MemoryTier.SEMANTIC, False) == "30d"

    def test_per_type_beats_per_tier(self) -> None:
        p = MemoryPolicy(
            default_ttl="30d",
            per_type={"transient": "1h"},
            per_tier={"working": "1d"},
        )
        # transient + working: type wins
        assert p.resolve(FactType.TRANSIENT, MemoryTier.WORKING, False) == "1h"


class TestPolicyProtected:
    def test_protected_facts_skip_policy_by_default(self) -> None:
        p = MemoryPolicy(default_ttl="30d")
        assert p.resolve(FactType.IDENTITY, MemoryTier.SEMANTIC, True) is None

    def test_can_opt_out_of_protection(self) -> None:
        p = MemoryPolicy(default_ttl="30d", respect_protected=False)
        assert p.resolve(FactType.IDENTITY, MemoryTier.SEMANTIC, True) == "30d"
