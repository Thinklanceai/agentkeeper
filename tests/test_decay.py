"""Tests for importance decay."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from agentkeeper.compression.decay import (
    DecayConfig,
    apply_decay_in_place,
    days_since_last_access,
    decayed_importance,
)
from agentkeeper.cso.types import CognitiveStateObject


def _utc(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


class TestDaysSinceLastAccess:
    def test_zero_days_for_fresh_fact(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        fact = cso.add_fact("x")
        assert days_since_last_access(fact) < 1.0

    def test_thirty_days_old_fact(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        fact = cso.add_fact("x")
        fact.last_accessed_at = _utc(days_ago=30)
        days = days_since_last_access(fact)
        assert 29.9 < days < 30.1


class TestDecayedImportance:
    def test_critical_facts_immortal(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        fact = cso.add_fact("never share PII", critical=True)
        fact.last_accessed_at = _utc(days_ago=10_000)
        # Critical facts ignore decay entirely
        assert decayed_importance(fact) == fact.importance

    def test_fresh_fact_unchanged(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        fact = cso.add_fact("budget: 50k")
        before = fact.importance
        assert math.isclose(decayed_importance(fact), before, rel_tol=1e-3)

    def test_half_life_decay(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        fact = cso.add_fact("budget: 50k")
        fact.importance = 0.8
        fact.last_accessed_at = _utc(days_ago=30)
        result = decayed_importance(fact, config=DecayConfig(half_life_days=30))
        assert math.isclose(result, 0.4, abs_tol=0.01)

    def test_floor_respected(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        fact = cso.add_fact("ancient note")
        fact.importance = 0.3
        fact.last_accessed_at = _utc(days_ago=10_000)
        result = decayed_importance(fact)
        assert result >= 0.05


class TestApplyDecayInPlace:
    def test_mutates_facts(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        a = cso.add_fact("old note")
        a.importance = 0.7
        a.last_accessed_at = _utc(days_ago=60)
        b = cso.add_fact("fresh note")
        b.importance = 0.7
        changed = apply_decay_in_place(cso.memory_facts)
        assert changed == 1  # only `a` decayed
        assert a.importance < 0.7
        assert b.importance == 0.7

    def test_critical_facts_skipped(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        crit = cso.add_fact("never share", critical=True)
        crit.last_accessed_at = _utc(days_ago=10_000)
        changed = apply_decay_in_place(cso.memory_facts)
        assert changed == 0
        assert crit.importance >= 0.9
