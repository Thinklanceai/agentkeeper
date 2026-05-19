"""Tests that decay respects per-type multipliers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from agentkeeper.compression.decay import DecayConfig, decayed_importance
from agentkeeper.cso.fact_types import FactType
from agentkeeper.cso.types import Fact


def _utc(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def _aged_fact(fact_type: FactType, days: float) -> Fact:
    f = Fact.create("content", fact_type=fact_type, importance=0.8)
    f.last_accessed_at = _utc(days_ago=days)
    return f


class TestTypeAwareDecay:
    def test_decision_decays_slower_than_fact(self) -> None:
        decision = _aged_fact(FactType.DECISION, days=60)
        fact = _aged_fact(FactType.FACT, days=60)
        # decision multiplier = 5.0, fact multiplier = 1.0
        assert decayed_importance(decision) > decayed_importance(fact)

    def test_transient_decays_faster_than_fact(self) -> None:
        transient = _aged_fact(FactType.TRANSIENT, days=30)
        fact = _aged_fact(FactType.FACT, days=30)
        # transient multiplier = 0.2, fact multiplier = 1.0
        assert decayed_importance(transient) < decayed_importance(fact)

    def test_decision_30_days_minimal_decay(self) -> None:
        # decision: half_life = 30d * 5.0 = 150d. After 30d, decay
        # factor ≈ 0.5^(30/150) = 0.871 → importance ≈ 0.697
        decision = _aged_fact(FactType.DECISION, days=30)
        result = decayed_importance(decision, config=DecayConfig(half_life_days=30))
        assert 0.65 < result < 0.75

    def test_transient_30_days_heavy_decay(self) -> None:
        # transient: half_life = 30d * 0.2 = 6d. After 30d, decay
        # factor = 0.5^(30/6) = 0.03125 → importance hits the floor 0.05
        transient = _aged_fact(FactType.TRANSIENT, days=30)
        result = decayed_importance(transient, config=DecayConfig(half_life_days=30))
        assert result <= 0.06  # floored

    def test_protected_fact_decay_untouched(self) -> None:
        # Even with a fast-decay type, protected facts are immortal.
        f = Fact.create(
            "principle",
            fact_type=FactType.TRANSIENT,
            protected=True,
        )
        f.last_accessed_at = _utc(days_ago=10_000)
        assert decayed_importance(f) >= 0.95
