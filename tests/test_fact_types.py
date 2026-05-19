"""Tests for the FactType enum and decay multipliers."""

from __future__ import annotations

import pytest

from agentkeeper.cso.fact_types import (
    TYPE_DECAY_MULTIPLIER,
    FactType,
    half_life_multiplier,
    is_valid_fact_type,
)


class TestFactTypeEnum:
    def test_known_values(self) -> None:
        for name in (
            "decision",
            "preference",
            "constraint",
            "relationship",
            "task_state",
            "transient",
            "identity",
            "event",
            "fact",
        ):
            assert FactType(name).value == name

    def test_is_valid_fact_type(self) -> None:
        assert is_valid_fact_type("decision") is True
        assert is_valid_fact_type("not-a-real-type") is False


class TestDecayMultipliers:
    @pytest.mark.parametrize(
        "fact_type,expected",
        [
            (FactType.DECISION, 5.0),
            (FactType.CONSTRAINT, 5.0),
            (FactType.IDENTITY, 10.0),
            (FactType.PREFERENCE, 2.0),
            (FactType.RELATIONSHIP, 1.5),
            (FactType.FACT, 1.0),
            (FactType.EVENT, 1.0),
            (FactType.TASK_STATE, 0.5),
            (FactType.TRANSIENT, 0.2),
        ],
    )
    def test_multiplier_values(self, fact_type: FactType, expected: float) -> None:
        assert half_life_multiplier(fact_type) == expected

    def test_multipliers_cover_all_types(self) -> None:
        for t in FactType:
            assert t in TYPE_DECAY_MULTIPLIER
            assert TYPE_DECAY_MULTIPLIER[t] > 0
