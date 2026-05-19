"""Tests for tier inference and tier metadata."""

from __future__ import annotations

import pytest

from agentkeeper.cso.inference import infer_importance, infer_tier
from agentkeeper.cso.tiers import (
    TIER_PRIORITY,
    MemoryTier,
    is_valid_tier,
)


class TestTierEnum:
    def test_values_are_string_compatible(self) -> None:
        assert MemoryTier.SEMANTIC.value == "semantic"
        assert MemoryTier.EPISODIC.value == "episodic"
        assert MemoryTier.WORKING.value == "working"
        assert MemoryTier.ARCHIVAL.value == "archival"

    def test_priority_defined_for_all_tiers(self) -> None:
        for tier in MemoryTier:
            assert tier in TIER_PRIORITY

    def test_is_valid_tier(self) -> None:
        assert is_valid_tier("semantic")
        assert is_valid_tier("episodic")
        assert not is_valid_tier("invalid-tier")


class TestInferTier:
    @pytest.mark.parametrize(
        "content",
        [
            "Meeting on 2025-03-15 was productive",
            "Yesterday we shipped the release",
            "Demain est un jour important",
            "Last week the client cancelled",
            "Contract signed on March 15",
            "Hier nous avons décidé du budget",
        ],
    )
    def test_temporal_content_routes_to_episodic(self, content: str) -> None:
        assert infer_tier(content) == MemoryTier.EPISODIC

    @pytest.mark.parametrize(
        "content",
        [
            "The contract was signed yesterday",
            "Client refused offer A",
            "We agreed on the budget",
            "Deployment occurred without issue",
            "Decided to switch providers",
        ],
    )
    def test_event_verbs_route_to_episodic(self, content: str) -> None:
        assert infer_tier(content) == MemoryTier.EPISODIC

    @pytest.mark.parametrize(
        "content",
        [
            "budget: 50000 EUR",
            "client name: Acme Corporation",
            "primary contact: Jean Dupont",
            "the database is PostgreSQL 15",
            "team size: 3 engineers",
        ],
    )
    def test_stable_content_routes_to_semantic(self, content: str) -> None:
        assert infer_tier(content) == MemoryTier.SEMANTIC

    def test_arbitrary_content_defaults_to_semantic(self) -> None:
        # Content with no temporal markers, no event verbs
        assert infer_tier("arbitrary opaque content") == MemoryTier.SEMANTIC


class TestInferImportance:
    def test_principles_get_high_importance(self) -> None:
        assert infer_importance("never share PII", MemoryTier.SEMANTIC) == 0.9
        assert infer_importance("always confirm before deletion", MemoryTier.SEMANTIC) == 0.9
        assert infer_importance("must respect GDPR Article 25", MemoryTier.SEMANTIC) == 0.9

    def test_french_principles(self) -> None:
        assert infer_importance("jamais partager les PII", MemoryTier.SEMANTIC) == 0.9
        assert infer_importance("toujours confirmer", MemoryTier.SEMANTIC) == 0.9

    def test_key_value_semantic_facts(self) -> None:
        assert infer_importance("budget: 50k", MemoryTier.SEMANTIC) == 0.6

    def test_stable_predicate_semantic(self) -> None:
        assert (
            infer_importance("the database is PostgreSQL", MemoryTier.SEMANTIC)
            == 0.6
        )

    def test_plain_semantic_default(self) -> None:
        assert infer_importance("xyzzy plover", MemoryTier.SEMANTIC) == 0.5

    def test_episodic_default(self) -> None:
        assert infer_importance("yesterday it rained", MemoryTier.EPISODIC) == 0.5
