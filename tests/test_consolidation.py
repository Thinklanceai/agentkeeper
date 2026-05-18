"""Tests for semantic consolidation."""

from __future__ import annotations

from agentkeeper.compression.consolidation import (
    ConsolidationConfig,
    consolidate,
)
from agentkeeper.cso.types import CognitiveStateObject
from agentkeeper.semantic.mock import MockEmbeddingProvider


class TestConsolidate:
    def test_no_op_on_unique_facts(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        cso.add_fact("budget: 50k EUR")
        cso.add_fact("client: Acme Corporation")
        cso.add_fact("technology stack: Python")
        provider = MockEmbeddingProvider()
        result = consolidate(cso.memory_facts, provider)
        assert result.facts_removed == 0
        assert len(cso.memory_facts) == 3

    def test_exact_duplicates_collapse(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        cso.add_fact("budget: 50000 EUR")
        cso.add_fact("budget: 50000 EUR")
        cso.add_fact("budget: 50000 EUR")
        provider = MockEmbeddingProvider()
        result = consolidate(cso.memory_facts, provider)
        # With the mock embedder, identical strings produce identical
        # vectors → similarity 1.0 → all collapse to one cluster.
        assert result.clusters_found >= 1
        assert result.facts_removed == 2
        assert len(cso.memory_facts) == 1

    def test_canonical_is_highest_importance(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        weak = cso.add_fact("budget: 50000 EUR")
        weak.importance = 0.3
        strong = cso.add_fact("budget: 50000 EUR")
        strong.importance = 0.9
        provider = MockEmbeddingProvider()
        consolidate(cso.memory_facts, provider)
        assert len(cso.memory_facts) == 1
        assert cso.memory_facts[0].id == strong.id

    def test_synthesiser_called_for_clusters(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        cso.add_fact("budget: 50k")
        cso.add_fact("budget: 50k")
        provider = MockEmbeddingProvider()

        synth_calls: list[int] = []

        def synth(cluster: list) -> str:
            synth_calls.append(len(cluster))
            return "consolidated: budget 50k"

        consolidate(cso.memory_facts, provider, synthesiser=synth)
        assert synth_calls == [2]
        assert cso.memory_facts[0].content == "consolidated: budget 50k"

    def test_archival_tier_not_touched_by_default(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        cso.add_fact("budget: 50k")
        # Default target_tiers excludes archival
        config = ConsolidationConfig(target_tiers=("semantic",))
        f = cso.add_fact("budget: 50k", tier="semantic")
        f.tier = type(f.tier).ARCHIVAL
        provider = MockEmbeddingProvider()
        result = consolidate(cso.memory_facts, provider, config=config)
        # Only the one in `semantic` tier is targetable → no clustering possible
        assert result.facts_removed == 0
