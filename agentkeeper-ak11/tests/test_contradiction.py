"""Tests for contradiction detection and arbitration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from agentkeeper.compression.contradiction import (
    ContradictionConfig,
    detect_and_resolve,
)
from agentkeeper.cso.types import CognitiveStateObject
from agentkeeper.semantic.mock import MockEmbeddingProvider


def _later(seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


class TestDetection:
    def test_no_contradictions_on_unrelated_facts(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        cso.add_fact("budget: 50000 EUR")
        cso.add_fact("client: Acme Corporation")
        provider = MockEmbeddingProvider()
        result = detect_and_resolve(cso.memory_facts, provider)
        assert result.pairs_found == 0

    def test_key_value_divergence_detected(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        cso.add_fact("budget: 50000 EUR")
        new = cso.add_fact("budget: 50000 EUR")
        # Force same key, different value
        new.content = "budget: 75000 EUR"
        new.last_accessed_at = _later(60)
        provider = MockEmbeddingProvider()
        # With identical key prefix the embeddings are close enough
        result = detect_and_resolve(
            cso.memory_facts,
            provider,
            config=ContradictionConfig(similarity_threshold=0.3),
        )
        assert result.pairs_found >= 1
        # The more recent fact should win
        winner_id = result.contradictions[0].winner_id
        assert winner_id == new.id

    def test_polarity_opposition_detected(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        cso.add_fact("client agreed to terms")
        cso.add_fact("client refused to terms")
        provider = MockEmbeddingProvider()
        result = detect_and_resolve(
            cso.memory_facts,
            provider,
            config=ContradictionConfig(similarity_threshold=0.3),
        )
        # The mock embedder gives moderate similarity for these. With
        # threshold 0.3 they should be considered "about the same subject"
        # and the polarity-opposition rule triggers.
        assert result.pairs_found >= 1


class TestArbitration:
    def test_critical_beats_non_critical(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        old_critical = cso.add_fact("budget: 50000 EUR", critical=True)
        new_non_critical = cso.add_fact("budget: 50000 EUR")
        new_non_critical.content = "budget: 75000 EUR"
        new_non_critical.last_accessed_at = _later(1000)
        provider = MockEmbeddingProvider()
        result = detect_and_resolve(
            cso.memory_facts,
            provider,
            config=ContradictionConfig(similarity_threshold=0.3),
        )
        assert result.pairs_found >= 1
        # Critical wins despite being older
        assert result.contradictions[0].winner_id == old_critical.id

    def test_loser_importance_reduced_not_deleted(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        a = cso.add_fact("budget: 50000 EUR")
        a.importance = 0.5
        b = cso.add_fact("budget: 50000 EUR")
        b.content = "budget: 75000 EUR"
        b.importance = 0.6  # b wins
        provider = MockEmbeddingProvider()
        detect_and_resolve(
            cso.memory_facts,
            provider,
            config=ContradictionConfig(similarity_threshold=0.3),
        )
        # Both facts are still present
        assert len(cso.memory_facts) == 2
        # The loser has reduced importance and contradiction metadata
        loser = next(f for f in cso.memory_facts if f.id == a.id)
        assert loser.importance < 0.5
        assert "contradicted_by" in loser.metadata
        assert "contradiction_reason" in loser.metadata
