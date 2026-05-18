"""End-to-end tests for the compression pipeline."""

from __future__ import annotations

from agentkeeper.compression.pipeline import CompressionConfig, compress
from agentkeeper.cso.types import CognitiveStateObject
from agentkeeper.semantic.mock import MockEmbeddingProvider


class TestPipeline:
    def test_empty_cso_runs_cleanly(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        report = compress(cso, embedding_provider=MockEmbeddingProvider())
        assert report.facts_before == 0
        assert report.facts_after == 0
        assert report.decayed_facts == 0

    def test_report_captures_all_phases(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        cso.add_fact("budget: 50000 EUR")
        cso.add_fact("budget: 50000 EUR")  # duplicate
        report = compress(cso, embedding_provider=MockEmbeddingProvider())
        d = report.to_dict()
        assert "consolidation" in d
        assert "contradictions" in d
        assert "decayed_facts" in d
        assert d["facts_before"] == 2

    def test_can_disable_individual_passes(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        cso.add_fact("budget: 50000 EUR")
        cso.add_fact("budget: 50000 EUR")
        config = CompressionConfig(
            run_decay=False,
            run_consolidation=False,
            run_contradiction=False,
        )
        report = compress(
            cso, embedding_provider=MockEmbeddingProvider(), config=config
        )
        assert report.decayed_facts == 0
        assert report.consolidation.facts_removed == 0
        assert report.contradictions.pairs_found == 0
        assert len(cso.memory_facts) == 2

    def test_consolidation_reduces_fact_count(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        cso.add_fact("budget: 50000 EUR")
        cso.add_fact("budget: 50000 EUR")
        cso.add_fact("budget: 50000 EUR")
        report = compress(cso, embedding_provider=MockEmbeddingProvider())
        assert report.facts_after < report.facts_before

    def test_updated_at_timestamp_bumped(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        cso.add_fact("x")
        before = cso.updated_at
        compress(cso, embedding_provider=MockEmbeddingProvider())
        assert cso.updated_at >= before
