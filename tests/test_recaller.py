"""Tests for the SemanticRecaller."""

from __future__ import annotations

from agentkeeper.cso.types import CognitiveStateObject
from agentkeeper.semantic.mock import MockEmbeddingProvider
from agentkeeper.semantic.recaller import SemanticRecaller


def _make_recaller() -> tuple[SemanticRecaller, CognitiveStateObject]:
    cso = CognitiveStateObject.create(agent_id="r")
    provider = MockEmbeddingProvider()
    return SemanticRecaller(provider, cso), cso


class TestIndexing:
    def test_empty_cso_indexes_zero(self) -> None:
        recaller, _ = _make_recaller()
        assert recaller.index_all() == 0

    def test_index_all_processes_each_fact_once(self) -> None:
        recaller, cso = _make_recaller()
        cso.add_fact("budget: 50k")
        cso.add_fact("client: Acme")
        assert recaller.index_all() == 2
        # Second call: nothing new to index
        assert recaller.index_all() == 0
        assert recaller.index_size == 2

    def test_changed_content_reindexes(self) -> None:
        recaller, cso = _make_recaller()
        f = cso.add_fact("v1")
        recaller.index_all()
        f.content = "v2"
        # content changed → re-embedding required
        assert recaller.index_all() == 1


class TestRecall:
    def test_recall_empty_query_returns_empty(self) -> None:
        recaller, cso = _make_recaller()
        cso.add_fact("budget: 50k")
        assert recaller.recall("") == []
        assert recaller.recall("   ") == []

    def test_recall_top_k_zero_returns_empty(self) -> None:
        recaller, cso = _make_recaller()
        cso.add_fact("budget: 50k")
        assert recaller.recall("budget", top_k=0) == []

    def test_recall_returns_pairs(self) -> None:
        recaller, cso = _make_recaller()
        cso.add_fact("budget: 50k")
        cso.add_fact("client: Acme")

        results = recaller.recall("budget", top_k=2)
        assert len(results) == 2
        # Tuple shape
        fact, score = results[0]
        assert isinstance(score, float)
        assert fact.id in [f.id for f in cso.memory_facts]

    def test_recall_exact_content_match_top1(self) -> None:
        # With the deterministic mock embedder, the same exact string
        # produces the same vector → cosine similarity == 1.0
        recaller, cso = _make_recaller()
        f_target = cso.add_fact("client: Acme Corporation")
        cso.add_fact("budget: 50k EUR")
        cso.add_fact("deploy target: AWS eu-west-1")

        results = recaller.recall("client: Acme Corporation", top_k=1)
        assert len(results) == 1
        fact, score = results[0]
        assert fact.id == f_target.id
        assert score > 0.99

    def test_recall_skips_deleted_fact(self) -> None:
        recaller, cso = _make_recaller()
        f1 = cso.add_fact("budget: 50k")
        cso.add_fact("client: Acme")
        recaller.index_all()
        # Simulate forget without going through Agent
        cso.memory_facts = [f for f in cso.memory_facts if f.id != f1.id]
        results = recaller.recall("budget: 50k", top_k=5)
        # The deleted fact should not appear in results
        assert f1.id not in [f.id for f, _ in results]
