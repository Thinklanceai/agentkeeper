"""Tests for the in-memory vector index."""

from __future__ import annotations

import pytest

from agentkeeper.semantic.index import InMemoryVectorIndex


class TestInMemoryVectorIndex:
    def test_starts_empty(self) -> None:
        idx = InMemoryVectorIndex(dimension=4)
        assert idx.size() == 0
        assert idx.search([1, 0, 0, 0]) == []

    def test_upsert_increases_size(self) -> None:
        idx = InMemoryVectorIndex(dimension=4)
        idx.upsert("a", [1, 0, 0, 0])
        idx.upsert("b", [0, 1, 0, 0])
        assert idx.size() == 2

    def test_upsert_dimension_mismatch_raises(self) -> None:
        idx = InMemoryVectorIndex(dimension=4)
        with pytest.raises(ValueError):
            idx.upsert("a", [1, 0, 0])

    def test_upsert_overwrites(self) -> None:
        idx = InMemoryVectorIndex(dimension=4)
        idx.upsert("a", [1, 0, 0, 0])
        idx.upsert("a", [0, 1, 0, 0])
        assert idx.size() == 1

    def test_delete_removes_vector(self) -> None:
        idx = InMemoryVectorIndex(dimension=4)
        idx.upsert("a", [1, 0, 0, 0])
        idx.delete("a")
        assert idx.size() == 0

    def test_delete_missing_is_noop(self) -> None:
        idx = InMemoryVectorIndex(dimension=4)
        idx.delete("never-existed")  # should not raise

    def test_search_returns_top_k_sorted(self) -> None:
        idx = InMemoryVectorIndex(dimension=4)
        idx.upsert("a", [1, 0, 0, 0])
        idx.upsert("b", [0.9, 0.1, 0, 0])
        idx.upsert("c", [0, 0, 0, 1])
        results = idx.search([1, 0, 0, 0], top_k=2)
        assert len(results) == 2
        assert results[0][0] == "a"
        assert results[0][1] > results[1][1]

    def test_search_respects_min_score(self) -> None:
        idx = InMemoryVectorIndex(dimension=4)
        idx.upsert("a", [1, 0, 0, 0])
        idx.upsert("b", [0, 0, 0, 1])
        results = idx.search([1, 0, 0, 0], top_k=5, min_score=0.5)
        ids = [r[0] for r in results]
        assert "a" in ids
        assert "b" not in ids

    def test_top_k_zero_returns_empty(self) -> None:
        idx = InMemoryVectorIndex(dimension=4)
        idx.upsert("a", [1, 0, 0, 0])
        assert idx.search([1, 0, 0, 0], top_k=0) == []

    def test_clear_resets_index(self) -> None:
        idx = InMemoryVectorIndex(dimension=4)
        idx.upsert("a", [1, 0, 0, 0])
        idx.clear()
        assert idx.size() == 0
