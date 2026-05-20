"""Tests for the optional numpy accelerator (agentkeeper._fastmath).

These verify both code paths produce identical results:
- the numpy fast path (when numpy is importable)
- the pure-Python fallback (forced by monkeypatching HAS_NUMPY)

The accelerator is correctness-critical: consolidation and
contradiction arbitration depend on dot products and centroid means.
A divergence between the two paths would mean compression behaves
differently depending on whether numpy happens to be installed.
"""

from __future__ import annotations

import pytest

from agentkeeper import _fastmath


def _approx_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


class TestDot:
    def test_basic(self) -> None:
        assert _approx_equal(_fastmath.dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0)

    def test_orthogonal(self) -> None:
        assert _approx_equal(_fastmath.dot([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            _fastmath.dot([1.0, 2.0], [1.0])

    def test_numpy_and_pure_agree(self, monkeypatch: pytest.MonkeyPatch) -> None:
        a = [0.1, 0.2, 0.3, 0.4]
        b = [0.5, 0.6, 0.7, 0.8]
        # Whatever the current state, compute both ways and compare.
        numpy_result = None
        pure_result = None

        if _fastmath.HAS_NUMPY:
            numpy_result = _fastmath.dot(a, b)
        monkeypatch.setattr(_fastmath, "HAS_NUMPY", False)
        pure_result = _fastmath.dot(a, b)

        # Pure result must equal the hand-computed value.
        expected = sum(x * y for x, y in zip(a, b, strict=True))
        assert _approx_equal(pure_result, expected)
        if numpy_result is not None:
            assert _approx_equal(numpy_result, pure_result)


class TestBatchDot:
    def test_empty_matrix(self) -> None:
        assert _fastmath.batch_dot([1.0, 2.0], []) == []

    def test_against_rows(self) -> None:
        query = [1.0, 0.0, 0.0]
        matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]
        scores = _fastmath.batch_dot(query, matrix)
        assert len(scores) == 3
        assert _approx_equal(scores[0], 1.0)
        assert _approx_equal(scores[1], 0.0)
        assert _approx_equal(scores[2], 0.5)

    def test_numpy_and_pure_agree(self, monkeypatch: pytest.MonkeyPatch) -> None:
        query = [0.1, 0.2, 0.3]
        matrix = [[0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.0, 0.0, 1.0]]

        numpy_scores = None
        if _fastmath.HAS_NUMPY:
            numpy_scores = _fastmath.batch_dot(query, matrix)

        monkeypatch.setattr(_fastmath, "HAS_NUMPY", False)
        pure_scores = _fastmath.batch_dot(query, matrix)

        expected = [
            sum(q * m for q, m in zip(query, row, strict=True)) for row in matrix
        ]
        for got, exp in zip(pure_scores, expected, strict=True):
            assert _approx_equal(got, exp)
        if numpy_scores is not None:
            for n, p in zip(numpy_scores, pure_scores, strict=True):
                assert _approx_equal(n, p)


class TestMeanVector:
    def test_empty(self) -> None:
        assert _fastmath.mean_vector([]) == []

    def test_single(self) -> None:
        assert _fastmath.mean_vector([[1.0, 2.0, 3.0]]) == [1.0, 2.0, 3.0]

    def test_mean(self) -> None:
        result = _fastmath.mean_vector([[0.0, 0.0], [2.0, 4.0]])
        assert _approx_equal(result[0], 1.0)
        assert _approx_equal(result[1], 2.0)

    def test_numpy_and_pure_agree(self, monkeypatch: pytest.MonkeyPatch) -> None:
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]

        numpy_mean = None
        if _fastmath.HAS_NUMPY:
            numpy_mean = _fastmath.mean_vector(vectors)

        monkeypatch.setattr(_fastmath, "HAS_NUMPY", False)
        pure_mean = _fastmath.mean_vector(vectors)

        expected = [4.0, 5.0, 6.0]
        for got, exp in zip(pure_mean, expected, strict=True):
            assert _approx_equal(got, exp)
        if numpy_mean is not None:
            for n, p in zip(numpy_mean, pure_mean, strict=True):
                assert _approx_equal(n, p)


class TestConsolidationPathParity:
    """The two code paths must consolidate identically on the same input."""

    def test_compression_consistent_across_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        import os

        os.environ["AGENTKEEPER_DB"] = str(tmp_path / "ak.db")
        os.environ["AGENTKEEPER_EMBEDDING_PROVIDER"] = "mock"
        os.environ["AGENTKEEPER_VECTOR_INDEX"] = "in_memory"

        import agentkeeper

        def build_and_compress() -> int:
            agentkeeper._storage = None
            agent = agentkeeper.create(agent_id="parity", provider="mock")
            # Deterministic content: 5 distinct groups, 4 dupes each.
            for group in range(5):
                for _ in range(4):
                    agent.fact(f"group {group} fact", importance=0.5)
            agent.compress()
            count = len(agent.facts)
            agentkeeper.delete("parity")
            return count

        # Record whether numpy is really available before we patch.
        had_numpy = _fastmath.HAS_NUMPY

        # Pure path
        monkeypatch.setattr(_fastmath, "HAS_NUMPY", False)
        pure_count = build_and_compress()

        # Numpy path (only if the machine actually has numpy)
        if had_numpy:
            monkeypatch.setattr(_fastmath, "HAS_NUMPY", True)
            numpy_count = build_and_compress()
            assert numpy_count == pure_count
