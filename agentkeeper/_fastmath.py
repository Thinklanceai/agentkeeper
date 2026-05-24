"""Optional numpy acceleration for vector math.

AgentKeeper's core has zero required dependencies. Vector math (dot
products, cosine similarity, clustering) falls back to pure Python when
numpy is unavailable. When numpy *is* installed — which is the case for
essentially every AI/ML project — the same operations run 50-100x
faster, which matters a lot for compression at scale (consolidation
and contradiction passes are vector-heavy).

This module exposes a single flag, `HAS_NUMPY`, and helpers that pick
the fast path when possible. Callers never import numpy directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

try:
    import numpy as _np

    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _np = None  # type: ignore[assignment]
    HAS_NUMPY = False


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product of two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
    if HAS_NUMPY:
        return float(_np.dot(a, b))
    return sum(x * y for x, y in zip(a, b, strict=True))


def batch_dot(
    query: Sequence[float],
    matrix: Sequence[Sequence[float]],
) -> list[float]:
    """Dot product of `query` against every row of `matrix`.

    Returns a list of scores, one per row. With numpy this is a single
    matrix-vector product; without, it's a Python loop.
    """
    if not matrix:
        return []
    if HAS_NUMPY:
        m = _np.asarray(matrix, dtype=float)
        q = _np.asarray(query, dtype=float)
        return cast("list[float]", _np.dot(m, q).tolist())
    return [dot(query, row) for row in matrix]


def mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Component-wise mean of a list of equal-length vectors."""
    if not vectors:
        return []
    if HAS_NUMPY:
        return cast(
            "list[float]",
            _np.asarray(vectors, dtype=float).mean(axis=0).tolist(),
        )
    n = len(vectors)
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            acc[i] += v[i]
    return [x / n for x in acc]
