"""Tests for the mock embedding provider."""

from __future__ import annotations

import math

import pytest

from agentkeeper.semantic.mock import MockEmbeddingProvider


class TestMockEmbeddingProvider:
    def test_dimension_default(self) -> None:
        p = MockEmbeddingProvider()
        assert p.dimension == 64

    def test_dimension_configurable(self) -> None:
        p = MockEmbeddingProvider(dimension=128)
        assert p.dimension == 128

    def test_rejects_tiny_dimension(self) -> None:
        with pytest.raises(ValueError):
            MockEmbeddingProvider(dimension=2)

    def test_embed_returns_correct_dimension(self) -> None:
        p = MockEmbeddingProvider(dimension=32)
        vec = p.embed_one("hello world")
        assert len(vec) == 32

    def test_embeddings_are_normalised(self) -> None:
        p = MockEmbeddingProvider()
        vec = p.embed_one("the quick brown fox")
        norm = math.sqrt(sum(v * v for v in vec))
        assert math.isclose(norm, 1.0, rel_tol=1e-6)

    def test_same_text_yields_same_vector(self) -> None:
        p = MockEmbeddingProvider()
        a = p.embed_one("budget: 50k EUR")
        b = p.embed_one("budget: 50k EUR")
        assert a == b

    def test_different_text_yields_different_vector(self) -> None:
        p = MockEmbeddingProvider()
        a = p.embed_one("budget: 50k EUR")
        b = p.embed_one("client: Acme Corporation")
        assert a != b

    def test_batch_preserves_order(self) -> None:
        p = MockEmbeddingProvider()
        out = p.embed(["x", "y", "z"])
        assert len(out) == 3
        assert out[0] == p.embed_one("x")
        assert out[1] == p.embed_one("y")
        assert out[2] == p.embed_one("z")

    def test_empty_string_does_not_crash(self) -> None:
        p = MockEmbeddingProvider()
        vec = p.embed_one("")
        assert len(vec) == p.dimension
