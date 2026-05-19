"""Tests for Triple type and RelationGraph traversal."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agentkeeper.cso.types import CognitiveStateObject
from agentkeeper.errors import ConfigurationError
from agentkeeper.graph import RelationGraph, Triple


class TestTripleConstruction:
    def test_basic_create(self) -> None:
        t = Triple.create("Acme", "owns", "Globex")
        assert t.subject == "Acme"
        assert t.predicate == "owns"
        assert t.object == "Globex"
        assert t.confidence == 1.0
        assert t.protected is False
        assert t.expires_at is None

    def test_create_with_confidence_clipping(self) -> None:
        t1 = Triple.create("a", "p", "b", confidence=1.5)
        assert t1.confidence == 1.0
        t2 = Triple.create("a", "p", "b", confidence=-0.3)
        assert t2.confidence == 0.0

    def test_empty_subject_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            Triple.create("", "p", "b")

    def test_empty_predicate_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            Triple.create("a", "", "b")

    def test_empty_object_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            Triple.create("a", "p", "")

    def test_whitespace_stripped(self) -> None:
        t = Triple.create("  Acme  ", "  owns  ", "  Globex  ")
        assert t.subject == "Acme"
        assert t.predicate == "owns"
        assert t.object == "Globex"

    def test_ttl(self) -> None:
        t = Triple.create("a", "p", "b", ttl="30d")
        assert t.expires_at is not None
        parsed = datetime.fromisoformat(t.expires_at)
        delta = parsed - datetime.now(timezone.utc)
        assert timedelta(days=29) < delta <= timedelta(days=30)

    def test_explicit_expires_at_wins_over_ttl(self) -> None:
        explicit = "2030-01-01T00:00:00+00:00"
        t = Triple.create("a", "p", "b", ttl="30d", expires_at=explicit)
        assert t.expires_at == explicit


class TestTripleSerialization:
    def test_roundtrip(self) -> None:
        t = Triple.create(
            "Acme", "owns", "Globex",
            confidence=0.85,
            protected=True,
            metadata={"source": "fact-123"},
        )
        d = t.to_dict()
        t2 = Triple.from_dict(d)
        assert t2.subject == t.subject
        assert t2.predicate == t.predicate
        assert t2.object == t.object
        assert t2.confidence == t.confidence
        assert t2.protected == t.protected
        assert t2.metadata == t.metadata


class TestRelationGraphBasics:
    def _build(self, triples: list[tuple[str, str, str]]) -> RelationGraph:
        cso = CognitiveStateObject.create(agent_id="g")
        for s, p, o in triples:
            cso.triples.append(Triple.create(s, p, o))
        return RelationGraph(cso)

    def test_empty_graph(self) -> None:
        cso = CognitiveStateObject.create(agent_id="g")
        graph = RelationGraph(cso)
        assert len(graph) == 0
        assert graph.entities() == set()

    def test_entities(self) -> None:
        graph = self._build([
            ("Alice", "works_at", "Acme"),
            ("Acme", "owned_by", "Globex"),
        ])
        entities = graph.entities()
        assert entities == {"Alice", "Acme", "Globex"}

    def test_neighbours_out(self) -> None:
        graph = self._build([
            ("Acme", "owns", "Globex"),
            ("Acme", "owns", "Initech"),
            ("Globex", "owns", "Initech"),
        ])
        out = graph.neighbours("Acme", direction="out")
        assert len(out) == 2
        objects = {t.object for t in out}
        assert objects == {"Globex", "Initech"}

    def test_neighbours_in(self) -> None:
        graph = self._build([
            ("Acme", "owns", "Globex"),
            ("Initech", "owns", "Globex"),
        ])
        inb = graph.neighbours("Globex", direction="in")
        assert len(inb) == 2
        subjects = {t.subject for t in inb}
        assert subjects == {"Acme", "Initech"}

    def test_neighbours_predicate_filter(self) -> None:
        graph = self._build([
            ("Alice", "works_at", "Acme"),
            ("Alice", "lives_in", "Brussels"),
        ])
        result = graph.neighbours("Alice", predicate="works_at")
        assert len(result) == 1
        assert result[0].object == "Acme"


class TestFindRelated:
    def test_single_hop(self) -> None:
        cso = CognitiveStateObject.create(agent_id="g")
        cso.triples.extend([
            Triple.create("Acme", "owns", "Globex"),
            Triple.create("Acme", "owns", "Initech"),
        ])
        graph = RelationGraph(cso)
        result = graph.find_related("Acme", max_hops=1)
        assert result["Acme"] == 0
        assert result["Globex"] == 1
        assert result["Initech"] == 1

    def test_two_hops(self) -> None:
        cso = CognitiveStateObject.create(agent_id="g")
        cso.triples.extend([
            Triple.create("Alice", "works_at", "Acme"),
            Triple.create("Acme", "owned_by", "Globex"),
            Triple.create("Globex", "located_in", "BE"),
        ])
        graph = RelationGraph(cso)
        result = graph.find_related("Alice", max_hops=2, direction="out")
        # At hop 2 we should see Globex but NOT BE (which is at hop 3)
        assert "Alice" in result
        assert "Acme" in result
        assert "Globex" in result
        assert "BE" not in result

    def test_bidirectional(self) -> None:
        cso = CognitiveStateObject.create(agent_id="g")
        cso.triples.extend([
            Triple.create("Alice", "works_at", "Acme"),
            Triple.create("Bob", "works_at", "Acme"),
        ])
        graph = RelationGraph(cso)
        # From Alice in direction "both" we should reach Bob in 2 hops
        result = graph.find_related("Alice", max_hops=2, direction="both")
        assert result.get("Bob") == 2

    def test_confidence_filter(self) -> None:
        cso = CognitiveStateObject.create(agent_id="g")
        cso.triples.extend([
            Triple.create("A", "p", "B", confidence=0.9),
            Triple.create("B", "p", "C", confidence=0.3),
        ])
        graph = RelationGraph(cso)
        result = graph.find_related("A", max_hops=2, min_confidence=0.5)
        assert "B" in result
        assert "C" not in result


class TestShortestPath:
    def test_same_node(self) -> None:
        cso = CognitiveStateObject.create(agent_id="g")
        graph = RelationGraph(cso)
        assert graph.shortest_path("A", "A") == []

    def test_direct_path(self) -> None:
        cso = CognitiveStateObject.create(agent_id="g")
        cso.triples.append(Triple.create("A", "p", "B"))
        graph = RelationGraph(cso)
        path = graph.shortest_path("A", "B")
        assert path is not None
        assert len(path) == 1
        assert path[0].object == "B"

    def test_two_hop_path(self) -> None:
        cso = CognitiveStateObject.create(agent_id="g")
        cso.triples.extend([
            Triple.create("A", "p", "B"),
            Triple.create("B", "p", "C"),
        ])
        graph = RelationGraph(cso)
        path = graph.shortest_path("A", "C")
        assert path is not None
        assert len(path) == 2

    def test_no_path(self) -> None:
        cso = CognitiveStateObject.create(agent_id="g")
        cso.triples.extend([
            Triple.create("A", "p", "B"),
            Triple.create("X", "p", "Y"),
        ])
        graph = RelationGraph(cso)
        assert graph.shortest_path("A", "Y") is None
