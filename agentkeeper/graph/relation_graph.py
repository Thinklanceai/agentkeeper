"""Relational graph operations over a Cognitive State Object.

`RelationGraph` provides traversal utilities (`neighbours`, `paths`,
`find_related`, `subgraph`) over the triples stored inside a CSO.
The graph itself is *not* a separate data structure — it's a *view*
over `cso.triples`. This guarantees consistency: when a triple is
added or removed (by `agent.link`, `agent.unlink`, `purge_expired`,
`gdpr_purge`), the graph view immediately reflects the change.

Traversal is breadth-first and depth-bounded by default (max_hops=2),
matching the "1-2 hops" pattern used by the agent helpers. Larger
depths are supported but rarely needed; the typical use case is
"give me everything connected to Acme up to 2 steps".

The graph is **directed**: a triple `(A, owns, B)` does not imply
`(B, owned_by, A)` automatically. Add the inverse explicitly if you
want bidirectional reachability. We don't auto-infer inverses because
predicates are free-form strings and we have no schema to compute
inverses from.

Performance: O(E) per traversal step where E is the triple count.
For agents with < 10k triples (the v1.1 sweet spot) this is
instantaneous. Beyond that, callers should consider denormalising
into sqlite-vec or an external graph DB — out of scope for v1.1.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING

from .triple import Triple

if TYPE_CHECKING:
    from ..cso.types import CognitiveStateObject


class RelationGraph:
    """A directed graph view over a CSO's triples.

    The graph is constructed lazily and re-indexes on demand. Indices
    are rebuilt cheaply (O(E)) — no incremental maintenance needed for
    the scale we target.
    """

    def __init__(self, cso: CognitiveStateObject) -> None:
        self._cso = cso
        # Lazily computed
        self._out_index: dict[str, list[Triple]] | None = None
        self._in_index: dict[str, list[Triple]] | None = None

    def _invalidate(self) -> None:
        self._out_index = None
        self._in_index = None

    def _ensure_indexed(self) -> None:
        if self._out_index is not None and self._in_index is not None:
            return
        out_idx: dict[str, list[Triple]] = defaultdict(list)
        in_idx: dict[str, list[Triple]] = defaultdict(list)
        for t in self._cso.triples:
            out_idx[t.subject].append(t)
            in_idx[t.object].append(t)
        self._out_index = dict(out_idx)
        self._in_index = dict(in_idx)

    # --- queries ----------------------------------------------------

    def neighbours(
        self,
        entity: str,
        direction: str = "out",
        predicate: str | None = None,
    ) -> list[Triple]:
        """Return triples adjacent to `entity`.

        Args:
            entity: The entity whose neighbours to return.
            direction: 'out' (subject == entity), 'in' (object == entity),
                or 'both'.
            predicate: Optional predicate filter (exact match).
        """
        self._ensure_indexed()
        assert self._out_index is not None
        assert self._in_index is not None

        triples: list[Triple] = []
        if direction in ("out", "both"):
            triples.extend(self._out_index.get(entity, []))
        if direction in ("in", "both"):
            triples.extend(self._in_index.get(entity, []))

        if predicate is not None:
            triples = [t for t in triples if t.predicate == predicate]

        return triples

    def find_related(
        self,
        entity: str,
        max_hops: int = 2,
        direction: str = "both",
        min_confidence: float = 0.0,
    ) -> dict[str, int]:
        """Breadth-first traversal returning {entity: hop_distance}.

        The starting entity is included at distance 0.

        Args:
            entity: Starting node.
            max_hops: Maximum BFS depth (inclusive).
            direction: 'out', 'in', or 'both'.
            min_confidence: Filter out low-confidence edges.
        """
        if max_hops < 0:
            return {}

        self._ensure_indexed()
        distances: dict[str, int] = {entity: 0}
        queue: deque[tuple[str, int]] = deque([(entity, 0)])

        while queue:
            node, dist = queue.popleft()
            if dist >= max_hops:
                continue
            for t in self.neighbours(node, direction=direction):
                if t.confidence < min_confidence:
                    continue
                next_nodes: list[str] = []
                if direction in ("out", "both") and t.subject == node:
                    next_nodes.append(t.object)
                if direction in ("in", "both") and t.object == node:
                    next_nodes.append(t.subject)
                for nxt in next_nodes:
                    if nxt not in distances:
                        distances[nxt] = dist + 1
                        queue.append((nxt, dist + 1))

        return distances

    def shortest_path(
        self,
        source: str,
        target: str,
        max_hops: int = 4,
        direction: str = "both",
    ) -> list[Triple] | None:
        """Return the shortest sequence of triples connecting source → target.

        BFS over the edges. Returns `None` if no path within max_hops.
        For ties, the first path found is returned (BFS guarantees
        minimal length).
        """
        if source == target:
            return []
        if max_hops <= 0:
            return None

        self._ensure_indexed()
        # Each entry: (node, path_of_triples)
        queue: deque[tuple[str, list[Triple]]] = deque([(source, [])])
        seen = {source}

        while queue:
            node, path = queue.popleft()
            if len(path) >= max_hops:
                continue
            for t in self.neighbours(node, direction=direction):
                next_node = (
                    t.object
                    if direction in ("out", "both") and t.subject == node
                    else t.subject
                )
                if next_node in seen:
                    continue
                new_path = path + [t]
                if next_node == target:
                    return new_path
                seen.add(next_node)
                queue.append((next_node, new_path))

        return None

    def subgraph(
        self,
        entities: list[str],
        max_hops: int = 1,
    ) -> list[Triple]:
        """Return every triple touching any of `entities` within `max_hops`.

        Useful for "give me the local neighbourhood of these nodes" —
        e.g. when reconstructing context, you can pass relevant
        entities mentioned in the user task.
        """
        reachable: set[str] = set()
        for e in entities:
            reachable.update(self.find_related(e, max_hops=max_hops).keys())

        self._ensure_indexed()
        return [
            t for t in self._cso.triples
            if t.subject in reachable or t.object in reachable
        ]

    def entities(self) -> set[str]:
        """Return every entity name appearing in any triple."""
        result: set[str] = set()
        for t in self._cso.triples:
            result.add(t.subject)
            result.add(t.object)
        return result

    def __len__(self) -> int:
        return len(self._cso.triples)

    def __repr__(self) -> str:
        entity_count = len(self.entities())
        return (
            f"RelationGraph(triples={len(self._cso.triples)}, "
            f"entities={entity_count})"
        )
