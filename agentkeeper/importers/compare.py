"""Cross-platform knowledge comparison between import scopes.

Answers the question imports exist for: *what does one AI platform know
about the user that another doesn't?* Operates on the scope convention
written by every importer (``metadata["scope"] == "platform:<name>"``).

Same philosophy as :mod:`agentkeeper.checkpoint.diff` — concrete,
defensible output only. Matching is deterministic: normalized content
equality first, then a ``difflib`` similarity ratio for near-duplicates
("Prefers short answers" vs "prefers short, direct answers"). No
embeddings, no LLM calls, no scores you can't reproduce.

Quick start::

    from agentkeeper.importers import compare_scopes, scope_inventory

    result = compare_scopes(agent, "platform:chatgpt", "platform:claude")
    print(result.to_dict()["summary"])
    for fact in result.only_in_a:
        print("ChatGPT only:", fact.content)

    print(scope_inventory(agent))
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from ..cso.types import Fact

if TYPE_CHECKING:
    from .. import Agent

DEFAULT_SIMILARITY = 0.82

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize(text: str) -> str:
    """Casefold, strip accents and punctuation, collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    no_punct = _PUNCT_RE.sub(" ", stripped.casefold())
    return _WS_RE.sub(" ", no_punct).strip()


def _scope_of(fact: Fact) -> str | None:
    scope = fact.metadata.get("scope")
    return scope if isinstance(scope, str) else None


def _facts_in_scope(
    agent: Agent,
    scope: str,
    include_kinds: tuple[str, ...] | None,
) -> list[Fact]:
    facts: list[Fact] = []
    for fact in agent._cso.memory_facts:
        if _scope_of(fact) != scope:
            continue
        if include_kinds is not None:
            kind = fact.metadata.get("kind")
            if not isinstance(kind, str) or kind not in include_kinds:
                continue
        facts.append(fact)
    return facts


@dataclass
class MatchedPair:
    """Two facts from different scopes judged to carry the same knowledge."""

    fact_a: Fact
    fact_b: Fact
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": {"id": self.fact_a.id, "content": self.fact_a.content},
            "b": {"id": self.fact_b.id, "content": self.fact_b.content},
            "similarity": round(self.similarity, 3),
        }


@dataclass
class ScopeComparison:
    """Result of comparing the knowledge held in two scopes."""

    scope_a: str
    scope_b: str
    only_in_a: list[Fact] = field(default_factory=list)
    only_in_b: list[Fact] = field(default_factory=list)
    shared: list[MatchedPair] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.only_in_a or self.only_in_b or self.shared)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_a": self.scope_a,
            "scope_b": self.scope_b,
            "only_in_a": [
                {"id": f.id, "content": f.content} for f in self.only_in_a
            ],
            "only_in_b": [
                {"id": f.id, "content": f.content} for f in self.only_in_b
            ],
            "shared": [p.to_dict() for p in self.shared],
            "summary": {
                "only_in_a": len(self.only_in_a),
                "only_in_b": len(self.only_in_b),
                "shared": len(self.shared),
            },
        }


def compare_scopes(
    agent: Agent,
    scope_a: str,
    scope_b: str,
    similarity: float = DEFAULT_SIMILARITY,
    include_kinds: tuple[str, ...] | None = ("memory_summary",),
) -> ScopeComparison:
    """Compare the knowledge an agent holds under two scopes.

    Args:
        agent: The agent whose memory holds the imported facts.
        scope_a: First scope, e.g. ``"platform:chatgpt"``.
        scope_b: Second scope, e.g. ``"platform:claude"``.
        similarity: Ratio (0-1) above which two near-identical contents
            count as shared knowledge. Exact normalized matches always
            count regardless of this value.
        include_kinds: Restrict the comparison to facts whose
            ``metadata["kind"]`` is in this tuple. Defaults to
            ``("memory_summary",)`` — actual knowledge about the user —
            because comparing conversation-event titles across platforms
            is noise, not signal. Pass ``None`` to compare everything
            in both scopes.

    Matching is greedy best-match: each fact in scope A pairs with the
    most similar unmatched fact in scope B. Deterministic for a given
    memory state.
    """
    if not 0.0 <= similarity <= 1.0:
        raise ValueError(f"similarity must be in [0, 1], got {similarity}")

    facts_a = _facts_in_scope(agent, scope_a, include_kinds)
    facts_b = _facts_in_scope(agent, scope_b, include_kinds)

    norms_a = [_normalize(f.content) for f in facts_a]
    norms_b = [_normalize(f.content) for f in facts_b]

    result = ScopeComparison(scope_a=scope_a, scope_b=scope_b)
    unmatched_b = set(range(len(facts_b)))

    for i, fact_a in enumerate(facts_a):
        best_j: int | None = None
        best_score = 0.0
        for j in sorted(unmatched_b):
            if norms_a[i] == norms_b[j]:
                best_j, best_score = j, 1.0
                break
            score = SequenceMatcher(None, norms_a[i], norms_b[j]).ratio()
            if score > best_score:
                best_j, best_score = j, score
        if best_j is not None and best_score >= similarity:
            unmatched_b.discard(best_j)
            result.shared.append(
                MatchedPair(
                    fact_a=fact_a,
                    fact_b=facts_b[best_j],
                    similarity=best_score,
                )
            )
        else:
            result.only_in_a.append(fact_a)

    result.only_in_b = [facts_b[j] for j in sorted(unmatched_b)]
    return result


def scope_inventory(agent: Agent) -> dict[str, dict[str, int]]:
    """Per-scope fact counts grouped by fact type.

    Returns a mapping like::

        {
            "platform:chatgpt": {"EVENT": 142, "PREFERENCE": 9, ...},
            "platform:claude": {"EVENT": 87, "FACT": 4, ...},
        }

    Facts without a scope are grouped under ``"unscoped"``. This is the
    one-call overview a dashboard needs before drilling into
    :func:`compare_scopes`.
    """
    inventory: dict[str, dict[str, int]] = {}
    for fact in agent._cso.memory_facts:
        scope = _scope_of(fact) or "unscoped"
        bucket = inventory.setdefault(scope, {})
        type_name = fact.fact_type.name
        bucket[type_name] = bucket.get(type_name, 0) + 1
    return inventory
