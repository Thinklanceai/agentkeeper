"""Fact verification utilities for benchmarks.

Given a list of expected facts and a model response, determine which
facts have been recovered. This is intentionally simple (substring /
keyword match); the v1.0 cognitive compression module will replace it
with embedding-based semantic verification.
"""

from __future__ import annotations

from ..cso.types import Fact


def extract_recovered_facts(response: str, expected_facts: list[Fact]) -> list[str]:
    """Return the IDs of facts whose content is detectable in the response.

    Heuristic:
    - For facts of the form "key: value", split and search for value
      keywords (length > 3) in the response.
    - For plain facts, fall back to a case-insensitive substring match.

    This is a coarse but provider-agnostic baseline. A real semantic
    recovery score will be added in Sprint AK-3 (Semantic Recall).
    """
    found: list[str] = []
    response_lower = response.lower()

    for fact in expected_facts:
        content = fact.content
        if ":" in content:
            _key, value = content.split(":", 1)
            keywords = [
                w.lower() for w in value.strip().split() if len(w) > 3
            ]
            if keywords and any(kw in response_lower for kw in keywords):
                found.append(fact.id)
        else:
            if content.lower() in response_lower:
                found.append(fact.id)

    return found
