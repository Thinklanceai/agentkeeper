"""Heuristic tier inference for incoming facts.

Given a raw textual fact, we attempt to classify it into the right
memory tier without invoking any LLM. Inference is purely pattern-based,
deterministic, fast, and overridable.

Rules in order of priority (first match wins):

1. Explicit temporal markers (dates, "yesterday", "last week", ...)
   → `episodic`
2. Event verbs in past tense ("happened", "occurred", "decided",
   "agreed", "refused", "signed", "shipped")
   → `episodic`
3. Stable predicate forms (`key: value`, "is/are/has", "the X is Y")
   → `semantic`
4. Imperative or principle-shaped statements ("never X", "always X",
   "must X", "do not X")
   → `semantic` with high importance (treated as principles upstream)
5. Default
   → `semantic`

Importance defaults are also inferred:
- principles / imperatives → 0.9
- semantic facts           → 0.6
- episodic events          → 0.5
- working / archival       → 0.4

The user can override any inference by passing `tier=` and/or
`importance=` explicitly.
"""

from __future__ import annotations

import re

from .tiers import MemoryTier

_DATE_PATTERNS = [
    # ISO dates
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    # numeric dates (en/fr)
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    # month names (en, fr abridged)
    re.compile(
        r"\b(?:january|february|march|april|may|june|july|august|"
        r"september|october|november|december|"
        r"janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[uû]t|"
        r"septembre|octobre|novembre|d[eé]cembre|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b"
        r"(?:\s+\d{1,2})?",
        re.IGNORECASE,
    ),
]

_RELATIVE_TIME_PATTERN = re.compile(
    r"\b(?:yesterday|today|tomorrow|last\s+(?:week|month|year)|"
    r"next\s+(?:week|month|year)|this\s+(?:morning|afternoon|evening)|"
    r"hier|aujourd'?hui|demain|"
    r"la\s+semaine\s+(?:derni[èe]re|prochaine)|"
    r"le\s+mois\s+(?:dernier|prochain))\b",
    re.IGNORECASE,
)

_EVENT_VERBS_PATTERN = re.compile(
    r"\b(?:happened|occurred|decided|agreed|refused|signed|shipped|"
    r"launched|released|deployed|cancell?ed|approved|rejected|"
    r"announced|met|called|sent|received|"
    r"d[ée]cid[ée]|sign[ée]|annul[ée]|envoy[ée]|re[çc]u|"
    r"approuv[ée]|refus[ée]|lanc[ée]|annonc[ée])\b",
    re.IGNORECASE,
)

_PRINCIPLE_PATTERN = re.compile(
    r"^(?:never|always|must|do\s+not|don'?t|should\s+(?:always|never)|"
    r"jamais|toujours|doit|ne\s+(?:pas|jamais))\b",
    re.IGNORECASE,
)

_KEY_VALUE_PATTERN = re.compile(r"^[A-Za-z][\w\s]{0,40}:\s*.+$")
_STABLE_PREDICATE_PATTERN = re.compile(
    r"\b(?:is|are|has|have|contains?|equals?|costs?|"
    r"est|sont|a|ont|contient|co[uû]te)\b",
    re.IGNORECASE,
)


def infer_tier(content: str) -> MemoryTier:
    """Classify a fact into the most likely tier.

    The function is pure: same input → same output. No side effects.
    Override the inference by passing `tier=` to `Fact.create` or
    `agent.remember`.
    """
    text = content.strip()

    # Rule 1: temporal markers → episodic
    if _RELATIVE_TIME_PATTERN.search(text):
        return MemoryTier.EPISODIC
    for pattern in _DATE_PATTERNS:
        if pattern.search(text):
            return MemoryTier.EPISODIC

    # Rule 2: event verbs → episodic
    if _EVENT_VERBS_PATTERN.search(text):
        return MemoryTier.EPISODIC

    # Rule 3 & 5: default semantic (key:value, stable predicates, anything else)
    return MemoryTier.SEMANTIC


def infer_importance(content: str, tier: MemoryTier) -> float:
    """Infer a default importance score (0.0 - 1.0).

    Principles get a very high importance because they encode behaviour
    the agent must respect across time and providers. The CRE force-includes
    them regardless of token budget.
    """
    text = content.strip()

    if _PRINCIPLE_PATTERN.match(text):
        return 0.9

    if tier == MemoryTier.SEMANTIC:
        if _KEY_VALUE_PATTERN.match(text) or _STABLE_PREDICATE_PATTERN.search(text):
            return 0.6
        return 0.5

    if tier == MemoryTier.EPISODIC:
        return 0.5

    if tier == MemoryTier.WORKING:
        return 0.4

    return 0.3  # archival
