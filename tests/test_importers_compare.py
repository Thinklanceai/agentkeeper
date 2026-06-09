"""Tests for agentkeeper.importers.compare."""

from __future__ import annotations

import pytest

import agentkeeper
from agentkeeper import Agent
from agentkeeper.importers import (
    compare_scopes,
    parse_memory_summary,
    scope_inventory,
)
from agentkeeper.importers.compare import _normalize


def _agent_with_two_platforms() -> Agent:
    agent = agentkeeper.create("test-compare")
    chatgpt = parse_memory_summary(
        """
        - My name is Tom and I work as a developer in Brussels
        - Prefers short, direct answers
        - Building a touring platform for electronic music
        - Likes techno and runs a music archive since 1992
        """,
        platform="chatgpt",
    )
    claude = parse_memory_summary(
        """
        - My name is Tom and I work as a developer in Brussels
        - prefers short direct answers
        - Working on an EU AI Act compliance observatory
        """,
        platform="claude",
    )
    for f in chatgpt + claude:
        agent._cso.add_fact(
            f.content,
            tier=f.tier,
            importance=f.importance,
            when=f.when,
            metadata=f.metadata,
            fact_type=f.fact_type,
        )
    return agent


def test_normalize() -> None:
    assert _normalize("Préfère des réponses COURTES!") == (
        "prefere des reponses courtes"
    )
    assert _normalize("  a,  b.c ") == "a b c"


def test_exact_and_fuzzy_shared() -> None:
    agent = _agent_with_two_platforms()
    result = compare_scopes(agent, "platform:chatgpt", "platform:claude")
    shared_contents = {p.fact_a.content for p in result.shared}
    assert any("name is Tom" in c for c in shared_contents)
    assert any("short" in c for c in shared_contents)
    assert len(result.shared) == 2


def test_exclusive_knowledge_detected() -> None:
    agent = _agent_with_two_platforms()
    result = compare_scopes(agent, "platform:chatgpt", "platform:claude")
    only_a = " ".join(f.content for f in result.only_in_a)
    only_b = " ".join(f.content for f in result.only_in_b)
    assert "touring platform" in only_a
    assert "1992" in only_a
    assert "compliance observatory" in only_b
    assert len(result.only_in_a) == 2
    assert len(result.only_in_b) == 1


def test_to_dict_summary() -> None:
    agent = _agent_with_two_platforms()
    result = compare_scopes(agent, "platform:chatgpt", "platform:claude")
    d = result.to_dict()
    assert d["summary"] == {"only_in_a": 2, "only_in_b": 1, "shared": 2}
    assert all(0.0 <= p["similarity"] <= 1.0 for p in d["shared"])


def test_symmetry_of_counts() -> None:
    agent = _agent_with_two_platforms()
    ab = compare_scopes(agent, "platform:chatgpt", "platform:claude")
    ba = compare_scopes(agent, "platform:claude", "platform:chatgpt")
    assert len(ab.shared) == len(ba.shared)
    assert len(ab.only_in_a) == len(ba.only_in_b)
    assert len(ab.only_in_b) == len(ba.only_in_a)


def test_empty_scope() -> None:
    agent = _agent_with_two_platforms()
    result = compare_scopes(agent, "platform:chatgpt", "platform:gemini")
    assert len(result.only_in_b) == 0
    assert len(result.shared) == 0
    assert len(result.only_in_a) == 4


def test_include_kinds_none_compares_everything() -> None:
    agent = _agent_with_two_platforms()
    agent._cso.add_fact(
        "ChatGPT conversation: 'Trip planning' (4 messages)",
        metadata={
            "source": "chatgpt",
            "scope": "platform:chatgpt",
            "kind": "conversation",
        },
    )
    default = compare_scopes(agent, "platform:chatgpt", "platform:claude")
    everything = compare_scopes(
        agent, "platform:chatgpt", "platform:claude", include_kinds=None
    )
    assert len(everything.only_in_a) == len(default.only_in_a) + 1


def test_invalid_similarity_rejected() -> None:
    agent = _agent_with_two_platforms()
    with pytest.raises(ValueError, match="similarity"):
        compare_scopes(
            agent, "platform:chatgpt", "platform:claude", similarity=1.5
        )


def test_scope_inventory() -> None:
    agent = _agent_with_two_platforms()
    agent.fact("An unscoped fact about nothing in particular")
    inv = scope_inventory(agent)
    assert "platform:chatgpt" in inv
    assert "platform:claude" in inv
    assert "unscoped" in inv
    assert sum(inv["platform:chatgpt"].values()) == 4
    assert sum(inv["platform:claude"].values()) == 3
    assert "IDENTITY" in inv["platform:chatgpt"]
    assert "PREFERENCE" in inv["platform:chatgpt"]


def test_deterministic() -> None:
    agent = _agent_with_two_platforms()
    r1 = compare_scopes(agent, "platform:chatgpt", "platform:claude")
    r2 = compare_scopes(agent, "platform:chatgpt", "platform:claude")
    assert r1.to_dict() == r2.to_dict()
