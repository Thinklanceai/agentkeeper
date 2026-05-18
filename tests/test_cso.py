"""Tests for the Cognitive State Object data model."""

from __future__ import annotations

from agentkeeper.cso.types import CognitiveStateObject, Fact


class TestFact:
    def test_create_generates_uuid_id(self) -> None:
        fact = Fact.create("hello")
        assert fact.id
        assert len(fact.id) == 36  # uuid4 string length
        assert "-" in fact.id

    def test_create_preserves_content(self) -> None:
        fact = Fact.create("project budget: 50k")
        assert fact.content == "project budget: 50k"

    def test_create_defaults_non_critical(self) -> None:
        fact = Fact.create("note")
        assert fact.critical is False

    def test_create_critical_flag(self) -> None:
        fact = Fact.create("important", critical=True)
        assert fact.critical is True

    def test_create_token_count_starts_at_zero(self) -> None:
        fact = Fact.create("any content")
        assert fact.token_count == 0


class TestCognitiveStateObject:
    def test_create_with_explicit_id(self) -> None:
        cso = CognitiveStateObject.create(agent_id="my-agent")
        assert cso.agent_id == "my-agent"
        assert cso.memory_facts == []

    def test_create_generates_id_when_omitted(self) -> None:
        cso = CognitiveStateObject.create()
        assert cso.agent_id
        assert len(cso.agent_id) == 36

    def test_add_fact_appends_to_memory(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        fact = cso.add_fact("hello")
        assert len(cso.memory_facts) == 1
        assert cso.memory_facts[0] is fact

    def test_add_fact_updates_timestamp(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        old_updated = cso.updated_at
        # ensure timestamps differ even on fast machines
        cso.add_fact("x")
        assert cso.updated_at >= old_updated

    def test_critical_facts_filters_correctly(self) -> None:
        cso = CognitiveStateObject.create(agent_id="a")
        cso.add_fact("non-critical")
        cso.add_fact("critical", critical=True)
        cso.add_fact("also critical", critical=True)
        critical = cso.critical_facts()
        assert len(critical) == 2
        assert all(f.critical for f in critical)

    def test_to_dict_roundtrip(self) -> None:
        cso = CognitiveStateObject.create(agent_id="round-trip")
        cso.add_fact("a", critical=True)
        cso.add_fact("b")
        d = cso.to_dict()
        restored = CognitiveStateObject.from_dict(d)
        assert restored.agent_id == cso.agent_id
        assert len(restored.memory_facts) == 2
        assert restored.memory_facts[0].content == "a"
        assert restored.memory_facts[0].critical is True
        assert restored.memory_facts[1].critical is False

    def test_from_dict_handles_missing_token_count(self) -> None:
        data = {
            "agent_id": "legacy",
            "memory_facts": [
                {"id": "f1", "content": "x", "critical": True}
                # no token_count field — old format
            ],
            "created_at": "2026-05-01T00:00:00+00:00",
            "updated_at": "2026-05-01T00:00:00+00:00",
        }
        restored = CognitiveStateObject.from_dict(data)
        assert restored.memory_facts[0].token_count == 0
