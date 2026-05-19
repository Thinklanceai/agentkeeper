"""Tests for benchmark utilities."""

from __future__ import annotations

from agentkeeper.benchmark.dataset import (
    CRITICAL_DATA,
    NON_CRITICAL_TEMPLATES,
    generate_test_facts,
)
from agentkeeper.benchmark.verification import extract_recovered_facts
from agentkeeper.cso.types import CognitiveStateObject, Fact


class TestDataset:
    def test_generate_default_size(self) -> None:
        cso = generate_test_facts()
        assert len(cso.memory_facts) == 100
        assert len(cso.critical_facts()) == 20

    def test_generate_custom_size(self) -> None:
        cso = generate_test_facts(n_total=50, n_critical=10)
        assert len(cso.memory_facts) == 50
        assert len(cso.critical_facts()) == 10

    def test_dataset_is_reproducible(self) -> None:
        a = generate_test_facts(agent_id="x")
        b = generate_test_facts(agent_id="x")
        assert [f.content for f in a.memory_facts] == [
            f.content for f in b.memory_facts
        ]

    def test_critical_data_exists(self) -> None:
        assert len(CRITICAL_DATA) >= 20
        assert all(":" in fact for fact in CRITICAL_DATA)

    def test_non_critical_templates_format_correctly(self) -> None:
        for tmpl in NON_CRITICAL_TEMPLATES:
            rendered = tmpl.format(i=1)
            assert "{i}" not in rendered


class TestVerification:
    def test_extracts_key_value_fact(self) -> None:
        cso = CognitiveStateObject.create(agent_id="t")
        cso.add_fact("project budget: 50000 EUR", critical=True)
        response = "The budget is 50000 EUR allocated for this quarter."
        recovered = extract_recovered_facts(response, cso.memory_facts)
        assert len(recovered) == 1

    def test_does_not_extract_unmatched_fact(self) -> None:
        cso = CognitiveStateObject.create(agent_id="t")
        cso.add_fact("client name: Acme Corporation", critical=True)
        response = "I have no specific information about that."
        recovered = extract_recovered_facts(response, cso.memory_facts)
        assert recovered == []

    def test_case_insensitive_match(self) -> None:
        fact = Fact.create("decision: use Anthropic Claude for production")
        response = "we decided to use anthropic claude for production."
        recovered = extract_recovered_facts(response, [fact])
        assert recovered == [fact.id]

    def test_plain_fact_substring_match(self) -> None:
        # Plain facts (no colon) fall back to substring search
        fact = Fact.create("the moon orbits the earth")
        response = "Everyone knows the moon orbits the earth at a stable distance."
        recovered = extract_recovered_facts(response, [fact])
        assert recovered == [fact.id]

    def test_empty_response_returns_empty(self) -> None:
        cso = generate_test_facts(n_total=5, n_critical=5)
        assert extract_recovered_facts("", cso.memory_facts) == []
