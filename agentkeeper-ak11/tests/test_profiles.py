"""Tests for cognitive profiles and format-specific renderers."""

from __future__ import annotations

from agentkeeper.cso.identity import AgentIdentity
from agentkeeper.cso.tiers import MemoryTier
from agentkeeper.cso.types import Fact
from agentkeeper.translation.profiles import (
    CognitiveProfile,
    PromptFormat,
    get_profile,
    known_providers,
    register_profile,
)
from agentkeeper.translation.renderers import render


def _identity() -> AgentIdentity:
    return AgentIdentity(
        name="Aria",
        role="copilot",
        principles=["never share PII"],
        constraints=["EU only"],
    )


def _grouped(facts: list[Fact]) -> dict[MemoryTier, list[Fact]]:
    g: dict[MemoryTier, list[Fact]] = {t: [] for t in MemoryTier}
    for f in facts:
        g[f.tier].append(f)
    return g


class TestProfiles:
    def test_known_providers_includes_majors(self) -> None:
        names = known_providers()
        for required in ("anthropic", "openai", "gemini", "ollama"):
            assert required in names

    def test_unknown_provider_falls_back_to_openai_format(self) -> None:
        profile = get_profile("some-random-llm")
        assert profile.format == PromptFormat.SECTIONS

    def test_anthropic_uses_xml(self) -> None:
        assert get_profile("anthropic").format == PromptFormat.XML

    def test_gemini_uses_narrative(self) -> None:
        assert get_profile("gemini").format == PromptFormat.NARRATIVE

    def test_ollama_uses_minimal(self) -> None:
        assert get_profile("ollama").format == PromptFormat.MINIMAL

    def test_register_profile(self) -> None:
        custom = CognitiveProfile(
            provider="custom-llm-test",
            format=PromptFormat.MINIMAL,
            effective_context_tokens=2_000,
        )
        register_profile(custom)
        assert get_profile("custom-llm-test") is custom


class TestRenderers:
    def test_xml_renderer_contains_xml_tags(self) -> None:
        identity = _identity()
        facts = [Fact.create("budget: 50k", critical=True)]
        prompt = render(
            get_profile("anthropic"),
            identity,
            _grouped(facts),
            "What is the budget?",
        )
        assert "<agent_identity>" in prompt
        assert "<memory>" in prompt
        assert "<task>" in prompt
        assert "Aria" in prompt
        assert "budget: 50k" in prompt
        assert 'critical="true"' in prompt

    def test_sections_renderer_contains_section_headers(self) -> None:
        identity = _identity()
        facts = [Fact.create("budget: 50k", critical=True)]
        prompt = render(
            get_profile("openai"),
            identity,
            _grouped(facts),
            "What is the budget?",
        )
        assert "AGENT IDENTITY" in prompt
        assert "MEMORY" in prompt
        assert "CURRENT TASK" in prompt
        assert "★" in prompt

    def test_narrative_renderer_reads_as_prose(self) -> None:
        identity = _identity()
        facts = [Fact.create("budget: 50k", critical=True)]
        prompt = render(
            get_profile("gemini"),
            identity,
            _grouped(facts),
            "Status?",
        )
        # Narrative speaks directly to the agent
        assert "You are Aria" in prompt
        # Lower-case markdown-ish headers
        assert "## Semantic memory" in prompt or "## semantic" in prompt.lower()

    def test_minimal_renderer_is_compact(self) -> None:
        identity = _identity()
        facts = [Fact.create("budget: 50k", critical=True)]
        prompt = render(
            get_profile("ollama"),
            identity,
            _grouped(facts),
            "Q?",
        )
        # No XML, no big headers
        assert "<agent_identity>" not in prompt
        assert "AGENT IDENTITY" not in prompt
        # Critical marker uses minimal "!" form
        assert "!" in prompt
        # Identity reduced to a single line
        assert "You: Aria" in prompt or "You:" in prompt

    def test_empty_identity_omits_identity_block(self) -> None:
        identity = AgentIdentity()
        facts = [Fact.create("x")]
        # XML format
        prompt = render(
            get_profile("anthropic"), identity, _grouped(facts), "task"
        )
        assert "<agent_identity>" not in prompt

    def test_empty_memory_renders_cleanly(self) -> None:
        identity = AgentIdentity(name="Aria")
        for provider in ("anthropic", "openai", "gemini", "ollama"):
            prompt = render(get_profile(provider), identity, _grouped([]), "task")
            assert "task" in prompt
            assert "Aria" in prompt
