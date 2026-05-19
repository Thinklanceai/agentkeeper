"""Tests for AgentIdentity."""

from __future__ import annotations

from agentkeeper.cso.identity import AgentIdentity


class TestAgentIdentity:
    def test_default_is_empty(self) -> None:
        identity = AgentIdentity()
        assert identity.is_empty()
        assert identity.render_for_prompt() == ""

    def test_non_empty_with_name(self) -> None:
        identity = AgentIdentity(name="Aria")
        assert not identity.is_empty()

    def test_to_from_dict_roundtrip(self) -> None:
        identity = AgentIdentity(
            name="Aria",
            role="EU insurance broker copilot",
            principles=["never share PII"],
            constraints=["EU data only"],
        )
        data = identity.to_dict()
        restored = AgentIdentity.from_dict(data)
        assert restored.name == "Aria"
        assert restored.role == "EU insurance broker copilot"
        assert restored.principles == ["never share PII"]
        assert restored.constraints == ["EU data only"]

    def test_from_dict_partial_data(self) -> None:
        # Old or partial data should not crash
        restored = AgentIdentity.from_dict({"name": "X"})
        assert restored.name == "X"
        assert restored.role == ""
        assert restored.principles == []
        assert restored.constraints == []

    def test_render_includes_all_sections(self) -> None:
        identity = AgentIdentity(
            name="Aria",
            role="copilot",
            principles=["p1", "p2"],
            constraints=["c1"],
        )
        rendered = identity.render_for_prompt()
        assert "Aria" in rendered
        assert "copilot" in rendered
        assert "p1" in rendered
        assert "p2" in rendered
        assert "c1" in rendered
        assert "AGENT IDENTITY" in rendered

    def test_render_skips_missing_sections(self) -> None:
        identity = AgentIdentity(name="X")
        rendered = identity.render_for_prompt()
        assert "X" in rendered
        assert "Principles" not in rendered
        assert "constraints" not in rendered.lower() or "X" in rendered
