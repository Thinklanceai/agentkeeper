"""Tests for the AK-4 Agent.compress() public API."""

from __future__ import annotations

from pathlib import Path

import pytest

import agentkeeper


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak-test.db"))
    monkeypatch.setenv("AGENTKEEPER_EMBEDDING_PROVIDER", "mock")
    monkeypatch.setattr(agentkeeper, "_storage", None)


class TestAgentCompress:
    def test_compress_returns_report(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.remember("budget: 50k EUR")
        agent.remember("budget: 50k EUR")
        report = agent.compress()
        assert report.facts_before == 2
        assert report.facts_after <= 2

    def test_compress_collapses_duplicates(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.remember("budget: 50k EUR")
        agent.remember("budget: 50k EUR")
        agent.remember("budget: 50k EUR")
        agent.compress()
        assert len(agent.facts) == 1

    def test_compress_preserves_critical(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.principle("never share PII")
        agent.remember("note 1")
        agent.remember("note 2")
        agent.compress()
        contents = [f.content for f in agent.facts]
        assert "never share PII" in contents

    def test_compress_preserves_identity(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.set_identity(
            name="Aria",
            role="copilot",
            principles=["never share PII"],
        )
        agent.remember("x")
        agent.remember("x")
        agent.compress()
        assert agent.identity.name == "Aria"
        assert agent.identity.role == "copilot"
        assert "never share PII" in agent.identity.principles

    def test_contradictions_lists_flagged_facts(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.fact("budget: 50000 EUR", importance=0.5)
        agent.fact("budget: 75000 EUR", importance=0.7)
        # Tune threshold so the mock embedder triggers
        from agentkeeper.compression.contradiction import ContradictionConfig
        from agentkeeper.compression.pipeline import (
            CompressionConfig,
        )

        config = CompressionConfig(
            run_decay=False,
            run_consolidation=False,
            contradiction=ContradictionConfig(similarity_threshold=0.3),
        )
        agent.compress(config=config)

        flagged = agent.contradictions()
        assert len(flagged) >= 1
        assert all("contradicted_by" in f.metadata for f in flagged)

    def test_compress_resets_recaller(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.remember("x")
        agent.recall("x")  # populates recaller
        recaller_before = agent._recaller
        agent.compress()
        # Compress should force a recaller rebuild
        assert agent._recaller is None or agent._recaller is not recaller_before

    def test_save_load_persists_metadata(self) -> None:
        agent = agentkeeper.create(agent_id="persist", provider="mock")
        agent.fact("budget: 50000 EUR", importance=0.5)
        agent.fact("budget: 75000 EUR", importance=0.7)

        from agentkeeper.compression.contradiction import ContradictionConfig
        from agentkeeper.compression.pipeline import CompressionConfig

        config = CompressionConfig(
            run_decay=False,
            run_consolidation=False,
            contradiction=ContradictionConfig(similarity_threshold=0.3),
        )
        agent.compress(config=config)
        agent.save()

        loaded = agentkeeper.load("persist", provider="mock")
        flagged = loaded.contradictions()
        assert len(flagged) >= 1
