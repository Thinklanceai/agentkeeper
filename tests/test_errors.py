"""Tests for the AK-7 typed exception hierarchy."""

from __future__ import annotations

from pathlib import Path

import pytest

import agentkeeper
from agentkeeper import (
    AgentKeeperError,
    AgentNotFoundError,
    UnknownProviderError,
    UnknownTierError,
)


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak-test.db"))
    monkeypatch.setattr(agentkeeper, "_storage", None)


class TestExceptionHierarchy:
    def test_all_inherit_from_root(self) -> None:
        for cls in (UnknownProviderError, UnknownTierError, AgentNotFoundError):
            assert issubclass(cls, AgentKeeperError)

    def test_value_error_compat(self) -> None:
        # Existing code catching ValueError should still work.
        assert issubclass(UnknownProviderError, ValueError)
        assert issubclass(UnknownTierError, ValueError)
        assert issubclass(AgentNotFoundError, ValueError)


class TestRaiseSites:
    def test_unknown_provider_in_create(self) -> None:
        with pytest.raises(UnknownProviderError):
            agentkeeper.create(provider="not-real")

    def test_unknown_provider_in_switch(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        with pytest.raises(UnknownProviderError):
            agent.switch_provider("nope")

    def test_unknown_tier_in_remember(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        with pytest.raises(UnknownTierError):
            agent.remember("x", tier="weird-tier")

    def test_agent_not_found_in_load(self) -> None:
        with pytest.raises(AgentNotFoundError):
            agentkeeper.load("does-not-exist", provider="mock")

    def test_old_value_error_catch_still_works(self) -> None:
        """Code written for v0.x that catches ValueError keeps working."""
        with pytest.raises(ValueError):
            agentkeeper.create(provider="not-real")
