"""Tests for expiration on Fact + the agent retention/GDPR surface."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import agentkeeper
from agentkeeper import MemoryPolicy
from agentkeeper.cso.types import Fact


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak-test.db"))
    monkeypatch.setenv("AGENTKEEPER_EMBEDDING_PROVIDER", "mock")
    monkeypatch.setattr(agentkeeper, "_storage", None)


class TestFactExpiresAt:
    def test_no_ttl_no_expiry(self) -> None:
        f = Fact.create("hello")
        assert f.expires_at is None

    def test_ttl_string_sets_expires_at(self) -> None:
        f = Fact.create("hello", ttl="30d")
        assert f.expires_at is not None
        # Should be roughly 30 days in the future
        parsed = datetime.fromisoformat(f.expires_at)
        delta = parsed - datetime.now(timezone.utc)
        assert timedelta(days=29) < delta <= timedelta(days=30)

    def test_ttl_timedelta(self) -> None:
        f = Fact.create("hello", ttl=timedelta(hours=1))
        assert f.expires_at is not None

    def test_explicit_expires_at_wins(self) -> None:
        explicit = "2030-01-01T00:00:00+00:00"
        f = Fact.create("hello", ttl="30d", expires_at=explicit)
        assert f.expires_at == explicit

    def test_serialise_includes_expires_at(self) -> None:
        f = Fact.create("hello", ttl="30d")
        d = f.to_dict()
        assert "expires_at" in d
        # Roundtrip
        f2 = Fact.from_dict(d)
        assert f2.expires_at == f.expires_at

    def test_legacy_dict_loads_with_no_expiration(self) -> None:
        legacy = {
            "id": "x",
            "content": "y",
            "tier": "semantic",
            "importance": 0.5,
            "critical": False,
        }
        f = Fact.from_dict(legacy)
        assert f.expires_at is None


class TestAgentRememberWithTTL:
    def test_remember_with_ttl(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.remember("ephemeral", ttl="1h")
        last = agent.last_fact()
        assert last is not None
        assert last.expires_at is not None


class TestPolicyDrivenTTL:
    def test_policy_default_applies_when_no_ttl(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.set_memory_policy(MemoryPolicy(default_ttl="30d"))
        agent.fact("ordinary fact")
        last = agent.last_fact()
        assert last is not None
        assert last.expires_at is not None

    def test_policy_per_type_applies(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.set_memory_policy(
            MemoryPolicy(default_ttl="30d", per_type={"transient": "1h"})
        )
        agent.transient("ephemeral")
        last = agent.last_fact()
        assert last is not None
        assert last.expires_at is not None
        # Should be ~1h, not ~30d
        delta = (
            datetime.fromisoformat(last.expires_at)
            - datetime.now(timezone.utc)
        )
        assert delta < timedelta(hours=2)

    def test_policy_skips_protected_by_default(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.set_memory_policy(MemoryPolicy(default_ttl="30d"))
        agent.principle("never share PII")
        last = agent.last_fact()
        assert last is not None
        assert last.expires_at is None

    def test_explicit_ttl_overrides_policy(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.set_memory_policy(MemoryPolicy(default_ttl="30d"))
        agent.remember("special", ttl="1h")
        last = agent.last_fact()
        assert last is not None
        delta = (
            datetime.fromisoformat(last.expires_at)
            - datetime.now(timezone.utc)
        )
        assert delta < timedelta(hours=2)


class TestPurgeExpired:
    def test_purges_expired_facts(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.fact("keeper")
        agent.remember("ephemeral", ttl="1h")
        # Force the second fact to be expired
        agent.facts[1].expires_at = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat()
        purged = agent.purge_expired()
        assert purged == 1
        assert len(agent.facts) == 1
        assert agent.facts[0].content == "keeper"

    def test_protected_facts_never_purged_by_purge_expired(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.principle("never share PII")
        # Even if we forcibly mark it expired
        agent.facts[0].expires_at = (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).isoformat()
        purged = agent.purge_expired()
        assert purged == 0
        assert len(agent.facts) == 1


class TestCompressionPurgesExpired:
    def test_compression_purges_first(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        for i in range(3):
            agent.fact(f"keeper {i}")
        agent.remember("ephemeral", ttl="1h")
        agent.facts[-1].expires_at = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat()
        report = agent.compress()
        assert report.expired_purged == 1


class TestGDPRExport:
    def test_export_includes_facts_and_identity(self) -> None:
        agent = agentkeeper.create(agent_id="gdpr-a", provider="mock")
        agent.set_identity(name="Aria", principles=["never share PII"])
        agent.fact("budget: 50k")
        agent.decision("use Anthropic")
        export = agent.gdpr_export()
        assert export["schema_version"] == "1.1"
        assert export["agent_id"] == "gdpr-a"
        assert export["identity"]["name"] == "Aria"
        assert len(export["facts"]) == 2 or len(export["facts"]) == 3
        # Each fact carries its full dict
        for f in export["facts"]:
            assert "content" in f
            assert "fact_type" in f


class TestGDPRPurge:
    def test_purge_all_non_protected(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.principle("never share PII")
        agent.fact("budget: 50k")
        agent.decision("d1")
        # Purge everything except protected
        removed = agent.gdpr_purge()
        assert removed == 2
        assert len(agent.facts) == 1
        assert agent.facts[0].protected is True

    def test_purge_with_predicate(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.fact("acme data")
        agent.fact("globex data")
        agent.fact("totally unrelated")
        removed = agent.gdpr_purge(lambda f: "acme" in f.content)
        assert removed == 1
        contents = [f.content for f in agent.facts]
        assert "acme data" not in contents
        assert "globex data" in contents

    def test_include_protected_flag(self) -> None:
        agent = agentkeeper.create(agent_id="a", provider="mock")
        agent.principle("never share PII")
        agent.fact("ordinary")
        removed = agent.gdpr_purge(include_protected=True)
        assert removed == 2
        assert len(agent.facts) == 0


class TestRoundTrip:
    def test_ttl_survives_save_load(self) -> None:
        agent = agentkeeper.create(agent_id="rt", provider="mock")
        agent.remember("ephemeral", ttl="30d")
        original_expiry = agent.facts[0].expires_at
        agent.save()
        loaded = agentkeeper.load("rt", provider="mock")
        assert loaded.facts[0].expires_at == original_expiry


class TestAsyncMirrors:
    def test_async_remember_with_ttl(self) -> None:
        agent = agentkeeper.create_async(agent_id="a", provider="mock")
        agent.remember("ephemeral", ttl="1h")
        last = agent.last_fact()
        assert last is not None
        assert last.expires_at is not None

    def test_async_purge_expired(self) -> None:
        agent = agentkeeper.create_async(agent_id="a", provider="mock")
        agent.fact("keeper")
        agent.remember("doomed", ttl="1h")
        agent.facts[1].expires_at = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat()
        assert agent.purge_expired() == 1

    def test_async_gdpr_export(self) -> None:
        agent = agentkeeper.create_async(agent_id="a", provider="mock")
        agent.fact("x")
        export = agent.gdpr_export()
        assert export["agent_id"] == "a"

    def test_async_gdpr_purge(self) -> None:
        agent = agentkeeper.create_async(agent_id="a", provider="mock")
        agent.fact("x")
        agent.fact("y")
        assert agent.gdpr_purge() == 2
