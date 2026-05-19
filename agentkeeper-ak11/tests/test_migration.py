"""Tests for v0.1 → v0.3 schema migration (backward compatibility)."""

from __future__ import annotations

from agentkeeper.cso.types import CognitiveStateObject


class TestLegacyMigration:
    def test_v01_fact_dict_loads_cleanly(self) -> None:
        # Schema as it was in v0.1: only id/content/critical/token_count
        legacy_data = {
            "agent_id": "legacy-agent",
            "memory_facts": [
                {
                    "id": "f1",
                    "content": "budget: 50k",
                    "critical": True,
                    "token_count": 3,
                },
                {
                    "id": "f2",
                    "content": "note",
                    "critical": False,
                    "token_count": 1,
                },
            ],
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }

        cso = CognitiveStateObject.from_dict(legacy_data)
        assert cso.agent_id == "legacy-agent"
        assert len(cso.memory_facts) == 2

        # Critical fact migrated to importance >= 0.9
        critical_fact = next(f for f in cso.memory_facts if f.content == "budget: 50k")
        assert critical_fact.critical is True
        assert critical_fact.importance >= 0.9

        # Non-critical fact has default importance
        non_critical = next(f for f in cso.memory_facts if f.content == "note")
        assert non_critical.critical is False
        assert non_critical.importance < 0.9

    def test_v01_data_without_identity_migrates(self) -> None:
        legacy_data = {
            "agent_id": "x",
            "memory_facts": [],
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
        cso = CognitiveStateObject.from_dict(legacy_data)
        assert cso.identity.is_empty()

    def test_roundtrip_preserves_new_fields(self) -> None:
        cso = CognitiveStateObject.create(agent_id="rt")
        cso.add_fact("budget: 50k", critical=True)
        cso.add_fact("client refused yesterday", tier="episodic")
        d = cso.to_dict()
        restored = CognitiveStateObject.from_dict(d)
        assert len(restored.memory_facts) == 2
        assert any(f.tier.value == "episodic" for f in restored.memory_facts)
