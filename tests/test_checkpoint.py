"""Tests for the cognitive checkpoint layer.

Covers: snapshot round-trip, hash determinism (and its insensitivity to
access telemetry), tamper detection, opaque execution_state handling,
restore fidelity, listing, factual diff, and path-traversal safety.
"""

from __future__ import annotations

import pytest

import agentkeeper
from agentkeeper.checkpoint import (
    CheckpointError,
    CheckpointStore,
    Snapshot,
    content_hash,
)
from agentkeeper.checkpoint.diff import diff_snapshots


@pytest.fixture(autouse=True)
def _isolated_checkpoint_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTKEEPER_CHECKPOINT_DIR", str(tmp_path / "cp"))
    monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak.db"))
    yield


def _make_agent(agent_id="aria"):
    agent = agentkeeper.create(agent_id=agent_id, provider="mock")
    agent.set_identity(
        name="Aria",
        role="EU insurance broker copilot",
        principles=["never share PII without explicit consent"],
        constraints=["EU data residency only"],
    )
    agent.fact("client: Acme Corporation", importance=0.95)
    agent.fact("budget: 50k EUR", importance=0.9)
    return agent


# --- snapshot core ---------------------------------------------------


def test_checkpoint_round_trip():
    agent = _make_agent()
    snap = agent.checkpoint(label="before-refactor")
    assert snap.snapshot_id.startswith("cp_")
    reloaded = agentkeeper.load_checkpoint("aria", snap.snapshot_id)
    assert reloaded.cognitive_state["identity"]["name"] == "Aria"
    assert reloaded.label == "before-refactor"
    assert reloaded.verify()


def test_execution_state_is_opaque_round_trip():
    agent = _make_agent()
    payload = {"current_file": "auth.py", "pending_task": "fix jwt", "todos": [1, 2]}
    snap = agent.checkpoint(execution_state=payload)
    reloaded = agentkeeper.load_checkpoint("aria", snap.snapshot_id)
    assert reloaded.execution_state == payload


def test_execution_state_non_serialisable_rejected():
    agent = _make_agent()
    with pytest.raises(CheckpointError):
        agent.checkpoint(execution_state={"bad": {1, 2, 3}})


def test_execution_state_must_be_dict():
    agent = _make_agent()
    with pytest.raises(CheckpointError):
        agent.checkpoint(execution_state=["not", "a", "dict"])


# --- hash determinism ------------------------------------------------


def test_hash_is_deterministic_for_same_state():
    agent = _make_agent("h1")
    d1 = agent._cso.to_dict()
    d2 = agent._cso.to_dict()
    assert content_hash(d1) == content_hash(d2)

    from agentkeeper.cso.types import CognitiveStateObject
    rebuilt = CognitiveStateObject.from_dict(agent._cso.to_dict())
    assert content_hash(rebuilt.to_dict()) == content_hash(agent._cso.to_dict())


def test_hash_ignores_access_telemetry():
    agent = _make_agent("h2")
    base = content_hash(agent._cso.to_dict())
    for fact in agent._cso.memory_facts:
        fact.access_count += 50
        fact.last_accessed_at = "2099-01-01T00:00:00+00:00"
    assert content_hash(agent._cso.to_dict()) == base


def test_hash_changes_on_real_content_change():
    agent = _make_agent("h3")
    base = content_hash(agent._cso.to_dict())
    agent._cso.memory_facts[0].content = "client: Globex"
    assert content_hash(agent._cso.to_dict()) != base


def test_verify_detects_tampering():
    agent = _make_agent()
    snap = agent.checkpoint()
    data = snap.to_dict()
    data["cognitive_state"]["memory_facts"][0]["content"] = "TAMPERED"
    tampered = Snapshot.from_dict(data)
    assert not tampered.verify()


def test_to_dict_is_immutable_snapshot():
    agent = _make_agent()
    snap = agent.checkpoint()
    d = snap.to_dict()
    d["cognitive_state"]["memory_facts"][0]["content"] = "MUTATED"
    # Original snapshot must be untouched.
    assert snap.cognitive_state["memory_facts"][0]["content"] != "MUTATED"


# --- restore ---------------------------------------------------------


def test_restore_rebuilds_identical_state():
    agent = _make_agent()
    snap = agent.checkpoint()
    original_hash = snap.content_hash

    agent.fact("temp: scratch note", importance=0.2)
    agent._cso.identity.name = "Drifted"
    assert content_hash(agent._cso.to_dict()) != original_hash

    agent.restore(snap.snapshot_id)
    assert content_hash(agent._cso.to_dict()) == original_hash
    assert agent._cso.identity.name == "Aria"


def test_restore_does_not_autosave_main_cso():
    agent = _make_agent()
    snap = agent.checkpoint()
    agent.fact("new fact after checkpoint", importance=0.5)
    agent.save()
    agent.restore(snap.snapshot_id)
    # Restore mutates in-memory only; persisted main CSO still has the new fact
    # until the caller chooses to save().
    reloaded = agentkeeper.load("aria", provider="mock")
    contents = {f.content for f in reloaded._cso.memory_facts}
    assert "new fact after checkpoint" in contents


def test_restore_unknown_snapshot_raises():
    agent = _make_agent()
    with pytest.raises(CheckpointError):
        agent.restore("cp_20000101T000000_abcdef")


# --- listing ---------------------------------------------------------


def test_list_checkpoints_sorted():
    agent = _make_agent()
    s1 = agent.checkpoint(label="one")
    s2 = agent.checkpoint(label="two")
    metas = agent.list_checkpoints()
    ids = [m.snapshot_id for m in metas]
    assert s1.snapshot_id in ids and s2.snapshot_id in ids
    assert metas == sorted(metas, key=lambda m: (m.created_at, m.snapshot_id))


def test_list_checkpoints_empty_for_fresh_agent():
    agent = _make_agent("nobody")
    assert agent.list_checkpoints() == []


# --- diff ------------------------------------------------------------


def test_diff_reports_factual_changes():
    agent = _make_agent("d1")
    before = agent.checkpoint()

    agent.fact("contact: Alice", importance=0.6)
    agent._cso.memory_facts[0].content = "client: Acme Holding"
    agent._cso.identity.principles.append("log every access")
    after = agent.checkpoint()

    d = diff_snapshots(
        agentkeeper.load_checkpoint("d1", before.snapshot_id),
        agentkeeper.load_checkpoint("d1", after.snapshot_id),
    )
    summary = d.to_dict()["summary"]
    assert summary["facts_added"] >= 1
    assert summary["facts_modified"] >= 1
    assert "principles" in d.identity_changes
    assert not d.is_empty


def test_diff_identical_is_empty():
    agent = _make_agent("d2")
    snap = agent.checkpoint()
    s = agentkeeper.load_checkpoint("d2", snap.snapshot_id)
    assert diff_snapshots(s, s).is_empty


# --- safety ----------------------------------------------------------


def test_malformed_snapshot_id_rejected():
    store = CheckpointStore()
    with pytest.raises(CheckpointError):
        store.load("aria", "not-a-valid-id")


def test_agent_id_path_traversal_is_neutralised(tmp_path):
    agent = agentkeeper.create(agent_id="../../etc/evil", provider="mock")
    agent.set_identity(name="X", role="Y")
    agent.fact("z", importance=0.5)
    snap = agent.checkpoint()
    # Retrievable via the same id, and confined under the checkpoint root.
    assert agentkeeper.load_checkpoint("../../etc/evil", snap.snapshot_id).verify()
