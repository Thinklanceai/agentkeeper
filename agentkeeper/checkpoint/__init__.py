"""Cognitive checkpoints: immutable, dated, content-hashed snapshots.

A checkpoint freezes an agent's *cognitive* state (identity, facts,
triples) at a point in time and lets you restore it later -- after a
crash, a context-window overflow, a model switch, or a process restart.

Design boundaries (deliberate):

- AgentKeeper persists, versions, hashes and restores cognitive state.
  It is *not* an execution runtime. The optional `execution_state`
  payload is an opaque, JSON-serialisable blob: AgentKeeper stores it
  and hands it back verbatim. It is never interpreted, validated beyond
  JSON-serialisability, or executed.

- Determinism applies to *reconstruction*, not behaviour. Restoring a
  snapshot always rebuilds an identical cognitive state (verified by a
  content hash). What the model decides to do next is out of scope.

- Checkpoints live in their own flat-file store, outside the storage
  backend contract (`BaseStorage`). This keeps that contract narrow and
  makes checkpoints behave identically across SQLite, encrypted SQLite
  and Postgres backends, with zero required dependencies.
"""

from __future__ import annotations

import hashlib
import copy
import json
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import AgentKeeperError

if TYPE_CHECKING:
    from ..cso.types import CognitiveStateObject

SNAPSHOT_SCHEMA_VERSION = 1

_ID_RE = re.compile(r"^cp_\d{8}T\d{6}_[0-9a-f]{6}$")
_SAFE_AGENT_ID_RE = re.compile(r"[^A-Za-z0-9._-]")


class CheckpointError(AgentKeeperError):
    """Raised for checkpoint storage, lookup or integrity failures."""


# --- helpers ---------------------------------------------------------


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_snapshot_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"cp_{stamp}_{secrets.token_hex(3)}"


def _canonical_cognitive_state(cso_dict: dict[str, Any]) -> dict[str, Any]:
    """Project a CSO dict onto its semantically meaningful fields.

    Access telemetry (`last_accessed_at`, `access_count`) and token
    counts are excluded: they change as a side effect of *reading*
    memory and would otherwise make two cognitively identical states
    hash differently. The hash answers "is the cognitive state the
    same?", not "has anyone touched it since?".
    """
    identity = cso_dict.get("identity", {}) or {}
    canonical_identity = {
        "name": identity.get("name", "") or "",
        "role": identity.get("role", "") or "",
        "principles": list(identity.get("principles", []) or []),
        "constraints": list(identity.get("constraints", []) or []),
    }

    facts = []
    for f in cso_dict.get("memory_facts", []) or []:
        facts.append(
            {
                "id": f.get("id"),
                "content": f.get("content"),
                "tier": f.get("tier"),
                "fact_type": f.get("fact_type"),
                "importance": f.get("importance"),
                "protected": bool(f.get("protected", False)),
                "when": f.get("when"),
                "expires_at": f.get("expires_at"),
                "metadata": f.get("metadata", {}) or {},
            }
        )
    facts.sort(key=lambda x: (x["id"] is None, x["id"]))

    triples = [dict(t) for t in (cso_dict.get("triples", []) or [])]
    triples.sort(key=lambda t: json.dumps(t, sort_keys=True, ensure_ascii=False))

    return {
        "agent_id": cso_dict.get("agent_id"),
        "identity": canonical_identity,
        "memory_facts": facts,
        "triples": triples,
    }


def content_hash(cso_dict: dict[str, Any]) -> str:
    """Deterministic SHA-256 over the canonical cognitive state."""
    canonical = _canonical_cognitive_state(cso_dict)
    encoded = json.dumps(
        canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_json_serialisable(value: Any, field: str) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(
            f"{field} must be JSON-serialisable; got {type(value).__name__}: {exc}"
        ) from exc
    return value


# --- snapshot --------------------------------------------------------


@dataclass(frozen=True)
class SnapshotMeta:
    """Lightweight checkpoint descriptor (no cognitive payload)."""

    snapshot_id: str
    agent_id: str
    created_at: str
    label: str | None
    content_hash: str
    fact_count: int
    triple_count: int
    has_execution_state: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "agent_id": self.agent_id,
            "created_at": self.created_at,
            "label": self.label,
            "content_hash": self.content_hash,
            "fact_count": self.fact_count,
            "triple_count": self.triple_count,
            "has_execution_state": self.has_execution_state,
        }


@dataclass(frozen=True)
class Snapshot:
    """An immutable, content-hashed cognitive checkpoint."""

    snapshot_id: str
    agent_id: str
    created_at: str
    label: str | None
    cognitive_state: dict[str, Any]
    execution_state: dict[str, Any] | None
    content_hash: str
    schema_version: int = SNAPSHOT_SCHEMA_VERSION

    @property
    def meta(self) -> SnapshotMeta:
        return SnapshotMeta(
            snapshot_id=self.snapshot_id,
            agent_id=self.agent_id,
            created_at=self.created_at,
            label=self.label,
            content_hash=self.content_hash,
            fact_count=len(self.cognitive_state.get("memory_facts", []) or []),
            triple_count=len(self.cognitive_state.get("triples", []) or []),
            has_execution_state=self.execution_state is not None,
        )

    def to_dict(self) -> dict[str, Any]:
        # Deep-copy mutable payloads so callers cannot mutate a snapshot
        # that is meant to be immutable.
        return {
            "snapshot_id": self.snapshot_id,
            "agent_id": self.agent_id,
            "created_at": self.created_at,
            "label": self.label,
            "cognitive_state": copy.deepcopy(self.cognitive_state),
            "execution_state": copy.deepcopy(self.execution_state),
            "content_hash": self.content_hash,
            "schema_version": self.schema_version,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Snapshot:
        return Snapshot(
            snapshot_id=data["snapshot_id"],
            agent_id=data["agent_id"],
            created_at=data["created_at"],
            label=data.get("label"),
            cognitive_state=data["cognitive_state"],
            execution_state=data.get("execution_state"),
            content_hash=data["content_hash"],
            schema_version=int(data.get("schema_version", SNAPSHOT_SCHEMA_VERSION)),
        )

    def verify(self) -> bool:
        """Return True if the stored hash matches the cognitive state."""
        return self.content_hash == content_hash(self.cognitive_state)


def build_snapshot(
    cso: CognitiveStateObject,
    label: str | None = None,
    execution_state: dict[str, Any] | None = None,
) -> Snapshot:
    """Freeze a CSO into an immutable Snapshot."""
    cso_dict = cso.to_dict()
    if execution_state is not None:
        if not isinstance(execution_state, dict):
            raise CheckpointError(
                "execution_state must be a dict or None; "
                f"got {type(execution_state).__name__}"
            )
        _assert_json_serialisable(execution_state, "execution_state")
        execution_state = json.loads(json.dumps(execution_state))

    return Snapshot(
        snapshot_id=_new_snapshot_id(),
        agent_id=cso.agent_id,
        created_at=_utcnow_iso(),
        label=label,
        cognitive_state=cso_dict,
        execution_state=execution_state,
        content_hash=content_hash(cso_dict),
    )


# --- flat-file store -------------------------------------------------


def _default_root() -> Path:
    env = os.environ.get("AGENTKEEPER_CHECKPOINT_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".agentkeeper" / "checkpoints"


def _safe_agent_dir(root: Path, agent_id: str) -> Path:
    if not agent_id:
        raise CheckpointError("agent_id must be a non-empty string.")
    safe = _SAFE_AGENT_ID_RE.sub("_", agent_id)
    return root / safe


class CheckpointStore:
    """Flat-file JSON store for snapshots, one directory per agent.

    Layout: <root>/<safe_agent_id>/<snapshot_id>.json
    Default root: ~/.agentkeeper/checkpoints (override via
    AGENTKEEPER_CHECKPOINT_DIR). This store is intentionally independent
    of BaseStorage so checkpoints behave identically across all backends.
    """

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root).expanduser() if root else _default_root()

    def _agent_dir(self, agent_id: str) -> Path:
        return _safe_agent_dir(self.root, agent_id)

    def save(self, snapshot: Snapshot) -> Snapshot:
        if not _ID_RE.match(snapshot.snapshot_id):
            raise CheckpointError(f"Malformed snapshot_id: {snapshot.snapshot_id!r}")
        agent_dir = self._agent_dir(snapshot.agent_id)
        agent_dir.mkdir(parents=True, exist_ok=True)
        target = agent_dir / f"{snapshot.snapshot_id}.json"
        tmp = agent_dir / f".{snapshot.snapshot_id}.json.tmp"
        payload = json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2)
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, target)  # atomic write
        return snapshot

    def load(self, agent_id: str, snapshot_id: str) -> Snapshot:
        if not _ID_RE.match(snapshot_id):
            raise CheckpointError(f"Malformed snapshot_id: {snapshot_id!r}")
        path = self._agent_dir(agent_id) / f"{snapshot_id}.json"
        if not path.is_file():
            raise CheckpointError(
                f"Checkpoint {snapshot_id!r} not found for agent {agent_id!r}."
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        return Snapshot.from_dict(data)

    def list(self, agent_id: str) -> list[SnapshotMeta]:
        agent_dir = self._agent_dir(agent_id)
        if not agent_dir.is_dir():
            return []
        metas: list[SnapshotMeta] = []
        for path in agent_dir.glob("cp_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                metas.append(Snapshot.from_dict(data).meta)
            except (ValueError, KeyError, OSError):
                continue
        metas.sort(key=lambda m: (m.created_at, m.snapshot_id))
        return metas

    def delete(self, agent_id: str, snapshot_id: str) -> bool:
        if not _ID_RE.match(snapshot_id):
            raise CheckpointError(f"Malformed snapshot_id: {snapshot_id!r}")
        path = self._agent_dir(agent_id) / f"{snapshot_id}.json"
        if path.is_file():
            path.unlink()
            return True
        return False


__all__ = [
    "CheckpointError",
    "CheckpointStore",
    "Snapshot",
    "SnapshotMeta",
    "SNAPSHOT_SCHEMA_VERSION",
    "build_snapshot",
    "content_hash",
]
