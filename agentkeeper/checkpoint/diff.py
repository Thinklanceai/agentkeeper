"""Factual diff between two cognitive snapshots.

Reports concrete, defensible changes only -- added/removed/modified
facts (keyed by stable fact id), added/removed triples, and identity
field changes. No subjective scores (no "drift", no "continuity %"):
every number here is a count of something you can point at.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from . import Snapshot

# Fact fields whose change constitutes a *semantic* modification.
# Access telemetry and token counts are excluded on purpose.
_FACT_COMPARE_FIELDS = (
    "content",
    "tier",
    "fact_type",
    "importance",
    "protected",
    "when",
    "expires_at",
    "metadata",
)


@dataclass
class FactChange:
    fact_id: str
    fields: dict[str, dict[str, Any]]  # field -> {"from": ..., "to": ...}

    def to_dict(self) -> dict[str, Any]:
        return {"fact_id": self.fact_id, "fields": self.fields}


@dataclass
class SnapshotDiff:
    from_id: str
    to_id: str
    facts_added: list[dict[str, Any]] = field(default_factory=list)
    facts_removed: list[dict[str, Any]] = field(default_factory=list)
    facts_modified: list[FactChange] = field(default_factory=list)
    triples_added: list[dict[str, Any]] = field(default_factory=list)
    triples_removed: list[dict[str, Any]] = field(default_factory=list)
    identity_changes: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (
            self.facts_added
            or self.facts_removed
            or self.facts_modified
            or self.triples_added
            or self.triples_removed
            or self.identity_changes
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_id": self.from_id,
            "to_id": self.to_id,
            "facts_added": self.facts_added,
            "facts_removed": self.facts_removed,
            "facts_modified": [c.to_dict() for c in self.facts_modified],
            "triples_added": self.triples_added,
            "triples_removed": self.triples_removed,
            "identity_changes": self.identity_changes,
            "summary": {
                "facts_added": len(self.facts_added),
                "facts_removed": len(self.facts_removed),
                "facts_modified": len(self.facts_modified),
                "triples_added": len(self.triples_added),
                "triples_removed": len(self.triples_removed),
                "identity_fields_changed": len(self.identity_changes),
            },
        }


def _facts_by_id(cog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for f in cog.get("memory_facts", []) or []:
        fid = f.get("id")
        if fid is not None:
            out[fid] = f
    return out


def _triple_key(t: dict[str, Any]) -> str:
    return json.dumps(t, sort_keys=True, ensure_ascii=False)


def diff_snapshots(a: Snapshot, b: Snapshot) -> SnapshotDiff:
    """Diff snapshot `a` (from) against snapshot `b` (to)."""
    cog_a = a.cognitive_state
    cog_b = b.cognitive_state

    result = SnapshotDiff(from_id=a.snapshot_id, to_id=b.snapshot_id)

    facts_a = _facts_by_id(cog_a)
    facts_b = _facts_by_id(cog_b)

    for fid, fact in facts_b.items():
        if fid not in facts_a:
            result.facts_added.append(fact)
    for fid, fact in facts_a.items():
        if fid not in facts_b:
            result.facts_removed.append(fact)
    for fid in facts_a.keys() & facts_b.keys():
        before, after = facts_a[fid], facts_b[fid]
        changed: dict[str, dict[str, Any]] = {}
        for fld in _FACT_COMPARE_FIELDS:
            if before.get(fld) != after.get(fld):
                changed[fld] = {"from": before.get(fld), "to": after.get(fld)}
        if changed:
            result.facts_modified.append(FactChange(fact_id=fid, fields=changed))

    triples_a = {_triple_key(t): t for t in (cog_a.get("triples", []) or [])}
    triples_b = {_triple_key(t): t for t in (cog_b.get("triples", []) or [])}
    for key, t in triples_b.items():
        if key not in triples_a:
            result.triples_added.append(t)
    for key, t in triples_a.items():
        if key not in triples_b:
            result.triples_removed.append(t)

    id_a = cog_a.get("identity", {}) or {}
    id_b = cog_b.get("identity", {}) or {}
    for fld in ("name", "role", "principles", "constraints"):
        if id_a.get(fld) != id_b.get(fld):
            result.identity_changes[fld] = {
                "from": id_a.get(fld),
                "to": id_b.get(fld),
            }

    return result


__all__ = ["FactChange", "SnapshotDiff", "diff_snapshots"]
