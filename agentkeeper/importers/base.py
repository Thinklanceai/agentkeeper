"""Shared primitives for platform memory importers.

Importers follow a strict two-phase contract:

1. ``parse()`` reads the export *read-only* and produces a list of
   normalized :class:`ImportedFact` objects. No agent state is touched.
2. ``apply(agent)`` writes the parsed facts into an
   :class:`~agentkeeper.Agent` and returns an :class:`ImportReport`.

This keeps imports deterministic, inspectable and testable: callers can
review exactly what would be written before committing anything.

Scope convention
----------------
Every imported fact carries metadata describing its provenance and its
visibility scope::

    {
        "source": "chatgpt",            # platform identifier
        "scope": "platform:chatgpt",    # strict visibility namespace
        "imported_at": "<ISO-8601 UTC>",
        "external_id": "<platform-side id, when available>",
    }

The ``scope`` key is the foundation for cross-platform isolation: a
context compiler can filter facts by scope and guarantee that knowledge
imported from one platform never leaks into another platform's context
unless the user explicitly promotes it to ``scope: global``.
"""

from __future__ import annotations

import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..cso.fact_types import FactType
from ..cso.tiers import MemoryTier
from ..errors import AgentKeeperError

if TYPE_CHECKING:
    from .. import Agent

SCOPE_GLOBAL = "global"

MAX_ZIP_MEMBERS = 10_000
MAX_MEMBER_UNCOMPRESSED_BYTES = 1_073_741_824
MAX_TOTAL_UNCOMPRESSED_BYTES = 2_147_483_648
SUSPICIOUS_COMPRESSION_RATIO = 200.0
RATIO_GUARD_MIN_BYTES = 52_428_800


class ImportError_(AgentKeeperError):
    """Raised when an export archive is invalid, unsafe or unreadable."""


def utcnow_iso() -> str:
    """Current UTC time as an ISO-8601 string with second precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def epoch_to_iso(epoch: float | int | None) -> str | None:
    """Convert a platform epoch timestamp to ISO-8601 UTC, if present."""
    if epoch is None:
        return None
    try:
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    except (OverflowError, OSError, ValueError):
        return None


def platform_scope(platform: str) -> str:
    """Strict visibility namespace for a given platform."""
    return f"platform:{platform}"


def validate_zip(zf: zipfile.ZipFile) -> None:
    """Reject archives that look hostile before reading any member.

    Guards against zip bombs (member count, uncompressed size, extreme
    compression ratios) and path traversal in member names. Members are
    only ever read in memory — nothing is extracted to disk — so the
    traversal check is defence in depth.
    """
    infos = zf.infolist()
    if len(infos) > MAX_ZIP_MEMBERS:
        raise ImportError_(
            f"Archive has {len(infos)} members "
            f"(limit {MAX_ZIP_MEMBERS}); refusing to read."
        )
    total = 0
    for info in infos:
        name = info.filename
        if name.startswith(("/", "\\")) or ".." in name.split("/"):
            raise ImportError_(f"Unsafe member path in archive: {name!r}")
        total += info.file_size
        if info.file_size > MAX_MEMBER_UNCOMPRESSED_BYTES:
            raise ImportError_(
                f"Member {name!r} declares {info.file_size} uncompressed "
                f"bytes (limit {MAX_MEMBER_UNCOMPRESSED_BYTES})."
            )
        if (
            info.file_size > RATIO_GUARD_MIN_BYTES
            and info.compress_size > 0
            and info.file_size / info.compress_size
            > SUSPICIOUS_COMPRESSION_RATIO
        ):
            raise ImportError_(
                f"Member {name!r} has a suspicious compression ratio; "
                "refusing to read (possible zip bomb)."
            )
    if total > MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise ImportError_(
            f"Archive declares {total} total uncompressed bytes "
            f"(limit {MAX_TOTAL_UNCOMPRESSED_BYTES})."
        )


@dataclass
class ImportedFact:
    """A normalized, platform-agnostic fact ready to be applied."""

    content: str
    fact_type: FactType = FactType.FACT
    tier: MemoryTier = MemoryTier.SEMANTIC
    importance: float = 0.5
    when: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImportReport:
    """Summary of an applied import."""

    platform: str
    conversations_parsed: int = 0
    messages_seen: int = 0
    facts_created: int = 0
    skipped: int = 0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"[{self.platform}] {self.conversations_parsed} conversations, "
            f"{self.messages_seen} messages seen, "
            f"{self.facts_created} facts created, "
            f"{self.skipped} skipped, {len(self.warnings)} warnings"
        )


class BaseImporter(ABC):
    """Contract shared by all platform importers."""

    platform: str = "unknown"

    @abstractmethod
    def parse(self) -> list[ImportedFact]:
        """Read the export and return normalized facts. Read-only."""

    def apply(self, agent: Agent) -> ImportReport:
        """Write parsed facts into ``agent`` and return a report.

        The agent is saved once at the end, not per fact.
        """
        facts = self.parse()
        report = self._make_report()
        for f in facts:
            agent._cso.add_fact(
                f.content,
                tier=f.tier,
                importance=f.importance,
                when=f.when,
                metadata=f.metadata,
                fact_type=f.fact_type,
            )
            report.facts_created += 1
        agent.save()
        return report

    def _make_report(self) -> ImportReport:
        return ImportReport(platform=self.platform)
