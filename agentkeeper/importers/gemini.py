"""Importer for Google Gemini data (Google Takeout).

Gemini has no dedicated export — conversations are logged as flat
activity records in Google Takeout, under
``Takeout/My Activity/Gemini Apps/MyActivity.json``. Each record is one
prompt event::

    {
        "header": "Gemini Apps",
        "title": "Prompted how do I center a div",
        "time": "2026-01-15T09:00:00.123Z",
        "products": ["Gemini Apps"],
        "subtitles": [{"name": "..."}]
    }

There is no conversation grouping and no guaranteed response text, so
the importable unit is the prompt activity itself. Supports the Takeout
ZIP (the file is located anywhere in the archive), an extracted
directory, or a direct path to ``MyActivity.json``.

HTML exports are rejected with an actionable message — the user must
re-run Takeout with the JSON format selected.

Everything is parsed in memory; the archive is never extracted to disk.
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from ..cso.fact_types import FactType
from ..cso.tiers import MemoryTier
from .base import (
    BaseImporter,
    ImportedFact,
    ImportError_,
    ImportReport,
    platform_scope,
    utcnow_iso,
    validate_zip,
)

PLATFORM = "gemini"

_ACTIVITY_FILE = "MyActivity.json"
_PROMPT_MAX_CHARS = 300
_TITLE_PREFIXES = ("Prompted ", "Asked ", "Said ")
_GEMINI_MARKERS = ("gemini", "bard")


def _normalize_iso(value: Any) -> str | None:
    """Normalize a Takeout timestamp to second-precision ISO-8601 UTC."""
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


def _strip_prefix(title: str) -> str:
    for prefix in _TITLE_PREFIXES:
        if title.startswith(prefix):
            return title[len(prefix):]
    return title


def _is_gemini_record(record: dict[str, Any]) -> bool:
    header = record.get("header")
    if isinstance(header, str) and any(
        m in header.lower() for m in _GEMINI_MARKERS
    ):
        return True
    products = record.get("products")
    if isinstance(products, list):
        return any(
            isinstance(p, str)
            and any(m in p.lower() for m in _GEMINI_MARKERS)
            for p in products
        )
    return False


class GeminiImporter(BaseImporter):
    """Parse Gemini activity from a Google Takeout export.

    Args:
        path: Takeout ZIP, extracted directory, or ``MyActivity.json``.
        max_activities: Optional hard cap on activity records parsed.
            Takeout activity logs can hold thousands of prompts; the cap
            keeps imports bounded. Most recent records come first in
            Takeout exports, so capping keeps the freshest activity.
    """

    platform = PLATFORM

    def __init__(
        self,
        path: str | Path,
        max_activities: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.max_activities = max_activities
        self._last_report: ImportReport | None = None

    def parse(self) -> list[ImportedFact]:
        records = self._load_raw()
        report = self._make_report()
        facts: list[ImportedFact] = []
        imported_at = utcnow_iso()

        gemini_records = [r for r in records if _is_gemini_record(r)]
        report.skipped = len(records) - len(gemini_records)

        if self.max_activities is not None:
            gemini_records = gemini_records[: self.max_activities]

        for record in gemini_records:
            fact = self._parse_record(record, imported_at)
            if fact is None:
                report.skipped += 1
                continue
            facts.append(fact)
            report.messages_seen += 1

        report.conversations_parsed = len(facts)
        report.facts_created = 0
        self._last_report = report
        return facts

    def apply(self, agent: Any) -> ImportReport:
        result = super().apply(agent)
        if self._last_report is not None:
            result.conversations_parsed = self._last_report.conversations_parsed
            result.messages_seen = self._last_report.messages_seen
            result.skipped = self._last_report.skipped
            result.warnings = list(self._last_report.warnings)
        return result

    def _load_raw(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            raise ImportError_(f"Path does not exist: {self.path}")

        if self.path.is_dir():
            raw = self._read_activity_dir(self.path)
        elif self.path.suffix.lower() == ".zip":
            raw = self._read_zip(self.path)
        elif self.path.name == _ACTIVITY_FILE:
            raw = self.path.read_bytes()
        elif self.path.suffix.lower() in (".html", ".htm"):
            raise ImportError_(
                "This is an HTML Takeout export. Re-run Google Takeout "
                "with the JSON format selected for My Activity."
            )
        else:
            raise ImportError_(
                f"Unsupported input: {self.path} "
                f"(expected Takeout .zip, directory, or {_ACTIVITY_FILE})"
            )

        try:
            decoded = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ImportError_(
                f"Invalid JSON in {_ACTIVITY_FILE}: {exc}"
            ) from exc
        if not isinstance(decoded, list):
            raise ImportError_(f"{_ACTIVITY_FILE} is not a JSON array.")
        return [r for r in decoded if isinstance(r, dict)]

    @staticmethod
    def _read_activity_dir(path: Path) -> bytes:
        direct = path / _ACTIVITY_FILE
        if direct.exists():
            return direct.read_bytes()
        matches = sorted(path.rglob(_ACTIVITY_FILE))
        for candidate in matches:
            if any(
                m in str(candidate).lower() for m in _GEMINI_MARKERS
            ):
                return candidate.read_bytes()
        if matches:
            return matches[0].read_bytes()
        html_matches = sorted(path.rglob("MyActivity.html"))
        if html_matches:
            raise ImportError_(
                "Found an HTML Takeout export. Re-run Google Takeout "
                "with the JSON format selected for My Activity."
            )
        raise ImportError_(f"No {_ACTIVITY_FILE} found under {path}")

    @staticmethod
    def _read_zip(path: Path) -> bytes:
        try:
            with zipfile.ZipFile(path) as zf:
                validate_zip(zf)
                candidates = [
                    n
                    for n in zf.namelist()
                    if Path(n).name == _ACTIVITY_FILE
                ]
                gemini_first = sorted(
                    candidates,
                    key=lambda n: not any(
                        m in n.lower() for m in _GEMINI_MARKERS
                    ),
                )
                if gemini_first:
                    return zf.read(gemini_first[0])
                if any(
                    Path(n).name == "MyActivity.html" for n in zf.namelist()
                ):
                    raise ImportError_(
                        "This Takeout archive contains an HTML export. "
                        "Re-run Google Takeout with the JSON format "
                        "selected for My Activity."
                    )
                raise ImportError_(
                    f"No {_ACTIVITY_FILE} found in Takeout archive."
                )
        except zipfile.BadZipFile as exc:
            raise ImportError_(f"Not a valid ZIP archive: {path}") from exc

    def _parse_record(
        self, record: dict[str, Any], imported_at: str
    ) -> ImportedFact | None:
        title = record.get("title")
        if not isinstance(title, str) or not title.strip():
            return None
        prompt = _strip_prefix(title.strip())[:_PROMPT_MAX_CHARS]
        if not prompt:
            return None
        when = _normalize_iso(record.get("time"))
        return ImportedFact(
            content=f"Gemini prompt: {prompt}",
            fact_type=FactType.EVENT,
            tier=MemoryTier.EPISODIC,
            importance=0.3,
            when=when,
            metadata=self._meta(imported_at, kind="activity"),
        )

    @staticmethod
    def _meta(imported_at: str, **extra: Any) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "source": PLATFORM,
            "scope": platform_scope(PLATFORM),
            "imported_at": imported_at,
        }
        meta.update(extra)
        return meta
