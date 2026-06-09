"""Importer for Claude.ai data exports.

Supports the official Claude export ZIP (Settings → Privacy → Export
data), a directory containing the extracted files, or a direct path to
``conversations.json`` / ``conversations.jsonl``. Both layouts are
handled: the classic JSON array and the newer JSONL variant with one
conversation object per line.

Conversation shape (both layouts)::

    {
        "uuid": "...",
        "name": "Conversation title",
        "created_at": "2026-01-15T09:00:00.000000Z",
        "updated_at": "...",
        "chat_messages": [
            {
                "uuid": "...",
                "sender": "human" | "assistant",
                "text": "...",
                "content": [{"type": "text", "text": "..."}],
                "created_at": "..."
            }
        ]
    }

``text`` is preferred when present; otherwise text blocks inside
``content`` are joined. ``users.json`` and ``projects.json`` are read
when available — account PII stays out unless ``include_pii=True``.

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

PLATFORM = "claude"

_FIRST_MESSAGE_MAX_CHARS = 500
_CONVERSATION_FILES = ("conversations.json", "conversations.jsonl")


def _normalize_iso(value: Any) -> str | None:
    """Normalize a Claude timestamp to second-precision ISO-8601 UTC."""
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


class ClaudeImporter(BaseImporter):
    """Parse a Claude.ai export into normalized facts.

    Args:
        path: Export ZIP, extracted directory, or conversations file.
        include_messages: When True, the first human message of each
            conversation is kept (truncated) as an archival fact.
            Default False.
        include_pii: When True, account full name and email from
            ``users.json`` are imported as identity facts. Default
            False — privacy first.
        max_conversations: Optional hard cap on conversations parsed.
    """

    platform = PLATFORM

    def __init__(
        self,
        path: str | Path,
        include_messages: bool = False,
        include_pii: bool = False,
        max_conversations: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.include_messages = include_messages
        self.include_pii = include_pii
        self.max_conversations = max_conversations
        self._last_report: ImportReport | None = None

    def parse(self) -> list[ImportedFact]:
        conversations, users, projects = self._load_raw()
        report = self._make_report()
        facts: list[ImportedFact] = []
        imported_at = utcnow_iso()

        facts.extend(self._user_facts(users, imported_at))
        facts.extend(self._project_facts(projects, imported_at))

        if self.max_conversations is not None:
            conversations = conversations[: self.max_conversations]

        for conv in conversations:
            parsed = self._parse_conversation(conv, imported_at, report)
            if parsed is None:
                report.skipped += 1
                continue
            facts.extend(parsed)
            report.conversations_parsed += 1

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

    def _load_raw(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        if not self.path.exists():
            raise ImportError_(f"Path does not exist: {self.path}")

        if self.path.is_dir():
            conv_bytes, conv_name = self._read_conversations_dir(self.path)
            users_bytes = self._read_optional(self.path / "users.json")
            projects_bytes = self._read_optional(self.path / "projects.json")
        elif self.path.suffix.lower() == ".zip":
            conv_bytes, conv_name, users_bytes, projects_bytes = (
                self._read_zip(self.path)
            )
        elif self.path.name in _CONVERSATION_FILES:
            conv_bytes = self.path.read_bytes()
            conv_name = self.path.name
            users_bytes = None
            projects_bytes = None
        else:
            raise ImportError_(
                f"Unsupported input: {self.path} "
                "(expected .zip, directory, conversations.json "
                "or conversations.jsonl)"
            )

        conversations = self._decode_conversations(conv_bytes, conv_name)
        users = self._decode_list(users_bytes, "users.json")
        projects = self._decode_list(projects_bytes, "projects.json")
        return conversations, users, projects

    @staticmethod
    def _read_conversations_dir(path: Path) -> tuple[bytes, str]:
        for name in _CONVERSATION_FILES:
            candidate = path / name
            if candidate.exists():
                return candidate.read_bytes(), name
        raise ImportError_(
            f"No conversations.json or conversations.jsonl in {path}"
        )

    @staticmethod
    def _read_zip(
        path: Path,
    ) -> tuple[bytes, str, bytes | None, bytes | None]:
        try:
            with zipfile.ZipFile(path) as zf:
                validate_zip(zf)
                names = {Path(n).name: n for n in zf.namelist()}
                conv_name = next(
                    (n for n in _CONVERSATION_FILES if n in names), None
                )
                if conv_name is None:
                    raise ImportError_(
                        "No conversations.json or conversations.jsonl "
                        "found in export archive."
                    )
                conv = zf.read(names[conv_name])
                users = (
                    zf.read(names["users.json"])
                    if "users.json" in names
                    else None
                )
                projects = (
                    zf.read(names["projects.json"])
                    if "projects.json" in names
                    else None
                )
                return conv, conv_name, users, projects
        except zipfile.BadZipFile as exc:
            raise ImportError_(f"Not a valid ZIP archive: {path}") from exc

    @staticmethod
    def _read_optional(path: Path) -> bytes | None:
        return path.read_bytes() if path.exists() else None

    @staticmethod
    def _decode_conversations(raw: bytes, name: str) -> list[dict[str, Any]]:
        if name.endswith(".jsonl"):
            conversations: list[dict[str, Any]] = []
            for line_no, line in enumerate(raw.splitlines(), start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ImportError_(
                        f"Invalid JSON on line {line_no} of {name}: {exc}"
                    ) from exc
                if isinstance(obj, dict):
                    conversations.append(obj)
            return conversations
        try:
            decoded = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ImportError_(f"Invalid JSON in {name}: {exc}") from exc
        if not isinstance(decoded, list):
            raise ImportError_(f"{name} is not a JSON array.")
        return [c for c in decoded if isinstance(c, dict)]

    @staticmethod
    def _decode_list(
        raw: bytes | None, label: str
    ) -> list[dict[str, Any]]:
        if raw is None:
            return []
        try:
            decoded = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return []
        if isinstance(decoded, list):
            return [d for d in decoded if isinstance(d, dict)]
        if isinstance(decoded, dict):
            return [decoded]
        return []

    def _user_facts(
        self, users: list[dict[str, Any]], imported_at: str
    ) -> list[ImportedFact]:
        if not self.include_pii:
            return []
        facts: list[ImportedFact] = []
        meta = self._meta(imported_at, kind="account")
        for user in users:
            full_name = user.get("full_name")
            if isinstance(full_name, str) and full_name:
                facts.append(
                    ImportedFact(
                        content=f"Claude account holder: {full_name}",
                        fact_type=FactType.IDENTITY,
                        importance=0.85,
                        metadata=dict(meta),
                    )
                )
            email = user.get("email_address")
            if isinstance(email, str) and email:
                facts.append(
                    ImportedFact(
                        content=f"Claude account email: {email}",
                        fact_type=FactType.IDENTITY,
                        importance=0.85,
                        metadata=dict(meta),
                    )
                )
        return facts

    def _project_facts(
        self, projects: list[dict[str, Any]], imported_at: str
    ) -> list[ImportedFact]:
        facts: list[ImportedFact] = []
        for project in projects:
            name = project.get("name")
            if not isinstance(name, str) or not name:
                continue
            description = project.get("description")
            suffix = (
                f" — {description}"
                if isinstance(description, str) and description
                else ""
            )
            facts.append(
                ImportedFact(
                    content=f"Claude project: '{name}'{suffix}",
                    fact_type=FactType.FACT,
                    importance=0.5,
                    when=_normalize_iso(project.get("created_at")),
                    metadata=self._meta(
                        imported_at,
                        kind="project",
                        external_id=str(project.get("uuid") or ""),
                    ),
                )
            )
        return facts

    def _parse_conversation(
        self,
        conv: dict[str, Any],
        imported_at: str,
        report: ImportReport,
    ) -> list[ImportedFact] | None:
        messages = conv.get("chat_messages")
        if not isinstance(messages, list):
            return None
        title = conv.get("name") or "Untitled"
        conv_id = conv.get("uuid") or ""
        when = _normalize_iso(conv.get("created_at"))

        human_msgs = 0
        assistant_msgs = 0
        first_human_text: str | None = None

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            sender = msg.get("sender")
            text = self._text(msg)
            if sender == "human" and text:
                human_msgs += 1
                if first_human_text is None:
                    first_human_text = text
            elif sender == "assistant" and text:
                assistant_msgs += 1

        total = human_msgs + assistant_msgs
        report.messages_seen += total
        if total == 0:
            return None

        meta = self._meta(
            imported_at,
            kind="conversation",
            external_id=str(conv_id),
            message_count=total,
        )
        facts = [
            ImportedFact(
                content=(
                    f"Claude conversation: '{title}' "
                    f"({total} messages, {human_msgs} from user)"
                ),
                fact_type=FactType.EVENT,
                tier=MemoryTier.EPISODIC,
                importance=0.35,
                when=when,
                metadata=meta,
            )
        ]
        if self.include_messages and first_human_text:
            snippet = first_human_text[:_FIRST_MESSAGE_MAX_CHARS]
            facts.append(
                ImportedFact(
                    content=(
                        f"In Claude conversation '{title}', "
                        f"the user opened with: {snippet}"
                    ),
                    fact_type=FactType.FACT,
                    tier=MemoryTier.ARCHIVAL,
                    importance=0.3,
                    when=when,
                    metadata=self._meta(
                        imported_at,
                        kind="first_message",
                        external_id=str(conv_id),
                    ),
                )
            )
        return facts

    @staticmethod
    def _text(msg: dict[str, Any]) -> str:
        text = msg.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        content = msg.get("content")
        if not isinstance(content, list):
            return ""
        chunks: list[str] = []
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].strip()
            ):
                chunks.append(block["text"].strip())
        return "\n".join(chunks)

    @staticmethod
    def _meta(imported_at: str, **extra: Any) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "source": PLATFORM,
            "scope": platform_scope(PLATFORM),
            "imported_at": imported_at,
        }
        meta.update(extra)
        return meta
