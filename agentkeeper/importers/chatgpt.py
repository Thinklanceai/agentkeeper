"""Importer for ChatGPT data exports.

Supports the official ChatGPT export ZIP (Settings → Data controls →
Export data), a directory containing the extracted files, or a direct
path to ``conversations.json``. Also supports the prompt-generated
"memory summary" text used by the cross-platform migration flow.

Everything is parsed in memory — the archive is never extracted to
disk. Personally identifiable account data (email, phone) is excluded
unless ``include_pii=True`` is passed explicitly.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any

from ..cso.fact_types import FactType
from ..cso.tiers import MemoryTier
from .base import (
    BaseImporter,
    ImportedFact,
    ImportError_,
    ImportReport,
    epoch_to_iso,
    platform_scope,
    utcnow_iso,
    validate_zip,
)

PLATFORM = "chatgpt"

_PREFERENCE_RE = re.compile(
    r"\b(prefer|prefers|likes?|enjoys?|favou?rite|style|tone|"
    r"préfère|aime|favori)\b",
    re.IGNORECASE,
)
_IDENTITY_RE = re.compile(
    r"\b(name is|named|works (as|at|in)|lives in|based in|is a|is an|"
    r"years old|profession|s'appelle|travaille|habite)\b",
    re.IGNORECASE,
)

_FIRST_MESSAGE_MAX_CHARS = 500


class ChatGPTImporter(BaseImporter):
    """Parse a ChatGPT export into normalized facts.

    Args:
        path: Export ZIP, extracted directory, or ``conversations.json``.
        include_messages: When True, the first user message of each
            conversation is kept (truncated) as an archival fact to give
            semantic recall something to bite on. Default False.
        include_pii: When True, account email is imported as an identity
            fact. Default False — privacy first.
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
        conversations, user = self._load_raw()
        report = self._make_report()
        facts: list[ImportedFact] = []
        imported_at = utcnow_iso()

        facts.extend(self._user_facts(user, imported_at))

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
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.path.exists():
            raise ImportError_(f"Path does not exist: {self.path}")

        if self.path.is_dir():
            conv_bytes = self._read_file(self.path / "conversations.json")
            user_bytes = self._read_optional(self.path / "user.json")
        elif self.path.suffix.lower() == ".zip":
            conv_bytes, user_bytes = self._read_zip(self.path)
        elif self.path.name == "conversations.json":
            conv_bytes = self._read_file(self.path)
            user_bytes = None
        else:
            raise ImportError_(
                f"Unsupported input: {self.path} "
                "(expected .zip, directory, or conversations.json)"
            )

        conversations = self._decode_json(conv_bytes, "conversations.json")
        if not isinstance(conversations, list):
            raise ImportError_("conversations.json is not a JSON array.")
        user: dict[str, Any] = {}
        if user_bytes is not None:
            decoded = self._decode_json(user_bytes, "user.json")
            if isinstance(decoded, dict):
                user = decoded
        return conversations, user

    @staticmethod
    def _read_zip(path: Path) -> tuple[bytes, bytes | None]:
        try:
            with zipfile.ZipFile(path) as zf:
                validate_zip(zf)
                names = {Path(n).name: n for n in zf.namelist()}
                if "conversations.json" not in names:
                    raise ImportError_(
                        "conversations.json not found in export archive."
                    )
                conv = zf.read(names["conversations.json"])
                user = (
                    zf.read(names["user.json"])
                    if "user.json" in names
                    else None
                )
                return conv, user
        except zipfile.BadZipFile as exc:
            raise ImportError_(f"Not a valid ZIP archive: {path}") from exc

    @staticmethod
    def _read_file(path: Path) -> bytes:
        if not path.exists():
            raise ImportError_(f"File not found: {path}")
        return path.read_bytes()

    @staticmethod
    def _read_optional(path: Path) -> bytes | None:
        return path.read_bytes() if path.exists() else None

    @staticmethod
    def _decode_json(raw: bytes, label: str) -> Any:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ImportError_(f"Invalid JSON in {label}: {exc}") from exc

    def _user_facts(
        self, user: dict[str, Any], imported_at: str
    ) -> list[ImportedFact]:
        facts: list[ImportedFact] = []
        meta = self._meta(imported_at, kind="account")
        email = user.get("email")
        if self.include_pii and isinstance(email, str) and email:
            facts.append(
                ImportedFact(
                    content=f"ChatGPT account email: {email}",
                    fact_type=FactType.IDENTITY,
                    importance=0.85,
                    metadata=meta,
                )
            )
        plus = user.get("chatgpt_plus_user")
        if isinstance(plus, bool):
            tier_name = "Plus" if plus else "Free"
            facts.append(
                ImportedFact(
                    content=f"User had a ChatGPT {tier_name} account.",
                    fact_type=FactType.IDENTITY,
                    importance=0.4,
                    metadata=meta,
                )
            )
        return facts

    def _parse_conversation(
        self,
        conv: dict[str, Any],
        imported_at: str,
        report: ImportReport,
    ) -> list[ImportedFact] | None:
        mapping = conv.get("mapping")
        if not isinstance(mapping, dict):
            return None
        title = conv.get("title") or "Untitled"
        conv_id = conv.get("id") or conv.get("conversation_id") or ""
        when = epoch_to_iso(conv.get("create_time"))

        user_msgs = 0
        assistant_msgs = 0
        first_user_text: str | None = None

        for node in mapping.values():
            if not isinstance(node, dict):
                continue
            msg = node.get("message")
            if not isinstance(msg, dict):
                continue
            role = self._role(msg)
            text = self._text(msg)
            if role == "user" and text:
                user_msgs += 1
                if first_user_text is None:
                    first_user_text = text
            elif role == "assistant" and text:
                assistant_msgs += 1

        total = user_msgs + assistant_msgs
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
                    f"ChatGPT conversation: '{title}' "
                    f"({total} messages, {user_msgs} from user)"
                ),
                fact_type=FactType.EVENT,
                tier=MemoryTier.EPISODIC,
                importance=0.35,
                when=when,
                metadata=meta,
            )
        ]
        if self.include_messages and first_user_text:
            snippet = first_user_text[:_FIRST_MESSAGE_MAX_CHARS]
            facts.append(
                ImportedFact(
                    content=(
                        f"In ChatGPT conversation '{title}', "
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
    def _role(msg: dict[str, Any]) -> str:
        author = msg.get("author")
        if isinstance(author, dict):
            role = author.get("role")
            if isinstance(role, str):
                return role
        return ""

    @staticmethod
    def _text(msg: dict[str, Any]) -> str:
        content = msg.get("content")
        if not isinstance(content, dict):
            return ""
        parts = content.get("parts")
        if not isinstance(parts, list):
            return ""
        chunks = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
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


def parse_memory_summary(
    text: str, platform: str = PLATFORM
) -> list[ImportedFact]:
    """Parse a prompt-generated memory summary into normalized facts.

    This is the text a user obtains by asking their assistant to
    summarize everything it remembers about them (the flow popularized
    by the 2026 cross-platform migration tools). Each meaningful line
    becomes one fact, classified deterministically:

    - identity patterns (name, job, location) → ``IDENTITY``
    - preference patterns (likes, style, tone) → ``PREFERENCE``
    - everything else → ``FACT``
    """
    imported_at = utcnow_iso()
    meta_base: dict[str, Any] = {
        "source": platform,
        "scope": platform_scope(platform),
        "imported_at": imported_at,
        "kind": "memory_summary",
    }
    facts: list[ImportedFact] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-*•·").strip()
        if len(line) < 8 or line.endswith(":"):
            continue
        if _IDENTITY_RE.search(line):
            fact_type, importance = FactType.IDENTITY, 0.85
        elif _PREFERENCE_RE.search(line):
            fact_type, importance = FactType.PREFERENCE, 0.6
        else:
            fact_type, importance = FactType.FACT, 0.55
        facts.append(
            ImportedFact(
                content=line,
                fact_type=fact_type,
                importance=importance,
                metadata=dict(meta_base),
            )
        )
    return facts
