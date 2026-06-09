"""Tests for agentkeeper.importers (Claude)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

import agentkeeper
from agentkeeper import FactType, MemoryTier
from agentkeeper.importers import ClaudeImporter, ImportError_


def _conversation(
    uuid: str,
    name: str,
    created_at: str,
    human_texts: list[str],
    assistant_texts: list[str],
    use_content_blocks: bool = False,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for sender, texts in (("human", human_texts), ("assistant", assistant_texts)):
        for text in texts:
            msg: dict[str, Any] = {
                "uuid": f"{uuid}-{sender}-{len(messages)}",
                "sender": sender,
                "created_at": created_at,
            }
            if use_content_blocks:
                msg["text"] = ""
                msg["content"] = [{"type": "text", "text": text}]
            else:
                msg["text"] = text
            messages.append(msg)
    return {
        "uuid": uuid,
        "name": name,
        "created_at": created_at,
        "updated_at": created_at,
        "chat_messages": messages,
    }


def _fixture_conversations() -> list[dict[str, Any]]:
    return [
        _conversation(
            "c-1",
            "AEGIS roadmap",
            "2026-02-10T09:00:00.000000Z",
            ["Plan the compliance bridge feature", "And the diff view?"],
            ["Here is the plan.", "Diff view next sprint."],
        ),
        _conversation(
            "c-2",
            "Globe rendering",
            "2026-03-05T14:30:00.000000Z",
            ["Why does the globe flicker?"],
            ["Z-fighting on overlapping meshes."],
            use_content_blocks=True,
        ),
        {"uuid": "c-empty", "name": "Empty", "chat_messages": []},
    ]


@pytest.fixture()
def export_zip(tmp_path: Path) -> Path:
    users = [
        {
            "uuid": "u-1",
            "full_name": "Tom Anciaux",
            "email_address": "tom@example.com",
        }
    ]
    projects = [
        {
            "uuid": "p-1",
            "name": "Circuit Live",
            "description": "Touring intelligence",
            "created_at": "2026-01-01T00:00:00.000000Z",
        }
    ]
    zip_path = tmp_path / "claude-export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("conversations.json", json.dumps(_fixture_conversations()))
        zf.writestr("users.json", json.dumps(users))
        zf.writestr("projects.json", json.dumps(projects))
    return zip_path


@pytest.fixture()
def export_jsonl(tmp_path: Path) -> Path:
    lines = "\n".join(json.dumps(c) for c in _fixture_conversations())
    zip_path = tmp_path / "claude-export-jsonl.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("conversations.jsonl", lines)
    return zip_path


def test_parse_zip_counts(export_zip: Path) -> None:
    facts = ClaudeImporter(export_zip).parse()
    events = [f for f in facts if f.fact_type is FactType.EVENT]
    assert len(events) == 2
    assert all(f.tier is MemoryTier.EPISODIC for f in events)
    assert all(f.metadata["scope"] == "platform:claude" for f in facts)
    assert all(f.metadata["source"] == "claude" for f in facts)


def test_timestamps_normalized(export_zip: Path) -> None:
    facts = ClaudeImporter(export_zip).parse()
    events = [f for f in facts if f.fact_type is FactType.EVENT]
    assert {f.when for f in events} == {
        "2026-02-10T09:00:00Z",
        "2026-03-05T14:30:00Z",
    }


def test_content_blocks_fallback(export_zip: Path) -> None:
    facts = ClaudeImporter(export_zip, include_messages=True).parse()
    archival = [f for f in facts if f.tier is MemoryTier.ARCHIVAL]
    assert any("globe flicker" in f.content for f in archival)


def test_jsonl_layout(export_jsonl: Path) -> None:
    facts = ClaudeImporter(export_jsonl).parse()
    events = [f for f in facts if f.fact_type is FactType.EVENT]
    assert len(events) == 2


def test_projects_imported(export_zip: Path) -> None:
    facts = ClaudeImporter(export_zip).parse()
    project_facts = [
        f for f in facts if f.metadata.get("kind") == "project"
    ]
    assert len(project_facts) == 1
    assert "Circuit Live" in project_facts[0].content
    assert "Touring intelligence" in project_facts[0].content


def test_pii_excluded_by_default(export_zip: Path) -> None:
    facts = ClaudeImporter(export_zip).parse()
    joined = " ".join(f.content for f in facts)
    assert "tom@example.com" not in joined
    assert "Tom Anciaux" not in joined
    facts_pii = ClaudeImporter(export_zip, include_pii=True).parse()
    identity = [f for f in facts_pii if f.fact_type is FactType.IDENTITY]
    assert any("tom@example.com" in f.content for f in identity)
    assert any("Tom Anciaux" in f.content for f in identity)


def test_apply_writes_and_reports(export_zip: Path) -> None:
    agent = agentkeeper.create("test-claude-importer-apply")
    report = ClaudeImporter(export_zip).apply(agent)
    assert report.platform == "claude"
    assert report.conversations_parsed == 2
    assert report.skipped == 1
    assert report.messages_seen == 6
    reloaded = agentkeeper.load("test-claude-importer-apply")
    stored = reloaded._cso.memory_facts
    scoped = [
        f for f in stored if f.metadata.get("scope") == "platform:claude"
    ]
    assert len(scoped) == report.facts_created


def test_max_conversations_cap(export_zip: Path) -> None:
    facts = ClaudeImporter(export_zip, max_conversations=1).parse()
    events = [f for f in facts if f.fact_type is FactType.EVENT]
    assert len(events) == 1


def test_directory_and_direct_file(tmp_path: Path, export_zip: Path) -> None:
    out = tmp_path / "extracted"
    out.mkdir()
    with zipfile.ZipFile(export_zip) as zf:
        for name in ("conversations.json", "users.json", "projects.json"):
            (out / name).write_bytes(zf.read(name))
    assert len(ClaudeImporter(out).parse()) > 0
    assert len(ClaudeImporter(out / "conversations.json").parse()) > 0


def test_rejects_missing_conversations(tmp_path: Path) -> None:
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("users.json", "[]")
    with pytest.raises(ImportError_, match="No conversations"):
        ClaudeImporter(zip_path).parse()


def test_rejects_invalid_jsonl_line(tmp_path: Path) -> None:
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("conversations.jsonl", '{"uuid": "ok"}\n{broken')
    with pytest.raises(ImportError_, match="line 2"):
        ClaudeImporter(zip_path).parse()


def test_cross_platform_scoping(export_zip: Path) -> None:
    agent = agentkeeper.create("test-cross-scope")
    ClaudeImporter(export_zip).apply(agent)
    stored = agent._cso.memory_facts
    claude_facts = [
        f for f in stored if f.metadata.get("scope") == "platform:claude"
    ]
    chatgpt_facts = [
        f for f in stored if f.metadata.get("scope") == "platform:chatgpt"
    ]
    assert len(claude_facts) > 0
    assert len(chatgpt_facts) == 0
