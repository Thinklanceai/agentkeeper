"""Tests for agentkeeper.importers (ChatGPT)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

import agentkeeper
from agentkeeper import FactType, MemoryTier
from agentkeeper.importers import (
    ChatGPTImporter,
    ImportError_,
    parse_memory_summary,
)


def _conversation(
    conv_id: str,
    title: str,
    create_time: float,
    user_texts: list[str],
    assistant_texts: list[str],
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    idx = 0
    for role, texts in (("user", user_texts), ("assistant", assistant_texts)):
        for text in texts:
            idx += 1
            mapping[f"node-{idx}"] = {
                "id": f"node-{idx}",
                "message": {
                    "author": {"role": role},
                    "content": {"content_type": "text", "parts": [text]},
                    "create_time": create_time + idx,
                },
                "parent": None,
                "children": [],
            }
    return {
        "id": conv_id,
        "title": title,
        "create_time": create_time,
        "update_time": create_time + 100,
        "mapping": mapping,
    }


@pytest.fixture()
def export_zip(tmp_path: Path) -> Path:
    conversations = [
        _conversation(
            "conv-1",
            "Trip to Lisbon",
            1768000000.0,
            ["Help me plan a trip to Lisbon", "What about food?"],
            ["Sure, here is a plan.", "Try the pastéis."],
        ),
        _conversation(
            "conv-2",
            "Python debugging",
            1769000000.0,
            ["Why does this raise KeyError?"],
            ["Because the key is missing."],
        ),
        {"id": "conv-empty", "title": "Empty", "mapping": {}},
    ]
    user = {"id": "u-1", "email": "tom@example.com", "chatgpt_plus_user": True}
    zip_path = tmp_path / "chatgpt-export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("conversations.json", json.dumps(conversations))
        zf.writestr("user.json", json.dumps(user))
    return zip_path


def test_parse_zip_counts(export_zip: Path) -> None:
    importer = ChatGPTImporter(export_zip)
    facts = importer.parse()
    events = [f for f in facts if f.fact_type is FactType.EVENT]
    assert len(events) == 2
    assert all(f.tier is MemoryTier.EPISODIC for f in events)
    assert all(f.metadata["scope"] == "platform:chatgpt" for f in facts)
    assert all(f.metadata["source"] == "chatgpt" for f in facts)


def test_pii_excluded_by_default(export_zip: Path) -> None:
    facts = ChatGPTImporter(export_zip).parse()
    assert not any("tom@example.com" in f.content for f in facts)
    facts_pii = ChatGPTImporter(export_zip, include_pii=True).parse()
    identity = [f for f in facts_pii if f.fact_type is FactType.IDENTITY]
    assert any("tom@example.com" in f.content for f in identity)


def test_include_messages_archival(export_zip: Path) -> None:
    facts = ChatGPTImporter(export_zip, include_messages=True).parse()
    archival = [f for f in facts if f.tier is MemoryTier.ARCHIVAL]
    assert len(archival) == 2
    assert any("Lisbon" in f.content for f in archival)


def test_apply_writes_and_reports(export_zip: Path) -> None:
    agent = agentkeeper.create("test-importer-apply")
    report = ChatGPTImporter(export_zip).apply(agent)
    assert report.platform == "chatgpt"
    assert report.conversations_parsed == 2
    assert report.skipped == 1
    assert report.messages_seen == 6
    assert report.facts_created == len(ChatGPTImporter(export_zip).parse())
    reloaded = agentkeeper.load("test-importer-apply")
    stored = reloaded._cso.memory_facts
    scoped = [
        f for f in stored if f.metadata.get("scope") == "platform:chatgpt"
    ]
    assert len(scoped) == report.facts_created


def test_max_conversations_cap(export_zip: Path) -> None:
    facts = ChatGPTImporter(export_zip, max_conversations=1).parse()
    events = [f for f in facts if f.fact_type is FactType.EVENT]
    assert len(events) == 1


def test_directory_and_direct_json(tmp_path: Path, export_zip: Path) -> None:
    out = tmp_path / "extracted"
    out.mkdir()
    with zipfile.ZipFile(export_zip) as zf:
        for name in ("conversations.json", "user.json"):
            (out / name).write_bytes(zf.read(name))
    assert len(ChatGPTImporter(out).parse()) > 0
    assert len(ChatGPTImporter(out / "conversations.json").parse()) > 0


def test_rejects_traversal_member(tmp_path: Path) -> None:
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("conversations.json", "[]")
        zf.writestr("../../etc/evil.txt", "x")
    with pytest.raises(ImportError_, match="Unsafe member path"):
        ChatGPTImporter(zip_path).parse()


def test_rejects_missing_conversations(tmp_path: Path) -> None:
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("user.json", "{}")
    with pytest.raises(ImportError_, match="conversations.json not found"):
        ChatGPTImporter(zip_path).parse()


def test_rejects_invalid_json(tmp_path: Path) -> None:
    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("conversations.json", "{not json")
    with pytest.raises(ImportError_, match="Invalid JSON"):
        ChatGPTImporter(zip_path).parse()


def test_memory_summary_classification() -> None:
    text = """
    Here is what I remember about you:
    - My name is Tom and I work as a developer in Brussels
    - Prefers short, direct answers without fluff
    - Working on a touring platform for electronic music
    """
    facts = parse_memory_summary(text)
    types = {f.fact_type for f in facts}
    assert FactType.IDENTITY in types
    assert FactType.PREFERENCE in types
    assert FactType.FACT in types
    assert all(f.metadata["kind"] == "memory_summary" for f in facts)
    assert not any(f.content.endswith(":") for f in facts)
