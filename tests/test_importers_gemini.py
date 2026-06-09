"""Tests for agentkeeper.importers (Gemini / Google Takeout)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

import agentkeeper
from agentkeeper import FactType, MemoryTier
from agentkeeper.importers import GeminiImporter, ImportError_


def _record(title: str, time: str, header: str = "Gemini Apps") -> dict[str, Any]:
    return {
        "header": header,
        "title": title,
        "time": time,
        "products": [header],
    }


def _fixture_records() -> list[dict[str, Any]]:
    return [
        _record("Prompted how do I center a div", "2026-02-01T10:00:00.123Z"),
        _record("Prompted plan a weekend in Vienna", "2026-02-02T18:30:00.000Z"),
        _record("Watched a video about cats", "2026-02-03T11:00:00Z", header="YouTube"),
        _record("", "2026-02-04T09:00:00Z"),
    ]


@pytest.fixture()
def takeout_zip(tmp_path: Path) -> Path:
    zip_path = tmp_path / "takeout.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "Takeout/My Activity/Gemini Apps/MyActivity.json",
            json.dumps(_fixture_records()),
        )
        zf.writestr(
            "Takeout/My Activity/Search/MyActivity.json",
            json.dumps([_record("Searched cats", "2026-01-01T00:00:00Z", header="Search")]),
        )
    return zip_path


def test_parse_takeout_zip(takeout_zip: Path) -> None:
    facts = GeminiImporter(takeout_zip).parse()
    assert len(facts) == 2
    assert all(f.fact_type is FactType.EVENT for f in facts)
    assert all(f.tier is MemoryTier.EPISODIC for f in facts)
    assert all(f.metadata["scope"] == "platform:gemini" for f in facts)
    assert any("center a div" in f.content for f in facts)
    assert not any("video about cats" in f.content for f in facts)


def test_prefix_stripped_and_timestamp_normalized(takeout_zip: Path) -> None:
    facts = GeminiImporter(takeout_zip).parse()
    vienna = next(f for f in facts if "Vienna" in f.content)
    assert vienna.content == "Gemini prompt: plan a weekend in Vienna"
    assert vienna.when == "2026-02-02T18:30:00Z"


def test_non_gemini_records_skipped(takeout_zip: Path) -> None:
    importer = GeminiImporter(takeout_zip)
    importer.parse()
    assert importer._last_report is not None
    assert importer._last_report.skipped == 2


def test_max_activities_cap(takeout_zip: Path) -> None:
    facts = GeminiImporter(takeout_zip, max_activities=1).parse()
    assert len(facts) == 1


def test_directory_nested_lookup(tmp_path: Path) -> None:
    nested = tmp_path / "Takeout" / "My Activity" / "Gemini Apps"
    nested.mkdir(parents=True)
    (nested / "MyActivity.json").write_text(json.dumps(_fixture_records()))
    facts = GeminiImporter(tmp_path).parse()
    assert len(facts) == 2


def test_direct_file(tmp_path: Path) -> None:
    direct = tmp_path / "MyActivity.json"
    direct.write_text(json.dumps(_fixture_records()))
    assert len(GeminiImporter(direct).parse()) == 2


def test_apply_writes_and_reports(takeout_zip: Path) -> None:
    agent = agentkeeper.create("test-gemini-importer-apply")
    report = GeminiImporter(takeout_zip).apply(agent)
    assert report.platform == "gemini"
    assert report.facts_created == 2
    reloaded = agentkeeper.load("test-gemini-importer-apply")
    stored = reloaded._cso.memory_facts
    scoped = [
        f for f in stored if f.metadata.get("scope") == "platform:gemini"
    ]
    assert len(scoped) == 2


def test_rejects_html_export(tmp_path: Path) -> None:
    zip_path = tmp_path / "takeout-html.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "Takeout/My Activity/Gemini Apps/MyActivity.html",
            "<html></html>",
        )
    with pytest.raises(ImportError_, match="JSON format"):
        GeminiImporter(zip_path).parse()


def test_rejects_html_direct(tmp_path: Path) -> None:
    html = tmp_path / "MyActivity.html"
    html.write_text("<html></html>")
    with pytest.raises(ImportError_, match="JSON format"):
        GeminiImporter(html).parse()


def test_rejects_missing_activity(tmp_path: Path) -> None:
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("Takeout/readme.txt", "nothing here")
    with pytest.raises(ImportError_, match="No MyActivity.json"):
        GeminiImporter(zip_path).parse()


def test_rejects_invalid_json(tmp_path: Path) -> None:
    direct = tmp_path / "MyActivity.json"
    direct.write_text("{not json")
    with pytest.raises(ImportError_, match="Invalid JSON"):
        GeminiImporter(direct).parse()
