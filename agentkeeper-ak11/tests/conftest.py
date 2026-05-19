"""Test-suite-wide fixtures.

Forces the in-memory vector index by default so existing tests don't
accidentally write a sqlite-vec table to disk. Tests that explicitly
want to exercise the persistent backend set
`AGENTKEEPER_VECTOR_INDEX=sqlite_vec` via `monkeypatch.setenv`.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _default_in_memory_vector_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only set if a test didn't already.
    import os

    if "AGENTKEEPER_VECTOR_INDEX" not in os.environ:
        monkeypatch.setenv("AGENTKEEPER_VECTOR_INDEX", "in_memory")
