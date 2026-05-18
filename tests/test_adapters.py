"""Tests for the MockAdapter and base adapter contract."""

from __future__ import annotations

import pytest

from agentkeeper.adapters.base import BaseAdapter, MockAdapter


class TestMockAdapter:
    def test_returns_response_containing_system_prompt(self) -> None:
        adapter = MockAdapter()
        response = adapter.query("you remember X", "what is X?")
        assert "you remember X" in response

    def test_records_last_system_prompt(self) -> None:
        adapter = MockAdapter()
        adapter.query("first prompt", "msg1")
        assert adapter._last_system_prompt == "first prompt"
        adapter.query("second prompt", "msg2")
        assert adapter._last_system_prompt == "second prompt"

    def test_is_instance_of_base_adapter(self) -> None:
        assert isinstance(MockAdapter(), BaseAdapter)


class TestBaseAdapterContract:
    def test_base_adapter_cannot_be_instantiated_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseAdapter()  # type: ignore[abstract]
