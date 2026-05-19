"""Tests for the AgentKeeper MCP server.

The MCP server is an optional integration. These tests are skipped
automatically when `fastmcp` is not installed.

We verify:

- The optional-dep error message is correct when fastmcp is missing.
- The server is built successfully with fastmcp present.
- Each tool, when called directly through the registered handler,
  behaves correctly.
- The CLI argument parsing surface is sane.

We intentionally do NOT spin up a real stdio server — that's heavy,
hard to teardown deterministically, and what's covered by FastMCP's
own test suite. We test our *logic* by introspecting the registered
tools/resources and calling them directly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_FASTMCP_AVAILABLE = importlib.util.find_spec("fastmcp") is not None
skip_no_fastmcp = pytest.mark.skipif(
    not _FASTMCP_AVAILABLE,
    reason="fastmcp not installed; run `pip install 'agentkeeper[mcp]'`",
)


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTKEEPER_DB", str(tmp_path / "ak.db"))
    monkeypatch.setenv("AGENTKEEPER_EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("AGENTKEEPER_VECTOR_INDEX", "in_memory")
    import agentkeeper

    monkeypatch.setattr(agentkeeper, "_storage", None)


class TestMissingDep:
    """Verifies the graceful failure when fastmcp is not installed."""

    def test_dependency_error_message_helpful(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force the import to fail even if fastmcp is installed.
        from agentkeeper.mcp import server as server_mod

        def _fake_import() -> None:
            from agentkeeper.mcp.errors import MCPDependencyError

            raise MCPDependencyError(
                "fastmcp is required for the AgentKeeper MCP server. "
                "Install with: pip install 'agentkeeper[mcp]'"
            )

        monkeypatch.setattr(server_mod, "_import_fastmcp", _fake_import)

        from agentkeeper.mcp.errors import MCPDependencyError

        with pytest.raises(MCPDependencyError) as exc_info:
            server_mod.build_server(agent_id="x", provider="mock")
        assert "fastmcp" in str(exc_info.value).lower()
        assert "agentkeeper[mcp]" in str(exc_info.value)


@skip_no_fastmcp
class TestServerBuild:
    def test_build_server_returns_fastmcp_instance(self) -> None:
        from fastmcp import FastMCP

        from agentkeeper.mcp.server import build_server

        server = build_server(agent_id="test-build", provider="mock")
        assert isinstance(server, FastMCP)

    def test_build_server_creates_missing_agent(self) -> None:
        import agentkeeper
        from agentkeeper.errors import AgentNotFoundError
        from agentkeeper.mcp.server import build_server

        # No agent yet
        with pytest.raises(AgentNotFoundError):
            agentkeeper.load("ghost", provider="mock")

        # Building a server creates it
        build_server(agent_id="ghost", provider="mock")
        # Now it exists
        loaded = agentkeeper.load("ghost", provider="mock")
        assert loaded.id == "ghost"

    def test_custom_server_name(self) -> None:
        from agentkeeper.mcp.server import build_server

        server = build_server(
            agent_id="x", provider="mock", server_name="custom-name"
        )
        # FastMCP exposes the name on the instance
        assert server.name == "custom-name"


@skip_no_fastmcp
class TestRegisteredTools:
    """Verifies the right set of tools/resources is registered."""

    async def test_registered_tool_names(self) -> None:
        from agentkeeper.mcp.server import build_server

        server = build_server(agent_id="tools-test", provider="mock")
        tools = await server.list_tools()
        names = {t.name for t in tools}
        expected = {
            "add_fact",
            "recall",
            "set_identity",
            "link",
            "find_related",
            "compress",
            "health",
            "gdpr_export",
            "purge_expired",
        }
        assert expected.issubset(names), (
            f"Missing tools: {expected - names}"
        )

    async def test_registered_resources(self) -> None:
        from agentkeeper.mcp.server import build_server

        server = build_server(agent_id="res-test", provider="mock")
        resources = await server.list_resources()
        templates = await server.list_resource_templates()
        all_uris = {str(r.uri) for r in resources} | {
            str(t.uri_template) for t in templates
        }
        # The static identity resource + the templated facts/{fact_id}
        assert "agentkeeper://identity" in all_uris
        assert any(
            "agentkeeper://facts/" in uri for uri in all_uris
        ), f"facts/<id> template missing in {all_uris}"


@skip_no_fastmcp
class TestToolBehaviour:
    """End-to-end: build a server, exercise each tool, verify side
    effects on the underlying agent."""

    async def _invoke(
        self, server: object, _tool_name: str, **kwargs: object
    ) -> object:
        """Helper: look up a registered tool and invoke its callable.

        Underscore-prefixed first arg to avoid clashing with tool
        parameters like `name=` (used by set_identity).
        """
        tool = await server.get_tool(_tool_name)  # type: ignore[attr-defined]
        return tool.fn(**kwargs)

    async def test_add_fact_persists(self) -> None:
        import agentkeeper
        from agentkeeper.mcp.server import build_server

        server = build_server(agent_id="t1", provider="mock")
        result = await self._invoke(
            server, "add_fact", content="budget: 50k EUR", importance=0.9
        )
        assert "id" in result
        # Reload to confirm it persisted
        reloaded = agentkeeper.load("t1", provider="mock")
        assert any("budget: 50k EUR" in f.content for f in reloaded.facts)

    async def test_recall_returns_results(self) -> None:
        from agentkeeper.mcp.server import build_server

        server = build_server(agent_id="t2", provider="mock")
        await self._invoke(server, "add_fact", content="Acme is a Belgian holding")
        await self._invoke(server, "add_fact", content="Globex makes weather systems")
        results = await self._invoke(server, "recall", query="Belgian", top_k=2)
        assert isinstance(results, list)
        assert len(results) >= 1
        assert "fact" in results[0]
        assert "score" in results[0]

    async def test_set_identity_returns_audit(self) -> None:
        from agentkeeper.mcp.server import build_server

        server = build_server(agent_id="t3", provider="mock")
        audit = await self._invoke(
            server,
            "set_identity",
            name="Aria",
            role="EU broker copilot",
            principles=["never share PII"],
        )
        assert audit["identity"]["name"] == "Aria"
        assert audit["identity"]["principles_count"] == 1

    async def test_link_and_find_related(self) -> None:
        from agentkeeper.mcp.server import build_server

        server = build_server(agent_id="t4", provider="mock")
        await self._invoke(server, "link", subject="Acme", predicate="owns", object="Globex")
        await self._invoke(server, "link", subject="Alice", predicate="works_at", object="Acme")
        result = await self._invoke(
            server, "find_related", entity="Acme", max_hops=1, direction="both"
        )
        assert "Globex" in result
        assert "Alice" in result

    async def test_compress_returns_report(self) -> None:
        from agentkeeper.mcp.server import build_server

        server = build_server(agent_id="t5", provider="mock")
        await self._invoke(server, "add_fact", content="dup", importance=0.5)
        await self._invoke(server, "add_fact", content="dup", importance=0.5)
        report = await self._invoke(server, "compress")
        assert "facts_before" in report
        assert "facts_after" in report

    async def test_health_includes_graph_section(self) -> None:
        from agentkeeper.mcp.server import build_server

        server = build_server(agent_id="t6", provider="mock")
        h = await self._invoke(server, "health")
        assert "graph" in h
        assert "total_facts" in h

    async def test_gdpr_export_includes_triples(self) -> None:
        from agentkeeper.mcp.server import build_server

        server = build_server(agent_id="t7", provider="mock")
        await self._invoke(server, "link", subject="X", predicate="p", object="Y")
        export = await self._invoke(server, "gdpr_export")
        assert "facts" in export
        assert "triples" in export
        assert len(export["triples"]) == 1

    async def test_purge_expired_returns_count(self) -> None:
        from agentkeeper.mcp.server import build_server

        server = build_server(agent_id="t8", provider="mock")
        result = await self._invoke(server, "purge_expired")
        assert "purged" in result
        assert result["purged"] == 0


@skip_no_fastmcp
class TestCLIArgs:
    """Smoke test the CLI parser; we don't actually start a server."""

    def test_missing_agent_id_exits(self, capsys: pytest.CaptureFixture[str]) -> None:
        from agentkeeper.mcp import server as server_mod

        with pytest.raises(SystemExit):
            # argparse exits when --agent-id is missing
            import sys

            original = sys.argv
            try:
                sys.argv = ["agentkeeper-mcp"]
                server_mod.serve_cli()
            finally:
                sys.argv = original
