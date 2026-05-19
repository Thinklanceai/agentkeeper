"""FastMCP-based MCP server for AgentKeeper.

Lazy-imports `fastmcp` to keep the dependency optional. The package
remains usable without `fastmcp` for everything except this server.

Lifecycle:

    server = build_server(agent_id="aria", provider="anthropic")
    server.run()  # stdio by default

Or via the CLI::

    agentkeeper-mcp --agent-id aria --provider anthropic
    agentkeeper-mcp --agent-id aria --transport streamable-http --port 8000

The CLI entry-point is `agentkeeper.mcp:serve_cli`, registered as
``agentkeeper-mcp`` in pyproject.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from ..logging import get_logger
from .errors import MCPDependencyError

_log = get_logger(__name__)


def _import_fastmcp() -> Any:
    """Lazy import of FastMCP, raising a clear error if missing."""
    try:
        from fastmcp import FastMCP
        return FastMCP
    except ImportError as exc:
        raise MCPDependencyError(
            "fastmcp is required for the AgentKeeper MCP server. "
            "Install with: pip install 'agentkeeper[mcp]'"
        ) from exc


def _resolve_agent(agent_id: str, provider: str) -> Any:
    """Load or create the agent the server will expose."""
    import agentkeeper
    from agentkeeper.errors import AgentNotFoundError

    try:
        return agentkeeper.load(agent_id, provider=provider)
    except AgentNotFoundError:
        _log.info(
            "Agent %r not found in storage; creating a fresh one.", agent_id
        )
        agent = agentkeeper.create(agent_id=agent_id, provider=provider)
        agent.save()
        return agent


def build_server(
    agent_id: str,
    provider: str = "anthropic",
    server_name: str | None = None,
) -> Any:
    """Build a FastMCP server exposing the given agent.

    Args:
        agent_id: The id of the agent to expose. Loaded from storage if
            it exists, otherwise created fresh and saved.
        provider: Default LLM provider for compress/use_llm operations.
        server_name: Optional custom server name shown to MCP clients.
            Defaults to ``"agentkeeper:<agent_id>"``.

    Returns:
        A configured ``FastMCP`` instance. Call ``.run()`` to start it.

    Raises:
        MCPDependencyError: when ``fastmcp`` is not installed.
    """
    FastMCP = _import_fastmcp()
    name = server_name or f"agentkeeper:{agent_id}"
    mcp = FastMCP(name)

    # Single mutable reference; the agent is rebuilt on demand inside
    # tools that need a fresh CSO view, but most ops can reuse it.
    agent = _resolve_agent(agent_id, provider)

    # ----- Tools -----

    @mcp.tool
    def add_fact(
        content: str,
        importance: float = 0.7,
        critical: bool = False,
        tier: str | None = None,
        fact_type: str | None = None,
        ttl: str | None = None,
    ) -> dict[str, Any]:
        """Add a fact to the agent's memory.

        Args:
            content: The fact's textual content.
            importance: 0-1, higher is harder to evict (default 0.7).
            critical: If True, treats the fact as critical (importance>=0.95).
            tier: One of 'working', 'episodic', 'semantic', 'archival'.
                If omitted, inferred from content.
            fact_type: One of 'decision', 'preference', 'constraint',
                'relationship', 'task_state', 'transient', 'identity',
                'event', 'fact'. Drives decay rate.
            ttl: Optional TTL like '30d', '12h', '7d12h', 'P30D'.

        Returns:
            The added fact as a dict.
        """
        agent.remember(
            content,
            critical=critical if critical else None,
            tier=tier,
            importance=importance if not critical else None,
            ttl=ttl,
        )
        last = agent.last_fact()
        if fact_type is not None and last is not None:
            from ..cso.fact_types import FactType

            last.fact_type = FactType(fact_type)
        agent.save()
        return last.to_dict() if last else {}

    @mcp.tool
    def recall(query: str, top_k: int = 5, min_score: float = 0.0) -> list[dict[str, Any]]:
        """Semantic recall — return top_k facts most relevant to `query`.

        Args:
            query: Natural-language question or topic.
            top_k: Maximum number of results.
            min_score: Filter out facts below this similarity score.

        Returns:
            List of {fact, score} dicts, sorted by score descending.
        """
        hits = agent.recall(query, top_k=top_k, min_score=min_score)
        return [
            {"fact": fact.to_dict(), "score": round(score, 4)}
            for fact, score in hits
        ]

    @mcp.tool
    def set_identity(
        name: str = "",
        role: str = "",
        principles: list[str] | None = None,
        constraints: list[str] | None = None,
        merge: bool = False,
    ) -> dict[str, Any]:
        """Set or merge the agent's persistent identity.

        Identity is always injected into reconstructed context and
        survives every form of compression. Principles are protected.

        Args:
            name, role: agent identity strings (replace existing).
            principles: behavioural commitments.
            constraints: hard limits.
            merge: when True, principles/constraints are appended
                (deduplicated) instead of replacing.

        Returns:
            The full identity audit snapshot.
        """
        agent.set_identity(
            name=name,
            role=role,
            principles=principles or [],
            constraints=constraints or [],
            merge=merge,
        )
        agent.save()
        return agent.identity_audit()

    @mcp.tool
    def link(
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 1.0,
        ttl: str | None = None,
    ) -> dict[str, Any]:
        """Add a directed graph relation (subject -[predicate]-> object).

        Args:
            subject, predicate, object: the relation.
            confidence: 0-1 (default 1.0).
            ttl: optional TTL like '30d'.

        Returns:
            The created triple as a dict.
        """
        agent.link(subject, predicate, object, confidence=confidence, ttl=ttl)
        agent.save()
        return agent.triples[-1].to_dict()

    @mcp.tool
    def find_related(
        entity: str,
        max_hops: int = 2,
        direction: str = "both",
    ) -> dict[str, int]:
        """Return entities reachable from `entity` within `max_hops`.

        Args:
            entity: starting node.
            max_hops: BFS depth (default 2).
            direction: 'out', 'in', or 'both'.

        Returns:
            A dict mapping each reachable entity to its hop distance.
        """
        return agent.find_related(
            entity, max_hops=max_hops, direction=direction
        )

    @mcp.tool
    def compress(use_llm: bool = False) -> dict[str, Any]:
        """Run the compression pipeline (expiration → decay → consolidation
        → contradiction). Returns the report.

        Args:
            use_llm: When True, uses the agent's current provider to
                synthesise consolidated facts. Otherwise consolidation
                keeps the canonical fact as-is.

        Returns:
            The CompressionReport as a dict.
        """
        report = agent.compress(use_llm=use_llm)
        agent.save()
        return report.to_dict()

    @mcp.tool
    def health() -> dict[str, Any]:
        """Return the cognitive observability snapshot.

        Reports total/critical/protected/contradicted/stale facts,
        importance stats, tier and fact-type distributions, graph
        size, and identity status.
        """
        return agent.health()

    @mcp.tool
    def gdpr_export() -> dict[str, Any]:
        """Return a JSON-serialisable export of every fact, triple, and
        identity field this agent holds. Fulfils GDPR Article 20.
        """
        return agent.gdpr_export()

    @mcp.tool
    def purge_expired() -> dict[str, int]:
        """Remove facts and triples whose TTL has elapsed.

        Returns the count of purged items. Protected items are
        always preserved.
        """
        return {"purged": agent.purge_expired()}

    # ----- Resources -----

    @mcp.resource("agentkeeper://identity")
    def get_identity() -> dict[str, Any]:
        """The agent's current identity snapshot."""
        return agent.identity_audit()

    @mcp.resource("agentkeeper://facts/{fact_id}")
    def get_fact(fact_id: str) -> dict[str, Any] | None:
        """Lookup a single fact by id."""
        for f in agent.facts:
            if f.id == fact_id:
                return f.to_dict()
        return None

    return mcp


def serve_cli() -> int:
    """CLI entry-point for the ``agentkeeper-mcp`` console script."""
    parser = argparse.ArgumentParser(
        prog="agentkeeper-mcp",
        description=(
            "Run an MCP server exposing an AgentKeeper agent's cognitive "
            "layer. Compatible with Claude Desktop, Claude Code, Cursor, "
            "and any MCP-aware client."
        ),
    )
    parser.add_argument(
        "--agent-id",
        required=True,
        help="The agent id to expose (will be created if missing).",
    )
    parser.add_argument(
        "--provider",
        default="anthropic",
        help="Default LLM provider for compress(use_llm=True). "
        "Vendor-agnostic — supports anthropic / openai / gemini / "
        "ollama / mock.",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "streamable-http", "sse"],
        help="MCP transport. 'stdio' (default) for local Claude Desktop / "
        "Cursor / Codex. 'streamable-http' for remote.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="HTTP host (HTTP transports only)."
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="HTTP port (HTTP transports only)."
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Server name shown to MCP clients. Defaults to "
        "'agentkeeper:<agent-id>'.",
    )
    args = parser.parse_args()

    try:
        server = build_server(
            agent_id=args.agent_id,
            provider=args.provider,
            server_name=args.name,
        )
    except MCPDependencyError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    run_kwargs: dict[str, Any] = {"transport": args.transport}
    if args.transport in ("streamable-http", "sse"):
        run_kwargs["host"] = args.host
        run_kwargs["port"] = args.port
    server.run(**run_kwargs)
    return 0
