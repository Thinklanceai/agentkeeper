"""Native MCP server for AgentKeeper.

Exposes AgentKeeper's cognitive layer as an MCP (Model Context Protocol)
server. Any MCP-compatible client — Claude Desktop, Claude Code,
Cursor, VS Code Copilot, ChatGPT desktop, JetBrains AI — can now use
AgentKeeper as its persistent memory and identity backbone.

This is the second of the two AK-15 differentiating features. The first
was the graph memory layer (AK-14). Together they answer two distinct
Twitter signals:

- "does this persist memory across different agent frameworks?"
  → Answered by graph memory + integrations (LangChain, CrewAI) AND
    now by MCP (Claude Code, Cursor, ChatGPT, ...).
- "infrastructure not memory"
  → MCP exposure positions AgentKeeper as an *infrastructure layer*,
    callable from any MCP-aware host, not a Python library you wire
    into one specific framework.

Design choices:

- **Built on FastMCP** (the official Python MCP framework, ~70% market
  share as of mid-2026). We do not reinvent the wire protocol.
- **Optional dependency** via the `[mcp]` extra:
  ``pip install agentkeeper[mcp]``. Without it, importing this module
  raises a clear ImportError pointing the user to the install command.
- **One agent per server instance** — keeps the boundary clean. To
  host many agents, run many servers (or use the `--agent-id` arg to
  swap which agent is active).
- **stdio transport by default** for Claude Desktop / Cursor / Codex
  compatibility. Streamable HTTP available via the CLI flag for
  remote deployments.
- **Vendor-agnostic** — the MCP server uses the agent's *current
  provider* for any LLM-backed operation (compression with `use_llm`).
  We never hardcode Anthropic or OpenAI here.

Public surface:

- ``build_server(agent_id, provider)`` — returns a configured
  ``FastMCP`` instance, ready to ``.run()``.
- ``serve_cli()`` — entry-point used by the ``agentkeeper-mcp``
  console script. Parses argv, builds the server, runs it.

Exposed tools (8):

- ``add_fact``       — add a fact with optional ttl, importance, type
- ``recall``         — semantic recall (top-k facts)
- ``set_identity``   — set name / role / principles / constraints
- ``link``           — add a (subject, predicate, object) triple
- ``find_related``   — graph traversal (BFS)
- ``compress``       — run the compression pipeline
- ``health``         — cognitive observability snapshot
- ``gdpr_export``    — full export (Article 20 RGPD)
- ``purge_expired``  — remove facts/triples whose TTL has elapsed

Exposed resources (2):

- ``agentkeeper://identity``         — current identity (JSON)
- ``agentkeeper://facts/{fact_id}``  — single fact by id
"""

from .errors import MCPDependencyError
from .server import build_server, serve_cli

__all__ = ["MCPDependencyError", "build_server", "serve_cli"]
