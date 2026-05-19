"""MCP-specific errors."""

from __future__ import annotations

from ..errors import ConfigurationError


class MCPDependencyError(ConfigurationError):
    """Raised when the `fastmcp` package is not installed.

    The MCP server is an optional integration. Install via::

        pip install 'agentkeeper[mcp]'
    """
