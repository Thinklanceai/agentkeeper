# AgentKeeper MCP — example configurations

This folder contains drop-in MCP client configurations for popular AI hosts.
The `agentkeeper-mcp` command becomes available once you install:

```bash
pip install 'agentkeeper[mcp]'
```

## Claude Desktop

Paste the contents of `claude_desktop_config.json` into:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

Restart Claude Desktop. The 9 AgentKeeper tools (`add_fact`, `recall`,
`set_identity`, `link`, `find_related`, `compress`, `health`,
`gdpr_export`, `purge_expired`) appear in the tool panel.

## Cursor

Paste the contents of `cursor_mcp_json.json` into:

- **Project-level**: `<project_root>/.cursor/mcp.json`
- **Global**: `~/.cursor/mcp.json`

Cursor picks it up automatically on next chat.

## Claude Code

Add this to your `claude_code_config.json` (location varies by OS,
see [Anthropic docs](https://docs.claude.com)):

```json
{
  "mcpServers": {
    "agentkeeper": {
      "command": "agentkeeper-mcp",
      "args": ["--agent-id", "claude-code", "--provider", "anthropic"]
    }
  }
}
```

## Customising

- **Agent id**: pick one per host context (`aria`, `cursor-default`,
  `claude-code`, …). Each gets its own cognitive state.
- **Provider**: any of `anthropic`, `openai`, `gemini`, `ollama`,
  `mock`. Only matters when a tool calls `compress(use_llm=True)`.
- **Storage**: set `AGENTKEEPER_DB` to control where the SQLite file
  lives. Default is the current working directory.
- **Embedding provider**: set `AGENTKEEPER_EMBEDDING_PROVIDER` to
  `sentence_transformers` (best, local, free), `openai` (cloud), or
  `mock` (deterministic test fallback).

## Remote deployment

For team-shared AgentKeeper instances, switch the transport:

```bash
agentkeeper-mcp \
  --agent-id team-shared \
  --provider anthropic \
  --transport streamable-http \
  --host 0.0.0.0 \
  --port 8000
```

Then point your MCP client at `http://your-host:8000/mcp`.
