# AgentKeeper

**Cognitive continuity infrastructure for long-lived AI agents.**

Your agent survives model switches, crashes, context-window limits, and restarts — with the same identity, memory, and priorities it had before.

[![PyPI version](https://img.shields.io/pypi/v/agentkeeper-ai.svg)](https://pypi.org/project/agentkeeper-ai/)
[![Python versions](https://img.shields.io/pypi/pyversions/agentkeeper-ai.svg)](https://pypi.org/project/agentkeeper-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/Thinklanceai/agentkeeper/actions/workflows/ci.yml/badge.svg)](https://github.com/Thinklanceai/agentkeeper/actions/workflows/ci.yml)
[![Built by ThinkLanceAI](https://img.shields.io/badge/built%20by-ThinkLanceAI-4f8cff)](https://thinklanceai.com)

---

## Why this exists

Agents don't fail because they forget facts. They fail because they lose **cognitive continuity** — their state, priorities, and identity drift the moment the model changes, the context window fills, or the process restarts.

AgentKeeper treats this as a systems problem, not a memory problem.

---

## Install

```bash
pip install agentkeeper-ai
```

Zero required dependencies. No external infrastructure. Storage defaults to local SQLite.

```bash
pip install 'agentkeeper-ai[anthropic]'   # Claude
pip install 'agentkeeper-ai[openai]'      # GPT + OpenAI embeddings
pip install 'agentkeeper-ai[gemini]'      # Gemini
pip install 'agentkeeper-ai[semantic]'    # Local embeddings (sentence-transformers)
pip install 'agentkeeper-ai[mcp]'         # MCP server (Claude Desktop, Cursor, Codex)
pip install 'agentkeeper-ai[encrypted]'   # Encrypted storage at rest
pip install 'agentkeeper-ai[all]'         # Everything
```

---

## Five things AgentKeeper does that nothing else does

### 1. Identity that survives everything

Principles and constraints are **protected** — exempt from every compression pass, injected into every reconstructed context, regardless of token budget. They survive decay, consolidation, contradiction arbitration, model switches, and process restarts.

```python
import agentkeeper

agent = agentkeeper.create(agent_id="aria", provider="anthropic")
agent.set_identity(
    name="Aria",
    role="EU insurance broker copilot",
    principles=["never share PII without explicit consent"],
    constraints=["EU data residency only"],
)
agent.principle("always confirm budget changes in writing")
agent.fact("client: Acme Corporation", importance=0.95)
agent.event("contract signed", when="2026-05-15")
agent.save()

# 100 compression cycles later — identity intact.
```

### 2. Same memory, switch models

The cognitive state is reconstructed in the format each model expects. XML for Claude, labelled sections for GPT-4, narrative prose for Gemini, terse tokens for Ollama. One agent, four runtimes, zero rewrites.

```python
agent = agentkeeper.load("aria", provider="anthropic")
response = agent.ask("What do we know about Acme?")

agent.switch_provider("openai").save()
response = agent.ask("Same question, different model.")
# Memory and identity are intact. Format has changed. Nothing broke.
```

### 3. TTL for GDPR — memory that expires itself

Facts and graph triples accept a TTL. When it lapses, `purge_expired()` removes them. No manual cleanup. Compliant by default.

```python
agent.fact("session token: abc123", ttl="1h")
agent.fact("audit log reference: Q1-2026", ttl="90d")
agent.link("Acme", "signed_contract", "ThinkLanceAI", ttl="2y")

agent.purge_expired()  # removes what's lapsed, keeps what's protected
```

### 4. Graph traversal — structured relations alongside prose memory

Facts are prose. Triples are structure. Both live in the same agent, with their own retention and TTL policy.

```python
agent.link("Acme", "owns", "Globex")
agent.link("Globex", "located_in", "BE")
agent.link("Alice", "works_at", "Acme", confidence=0.9)

related = agent.find_related("Acme", max_hops=2, direction="out")
# {"Globex": 1, "BE": 2, "Alice": 1}
```

### 5. Plug into Claude Desktop via MCP

AgentKeeper ships an MCP server. Any MCP-aware client — Claude Desktop, Cursor, Claude Code — gets full access to the agent's cognitive layer without writing a line of integration code.

```bash
agentkeeper-mcp --agent-id aria --provider anthropic
```

`claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "aria": {
      "command": "agentkeeper-mcp",
      "args": ["--agent-id", "aria", "--provider", "anthropic"]
    }
  }
}
```

Available tools over MCP: `add_fact`, `recall`, `set_identity`, `link`, `find_related`, `compress`, `health`, `gdpr_export`, `purge_expired`.

---

## Architecture

```
       ┌──────────────────────────────────────────────────────────────┐
       │                  AgentKeeper Public API                       │
       │  agent.remember() · agent.recall() · agent.ask()             │
       │  agent.compress() · agent.link() · agent.find_related()      │
       │  agent.set_identity() · agent.purge_expired() · agent.save() │
       └────────────────────────────┬─────────────────────────────────┘
                                    │
       ┌────────────────────────────▼─────────────────────────────────┐
       │           Cognitive Reconstruction Engine (CRE)              │
       │  Identity injection · importance ranking · semantic boost    │
       │  Token budget · profile-driven rendering                     │
       └─┬───────────┬────────────┬────────────┬──────────────────────┘
         │           │            │            │
   ┌─────▼─────┐ ┌──▼─────────┐ ┌▼──────────┐ ┌▼──────────────┐
   │ Memory    │ │ Semantic   │ │ Cognitive │ │ Cross-Model   │
   │ Hierarchy │ │ Recall     │ │ Compress  │ │ Translation   │
   │           │ │            │ │           │ │               │
   │ working   │ │ embeddings │ │ decay     │ │ XML (Claude)  │
   │ episodic  │ │ vector idx │ │ consol.   │ │ sections (GPT)│
   │ semantic  │ │ sqlite-vec │ │ contradic │ │ narrative (G.)│
   │ archival  │ │            │ │           │ │ minimal (Oll.)│
   └───────────┘ └────────────┘ └───────────┘ └───────────────┘
         │                                            │
   ┌─────▼────────────────────────────────────────────▼────────────┐
   │  Graph Layer             Storage (pluggable)                   │
   │  Triple, TTL, BFS        SQLite · Encrypted SQLite · Postgres* │
   │  agent.link()            AGENTKEEPER_DB, AGENTKEEPER_ENC_KEY   │
   └───────────────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────▼───────────────────┐
              │  MCP Server · LangChain · CrewAI         │
              │  agentkeeper-mcp · langchain_system_prompt│
              └──────────────────────────────────────────┘
```

*Postgres stub available; full implementation in v1.2.

---

## Full API tour

### Memory primitives

```python
agent.fact("budget: 50k EUR", importance=0.9)          # stable semantic fact
agent.event("contract signed", when="2026-05-15")      # episodic, time-anchored
agent.principle("never share PII")                     # protected, survives all compression
agent.remember("favourite colour: blue")               # tier inferred automatically
```

### Semantic recall

```python
results = agent.recall("money allocated to the project", top_k=5)
for fact, score in results:
    print(f"{score:.2f}  {fact.content}")
```

Pluggable backends: local `sentence-transformers` (default, free, offline), OpenAI, or your own. Persistent index via `sqlite-vec` — survives process restarts without rebuild.

### Cognitive compression

```python
report = agent.compress()
# CompressionReport(
#   decayed_facts=12,
#   consolidation={'clusters_found': 3, 'facts_removed': 7},
#   contradictions={'pairs_found': 2, 'resolutions': 2},
#   facts_before=120, facts_after=102,
# )
```

Three independent passes: **decay** (exponential half-life on unused facts), **consolidation** (embedding-based clustering, optional LLM synthesiser), **contradiction arbitration** (key-value divergence + polarity detection, deterministic winner). Protected facts are immortal.

### Graph relations

```python
agent.link("Acme", "owns", "Globex", confidence=1.0)
agent.link("Acme", "signed_contract", "ThinkLanceAI", ttl="2y")

# BFS traversal
related = agent.find_related("Acme", max_hops=2, direction="out")

# Introspect triples
for triple in agent.triples:
    print(triple)  # Triple('Acme' -[owns]-> 'Globex', conf=1.00)
```

### TTL on facts and triples

```python
agent.fact("session token: abc123", ttl="1h")
agent.fact("temp context: negotiation", ttl="30d")
agent.link("Project Phoenix", "uses_provider", "Anthropic", ttl="P90D")

agent.purge_expired()  # returns count of removed items
```

### Encrypted storage

```python
from agentkeeper.storage.encrypted_sqlite import EncryptedSQLiteStorage

key = EncryptedSQLiteStorage.generate_key()
storage = EncryptedSQLiteStorage(encryption_key=key)
# or via env: AGENTKEEPER_ENCRYPTION_KEY=...
```

AES-128-CBC + HMAC-SHA256 at rest. Progressive migration — plain SQLite rows are upgraded on next save.

### Async API

```python
import asyncio, agentkeeper

async def main():
    agent = agentkeeper.create_async(agent_id="aria", provider="anthropic")
    agent.set_identity(name="Aria", role="copilot")
    agent.fact("budget: 50k EUR", importance=0.95)

    answers = await asyncio.gather(
        agent.ask("status?", provider="anthropic"),
        agent.ask("status?", provider="openai"),
    )

asyncio.run(main())
```

Sync and async agents share the same storage. Save with one, load with the other.

### LangChain integration

```python
from agentkeeper.integrations.langchain import LangChainCognitiveProvider
from langchain_core.prompts import ChatPromptTemplate

provider = LangChainCognitiveProvider(agent, model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", provider(message="What's our budget?")),
    ("human", "{input}"),
])
```

### Custom provider profile

```python
from agentkeeper import CognitiveProfile, PromptFormat, register_profile

register_profile(CognitiveProfile(
    provider="my-llm",
    format=PromptFormat.SECTIONS,
    effective_context_tokens=10_000,
))
```

---

## GDPR / compliance

```python
export = agent.gdpr_export()   # all facts, triples, identity — JSON-serialisable
agent.purge_expired()          # remove anything whose TTL has elapsed
agent.forget(fact_id)          # remove a single fact by id
agentkeeper.delete("aria")     # permanently remove an agent from storage
```

---

## Observability

```python
snapshot = agent.health()
# {
#   "total_facts": 42,
#   "critical_facts": 3,
#   "protected_facts": 5,
#   "contradicted_facts": 1,
#   "stale_facts": 4,
#   "importance_stats": {"mean": 0.71, "min": 0.10, "max": 0.95},
#   "tier_distribution": {"semantic": 28, "episodic": 10, ...},
#   "graph": {"total_triples": 8, "protected_triples": 2},
#   "identity": {"name": "Aria", "principles_count": 3, ...}
# }
```

---

## Production checklist

- **Type-safe** — `py.typed` shipped, mypy-strict compatible.
- **Typed exceptions** — `AgentKeeperError` root, subclasses for every failure mode.
- **Structured logging** — namespaced under `agentkeeper.*`, opt-in, `NullHandler` default.
- **Retries** — exponential backoff + jitter via `with_retry` / `with_async_retry`.
- **Tested** — 459 tests, CI on Python 3.10 / 3.11 / 3.12.
- **Zero breaking changes from v0.1** — `agent.remember(content, critical=True)` still works.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI provider |
| `ANTHROPIC_API_KEY` | — | Anthropic provider |
| `GEMINI_API_KEY` | — | Gemini provider |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server |
| `AGENTKEEPER_DB` | `agentkeeper.db` | SQLite path |
| `AGENTKEEPER_ENCRYPTION_KEY` | — | Fernet key for encrypted storage |
| `AGENTKEEPER_EMBEDDING_PROVIDER` | auto | `sentence-transformers` › `openai` › `mock` |

---

## Try it in 30 seconds — no API key needed

```bash
pip install agentkeeper-ai
python examples/demo.py
```

The demo runs entirely on `provider="mock"` and `AGENTKEEPER_EMBEDDING_PROVIDER=mock`. No keys. No network. Shows identity hardening, compression, cross-model translation, and async — in one file.

---

## Roadmap

- **v1.1** ✅ — TTL, graph layer, encrypted storage, MCP server, LangChain + CrewAI integrations, persistent `sqlite-vec` index, GDPR export, health snapshot, 459 tests.
- **v1.2** — Postgres backend (full), async LLM synthesiser.
- **v1.3** — AgentKeeper Cloud (managed sync). OSS stays feature-complete.

---

## Contributing

Issues, ideas, and PRs welcome. See [CONTRIBUTING.md](./CONTRIBUTING.md).

---

## License

MIT. See [LICENSE](./LICENSE).

---

## Built by

[**ThinkLanceAI**](https://thinklanceai.com) — Tom Anciaux Berner — cognitive infrastructure for AI systems.

`tom@thinklanceai.com`
