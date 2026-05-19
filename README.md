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

<img width="159" height="150" alt="agentkeeper_v11_dark" src="https://github.com/user-attachments/assets/07248445-4af2-4ffb-8b5c-b92f9452b12a" />
<svg width="100%" viewBox="0 0 680 640" role="img" style="background: rgb(13, 17, 23); border-radius: 12px;" xmlns="http://www.w3.org/2000/svg">
<title style="fill:rgb(0, 0, 0);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">AgentKeeper v1.1 — Cognitive Continuity Infrastructure</title>
<desc style="fill:rgb(0, 0, 0);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto">AgentKeeper v1.1 full architecture on dark background</desc>
<defs>
<marker id="arr2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M2 1L8 5L2 9" fill="none" stroke="#4fc3f7" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</marker>
</defs>

<!-- Title -->
<text x="340" y="30" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="600" fill="#e6edf3" style="fill:rgb(230, 237, 243);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:15px;font-weight:600;text-anchor:middle;dominant-baseline:auto">AgentKeeper — Cognitive Continuity Infrastructure</text>
<text x="340" y="50" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#8b949e" style="fill:rgb(139, 148, 158);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:11px;font-weight:400;text-anchor:middle;dominant-baseline:auto">Reconstructs persistent cognitive state across model switches, crashes, restarts, and constrained contexts</text>

<!-- PUBLIC API -->
<rect x="40" y="64" width="600" height="50" rx="8" fill="#0d2137" stroke="#4fc3f7" stroke-width="1" style="fill:rgb(13, 33, 55);stroke:rgb(79, 195, 247);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="340" y="84" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#4fc3f7" style="fill:rgb(79, 195, 247);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:12px;font-weight:600;text-anchor:middle;dominant-baseline:auto">PUBLIC API</text>
<text x="340" y="102" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">agent.remember() · agent.recall() · agent.ask() · agent.compress() · agent.link() · agent.find_related() · agent.purge_expired()</text>

<!-- Arrow -->
<line x1="340" y1="114" x2="340" y2="132" stroke="#4fc3f7" stroke-width="1" marker-end="url(#arr2)" style="fill:rgb(0, 0, 0);stroke:rgb(79, 195, 247);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<!-- CRE -->
<rect x="80" y="132" width="520" height="50" rx="8" fill="#0d2137" stroke="#4fc3f7" stroke-width="1" style="fill:rgb(13, 33, 55);stroke:rgb(79, 195, 247);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="340" y="152" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#4fc3f7" style="fill:rgb(79, 195, 247);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:12px;font-weight:600;text-anchor:middle;dominant-baseline:auto">COGNITIVE RECONSTRUCTION ENGINE (CRE)</text>
<text x="340" y="170" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">Identity injection · Importance ranking · Semantic boost · Token budget · Profile-driven rendering · Critical-fact eviction</text>

<!-- 4 arrows down -->
<line x1="117" y1="182" x2="117" y2="206" stroke="#4fc3f7" stroke-width="1" marker-end="url(#arr2)" style="fill:rgb(0, 0, 0);stroke:rgb(79, 195, 247);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<line x1="270" y1="182" x2="270" y2="206" stroke="#4fc3f7" stroke-width="1" marker-end="url(#arr2)" style="fill:rgb(0, 0, 0);stroke:rgb(79, 195, 247);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<line x1="423" y1="182" x2="423" y2="206" stroke="#4fc3f7" stroke-width="1" marker-end="url(#arr2)" style="fill:rgb(0, 0, 0);stroke:rgb(79, 195, 247);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<line x1="575" y1="182" x2="575" y2="206" stroke="#4fc3f7" stroke-width="1" marker-end="url(#arr2)" style="fill:rgb(0, 0, 0);stroke:rgb(79, 195, 247);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<!-- Memory Hierarchy -->
<rect x="40" y="206" width="154" height="110" rx="6" fill="#0a1929" stroke="#1e4976" stroke-width="1" style="fill:rgb(10, 25, 41);stroke:rgb(30, 73, 118);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="117" y="226" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#4fc3f7" style="fill:rgb(79, 195, 247);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:11px;font-weight:600;text-anchor:middle;dominant-baseline:auto">MEMORY HIERARCHY</text>
<text x="117" y="244" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">· working</text>
<text x="117" y="260" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">· episodic</text>
<text x="117" y="276" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">· semantic</text>
<text x="117" y="292" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">· archival</text>
<text x="117" y="308" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#4a7fa5" style="fill:rgb(74, 127, 165);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:9px;font-weight:400;text-anchor:middle;dominant-baseline:auto">Importance + tier-aware retention</text>

<!-- Semantic Recall -->
<rect x="203" y="206" width="134" height="110" rx="6" fill="#0a1929" stroke="#1e4976" stroke-width="1" style="fill:rgb(10, 25, 41);stroke:rgb(30, 73, 118);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="270" y="226" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#4fc3f7" style="fill:rgb(79, 195, 247);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:11px;font-weight:600;text-anchor:middle;dominant-baseline:auto">SEMANTIC RECALL</text>
<text x="270" y="244" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">· Embeddings</text>
<text x="270" y="260" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">· Vector index</text>
<text x="270" y="276" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">· sqlite-vec</text>
<text x="270" y="292" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">· Pluggable</text>
<text x="270" y="308" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#4a7fa5" style="fill:rgb(74, 127, 165);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:9px;font-weight:400;text-anchor:middle;dominant-baseline:auto">sentence-transformers · OpenAI · custom</text>

<!-- Compression -->
<rect x="346" y="206" width="154" height="110" rx="6" fill="#0a1929" stroke="#1e4976" stroke-width="1" style="fill:rgb(10, 25, 41);stroke:rgb(30, 73, 118);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="423" y="226" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#4fc3f7" style="fill:rgb(79, 195, 247);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:11px;font-weight:600;text-anchor:middle;dominant-baseline:auto">COMPRESSION</text>
<text x="423" y="244" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">· Decay</text>
<text x="423" y="260" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">· Consolidation</text>
<text x="423" y="276" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">· Contradictions</text>
<text x="423" y="292" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">· Pipeline</text>
<text x="423" y="308" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#4a7fa5" style="fill:rgb(74, 127, 165);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:9px;font-weight:400;text-anchor:middle;dominant-baseline:auto">Identity exempt · LLM synth optional</text>

<!-- Translation -->
<rect x="509" y="206" width="131" height="110" rx="6" fill="#0a1929" stroke="#1e4976" stroke-width="1" style="fill:rgb(10, 25, 41);stroke:rgb(30, 73, 118);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="575" y="226" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#4fc3f7" style="fill:rgb(79, 195, 247);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:11px;font-weight:600;text-anchor:middle;dominant-baseline:auto">TRANSLATION</text>
<text x="575" y="244" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">· XML (Claude)</text>
<text x="575" y="260" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">· Sections (GPT)</text>
<text x="575" y="276" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">· Narrative (Gemini)</text>
<text x="575" y="292" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">· Minimal (Ollama)</text>
<text x="575" y="308" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#4a7fa5" style="fill:rgb(74, 127, 165);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:9px;font-weight:400;text-anchor:middle;dominant-baseline:auto">Per-provider cognitive profile</text>

<!-- Arrows down to graph+storage -->
<line x1="200" y1="316" x2="200" y2="340" stroke="#4fc3f7" stroke-width="1" marker-end="url(#arr2)" style="fill:rgb(0, 0, 0);stroke:rgb(79, 195, 247);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<line x1="460" y1="316" x2="460" y2="340" stroke="#4fc3f7" stroke-width="1" marker-end="url(#arr2)" style="fill:rgb(0, 0, 0);stroke:rgb(79, 195, 247);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<!-- Graph Layer -->
<rect x="40" y="340" width="300" height="72" rx="6" fill="#0a1929" stroke="#1e4976" stroke-width="1" style="fill:rgb(10, 25, 41);stroke:rgb(30, 73, 118);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="190" y="360" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#4fc3f7" style="fill:rgb(79, 195, 247);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:11px;font-weight:600;text-anchor:middle;dominant-baseline:auto">GRAPH MEMORY LAYER</text>
<text x="190" y="378" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">agent.link(subject, predicate, object) · BFS traversal</text>
<text x="190" y="394" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">Triple · confidence · TTL · protected</text>

<!-- Storage -->
<rect x="350" y="340" width="290" height="72" rx="6" fill="#0a1929" stroke="#1e4976" stroke-width="1" style="fill:rgb(10, 25, 41);stroke:rgb(30, 73, 118);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="495" y="360" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#4fc3f7" style="fill:rgb(79, 195, 247);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:11px;font-weight:600;text-anchor:middle;dominant-baseline:auto">STORAGE — SQLite-first · Vendor-agnostic</text>
<text x="495" y="378" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">SQLite · Encrypted SQLite (AES-128-CBC) · Postgres*</text>
<text x="495" y="394" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">Zero infrastructure · MIT license · BaseStorage ABC</text>

<!-- Arrow down -->
<line x1="340" y1="412" x2="340" y2="432" stroke="#4fc3f7" stroke-width="1" marker-end="url(#arr2)" style="fill:rgb(0, 0, 0);stroke:rgb(79, 195, 247);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<!-- TTL + Retention -->
<rect x="40" y="432" width="600" height="44" rx="6" fill="#0a1929" stroke="#1e4976" stroke-width="1" style="fill:rgb(10, 25, 41);stroke:rgb(30, 73, 118);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="340" y="451" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#4fc3f7" style="fill:rgb(79, 195, 247);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:11px;font-weight:600;text-anchor:middle;dominant-baseline:auto">TTL + RETENTION · GDPR</text>
<text x="340" y="468" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">agent.fact(ttl="1h") · agent.purge_expired() · agent.gdpr_export() · ISO-8601 + shorthand parser</text>

<!-- Arrow down -->
<line x1="340" y1="476" x2="340" y2="496" stroke="#4fc3f7" stroke-width="1" marker-end="url(#arr2)" style="fill:rgb(0, 0, 0);stroke:rgb(79, 195, 247);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>

<!-- MCP -->
<rect x="40" y="496" width="185" height="66" rx="6" fill="#0a1929" stroke="#1e4976" stroke-width="1" style="fill:rgb(10, 25, 41);stroke:rgb(30, 73, 118);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="132" y="516" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#4fc3f7" style="fill:rgb(79, 195, 247);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:11px;font-weight:600;text-anchor:middle;dominant-baseline:auto">MCP SERVER</text>
<text x="132" y="534" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">Claude Desktop · Cursor · Claude Code</text>
<text x="132" y="550" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">stdio · streamable-http · sse</text>

<!-- Integrations -->
<rect x="235" y="496" width="205" height="66" rx="6" fill="#0a1929" stroke="#1e4976" stroke-width="1" style="fill:rgb(10, 25, 41);stroke:rgb(30, 73, 118);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="337" y="516" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#4fc3f7" style="fill:rgb(79, 195, 247);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:11px;font-weight:600;text-anchor:middle;dominant-baseline:auto">INTEGRATIONS</text>
<text x="337" y="534" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">LangChain cognitive provider</text>
<text x="337" y="550" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">CrewAI · async API</text>

<!-- Observability -->
<rect x="450" y="496" width="190" height="66" rx="6" fill="#0a1929" stroke="#1e4976" stroke-width="1" style="fill:rgb(10, 25, 41);stroke:rgb(30, 73, 118);color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:&quot;Anthropic Sans&quot;, -apple-system, &quot;system-ui&quot;, &quot;Segoe UI&quot;, sans-serif;font-size:16px;font-weight:400;text-anchor:start;dominant-baseline:auto"/>
<text x="545" y="516" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#4fc3f7" style="fill:rgb(79, 195, 247);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:11px;font-weight:600;text-anchor:middle;dominant-baseline:auto">OBSERVABILITY</text>
<text x="545" y="534" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">agent.health() · identity_audit()</text>
<text x="545" y="550" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#8ec8f0" style="fill:rgb(142, 200, 240);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">459 tests · py.typed · MIT</text>

<!-- Bottom label -->
<text x="340" y="590" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#4a7fa5" style="fill:rgb(74, 127, 165);stroke:none;color:rgb(0, 0, 0);stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;opacity:1;font-family:sans-serif;font-size:10px;font-weight:400;text-anchor:middle;dominant-baseline:auto">pip install agentkeeper-ai · github.com/Thinklanceai/agentkeeper · Built by ThinkLanceAI — Tom Anciaux Berner</text>
</svg>

`tom@thinklanceai.com`
