# AgentKeeper

**Cognitive continuity infrastructure for long-lived AI agents.**

AgentKeeper reconstructs persistent cognitive state across model switches, crashes, restarts, and constrained contexts.
Built for agents that must survive longer than a single context window.

[![PyPI version](https://img.shields.io/pypi/v/agentkeeper.svg)](https://pypi.org/project/agentkeeper/)
[![Python versions](https://img.shields.io/pypi/pyversions/agentkeeper.svg)](https://pypi.org/project/agentkeeper/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/Thinklanceai/agentkeeper/actions/workflows/ci.yml/badge.svg)](https://github.com/Thinklanceai/agentkeeper/actions/workflows/ci.yml)
[![Built by ThinkLanceAI](https://img.shields.io/badge/built%20by-ThinkLanceAI-4f8cff)](https://thinklanceai.com)

---

## Why AgentKeeper exists

Agents don't fail because they forget facts.
They fail because they lose **cognitive continuity** — their state, priorities, identity, and decision context drift the moment the underlying model changes, the context window fills, or the process restarts.

AgentKeeper treats this as a systems problem, not a memory problem.

It provides:

- A **Cognitive Reconstruction Engine** that rebuilds an agent's state for the target model, every call.
- A **memory hierarchy** (working / episodic / semantic / archival) with importance-aware retention.
- **Semantic recall** based on embeddings — meaning, not keywords.
- **Cognitive compression** — decay, consolidation, contradiction arbitration.
- **Identity persistence** — principles and hard constraints that survive every form of compression.
- **Cross-model translation** — XML for Claude, sections for GPT-4, narrative for Gemini, minimal for local models.

Continuity, not just memory.

## Install

```bash
pip install agentkeeper-ai
```

Optional extras (only install what you need):

```bash
pip install 'agentkeeper[anthropic]'    # Claude
pip install 'agentkeeper[openai]'       # GPT models + OpenAI embeddings
pip install 'agentkeeper[gemini]'       # Google Gemini
pip install 'agentkeeper[semantic]'     # Local embeddings (sentence-transformers)
pip install 'agentkeeper[all]'          # everything
```

No external infrastructure required. Storage defaults to local SQLite. Vendor-agnostic by design.

## 60-second tour

```python
import agentkeeper

# 1. Create an agent and define its persistent identity
agent = agentkeeper.create(agent_id="aria", provider="anthropic")
agent.set_identity(
    name="Aria",
    role="EU insurance broker copilot",
    principles=["never share PII without explicit consent"],
    constraints=["EU data residency only"],
)

# 2. Teach it about the world
agent.principle("always confirm budget changes in writing")
agent.fact("client: Acme Corporation", importance=0.95)
agent.event("contract signed", when="2026-05-15")
agent.remember("favourite colour: blue")  # tier inferred automatically

# 3. Ask — context is reconstructed for the target model
response = agent.ask("What do we know about the Acme deal?")

# 4. Switch providers — memory and identity survive
agent.switch_provider("openai").save()
response = agent.ask("Same question, different model.")
```

## Architecture

```
       ┌──────────────────────────────────────────────────────────────┐
       │                  AgentKeeper Public API                       │
       │  agent.remember() · agent.recall() · agent.ask()              │
       │  agent.compress() · agent.set_identity() · agent.save()       │
       └────────────────────────────┬──────────────────────────────────┘
                                    │
       ┌────────────────────────────▼──────────────────────────────────┐
       │           Cognitive Reconstruction Engine (CRE)               │
       │  Identity injection · importance ranking · semantic boost     │
       │  Token budget · profile-driven rendering                      │
       └─┬────────────┬────────────┬────────────┬─────────────────────┘
         │            │            │            │
   ┌─────▼─────┐ ┌───▼────────┐ ┌─▼─────────┐ ┌▼──────────────┐
   │ Memory    │ │ Semantic   │ │ Cognitive │ │ Cross-Model    │
   │ Hierarchy │ │ Recall     │ │ Compress  │ │ Translation    │
   │           │ │            │ │           │ │                │
   │ working   │ │ embeddings │ │ decay     │ │ XML / sections │
   │ episodic  │ │ vector     │ │ consol.   │ │ narrative      │
   │ semantic  │ │ index      │ │ contradic │ │ minimal        │
   │ archival  │ │            │ │           │ │                │
   └───────────┘ └────────────┘ └───────────┘ └────────────────┘
                                    │
                      ┌─────────────▼─────────────┐
                      │   Storage (SQLite-first)  │
                      │   Vendor-agnostic         │
                      └───────────────────────────┘
```

Every layer is **interchangeable** and **opt-in**. The base CRE works with no embeddings, no compression, no profile customisation. Layers stack as you need them.

## Cognitive continuity in five primitives

### 1. Identity that never erodes

```python
agent.set_identity(
    name="Aria",
    role="EU broker copilot",
    principles=["never share PII"],
    constraints=["EU data residency only"],
)
```

Identity is injected into every reconstructed context, regardless of token budget. It survives compression, model switches, and restarts. Principles are **protected** — never decayed, never consolidated, never flagged.

### 2. Memory organised by tier

```python
agent.fact("budget: 50k EUR")                      # semantic (stable)
agent.event("contract signed", when="2026-05-15")  # episodic (time-anchored)
agent.principle("always confirm changes")          # protected, identity-level
agent.remember("favourite colour: blue")           # tier inferred automatically
```

Tiers shape narration in reconstructed prompts and drive retention policy under compression.

### 3. Semantic recall

```python
results = agent.recall("money allocated to the project", top_k=5)
for fact, score in results:
    print(f"{score:.2f}  {fact.content}")
```

Pluggable embedding providers: local `sentence-transformers` (default, free, no lock-in), OpenAI, or your own. Recall biases context reconstruction toward facts that actually matter for the current question.

### 4. Cognitive compression

```python
report = agent.compress()
# CompressionReport(
#   decayed_facts=12,
#   consolidation={'clusters_found': 3, 'facts_removed': 7, ...},
#   contradictions={'pairs_found': 2, 'resolutions': 2},
#   facts_before=120, facts_after=102,
# )
```

Three independent passes:

- **Decay** — exponential half-life on unused non-critical facts. Critical and protected facts are immortal.
- **Consolidation** — embedding-based clustering merges near-duplicates. Optional LLM-backed synthesiser.
- **Contradiction arbitration** — key-value divergence and polarity-opposition detection. Deterministic winner (critical > importance > recency). Loser is flagged, not deleted.

### 5. Cross-model translation

The same cognitive state, four formats:

| Provider   | Format     | Why                                          |
|------------|------------|-----------------------------------------------|
| Anthropic  | XML        | Claude excels with structured `<agent_identity>` and `<memory>` blocks |
| OpenAI     | Sections   | GPT family responds well to labelled sections (`AGENT IDENTITY`, `MEMORY`, `CURRENT TASK`) |
| Gemini     | Narrative  | Long-context model — prefers prose framing |
| Ollama     | Minimal    | Small/local models — aggressive compression, terse tokens |

Custom providers? Register your own:

```python
from agentkeeper import CognitiveProfile, PromptFormat, register_profile

register_profile(CognitiveProfile(
    provider="my-llm",
    format=PromptFormat.SECTIONS,
    effective_context_tokens=10_000,
))
```

## Async API

```python
import asyncio
import agentkeeper

async def main() -> None:
    agent = agentkeeper.create_async(agent_id="aria", provider="anthropic")
    agent.set_identity(name="Aria", role="copilot")
    agent.fact("budget: 50k", importance=0.95)

    # Parallel asks across providers
    answers = await asyncio.gather(
        agent.ask("status?", provider="anthropic"),
        agent.ask("status?", provider="openai"),
    )

asyncio.run(main())
```

Sync and async agents share storage — you can save with one and load with the other.

## Production-grade

- **Type-safe**: `py.typed` marker shipped, mypy-strict friendly.
- **Typed exceptions**: `AgentKeeperError` root, with subclasses for every failure mode (provider down, agent missing, retriable network errors).
- **Structured logging**: namespaced under `agentkeeper.*`, opt-in.
- **Retries**: built-in exponential backoff for transient provider errors via `with_retry` / `with_async_retry`.
- **Tested**: 267 tests, CI on Python 3.10 / 3.11 / 3.12.

## Configuration

Environment variables (all optional):

| Variable                          | Default                          | Purpose |
|-----------------------------------|----------------------------------|---------|
| `OPENAI_API_KEY`                  | —                                | Required for OpenAI provider |
| `ANTHROPIC_API_KEY`               | —                                | Required for Anthropic provider |
| `GEMINI_API_KEY`                  | —                                | Required for Gemini provider |
| `OPENAI_MODEL`                    | `gpt-4-turbo`                    | OpenAI model name |
| `ANTHROPIC_MODEL`                 | `claude-sonnet-4-5-20250929`     | Anthropic model name |
| `OLLAMA_HOST`                     | `http://localhost:11434`         | Ollama server URL |
| `AGENTKEEPER_DB`                  | `agentkeeper.db`                 | SQLite path |
| `AGENTKEEPER_EMBEDDING_PROVIDER`  | auto (sentence-transformers > openai > mock) | Override embedding backend |

## Roadmap (v1.x)

- **v1.1** — Persistent vector index (`sqlite-vec`) so the recaller survives restarts without rebuild.
- **v1.2** — Async LLM-backed synthesiser in compression.
- **v1.3** — Pluggable storage backends (Postgres, encrypted).
- **v1.4** — `AgentKeeper Cloud` (managed sync). *Stays optional; OSS remains feature-complete.*

## Featured on

AgentKeeper was vouched for in March 2026 by:

- Shruti Codes ✓
- Chidanand Tripathi ✓ ([80K-view thread](#))
- Kayvon Jafarzadeh ✓
- Leonard Rodman ✓
- Martin Szerment ✓
- Bespoke AI Solutions Inc ✓
- @grok (xAI)

If you've shipped agents that needed to survive across model changes, you understand why this matters. Thank you to everyone who saw the project early.

## Contributing

Issues, ideas, and PRs welcome. See [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

MIT. See [LICENSE](./LICENSE).

## Built by

[**ThinkLanceAI**](https://thinklanceai.com) — cognitive infrastructure for AI systems.
Need this in production with custom integrations? `hello@thinklanceai.com`.
