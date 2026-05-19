# AgentKeeper 1.0 — Cognitive continuity infrastructure for long-lived AI agents

When AgentKeeper first shipped in February, the README described it as
"cross-model memory persistence." That framing was too narrow.

What the code actually does — reconstruct an agent's full cognitive
state across model switches, crashes, restarts, and constrained contexts
— deserved a different name. v1.0 ships that name and the missing layers
that make the claim true.

## What's in 1.0

**Memory hierarchy.** Working / episodic / semantic / archival tiers.
Importance scoring replaces the binary critical flag (the old API still
works). Smart tier inference from content.

**Persistent identity.** Name, role, principles, hard constraints.
Always injected, never decayed, never consolidated, never flagged as
contradicted. Survives every form of compression.

**Semantic recall.** `agent.recall(query, top_k=5)` finds facts by
meaning. Pluggable embeddings: `sentence-transformers` by default
(local, free, no lock-in), OpenAI/Voyage available as opt-in.

**Cognitive compression.** Three independent passes — decay
(exponential half-life), consolidation (cosine clustering, optional
LLM synth), contradiction arbitration (key-value divergence + polarity
opposition). Identity exempt from all three.

**Cross-model translation.** Same cognitive state, four formats:
XML for Claude, sections for GPT-4, narrative for Gemini, minimal for
local models. Custom providers via `register_profile()`.

**Async API.** `AsyncAgent`, `create_async()`, `load_async()` parallel
to the sync API. Storage is shared.

**Production polish.** Typed exception hierarchy, namespaced stdlib
logger, retry decorators with exponential backoff + jitter, mypy-strict
friendly, `py.typed` shipped, CI on Python 3.10/3.11/3.12, 267 tests.

## Backward compatibility

Code written for v0.1 continues to work. `agent.remember(content,
critical=True)` still maps to the right place internally. Sync API
surface unchanged. v0.1 SQLite databases load automatically into the
v1.0 schema. The new helpers (`agent.principle()`, `agent.event()`,
`agent.fact()`) are additive.

## Install

```bash
pip install agentkeeper
```

Extras: `[anthropic]`, `[openai]`, `[gemini]`, `[semantic]`, `[all]`.

## Thanks

Special thanks to everyone who saw the project early in March 2026,
posted about it, and starred the repo without being asked: Shruti
Codes, Chidanand Tripathi, Kayvon Jafarzadeh, Leonard Rodman, Martin
Szerment, Mustapha Elouardi, Bespoke AI Solutions, and @grok. v1.0
is the version that lives up to what you were already seeing.

## Roadmap

- v1.1 — Persistent vector index (`sqlite-vec`) for recall across
  restarts.
- v1.2 — Async LLM-backed synthesiser.
- v1.3 — Pluggable storage backends.

## Links

- README: https://github.com/Thinklanceai/agentkeeper#readme
- PyPI: https://pypi.org/project/agentkeeper/
- CHANGELOG: https://github.com/Thinklanceai/agentkeeper/blob/main/CHANGELOG.md

Built by [ThinkLanceAI](https://thinklanceai.com).
