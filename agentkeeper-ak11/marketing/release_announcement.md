# Why I rebuilt AgentKeeper as a cognitive continuity layer

Three months ago I shipped AgentKeeper as "cross-model memory
persistence for AI agents." A few people saw the project early and
posted about it without being asked — Shruti, Chidanand (80K views),
Kayvon, Leonard, Martin, Bespoke AI, and @grok itself.

I didn't see the wave until last week. Notifications were off on the
repo, my fault entirely. By the time I caught up there were 117 stars,
16 forks, an open issue I had never answered, and three months of
silence that looked deliberate from the outside.

So I had a choice: ship a quiet v0.2 patch and move on, or do the
version that the early supporters had already imagined the project
was. I chose the second.

v1.0 ships today.

## The framing changed

The original README pitched AgentKeeper as a "memory wrapper for
cross-provider agents." That framing put it in a category that is
commoditising fast: vector DBs with extra glue, memory primitives
inside agent frameworks, half a dozen projects with similar names.

But that was never what the code did. The Cognitive Reconstruction
Engine sitting at the centre of v0.1 was already doing prioritisation,
token-aware selection, force-inclusion of critical facts, and
reconstruction across model switches. The bones of an OS-style memory
layer, not a wrapper.

The new framing is **cognitive continuity infrastructure for
long-lived AI agents**. The job: reconstruct an agent's full cognitive
state — memory, identity, priorities, decision context — across model
switches, crashes, restarts, and constrained contexts. Continuity, not
just memory.

## What's actually new

Seven sprints worth of work, all backward-compatible with v0.1.

**Memory hierarchy.** Working / episodic / semantic / archival tiers,
inspired by Tulving's distinctions. Smart routing infers the right tier
from content. `Fact.importance` (float 0–1) replaces the binary
`critical` flag internally — the old API still works.

**Persistent identity.** `agent.set_identity(name, role, principles,
constraints)`. Identity is always injected, always honoured, and —
crucially — exempt from every form of compression. Run 100 consecutive
compression cycles and the principles are still there.

**Semantic recall.** `agent.recall(query, top_k=5)` finds facts by
meaning, not keywords. Default embeddings come from local
`sentence-transformers` (free, no API call, no vendor lock-in). OpenAI
and custom providers available as opt-in.

**Cognitive compression.** Three independent passes:

- Decay — exponential half-life on unused non-critical facts.
- Consolidation — cosine-similarity clustering collapses near-duplicates.
  Optional LLM-backed synthesiser using the agent's own provider.
- Contradiction arbitration — detects key-value divergence and
  polarity opposition. Deterministic winner (critical > importance >
  recency). Loser is flagged, not deleted.

**Cross-model translation.** The same cognitive state, rendered
differently per provider: XML for Claude (`<agent_identity>`,
`<memory>`), labelled sections for GPT-4, narrative prose for Gemini's
long context window, minimal tokens for local models running through
Ollama. Custom providers via `register_profile()`.

**Async API.** `AsyncAgent`, `create_async()`, `load_async()`.
Storage is shared with the sync agent — save with one, load with the
other.

**Production polish.** Typed exception hierarchy, namespaced stdlib
logger, retry decorators with exponential backoff and jitter, full
`py.typed` typing, mypy-strict friendly, 267 tests, CI on
Python 3.10/3.11/3.12.

## Backward compatibility

Code written for v0.1 continues to work. Every existing test from the
original repo still passes. `agent.remember(content, critical=True)`
still does the right thing. v0.1 SQLite databases load automatically
into the v1.0 schema.

The new helpers — `agent.principle()`, `agent.event()`, `agent.fact()`
— are additive. Use them when you want clearer intent; ignore them if
you don't.

## What it isn't

It is not a vector DB. It is not an agent framework. It is not a
memory wrapper. It is not a managed service.

It is the layer that lives between your agent and whichever LLM you
happen to be calling this week. It owns the question: *given everything
this agent has ever known, what cognitive state should I reconstruct
for this specific call to this specific model?*

## What's next

v1.1 will add a persistent vector index via `sqlite-vec` so the
recaller survives restarts without a rebuild. v1.2 will add an async
LLM-backed synthesiser. v1.3 will add pluggable storage backends.
v1.4 may add a managed sync product — staying optional, with the OSS
remaining feature-complete.

## Thanks

To everyone who saw it early and posted about it: thank you. You were
right about what the project could be before I had finished it.
v1.0 is the version that earns the framing.

→ `pip install agentkeeper`
→ https://github.com/Thinklanceai/agentkeeper

Built by [ThinkLanceAI](https://thinklanceai.com).
Tom — Brussels, May 2026.
