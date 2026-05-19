# X / Twitter launch thread

Format: 5 posts, each ≤280 characters. Post one at a time, ~2 min apart.
Pin the first post. Add the architecture image to post #1 or #4.

---

**1/**

AgentKeeper 1.0 is out.

Cognitive continuity infrastructure for long-lived AI agents — reconstructs persistent state across model switches, crashes, restarts, and constrained contexts.

`pip install agentkeeper`

https://github.com/Thinklanceai/agentkeeper

---

**2/**

What changed since v0.1:

— Memory hierarchy: working / episodic / semantic / archival
— Persistent identity (principles, hard constraints) immune to compression
— Semantic recall (local embeddings by default)
— Decay + consolidation + contradiction arbitration
— Async API

---

**3/**

Cross-model translation:

Same cognitive state, rendered differently per provider.

— XML for Claude
— Sections for GPT-4
— Narrative for Gemini
— Minimal for Ollama / local models

One agent, four runtimes, no rewrites.

---

**4/**

In March, a few people saw this project and posted about it before I'd even finished v1. Shruti Codes, @cstripathi, @Kayvonjafarzadeh, Leonard Rodman, Martin Szerment, Bespoke AI, and @grok itself.

Thanks for catching it early. v1.0 is the version that earns the framing.

---

**5/**

Open source, MIT, zero infrastructure required. SQLite-first.

Built by @thinklanceai.

If you ship agents that need to survive across model changes, this is the layer you've been writing by hand.

Issues, PRs, ideas welcome.
