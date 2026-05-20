# Changelog

All notable changes to AgentKeeper are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) · [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.2] — 2026-05-20

Performance release. No public API changes.

### Changed

- **Compression is dramatically faster at scale.** An optional numpy
  accelerator (`agentkeeper._fastmath`) vectorises the dot products and
  cluster-centroid math used by consolidation and contradiction
  arbitration. With numpy installed, a full compression pass over an
  agent holding 10,000 facts drops from ~118s to ~5s (about 23x). Without
  numpy, behaviour is unchanged — the pure-Python fallback is preserved,
  so the core retains zero required dependencies.
- Install the accelerator with the new extra: `pip install
  'agentkeeper-ai[fast]'` (also included in `[all]`).
- Consolidation clustering now assigns each fact to the **best** matching
  centroid above the similarity threshold rather than the first one
  encountered. Tighter, more stable clusters; output on typical inputs is
  equal or better.

### Added

- `benchmark/stress_test.py` — reproducible scaling benchmark.
- `tests/test_fastmath.py` — verifies the numpy and pure-Python paths agree.

## [1.1.0] — 2026-05-19

### Added

**TTL and retention (AK-9)**
- `Fact.expires_at` and `Triple.expires_at` — absolute expiration timestamps.
- `agent.fact(..., ttl="30d")` — shorthand TTL on any memory primitive.
- `agent.link(..., ttl="P90D")` — TTL on graph triples.
- `agent.purge_expired()` — removes lapsed facts and triples, keeps protected ones.
- `agentkeeper.retention.ttl` — pure TTL parser: `timedelta`, shorthand (`"7d12h"`), ISO-8601 (`"P30D"`).
- `agentkeeper.retention.policy` — per-tier retention policy engine.

**Graph memory layer (AK-10)**
- `Triple` — directed relation type: `subject -[predicate]-> object`, with `confidence`, `protected`, `ttl`, `metadata`.
- `agent.link(subject, predicate, object, ...)` — add a structured relation.
- `agent.find_related(entity, max_hops, direction)` — BFS traversal returning `{entity: hop_distance}`.
- `agent.triples` — direct access to the triple store.
- Graph triples participate in TTL expiration and compression.

**Typed fact taxonomy (AK-11)**
- `FactType` enum: `decision`, `preference`, `constraint`, `relationship`, `task_state`, `transient`, `identity`, `event`, `fact`.
- Per-type decay rates — `transient` decays fast, `constraint` and `identity` are immortal by default.
- `fact_type` accepted by `agent.remember()` and the MCP `add_fact` tool.

**Persistent vector index (AK-12)**
- `SqliteVecIndex` — `sqlite-vec`-backed vector index that persists across restarts without rebuild.
- Auto-selected when `sqlite-vec` is installed; falls back to `InMemoryVectorIndex` otherwise.
- New extra: `pip install 'agentkeeper[vec]'`.

**Encrypted storage (AK-13)**
- `EncryptedSQLiteStorage` — payload-at-rest encryption via Fernet (AES-128-CBC + HMAC-SHA256).
- `EncryptedSQLiteStorage.generate_key()` convenience factory.
- Progressive migration: plain SQLite rows are detected and re-encrypted on next save.
- New extra: `pip install 'agentkeeper[encrypted]'`.
- `AGENTKEEPER_ENCRYPTION_KEY` env var.

**Pluggable storage backends (AK-13)**
- `BaseStorage` ABC — four-method contract for custom backends.
- `StorageFactory` — auto-selects backend from env vars.
- `PostgresStorage` stub declared (full implementation in v1.2).

**MCP server (AK-15)**
- `agentkeeper.mcp.build_server(agent_id, provider)` — FastMCP server exposing the agent's full cognitive layer.
- `agentkeeper-mcp` CLI entry-point — `stdio`, `streamable-http`, `sse` transports.
- Tools: `add_fact`, `recall`, `set_identity`, `link`, `find_related`, `compress`, `health`, `gdpr_export`, `purge_expired`.
- Resources: `agentkeeper://identity`, `agentkeeper://facts/{fact_id}`.
- Compatible with Claude Desktop, Claude Code, Cursor, and any MCP-aware client.
- New extra: `pip install 'agentkeeper[mcp]'`.

**Framework integrations (AK-14)**
- `agentkeeper.integrations.langchain` — `langchain_system_prompt()` and `LangChainCognitiveProvider`.
- `agentkeeper.integrations.crewai` — CrewAI cognitive provider.
- No hard dependency on either framework — integrations work as pure string helpers.

**Observability and GDPR**
- `agent.health()` — cognitive snapshot: fact counts, importance stats, tier distribution, graph size, identity status.
- `agent.gdpr_export()` — JSON-serialisable export of all facts, triples, and identity (GDPR Article 20).
- `agent.forget(fact_id)` — remove a single fact and purge it from the vector index.

### Changed

- `pyproject.toml` version bumped to `1.1.0`.
- New extras: `[vec]`, `[encrypted]`, `[mcp]`; `[all]` now includes all three.
- `agentkeeper-mcp` registered as console script entry-point.
- Test count: 267 → 459.

### Migration

Zero breaking changes. All v1.0 and v0.1 code continues to work unchanged.

---

## [1.0.0] — 2026-05-18

Full v1.0 release. See [CHANGELOG_v1.0.md](./CHANGELOG_v1.0.md) or the [GitHub release notes](https://github.com/Thinklanceai/agentkeeper/releases/tag/v1.0.0) for the complete history.

## [0.1.0] — 2026-02

Initial release: cross-model memory continuity, CSO persistence, adapters for OpenAI / Anthropic / Gemini / Ollama, 95% critical-fact recovery benchmark, MIT license.
