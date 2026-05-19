# Changelog

All notable changes to AgentKeeper are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-05-18

The v1.0 release reframes AgentKeeper from "cross-model memory wrapper"
to **cognitive continuity infrastructure for long-lived AI agents**.
Every layer of the original v0.1 architecture is preserved; the
public API is fully backward-compatible.

### Added

**Memory hierarchy and identity (AK-2)**
- `MemoryTier` enum: `working`, `episodic`, `semantic`, `archival`.
- `Fact.importance` (float 0–1) replaces internal binary `critical` flag.
- `Fact.tier`, `Fact.created_at`, `Fact.last_accessed_at`,
  `Fact.access_count`, `Fact.when`, `Fact.metadata`.
- Smart tier inference from content (deterministic, no LLM).
- `AgentIdentity` (name, role, principles, constraints) — always injected.
- `agent.fact()`, `agent.event()`, `agent.principle()` explicit helpers.
- `agent.set_identity()`, `agent.identity_audit()`.

**Semantic recall (AK-3)**
- `agentkeeper.semantic` package: `EmbeddingProvider` ABC,
  `MockEmbeddingProvider`, `SentenceTransformerProvider`,
  `OpenAIEmbeddingProvider`, `InMemoryVectorIndex`, `SemanticRecaller`.
- `agent.recall(query, top_k, min_score)` for meaning-based retrieval.
- Auto-detection: sentence-transformers → OpenAI → Mock fallback.
- Override via `AGENTKEEPER_EMBEDDING_PROVIDER` env var.

**Cognitive compression (AK-4)**
- `agentkeeper.compression` package with four passes:
  - **Decay**: exponential importance decay with 30-day half-life.
  - **Consolidation**: cosine-similarity clustering with optional
    LLM-backed synthesiser.
  - **Contradiction arbitration**: key-value divergence + polarity
    opposition, deterministic winner.
  - **Pipeline**: orchestrates all three with toggles +
    `CompressionReport`.
- `agent.compress()` and `agent.contradictions()` public API.

**Identity hardening (AK-5)**
- `Fact.protected` flag: principles and hard constraints exempt
  from every compression pass.
- `agent.principle()` sets `protected=True` automatically.
- `agent.set_identity(merge=True)` to append principles without
  replacing the existing identity.
- 100-iteration compression identity-survival test.

**Cross-model translation (AK-6)**
- `agentkeeper.translation` package: `CognitiveProfile`,
  `PromptFormat`, four format-specific renderers.
- Per-provider rendering: XML (Claude), sections (GPT-4), narrative
  (Gemini), minimal (Ollama).
- Profile-driven budgets replace static `DEFAULT_TOKEN_LIMIT`.
- `register_profile()` for custom providers.
- `run_cross_model_benchmark()` for parallel provider evaluation.

**Async API + production polish (AK-7)**
- `AsyncAgent`, `create_async()`, `load_async()` — full async-native
  facade. Shares storage with sync `Agent`.
- `AsyncBaseAdapter`, `AsyncOpenAIAdapter`, `AsyncAnthropicAdapter`,
  `AsyncMockAdapter`.
- `agentkeeper.errors`: typed exception hierarchy
  (`AgentKeeperError`, `UnknownProviderError`, `UnknownTierError`,
  `AgentNotFoundError`, `ProviderError`, `RetriableProviderError`,
  `ConfigurationError`, `CompressionError`, `EmbeddingError`).
- `agentkeeper.logging`: stdlib namespaced logger with
  `NullHandler` default.
- `agentkeeper.retry`: `with_retry` / `with_async_retry` decorators
  with exponential backoff + jitter.
- Sync errors keep subclassing `ValueError` for backward compatibility.

**Foundations (AK-1)**
- Proper Python package layout (`src/*` → `agentkeeper/*`).
- Fixed `pyproject.toml` build backend.
- Extras: `[openai]`, `[anthropic]`, `[gemini]`, `[semantic]`,
  `[dotenv]`, `[all]`, `[dev]`.
- GitHub Actions CI on Python 3.10 / 3.11 / 3.12.
- Adapter caching (no per-call reconnects).
- `py.typed` marker for downstream type checkers.

### Changed

- Default reconstruction prompt format now adapts to the target
  provider. v0.1 produced a single hard-coded format; v1.0 produces
  XML for Claude, sections for OpenAI, narrative for Gemini, minimal
  for Ollama.
- Token budgets are now profile-driven when the model is unknown
  (fall back to provider profile instead of `DEFAULT_TOKEN_LIMIT`).
- Embedding resolution favours local `sentence-transformers` over
  cloud OpenAI when both are available — preserves the "no vendor
  lock-in" principle.

### Migration

**Zero breaking changes.** Code written for v0.1 continues to work:

- `agent.remember(content, critical=True)` still works, and is
  internally mapped to `tier=semantic, importance=0.95`.
- `Fact.critical` remains a readable property
  (`importance >= 0.9`).
- The sync `Agent` API is unchanged.
- v0.1 SQLite databases load automatically into v1.0 schema.

Optional: adopt the new helpers gradually (`agent.principle()`,
`agent.event()`, `agent.fact()`) for clearer intent.

## [0.1.0] — 2026-02 (initial release)

- Cross-model memory continuity via a Cognitive Reconstruction Engine.
- Cognitive State Object (CSO) persistence in local SQLite.
- Adapters for OpenAI, Anthropic, Gemini, Ollama.
- 95% critical-fact recovery benchmark across provider switches.
- MIT license, zero infrastructure, vendor-agnostic.
