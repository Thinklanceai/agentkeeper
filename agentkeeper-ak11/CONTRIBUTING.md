# Contributing to AgentKeeper

Thanks for considering a contribution. AgentKeeper is small enough that
most decisions get made quickly — open an issue first to discuss anything
non-trivial.

## Development setup

```bash
git clone https://github.com/Thinklanceai/agentkeeper.git
cd agentkeeper
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,all]"
pytest -q
ruff check agentkeeper tests
```

## Principles we hold

1. **No breaking changes to the public API.** Every release since v0.1
   has preserved the surface (`agent.remember(content, critical=True)`
   still works in v1.0). New features are additive.
2. **Vendor-agnostic.** Anything that locks the library to a specific
   provider (OpenAI-only, Anthropic-only) goes in an *optional extra*,
   never in the core path.
3. **No central infrastructure.** Storage stays local-first (SQLite).
   Cloud / managed features are downstream products, not in this repo.
4. **Tests come with the code.** PRs without tests for the new behaviour
   get a polite ping.
5. **Narrative consistency.** AgentKeeper is *cognitive continuity
   infrastructure* — not a "memory wrapper", not a "vector DB", not an
   "agent framework". Keep terminology aligned.

## Workflow

- One feature/fix per branch.
- Commit messages: present-tense imperative
  (`add semantic recaller cache`, not `added a cache`).
- Run `pytest -q && ruff check agentkeeper tests` before pushing.
- Open a PR; CI must pass.

## Project layout

```
agentkeeper/
  __init__.py          # public API, sync Agent
  async_agent.py       # AsyncAgent
  errors.py            # typed exceptions
  logging.py           # stdlib logger namespace
  retry.py             # backoff decorators
  cso/                 # CognitiveStateObject, Fact, identity, tiers
  cre/                 # Cognitive Reconstruction Engine
  semantic/            # Embeddings + vector index + recaller
  compression/         # Decay + consolidation + contradiction + pipeline
  translation/         # Per-provider cognitive profiles + renderers
  adapters/            # LLM provider adapters (sync + async)
  storage/             # SQLite persistence
  benchmark/           # Reproducible benchmarks
tests/                 # pytest suite
examples/              # Self-contained demos
```

## Questions?

`hello@thinklanceai.com` or open an issue.
