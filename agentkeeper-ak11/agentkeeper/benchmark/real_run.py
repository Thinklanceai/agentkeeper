"""Real cross-model recovery benchmark.

Runs the cross-model benchmark with live providers when API keys are
available, falling back to MockAdapter for any provider whose key is
missing. Outputs a markdown table suitable for the README.

Usage:
    python -m agentkeeper.benchmark.real_run

Environment:
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY: optional, real
        provider keys. When set, the corresponding live adapter is used.
    AGENTKEEPER_BENCH_OUT: optional path to write the markdown table to.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

from ..adapters.base import BaseAdapter, MockAdapter
from .cross_model import run_cross_model_benchmark


def _try_live_adapter(provider: str) -> BaseAdapter | None:
    """Return a real adapter when its API key is present, else None."""
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        if key and "..." not in key:
            from ..adapters.openai import OpenAIAdapter
            return OpenAIAdapter(
                api_key=key,
                model=os.getenv("OPENAI_MODEL", "gpt-4-turbo"),
            )
    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if key and "..." not in key:
            from ..adapters.anthropic import AnthropicAdapter
            return AnthropicAdapter(
                api_key=key,
                model=os.getenv(
                    "ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"
                ),
            )
    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY", "")
        if key and "..." not in key:
            from ..adapters.gemini import GeminiAdapter
            return GeminiAdapter(
                api_key=key,
                model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
            )
    if provider == "ollama":
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        # Only use Ollama if explicitly requested via OLLAMA_USE=1, since
        # we cannot easily probe the local server here.
        if os.getenv("OLLAMA_USE") == "1":
            from ..adapters.ollama import OllamaAdapter
            return OllamaAdapter(
                model=os.getenv("OLLAMA_MODEL", "llama3"),
                host=host,
            )
    return None


def _build_adapter_map() -> dict[str, BaseAdapter]:
    """Build the per-provider adapter map.

    Live adapters when API keys are present, MockAdapter as deterministic
    fallback otherwise.
    """
    mapping: dict[str, BaseAdapter] = {}
    used_live: list[str] = []
    used_mock: list[str] = []
    for provider in ("anthropic", "openai", "gemini", "ollama"):
        live = _try_live_adapter(provider)
        if live is not None:
            mapping[provider] = live
            used_live.append(provider)
        else:
            mapping[provider] = MockAdapter()
            used_mock.append(provider)
    print(f"Live adapters: {used_live or '(none — set API keys to enable)'}")
    print(f"Mock fallback: {used_mock}")
    return mapping


def _to_markdown(report) -> str:
    """Render a CrossModelReport as a markdown table."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        f"# AgentKeeper cross-model recovery — {now}",
        "",
        f"**Task**: {report.task}",
        "",
        "| Provider | Format | Token budget | Critical injected | Recovery |",
        "|----------|--------|-------------:|------------------:|---------:|",
    ]
    for r in report.results:
        inject_pct = (
            100.0 * r.critical_injected / r.critical_total
            if r.critical_total
            else 0.0
        )
        recov_pct = r.recovery_rate * 100
        lines.append(
            f"| {r.provider} | {r.format} | {r.token_budget} | "
            f"{r.critical_injected}/{r.critical_total} ({inject_pct:.0f}%) "
            f"| {recov_pct:.1f}% |"
        )
    lines.append("")
    lines.append(
        "_Recovery measured by keyword-based detection in the model's "
        "response. With MockAdapter, recovery equals injection — the "
        "score reflects CRE prioritisation quality. With live providers, "
        "it reflects actual LLM recall._"
    )
    return "\n".join(lines)


def main() -> int:
    print("=" * 60)
    print("AgentKeeper cross-model benchmark")
    print("=" * 60)
    adapters = _build_adapter_map()
    report = run_cross_model_benchmark(
        providers=["anthropic", "openai", "gemini", "ollama"],
        adapters=adapters,
    )
    print()
    print(report.summary())
    md = _to_markdown(report)
    out_path = os.getenv("AGENTKEEPER_BENCH_OUT")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"\n✓ Markdown report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
