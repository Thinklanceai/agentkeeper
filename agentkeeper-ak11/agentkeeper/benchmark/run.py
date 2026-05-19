"""Benchmark runner for AgentKeeper cross-model reconstruction.

This module is intentionally a CLI entry point. Library users do not
need to import it. Run it directly:

    python -m agentkeeper.benchmark.run

It will fall back to MockAdapter for any provider whose API key is missing.
"""

from __future__ import annotations

import os
from typing import Any

from ..adapters.base import BaseAdapter, MockAdapter
from ..cre.engine import CognitiveReconstructionEngine
from .dataset import generate_test_facts
from .verification import extract_recovered_facts


def run_benchmark(
    source_model: str,
    target_model: str,
    source_adapter: BaseAdapter,
    target_adapter: BaseAdapter,
    token_budget: int = 2_000,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run one cross-model reconstruction benchmark.

    The source model is the agent's prior provider; the target is the
    one we reconstruct context for. We measure how many critical facts
    survive the reconstruction by checking the target's response.
    """
    if verbose:
        print()
        print("=" * 60)
        print("AgentKeeper Benchmark")
        print(f"Source: {source_model} → Target: {target_model}")
        print(f"Token budget: {token_budget}")
        print("=" * 60)
        print()

    cso = generate_test_facts(n_total=100, n_critical=20)
    if verbose:
        print(
            f"✓ Generated {len(cso.memory_facts)} facts "
            f"({len(cso.critical_facts())} critical)\n"
        )

    cre = CognitiveReconstructionEngine(cso)
    stats = cre.reconstruction_stats(target_model, max_tokens=token_budget)

    if verbose:
        print("CRE Analysis:")
        print(f"  Total facts:       {stats['total_facts']}")
        print(f"  Selected facts:    {stats['selected_facts']}")
        print(f"  Critical total:    {stats['critical_total']}")
        print(f"  Critical selected: {stats['critical_selected']}")
        print(f"  Token budget:      {stats['token_budget']}")
        print(f"  Tokens used:       {stats['tokens_used']}")
        print(
            f"  Critical recovery: "
            f"{stats['critical_recovery_rate'] * 100:.1f}%\n"
        )

    task = "What are the key project details I should know?"
    context_prompt = cre.build_context_prompt(
        target_model, task, max_tokens=token_budget
    )

    if verbose:
        print(f"Querying {target_model}...")
    response = target_adapter.query(context_prompt, task)

    selected_facts = cre.prioritize(target_model, max_tokens=token_budget)
    critical_selected = [f for f in selected_facts if f.critical]
    recovered_ids = extract_recovered_facts(response, critical_selected)
    recovery_score = (
        len(recovered_ids) / len(critical_selected)
        if critical_selected
        else 0.0
    )

    if verbose:
        print("\nResults:")
        print(f"  Facts injected (critical):  {len(critical_selected)}")
        print(f"  Facts verified in response: {len(recovered_ids)}")
        print(
            f"  Recovery score: "
            f"{len(recovered_ids)}/{len(critical_selected)} = "
            f"{recovery_score * 100:.1f}%"
        )
        print(f"\nSample response:\n{response[:500]}...")

    return {
        "source_model": source_model,
        "target_model": target_model,
        "token_budget": token_budget,
        "cre_stats": stats,
        "recovery_score": recovery_score,
        "recovered_count": len(recovered_ids),
        "critical_injected": len(critical_selected),
    }


def _build_adapter_or_mock(
    provider: str, env_key: str, model_env: str, default_model: str
) -> tuple[BaseAdapter, str]:
    """Return (adapter, model_name) — falls back to MockAdapter if no key."""
    key = os.getenv(env_key, "")
    if not key or "REMPLACE" in key.upper() or key.lower().startswith("sk-..."):
        print(f"⚠️  {provider} key missing — using MockAdapter")
        return MockAdapter(), f"{provider.lower()}-mock"

    model = os.getenv(model_env, default_model)
    if provider == "OpenAI":
        from ..adapters.openai import OpenAIAdapter
        return OpenAIAdapter(api_key=key, model=model), model
    if provider == "Anthropic":
        from ..adapters.anthropic import AnthropicAdapter
        return AnthropicAdapter(api_key=key, model=model), model
    raise ValueError(f"Unsupported provider for benchmark CLI: {provider}")


def main() -> None:
    """CLI entry point."""
    # Try to load .env if python-dotenv is installed, but don't require it.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    source_adapter, source_model = _build_adapter_or_mock(
        "OpenAI", "OPENAI_API_KEY", "OPENAI_MODEL", "gpt-4-turbo"
    )
    target_adapter, target_model = _build_adapter_or_mock(
        "Anthropic",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "claude-sonnet-4-5-20250929",
    )

    result = run_benchmark(
        source_model=source_model,
        target_model=target_model,
        source_adapter=source_adapter,
        target_adapter=target_adapter,
        token_budget=2_000,
        verbose=True,
    )

    print()
    print("=" * 60)
    print("BENCHMARK COMPLETE")
    print(
        "Critical recovery rate: "
        f"{result['cre_stats']['critical_recovery_rate'] * 100:.1f}%"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
