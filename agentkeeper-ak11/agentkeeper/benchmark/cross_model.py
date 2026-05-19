"""Cross-model reconstruction benchmark.

Measures how well an agent's cognitive state survives reconstruction
across heterogeneous providers (Claude / GPT / Gemini / Ollama).

The benchmark seeds a fixed set of critical facts, asks each target
provider the same question, and reports the fraction of critical facts
that were preserved through the reconstruction. With the MockAdapter
(default fallback) the score equals the *injection* rate — useful as a
deterministic floor. With real providers, the score measures actual
LLM recovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..adapters.base import BaseAdapter, MockAdapter
from ..cre.engine import CognitiveReconstructionEngine
from ..cso.types import CognitiveStateObject
from ..translation.profiles import get_profile
from .dataset import generate_test_facts
from .verification import extract_recovered_facts


@dataclass
class ProviderResult:
    provider: str
    model: str
    format: str
    token_budget: int
    critical_total: int
    critical_injected: int
    recovered_count: int
    recovery_rate: float
    sample_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "format": self.format,
            "token_budget": self.token_budget,
            "critical_total": self.critical_total,
            "critical_injected": self.critical_injected,
            "recovered_count": self.recovered_count,
            "recovery_rate": self.recovery_rate,
        }


@dataclass
class CrossModelReport:
    task: str
    results: list[ProviderResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "results": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:
        if not self.results:
            return "(no providers evaluated)"
        lines = [
            f"{'PROVIDER':<12} {'FORMAT':<10} {'BUDGET':>7} "
            f"{'INJECT':>8} {'RECOV':>8}"
        ]
        for r in self.results:
            inject_pct = (
                100.0 * r.critical_injected / r.critical_total
                if r.critical_total
                else 0.0
            )
            lines.append(
                f"{r.provider:<12} {r.format:<10} {r.token_budget:>7} "
                f"{inject_pct:>7.1f}% {r.recovery_rate * 100:>7.1f}%"
            )
        return "\n".join(lines)


# Concrete model identifiers used when the user does not pass an
# explicit map. These are the *latest* models per provider as of v1.0.
_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-5-20250929",
    "openai": "gpt-4o",
    "gemini": "gemini-1.5-pro",
    "ollama": "llama3",
    "mock": "mock-v1",
}


def run_cross_model_benchmark(
    providers: list[str] | None = None,
    adapters: dict[str, BaseAdapter] | None = None,
    cso: CognitiveStateObject | None = None,
    task: str = "What are the key project details I should know?",
    model_map: dict[str, str] | None = None,
) -> CrossModelReport:
    """Run the cross-model benchmark.

    Args:
        providers: Provider names to evaluate. Defaults to all known.
        adapters: Per-provider live adapters. Any missing adapter falls
            back to MockAdapter (still useful: it deterministically
            measures the injection rate — i.e. CRE's prioritization
            quality).
        cso: Optional pre-seeded cognitive state. Defaults to the
            standard 100-fact / 20-critical synthetic dataset.
        task: User-facing question.
        model_map: Per-provider model name override.

    Returns:
        A `CrossModelReport` with one `ProviderResult` per provider.
    """
    providers = providers or ["anthropic", "openai", "gemini", "ollama"]
    adapters = adapters or {}
    model_map = model_map or {}
    cso = cso or generate_test_facts(n_total=100, n_critical=20)

    report = CrossModelReport(task=task)

    for provider in providers:
        model = model_map.get(provider, _DEFAULT_MODELS.get(provider, provider))
        profile = get_profile(provider)
        adapter = adapters.get(provider) or MockAdapter()

        cre = CognitiveReconstructionEngine(cso)
        prompt = cre.build_context_prompt(model, task)
        selected = cre.prioritize(model)
        critical_selected = [f for f in selected if f.importance >= 0.9]
        all_critical = [f for f in cso.memory_facts if f.importance >= 0.9]

        response = adapter.query(prompt, task)
        recovered_ids = extract_recovered_facts(response, critical_selected)

        recovery_rate = (
            len(recovered_ids) / len(critical_selected)
            if critical_selected
            else 0.0
        )

        report.results.append(
            ProviderResult(
                provider=provider,
                model=model,
                format=profile.format.value,
                token_budget=cre._budget_for(model, None),
                critical_total=len(all_critical),
                critical_injected=len(critical_selected),
                recovered_count=len(recovered_ids),
                recovery_rate=recovery_rate,
                sample_response=response[:300],
            )
        )

    return report
