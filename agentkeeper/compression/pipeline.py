"""Top-level cognitive compression pipeline.

The pipeline orchestrates three independent passes that together keep
an agent's cognitive state coherent over millions of tokens of life:

1. **Decay** — non-critical facts lose importance with age.
2. **Consolidation** — near-duplicate facts collapse into a canonical
   representative; optional LLM synthesis for richer summaries.
3. **Contradiction arbitration** — divergent facts about the same
   subject are resolved deterministically.

Each pass is independently controllable and reportable. The pipeline
is the single entry point used by `agent.compress()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..cso.types import CognitiveStateObject
from ..semantic.base import EmbeddingProvider
from .consolidation import (
    ConsolidationConfig,
    ConsolidationResult,
    Synthesiser,
    consolidate,
)
from .contradiction import (
    ContradictionConfig,
    ContradictionResult,
    detect_and_resolve,
)
from .decay import DEFAULT_DECAY, DecayConfig, apply_decay_in_place


@dataclass
class CompressionReport:
    """Diagnostic report returned by `compress()`.

    Captures what happened during a compression run so callers can audit
    the agent's state evolution.
    """

    started_at: str = ""
    finished_at: str = ""
    decayed_facts: int = 0
    consolidation: ConsolidationResult = field(default_factory=ConsolidationResult)
    contradictions: ContradictionResult = field(default_factory=ContradictionResult)
    facts_before: int = 0
    facts_after: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "decayed_facts": self.decayed_facts,
            "consolidation": {
                "clusters_found": self.consolidation.clusters_found,
                "facts_removed": self.consolidation.facts_removed,
                "facts_synthesised": self.consolidation.facts_synthesised,
            },
            "contradictions": {
                "pairs_found": self.contradictions.pairs_found,
                "resolutions": len(self.contradictions.contradictions),
            },
            "facts_before": self.facts_before,
            "facts_after": self.facts_after,
        }


@dataclass
class CompressionConfig:
    """Top-level toggle for each compression pass."""

    run_decay: bool = True
    run_consolidation: bool = True
    run_contradiction: bool = True
    decay: DecayConfig = field(default_factory=lambda: DEFAULT_DECAY)
    consolidation: ConsolidationConfig = field(
        default_factory=ConsolidationConfig
    )
    contradiction: ContradictionConfig = field(
        default_factory=ContradictionConfig
    )


def compress(
    cso: CognitiveStateObject,
    embedding_provider: EmbeddingProvider,
    config: CompressionConfig | None = None,
    synthesiser: Synthesiser | None = None,
    now: datetime | None = None,
) -> CompressionReport:
    """Run the compression pipeline against an agent's CSO.

    Args:
        cso: The cognitive state to compress (mutated in place).
        embedding_provider: Required for semantic passes (consolidation,
            contradiction). Use the same provider as the agent's recall
            for consistency.
        config: Override individual pass toggles and thresholds.
        synthesiser: Optional LLM-backed summariser for consolidation.
            If omitted, consolidation keeps the canonical fact as-is.
        now: Override the current time (useful for testing decay).

    Returns:
        A `CompressionReport` describing what changed.
    """
    config = config or CompressionConfig()
    now = now or datetime.now(timezone.utc)

    report = CompressionReport(
        started_at=now.isoformat(),
        facts_before=len(cso.memory_facts),
    )

    if config.run_decay:
        report.decayed_facts = apply_decay_in_place(
            cso.memory_facts, now=now, config=config.decay
        )

    if config.run_consolidation and len(cso.memory_facts) >= 2:
        report.consolidation = consolidate(
            cso.memory_facts,
            provider=embedding_provider,
            config=config.consolidation,
            synthesiser=synthesiser,
        )

    if config.run_contradiction and len(cso.memory_facts) >= 2:
        report.contradictions = detect_and_resolve(
            cso.memory_facts,
            provider=embedding_provider,
            config=config.contradiction,
            now=now,
        )

    report.facts_after = len(cso.memory_facts)
    report.finished_at = datetime.now(timezone.utc).isoformat()
    cso.updated_at = report.finished_at
    return report
