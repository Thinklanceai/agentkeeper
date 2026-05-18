"""Cognitive compression — decay, consolidation, contradiction resolution."""

from .consolidation import (
    ConsolidationConfig,
    ConsolidationResult,
    Synthesiser,
    consolidate,
)
from .contradiction import (
    Contradiction,
    ContradictionConfig,
    ContradictionResult,
    detect_and_resolve,
)
from .decay import (
    DEFAULT_DECAY,
    DecayConfig,
    apply_decay_in_place,
    days_since_last_access,
    decayed_importance,
)
from .llm_synth import make_llm_synthesiser
from .pipeline import CompressionConfig, CompressionReport, compress

__all__ = [
    "CompressionConfig",
    "CompressionReport",
    "ConsolidationConfig",
    "ConsolidationResult",
    "Contradiction",
    "ContradictionConfig",
    "ContradictionResult",
    "DEFAULT_DECAY",
    "DecayConfig",
    "Synthesiser",
    "apply_decay_in_place",
    "compress",
    "consolidate",
    "days_since_last_access",
    "decayed_importance",
    "detect_and_resolve",
    "make_llm_synthesiser",
]
