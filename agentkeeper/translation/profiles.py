"""Cognitive profiles for LLM providers.

Different LLMs respond optimally to different prompt structures.
A `CognitiveProfile` captures the rendering preferences of a target
provider so the CRE can produce context that matches the model's
strengths, not a one-size-fits-all blob.

This is what allows AgentKeeper to switch an agent's runtime from
Claude to GPT-4 to Gemini to Ollama without rewriting the system
prompt by hand each time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PromptFormat(str, Enum):
    """Top-level prompt-rendering format."""

    # XML-flavoured structured blocks (Claude's preferred form).
    XML = "xml"
    # Sectioned markdown-ish (good default for GPT-4 family).
    SECTIONS = "sections"
    # Long-form structured for high-context models (Gemini).
    NARRATIVE = "narrative"
    # Aggressive compression for small/local models (Ollama).
    MINIMAL = "minimal"


@dataclass(frozen=True)
class CognitiveProfile:
    """Per-provider reconstruction profile.

    Attributes:
        provider: Logical provider name ("anthropic", "openai", ...).
        format: Top-level prompt rendering style.
        effective_context_tokens: Conservative tokens we feel safe
            spending on the reconstructed context — *below* the model's
            theoretical max to leave room for the user task and reply.
        verbosity: How much explanatory framing to include around facts.
            0.0 = minimal headers, 1.0 = full prose.
        identity_emphasis: Multiplier on identity rendering. >1.0 means
            identity gets repeated/expanded; <1.0 means terse identity.
        tier_order: Order in which tiers are rendered, from first
            (highest priority) to last.
        compression_bias: Multiplier on token compression aggressiveness.
            1.0 = default. 0.5 = give the model a lot, trust it.
            2.0 = aggressively trim for small-context models.
    """

    provider: str
    format: PromptFormat = PromptFormat.SECTIONS
    effective_context_tokens: int = 8_000
    verbosity: float = 0.5
    identity_emphasis: float = 1.0
    tier_order: tuple[str, ...] = ("semantic", "episodic", "working", "archival")
    compression_bias: float = 1.0


# Built-in defaults. Tuned conservatively. Users can register their own
# via `register_profile()` (e.g. fine-tuned models with non-standard
# context windows or specialised structure preferences).
_DEFAULT_PROFILES: dict[str, CognitiveProfile] = {
    "anthropic": CognitiveProfile(
        provider="anthropic",
        format=PromptFormat.XML,
        effective_context_tokens=12_000,
        verbosity=0.7,
        identity_emphasis=1.2,
    ),
    "openai": CognitiveProfile(
        provider="openai",
        format=PromptFormat.SECTIONS,
        effective_context_tokens=8_000,
        verbosity=0.5,
        identity_emphasis=1.0,
    ),
    "gemini": CognitiveProfile(
        provider="gemini",
        format=PromptFormat.NARRATIVE,
        effective_context_tokens=16_000,
        verbosity=0.8,
        identity_emphasis=1.0,
    ),
    "ollama": CognitiveProfile(
        provider="ollama",
        format=PromptFormat.MINIMAL,
        effective_context_tokens=3_000,
        verbosity=0.1,
        identity_emphasis=0.8,
        compression_bias=2.0,
    ),
    "mock": CognitiveProfile(
        provider="mock",
        format=PromptFormat.SECTIONS,
        effective_context_tokens=4_000,
        verbosity=0.5,
        identity_emphasis=1.0,
    ),
}


def get_profile(provider: str) -> CognitiveProfile:
    """Return the cognitive profile for a provider, or a safe fallback.

    Unknown providers fall back to the OpenAI-style sections profile,
    which is the most generic and works reasonably with any chat LLM.
    """
    return _DEFAULT_PROFILES.get(provider, _DEFAULT_PROFILES["openai"])


def register_profile(profile: CognitiveProfile) -> None:
    """Register a custom profile, replacing any existing entry for
    `profile.provider`."""
    _DEFAULT_PROFILES[profile.provider] = profile


def known_providers() -> list[str]:
    """Return the list of providers that have a built-in profile."""
    return sorted(_DEFAULT_PROFILES.keys())
