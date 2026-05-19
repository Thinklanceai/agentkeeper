"""Cross-model cognitive translation.

Different LLM providers respond optimally to different prompt formats.
This package provides per-provider `CognitiveProfile`s and format-specific
renderers, so the CRE can reconstruct context that matches each model's
strengths.
"""

from .profiles import (
    CognitiveProfile,
    PromptFormat,
    get_profile,
    known_providers,
    register_profile,
)
from .renderers import render

__all__ = [
    "CognitiveProfile",
    "PromptFormat",
    "get_profile",
    "known_providers",
    "register_profile",
    "render",
]
