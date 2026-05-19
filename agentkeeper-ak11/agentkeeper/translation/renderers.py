"""Format-specific renderers for reconstructed cognitive context.

Each renderer takes the structured pieces of a reconstruction (identity,
facts grouped by tier, task) and produces the final string to send to
an LLM. The format is chosen per-provider via `CognitiveProfile`.

All renderers are pure functions: same input → same output, no I/O.
"""

from __future__ import annotations

from ..cso.identity import AgentIdentity
from ..cso.tiers import MemoryTier
from ..cso.types import Fact
from .profiles import CognitiveProfile, PromptFormat


def _critical_marker(fact: Fact, profile: CognitiveProfile) -> str:
    if not (fact.protected or fact.importance >= 0.9):
        return ""
    if profile.format == PromptFormat.XML:
        return ' critical="true"'
    if profile.format == PromptFormat.MINIMAL:
        return "!"
    return " ★"


def _render_episodic_when(fact: Fact) -> str:
    return f"({fact.when}) " if fact.when else ""


# --- XML (Claude-friendly) --------------------------------------------


def _render_xml(
    identity: AgentIdentity,
    grouped: dict[MemoryTier, list[Fact]],
    task: str,
    profile: CognitiveProfile,
) -> str:
    parts: list[str] = []

    if not identity.is_empty():
        parts.append("<agent_identity>")
        if identity.name:
            parts.append(f"  <name>{identity.name}</name>")
        if identity.role:
            parts.append(f"  <role>{identity.role}</role>")
        if identity.principles:
            parts.append("  <principles>")
            for p in identity.principles:
                parts.append(f"    <principle>{p}</principle>")
            parts.append("  </principles>")
        if identity.constraints:
            parts.append("  <hard_constraints>")
            for c in identity.constraints:
                parts.append(f"    <constraint>{c}</constraint>")
            parts.append("  </hard_constraints>")
        parts.append("</agent_identity>")

    if any(grouped.values()):
        parts.append("")
        parts.append("<memory>")
        for tier_name in profile.tier_order:
            try:
                tier = MemoryTier(tier_name)
            except ValueError:
                continue
            facts = grouped.get(tier, [])
            if not facts:
                continue
            parts.append(f'  <tier name="{tier.value}">')
            for f in facts:
                marker = _critical_marker(f, profile)
                when = _render_episodic_when(f) if tier == MemoryTier.EPISODIC else ""
                parts.append(f"    <fact{marker}>{when}{f.content}</fact>")
            parts.append("  </tier>")
        parts.append("</memory>")

    parts.append("")
    parts.append(f"<task>{task}</task>")
    parts.append("")
    parts.append(
        "Use your identity and memory to maintain continuity across sessions. "
        "Do not ask for information you already have."
    )
    return "\n".join(parts)


# --- SECTIONS (GPT-4 family) ------------------------------------------


def _render_sections(
    identity: AgentIdentity,
    grouped: dict[MemoryTier, list[Fact]],
    task: str,
    profile: CognitiveProfile,
) -> str:
    parts: list[str] = []

    identity_block = identity.render_for_prompt()
    if identity_block:
        parts.append(identity_block)

    if any(grouped.values()):
        if parts:
            parts.append("")
        parts.append("MEMORY (reconstructed from prior sessions):")
        for tier_name in profile.tier_order:
            try:
                tier = MemoryTier(tier_name)
            except ValueError:
                continue
            facts = grouped.get(tier, [])
            if not facts:
                continue
            parts.append(f"  [{tier.value}]")
            for f in facts:
                marker = _critical_marker(f, profile)
                when = _render_episodic_when(f) if tier == MemoryTier.EPISODIC else ""
                parts.append(f"    -{marker} {when}{f.content}")
    else:
        if parts:
            parts.append("")
        parts.append("MEMORY: (empty)")

    parts.append("")
    parts.append(f"CURRENT TASK: {task}")
    parts.append(
        "Use your identity and memory to maintain continuity. "
        "Do not ask for information you already have."
    )
    return "\n".join(parts)


# --- NARRATIVE (Gemini, long-context) ---------------------------------


def _render_narrative(
    identity: AgentIdentity,
    grouped: dict[MemoryTier, list[Fact]],
    task: str,
    profile: CognitiveProfile,
) -> str:
    parts: list[str] = []

    if not identity.is_empty():
        intro = (
            f"You are {identity.name or 'an AI agent'}"
            + (f", acting as {identity.role}" if identity.role else "")
            + "."
        )
        parts.append(intro)
        if identity.principles:
            parts.append("Your core principles, which you always honour:")
            for p in identity.principles:
                parts.append(f"  • {p}")
        if identity.constraints:
            parts.append("Your hard constraints, which you never violate:")
            for c in identity.constraints:
                parts.append(f"  • {c}")
        parts.append("")

    if any(grouped.values()):
        parts.append(
            "Below is your reconstructed memory from prior sessions, "
            "organised by cognitive tier:"
        )
        parts.append("")
        for tier_name in profile.tier_order:
            try:
                tier = MemoryTier(tier_name)
            except ValueError:
                continue
            facts = grouped.get(tier, [])
            if not facts:
                continue
            parts.append(f"## {tier.value.title()} memory")
            for f in facts:
                marker = _critical_marker(f, profile)
                when = _render_episodic_when(f) if tier == MemoryTier.EPISODIC else ""
                parts.append(f"  -{marker} {when}{f.content}")
            parts.append("")

    parts.append(f"Current task: {task}")
    parts.append(
        "Maintain continuity with your prior cognitive state. "
        "Do not ask for information already present above."
    )
    return "\n".join(parts)


# --- MINIMAL (Ollama, small/local models) -----------------------------


def _render_minimal(
    identity: AgentIdentity,
    grouped: dict[MemoryTier, list[Fact]],
    task: str,
    profile: CognitiveProfile,
) -> str:
    parts: list[str] = []

    if not identity.is_empty():
        identity_line = identity.name or "agent"
        if identity.role:
            identity_line += f" ({identity.role})"
        parts.append(f"You: {identity_line}")
        for p in identity.principles:
            parts.append(f"! {p}")
        for c in identity.constraints:
            parts.append(f"! {c}")

    if any(grouped.values()):
        for tier_name in profile.tier_order:
            try:
                tier = MemoryTier(tier_name)
            except ValueError:
                continue
            facts = grouped.get(tier, [])
            if not facts:
                continue
            for f in facts:
                marker = _critical_marker(f, profile)
                when = f.when + " " if (tier == MemoryTier.EPISODIC and f.when) else ""
                parts.append(f"-{marker} {when}{f.content}")

    parts.append(f"Q: {task}")
    return "\n".join(parts)


# --- dispatch ---------------------------------------------------------


_RENDERERS = {
    PromptFormat.XML: _render_xml,
    PromptFormat.SECTIONS: _render_sections,
    PromptFormat.NARRATIVE: _render_narrative,
    PromptFormat.MINIMAL: _render_minimal,
}


def render(
    profile: CognitiveProfile,
    identity: AgentIdentity,
    grouped: dict[MemoryTier, list[Fact]],
    task: str,
) -> str:
    """Render a reconstructed context for the given profile."""
    renderer = _RENDERERS[profile.format]
    return renderer(identity, grouped, task, profile)
