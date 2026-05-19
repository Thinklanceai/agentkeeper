"""Agent identity: the persistent self-model of an agent.

While facts come and go (and may be compressed or evicted), an agent's
identity must remain stable across model switches, restarts, and
years of operation. The CRE force-injects the identity into every
reconstructed context, regardless of token budget.

An identity is intentionally lightweight: name, role, principles,
constraints. Anything fancier (tone, persona, voice) is layered on top
in v1.x and beyond.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentIdentity:
    """The stable self-model of an agent.

    Attributes:
        name: Human-readable agent name (e.g. "Aria Assistant").
        role: Short role description (e.g. "EU insurance broker copilot").
        principles: Behavioural commitments the agent must respect
                    ("never recommend competitor products without consent").
        constraints: Hard limits ("never share PII outside the EU").
    """

    name: str = ""
    role: str = ""
    principles: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return (
            not self.name
            and not self.role
            and not self.principles
            and not self.constraints
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "principles": list(self.principles),
            "constraints": list(self.constraints),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> AgentIdentity:
        return AgentIdentity(
            name=data.get("name", "") or "",
            role=data.get("role", "") or "",
            principles=list(data.get("principles", []) or []),
            constraints=list(data.get("constraints", []) or []),
        )

    def render_for_prompt(self) -> str:
        """Render the identity as a system-prompt block.

        Empty identities render as an empty string so the CRE can skip
        injection cleanly.
        """
        if self.is_empty():
            return ""

        lines: list[str] = ["AGENT IDENTITY (immutable, always honoured):"]
        if self.name:
            lines.append(f"- Name: {self.name}")
        if self.role:
            lines.append(f"- Role: {self.role}")
        if self.principles:
            lines.append("- Principles:")
            for p in self.principles:
                lines.append(f"    • {p}")
        if self.constraints:
            lines.append("- Hard constraints:")
            for c in self.constraints:
                lines.append(f"    • {c}")
        return "\n".join(lines)
