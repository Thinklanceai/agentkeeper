"""Platform memory importers — bring your AI memory into AgentKeeper.

Quick start::

    import agentkeeper
    from agentkeeper.importers import ChatGPTImporter

    agent = agentkeeper.create("my-memory")
    report = ChatGPTImporter("~/Downloads/chatgpt-export.zip").apply(agent)
    print(report.summary())

Or from a pasted memory summary::

    from agentkeeper.importers import parse_memory_summary

    facts = parse_memory_summary(open("summary.txt").read())
"""

from .base import (
    SCOPE_GLOBAL,
    BaseImporter,
    ImportedFact,
    ImportError_,
    ImportReport,
    platform_scope,
)
from .chatgpt import ChatGPTImporter, parse_memory_summary
from .claude import ClaudeImporter
from .gemini import GeminiImporter

__all__ = [
    "SCOPE_GLOBAL",
    "BaseImporter",
    "ChatGPTImporter",
    "ClaudeImporter",
    "GeminiImporter",
    "ImportError_",
    "ImportReport",
    "ImportedFact",
    "parse_memory_summary",
    "platform_scope",
]
