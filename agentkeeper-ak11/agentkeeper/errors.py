"""AgentKeeper exception hierarchy.

A typed exception tree gives library users a way to handle failures
precisely (catch `AgentNotFoundError` separately from `ProviderError`,
distinguish retriable network issues from configuration errors, etc.).

All public AgentKeeper exceptions inherit from `AgentKeeperError`,
which itself inherits from the standard `Exception`. We never raise
bare exceptions across the public API.
"""

from __future__ import annotations


class AgentKeeperError(Exception):
    """Base class for every AgentKeeper-raised exception.

    Library users can `except AgentKeeperError:` to catch all
    AgentKeeper failures in one statement, while still being able to
    target subclasses for finer handling.
    """


# --- configuration & lookup -----------------------------------------


class ConfigurationError(AgentKeeperError):
    """A required setting (env var, provider name, model) is missing or invalid."""


class UnknownProviderError(ConfigurationError, ValueError):
    """The requested provider name has no registered factory.

    Also subclasses `ValueError` so existing code that catches
    `ValueError` keeps working (backward compat).
    """


class UnknownTierError(ConfigurationError, ValueError):
    """The requested memory tier name is not a valid MemoryTier.

    Also subclasses `ValueError` for backward compatibility.
    """


class AgentNotFoundError(AgentKeeperError, ValueError):
    """No agent with the given ID is present in storage.

    Also subclasses `ValueError` for backward compatibility.
    """


# --- runtime --------------------------------------------------------


class ProviderError(AgentKeeperError):
    """A real LLM provider failed in a way we cannot recover from.

    Includes the underlying provider name for diagnostics.
    """

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider


class RetriableProviderError(ProviderError):
    """A transient provider failure that may succeed on retry.

    Raised by adapters on rate-limit, timeout, or 5xx responses. The
    `with_retry` decorator catches this class specifically.
    """


class EmbeddingError(AgentKeeperError):
    """The embedding provider failed to vectorise a batch of texts."""


class CompressionError(AgentKeeperError):
    """The compression pipeline encountered an unrecoverable error."""
