"""Structured logging for AgentKeeper.

AgentKeeper uses Python's standard `logging` module under the
`agentkeeper.*` namespace. By default the library installs a single
`NullHandler` on the root namespace so that applications see no output
until they opt in.

Application code can:

    import logging
    logging.getLogger("agentkeeper").setLevel(logging.INFO)
    logging.getLogger("agentkeeper").addHandler(logging.StreamHandler())

…or, for JSON output, attach any structured-logging handler of their
choice. We do **not** ship a JSON formatter to keep the dependency
graph minimal — `structlog` and `loguru` integrations are documented
in the README.
"""

from __future__ import annotations

import logging

_ROOT_LOGGER_NAME = "agentkeeper"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a namespaced logger.

    Example::

        log = get_logger(__name__)
        log.info("compressed %d facts", n)
    """
    if name is None or name == _ROOT_LOGGER_NAME:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    if name.startswith(_ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


# Install a NullHandler on the root namespace so library users never see
# "No handlers could be found" warnings.
logging.getLogger(_ROOT_LOGGER_NAME).addHandler(logging.NullHandler())
