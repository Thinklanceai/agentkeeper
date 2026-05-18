"""Retry helpers for transient provider failures.

Provides two decorators:

- `with_retry(fn)`        — wraps a sync function
- `with_async_retry(fn)`  — wraps an `async def`

Both catch `RetriableProviderError` and retry up to `max_attempts`
times with exponential backoff plus jitter, then re-raise the last
error. Non-retriable exceptions propagate immediately.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import TypeVar

from .errors import RetriableProviderError
from .logging import get_logger

T = TypeVar("T")
A = TypeVar("A")

_log = get_logger(__name__)


def _backoff_seconds(
    attempt: int, base: float, cap: float, jitter: float
) -> float:
    """Compute a backoff delay with bounded exponential growth + jitter."""
    raw = min(cap, base * (2 ** max(0, attempt - 1)))
    return raw + random.random() * jitter


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    jitter: float = 0.25,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry a sync function on `RetriableProviderError`.

    Other exceptions propagate immediately. After the final failed
    attempt, the original exception is re-raised.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> T:
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except RetriableProviderError as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    delay = _backoff_seconds(
                        attempt, base_delay, max_delay, jitter
                    )
                    _log.warning(
                        "retriable error from %s (attempt %d/%d): %s; "
                        "retrying in %.2fs",
                        getattr(exc, "provider", "?"),
                        attempt,
                        max_attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            assert last_exc is not None  # for type checkers
            raise last_exc

        return wrapper

    return decorator


def with_async_retry(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    jitter: float = 0.25,
) -> Callable[
    [Callable[..., Awaitable[A]]], Callable[..., Awaitable[A]]
]:
    """Retry an async function on `RetriableProviderError`."""

    def decorator(
        fn: Callable[..., Awaitable[A]],
    ) -> Callable[..., Awaitable[A]]:
        @wraps(fn)
        async def wrapper(*args: object, **kwargs: object) -> A:
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except RetriableProviderError as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    delay = _backoff_seconds(
                        attempt, base_delay, max_delay, jitter
                    )
                    _log.warning(
                        "retriable error from %s (attempt %d/%d): %s; "
                        "retrying in %.2fs",
                        getattr(exc, "provider", "?"),
                        attempt,
                        max_attempts,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator
