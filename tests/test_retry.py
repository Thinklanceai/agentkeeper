"""Tests for retry decorators."""

from __future__ import annotations

import asyncio
import time

import pytest

from agentkeeper.errors import ProviderError, RetriableProviderError
from agentkeeper.retry import with_async_retry, with_retry


class TestSyncRetry:
    def test_succeeds_on_first_try(self) -> None:
        calls = {"n": 0}

        @with_retry(max_attempts=3, base_delay=0.01, max_delay=0.01, jitter=0)
        def fn() -> str:
            calls["n"] += 1
            return "ok"

        assert fn() == "ok"
        assert calls["n"] == 1

    def test_retries_on_retriable(self) -> None:
        calls = {"n": 0}

        @with_retry(max_attempts=3, base_delay=0.01, max_delay=0.01, jitter=0)
        def fn() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise RetriableProviderError("test", "transient")
            return "recovered"

        assert fn() == "recovered"
        assert calls["n"] == 3

    def test_re_raises_after_max_attempts(self) -> None:
        @with_retry(max_attempts=2, base_delay=0.01, max_delay=0.01, jitter=0)
        def fn() -> str:
            raise RetriableProviderError("test", "down")

        with pytest.raises(RetriableProviderError):
            fn()

    def test_non_retriable_propagates_immediately(self) -> None:
        calls = {"n": 0}

        @with_retry(max_attempts=5, base_delay=0.01, max_delay=0.01, jitter=0)
        def fn() -> str:
            calls["n"] += 1
            raise ProviderError("test", "boom")

        with pytest.raises(ProviderError):
            fn()
        assert calls["n"] == 1

    def test_backoff_is_bounded(self) -> None:
        """Backoff should never exceed max_delay even at high attempt counts."""
        start = time.monotonic()

        @with_retry(max_attempts=2, base_delay=0.05, max_delay=0.05, jitter=0)
        def fn() -> str:
            raise RetriableProviderError("test", "boom")

        with pytest.raises(RetriableProviderError):
            fn()
        elapsed = time.monotonic() - start
        # max one sleep of ~0.05s between the two attempts
        assert elapsed < 0.5


class TestAsyncRetry:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self) -> None:
        @with_async_retry(max_attempts=3, base_delay=0.01, max_delay=0.01, jitter=0)
        async def fn() -> str:
            return "ok"

        assert await fn() == "ok"

    @pytest.mark.asyncio
    async def test_retries_on_retriable(self) -> None:
        calls = {"n": 0}

        @with_async_retry(max_attempts=3, base_delay=0.01, max_delay=0.01, jitter=0)
        async def fn() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise RetriableProviderError("test", "transient")
            return "recovered"

        assert await fn() == "recovered"
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_non_retriable_propagates(self) -> None:
        @with_async_retry(max_attempts=5, base_delay=0.01, max_delay=0.01, jitter=0)
        async def fn() -> str:
            raise ProviderError("test", "boom")

        with pytest.raises(ProviderError):
            await fn()

    def test_async_retry_runs_in_event_loop(self) -> None:
        # Smoke test: the async retry works inside asyncio.run
        @with_async_retry(max_attempts=2, base_delay=0.01, max_delay=0.01, jitter=0)
        async def fn() -> int:
            return 42

        assert asyncio.run(fn()) == 42
