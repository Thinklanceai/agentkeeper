"""Tests for TTL parsing and expiration arithmetic."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agentkeeper.errors import ConfigurationError
from agentkeeper.retention.ttl import (
    compute_expires_at,
    is_expired,
    parse_ttl,
)


class TestParseTTLPrimitives:
    def test_timedelta_passthrough(self) -> None:
        td = timedelta(days=30)
        assert parse_ttl(td) is td

    def test_int_as_seconds(self) -> None:
        assert parse_ttl(60) == timedelta(seconds=60)

    def test_float_as_seconds(self) -> None:
        assert parse_ttl(1.5) == timedelta(seconds=1.5)

    def test_negative_int_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            parse_ttl(-1)

    def test_negative_timedelta_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            parse_ttl(timedelta(seconds=-1))


class TestParseTTLShorthand:
    @pytest.mark.parametrize(
        "text,expected_seconds",
        [
            ("30s", 30),
            ("90m", 90 * 60),
            ("12h", 12 * 3600),
            ("7d", 7 * 86_400),
            ("2w", 14 * 86_400),
            ("1d2h", 86_400 + 2 * 3600),
            ("7d12h30m", 7 * 86_400 + 12 * 3600 + 30 * 60),
        ],
    )
    def test_shorthand_combinations(self, text: str, expected_seconds: int) -> None:
        assert parse_ttl(text) == timedelta(seconds=expected_seconds)

    def test_shorthand_case_insensitive(self) -> None:
        assert parse_ttl("30D") == timedelta(days=30)
        assert parse_ttl("12H") == timedelta(hours=12)

    def test_shorthand_garbage_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            parse_ttl("hello")

    def test_shorthand_zero_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            parse_ttl("0d")

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(ConfigurationError):
            parse_ttl("")


class TestParseTTLISO:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("P30D", timedelta(days=30)),
            ("PT12H", timedelta(hours=12)),
            ("PT15M", timedelta(minutes=15)),
            ("P7DT12H", timedelta(days=7, hours=12)),
            ("P1DT2H3M4S", timedelta(days=1, hours=2, minutes=3, seconds=4)),
        ],
    )
    def test_iso_durations(self, text: str, expected: timedelta) -> None:
        assert parse_ttl(text) == expected


class TestComputeExpiresAt:
    def test_returns_none_for_none(self) -> None:
        assert compute_expires_at(None) is None

    def test_iso_format(self) -> None:
        base = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
        result = compute_expires_at("7d", base=base)
        assert result is not None
        # Should land 7 days later, parseable back
        parsed = datetime.fromisoformat(result)
        assert (parsed - base) == timedelta(days=7)

    def test_invalid_ttl_propagates(self) -> None:
        with pytest.raises(ConfigurationError):
            compute_expires_at("never")


class TestIsExpired:
    def test_none_never_expired(self) -> None:
        assert is_expired(None) is False
        assert is_expired("") is False

    def test_past_timestamp_expired(self) -> None:
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        assert is_expired(past) is True

    def test_future_timestamp_not_expired(self) -> None:
        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        assert is_expired(future) is False

    def test_malformed_timestamp_treated_as_not_expired(self) -> None:
        # Defensive: never crash on garbage
        assert is_expired("not-a-date") is False

    def test_naive_datetime_assumed_utc(self) -> None:
        # Strings without tzinfo should still work
        past = (
            datetime.now(timezone.utc).replace(tzinfo=None)
            - timedelta(days=1)
        ).isoformat()
        assert is_expired(past) is True
