"""TTL (time-to-live) parsing and expiration arithmetic.

AgentKeeper accepts TTLs as either:

- `datetime.timedelta`
- ISO-8601 duration strings (`PT30M`, `P7D`, `P30DT12H`)
- shorthand strings (`"30d"`, `"12h"`, `"90m"`, `"45s"`, `"7d12h"`)

Shorthand parsing is intentionally permissive — most user-facing TTLs
are written casually. Combinations like `"7d12h30m"` are supported.

This module is pure: no I/O, no datetime mutation, no side effects.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from ..errors import ConfigurationError

# Supported units → seconds. Order matters for the regex below.
_UNIT_SECONDS: dict[str, int] = {
    "w": 7 * 86_400,
    "d": 86_400,
    "h": 3_600,
    "m": 60,
    "s": 1,
}

# Match each "number+unit" pair in a shorthand string.
_SHORTHAND_PAIR = re.compile(r"(\d+)\s*([wdhms])", re.IGNORECASE)

# ISO-8601 duration: simplified to days/hours/minutes/seconds. We don't
# support months/years on purpose — they're ambiguous (28 vs 31 days).
_ISO_DURATION = re.compile(
    r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$",
    re.IGNORECASE,
)


def parse_ttl(value: timedelta | str | int | float) -> timedelta:
    """Convert a user-supplied TTL into a `timedelta`.

    Args:
        value: One of:
            - `timedelta` (returned as-is, modulo a non-negative check)
            - `int`/`float`: interpreted as seconds
            - `str`: ISO-8601 duration or shorthand (`"30d"`, `"12h"`,
              `"7d12h"`, `"90m"`, `"PT15M"`, ...)

    Returns:
        A non-negative `timedelta`.

    Raises:
        ConfigurationError: if the string is unparseable or the
            resulting duration is negative.
    """
    if isinstance(value, timedelta):
        if value.total_seconds() < 0:
            raise ConfigurationError(
                f"TTL must be non-negative; got {value}"
            )
        return value

    if isinstance(value, (int, float)):
        if value < 0:
            raise ConfigurationError(f"TTL must be non-negative; got {value}")
        return timedelta(seconds=float(value))

    if isinstance(value, str):
        return _parse_ttl_str(value)

    raise ConfigurationError(
        f"TTL must be timedelta, int, float, or str; got {type(value).__name__}"
    )


def _parse_ttl_str(value: str) -> timedelta:
    text = value.strip()
    if not text:
        raise ConfigurationError("TTL string is empty")

    # ISO-8601 duration?
    iso = _ISO_DURATION.match(text)
    if iso:
        days, hours, minutes, seconds = (
            int(g) if g else 0 for g in iso.groups()
        )
        td = timedelta(
            days=days, hours=hours, minutes=minutes, seconds=seconds
        )
        if td.total_seconds() == 0:
            raise ConfigurationError(
                f"TTL {value!r} parses to zero duration"
            )
        return td

    # Shorthand: 30d, 12h, 7d12h30m, ...
    total = 0
    matched_any = False
    cursor = 0
    for match in _SHORTHAND_PAIR.finditer(text):
        if match.start() != cursor:
            raise ConfigurationError(
                f"Unexpected characters in TTL {value!r} at position {cursor}"
            )
        amount = int(match.group(1))
        unit = match.group(2).lower()
        total += amount * _UNIT_SECONDS[unit]
        cursor = match.end()
        matched_any = True

    if not matched_any or cursor != len(text):
        raise ConfigurationError(
            f"Could not parse TTL {value!r}. "
            "Use shorthand like '30d', '12h', '7d12h' or ISO 'P30D' / 'PT12H'."
        )
    if total == 0:
        raise ConfigurationError(f"TTL {value!r} parses to zero duration")
    return timedelta(seconds=total)


def compute_expires_at(
    ttl: timedelta | str | int | float | None,
    *,
    base: datetime | None = None,
) -> str | None:
    """Compute the absolute expiration timestamp from a TTL.

    Returns `None` if `ttl` is `None` (meaning: no expiration).
    Otherwise returns an ISO-8601 UTC timestamp string.
    """
    if ttl is None:
        return None
    duration = parse_ttl(ttl)
    moment = base or datetime.now(timezone.utc)
    return (moment + duration).isoformat()


def is_expired(expires_at: str | None, now: datetime | None = None) -> bool:
    """Return True if the ISO timestamp `expires_at` is in the past."""
    if not expires_at:
        return False
    try:
        dt = datetime.fromisoformat(expires_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return False
    moment = now or datetime.now(timezone.utc)
    return dt <= moment
