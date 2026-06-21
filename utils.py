# utils.py
"""
Small shared helpers for the DeepSeek Telegram bot.
Keeps timezone-aware time logic in one place so every module agrees on "now".
"""

import logging
from datetime import datetime, timezone as _dt_timezone
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)

# Cached zone objects (ZoneInfo caches internally, but we cache the fallback
# result too so a bad/missing timezone never crashes the bot).
_zone_cache: dict = {}

# Common deprecated aliases -> current IANA names (tzdata ships these as links,
# but we normalize anyway so lookups work even on minimal installs).
_TZ_ALIASES = {
    "europe/kiev": "Europe/Kyiv",
    "kiev": "Europe/Kyiv",
}


def _normalize(timezone: str) -> str:
    tz = timezone.strip()
    return _TZ_ALIASES.get(tz.lower(), tz)


def get_zone(timezone: str):
    """
    Get a tzinfo object for a timezone name, caching the result.

    Falls back gracefully: invalid name -> UTC (ZoneInfo); if even UTC ZoneInfo
    is unavailable (no tzdata and no system TZ database, e.g. a bare Windows
    install) -> a stdlib fixed-offset UTC. Never raises.

    Args:
        timezone: IANA timezone name (e.g. "Europe/Kyiv", "UTC")

    Returns:
        A tzinfo instance (ZoneInfo when available, otherwise datetime.timezone.utc)
    """
    tz = _normalize(timezone)
    if tz in _zone_cache:
        return _zone_cache[tz]

    zone = None
    try:
        zone = ZoneInfo(tz)
    except ZoneInfoNotFoundError:
        logger.warning(
            f"Unknown timezone '{timezone}', falling back to UTC. "
            "Install the 'tzdata' package for full IANA support."
        )
        try:
            zone = ZoneInfo("UTC")
        except ZoneInfoNotFoundError:
            zone = _dt_timezone.utc

    _zone_cache[tz] = zone
    return zone


def get_now(timezone: str = "UTC") -> datetime:
    """
    Get the current time as a timezone-aware datetime.

    Args:
        timezone: IANA timezone name (default: UTC)

    Returns:
        Timezone-aware datetime for "now"
    """
    return datetime.now(get_zone(timezone))


def to_aware(dt: datetime, timezone: str = "UTC") -> datetime:
    """
    Ensure a datetime is timezone-aware.

    Naive datetimes are assumed to already be in the given timezone and are
    attached to it. Aware datetimes are converted to the given timezone.

    Args:
        dt: Datetime to normalize (naive or aware)
        timezone: IANA timezone name to use/convert to

    Returns:
        Timezone-aware datetime
    """
    zone = get_zone(timezone)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=zone)
    return dt.astimezone(zone)
