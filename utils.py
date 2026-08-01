# utils.py
"""
Small shared helpers for the DeepSeek Telegram bot.
Keeps timezone-aware time logic in one place so every module agrees on "now".
"""

import asyncio
import contextlib
import logging
from datetime import datetime, timezone as _dt_timezone
from typing import AsyncIterator, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)

# Telegram clears the "typing" bubble ~5s after a sendChatAction call, but the
# bot needs 10-30s (classify -> LightRAG -> generate). Re-send a bit under that
# window so the indicator never visibly blinks off mid-thought.
_TYPING_REFRESH_SECONDS = 4.0

# Hard stop for the refresh loop. If generation somehow never finishes, we stop
# claiming to type rather than pulsing at Telegram forever.
_TYPING_MAX_SECONDS = 300.0

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


async def _typing_loop(bot, chat_id: int) -> None:
    """Re-send the typing action until cancelled (see start_typing/keep_typing)."""
    deadline = asyncio.get_running_loop().time() + _TYPING_MAX_SECONDS
    while True:
        try:
            await bot.send_chat_action(chat_id=chat_id, action="typing")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # A dropped indicator must never cost us the reply — it's cosmetic.
            # Keep looping: a single network blip shouldn't kill the bubble for
            # the rest of a 30-second generation.
            logger.debug("typing indicator refresh failed: %s", e)

        if asyncio.get_running_loop().time() >= deadline:
            logger.debug("typing indicator gave up after %.0fs", _TYPING_MAX_SECONDS)
            return
        await asyncio.sleep(_TYPING_REFRESH_SECONDS)


def start_typing(bot, chat_id: int) -> "asyncio.Task[None]":
    """
    Start showing "typing" in a chat until the returned task is cancelled.

    Use this when the start and stop points aren't a single block (e.g. typing
    begins inside a callback but must survive until the reply is sent). Always
    pair it with stop_typing() in a finally block. For a plain block, prefer
    keep_typing().
    """
    return asyncio.create_task(_typing_loop(bot, chat_id))


async def stop_typing(task: "Optional[asyncio.Task[None]]") -> None:
    """Cancel a start_typing() task and wait for it to unwind. Safe on None."""
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@contextlib.asynccontextmanager
async def keep_typing(bot, chat_id: int) -> AsyncIterator[None]:
    """
    Show "typing" in a chat for as long as the wrapped block runs.

    Telegram treats sendChatAction as a ~5 second hint, so a single call makes
    the bubble flash and vanish while the bot is still working. This keeps
    re-sending it in the background and stops the moment the block exits.

    Usage:
        async with keep_typing(context.bot, chat_id):
            answer = await slow_generation()
    """
    task = start_typing(bot, chat_id)
    try:
        yield
    finally:
        await stop_typing(task)
