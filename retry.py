# retry.py
"""
Small hand-rolled retry/backoff helpers for the DeepSeek API calls.

Why not a library (tenacity/backoff)
-------------------------------------
The codebase already hand-rolls comparable small utilities (GrudgeTracker,
RecentResponseTracker, TaskScheduler) instead of pulling in dependencies for
them. The need here is narrow — two call sites, one sync one async, simple
exponential backoff with jitter, and a small allowlist of retryable
exceptions — so owning `is_retryable()` directly keeps the "never retry a
400/auth error" requirement explicit and easy to unit test, rather than
buried in a decorator's configuration.
"""

import asyncio
import logging
import random
import time
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Transient HTTP status codes worth retrying (rate limit + server errors).
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class DeepSeekHTTPError(RuntimeError):
    """
    Raised by ConversationAnalyzer._call_llm() on a non-2xx response.

    Carries the numeric status separately from the message so is_retryable()
    doesn't have to string-sniff the exception text.
    """

    def __init__(self, status: int, body: str):
        self.status = status
        super().__init__(f"DeepSeek HTTP {status}: {body}")


def is_retryable(exc: Exception) -> bool:
    """
    True only for transient failures — timeouts, connection errors, 429, 5xx.
    False (don't retry) for anything that indicates a bad request/auth/etc,
    since retrying those just wastes time and will fail again identically.
    """
    if isinstance(exc, DeepSeekHTTPError):
        return exc.status in RETRYABLE_STATUS

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True

    try:
        import aiohttp

        if isinstance(exc, aiohttp.ClientError):
            return True
    except ImportError:
        pass

    try:
        import openai

        if isinstance(
            exc,
            (
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
                openai.RateLimitError,
            ),
        ):
            return True
        if isinstance(exc, openai.APIStatusError):
            return exc.status_code in RETRYABLE_STATUS
    except ImportError:
        pass

    return False


def _backoff_delay(attempt: int, base_delay: float, max_delay: float, jitter: float) -> float:
    """attempt is 1-indexed (delay before retrying attempt N+1)."""
    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
    return delay + random.uniform(0, jitter)


def retry_sync(
    fn: Callable[[], T],
    *,
    attempts: int,
    base_delay: float,
    max_delay: float = 4.0,
    jitter: float = 0.3,
    sleep: Callable[[float], None] = time.sleep,
    is_retryable: Callable[[Exception], bool] = is_retryable,
) -> T:
    """
    Call fn(), retrying on transient failures up to `attempts` total tries.

    Re-raises the last exception if every attempt fails, or immediately on a
    non-retryable exception — callers keep their existing except/fallback
    logic unchanged.
    """
    last_exc: Exception = RuntimeError("retry_sync called with attempts < 1")
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - re-raised below when done
            last_exc = e
            if attempt >= attempts or not is_retryable(e):
                raise
            delay = _backoff_delay(attempt, base_delay, max_delay, jitter)
            logger.warning(
                "retry_sync: attempt %d/%d failed (%s), retrying in %.2fs",
                attempt, attempts, e, delay,
            )
            sleep(delay)
    raise last_exc  # pragma: no cover - loop always returns or raises above


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    base_delay: float,
    max_delay: float = 4.0,
    jitter: float = 0.3,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    is_retryable: Callable[[Exception], bool] = is_retryable,
) -> T:
    """Async twin of retry_sync() — see its docstring."""
    last_exc: Exception = RuntimeError("retry_async called with attempts < 1")
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except Exception as e:  # noqa: BLE001 - re-raised below when done
            last_exc = e
            if attempt >= attempts or not is_retryable(e):
                raise
            delay = _backoff_delay(attempt, base_delay, max_delay, jitter)
            logger.warning(
                "retry_async: attempt %d/%d failed (%s), retrying in %.2fs",
                attempt, attempts, e, delay,
            )
            await sleep(delay)
    raise last_exc  # pragma: no cover - loop always returns or raises above
