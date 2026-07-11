# tests/test_retry.py
"""Unit tests for retry.py — no real sleeping, no real network."""

import asyncio

import pytest

from retry import DeepSeekHTTPError, is_retryable, retry_async, retry_sync


class Boom(Exception):
    """A plain exception retry.py's is_retryable() doesn't know about."""


def test_retry_sync_succeeds_first_try():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    sleeps = []
    result = retry_sync(fn, attempts=3, base_delay=0.1, sleep=sleeps.append)

    assert result == "ok"
    assert len(calls) == 1
    assert sleeps == []


def test_retry_sync_recovers_after_transient_failures():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise DeepSeekHTTPError(503, "unavailable")
        return "recovered"

    sleeps = []
    result = retry_sync(fn, attempts=3, base_delay=0.1, sleep=sleeps.append)

    assert result == "recovered"
    assert calls["n"] == 3
    assert len(sleeps) == 2  # slept before retry #2 and #3


def test_retry_sync_non_retryable_raises_immediately():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise DeepSeekHTTPError(400, "bad request")

    sleeps = []
    with pytest.raises(DeepSeekHTTPError) as exc_info:
        retry_sync(fn, attempts=3, base_delay=0.1, sleep=sleeps.append)

    assert exc_info.value.status == 400
    assert calls["n"] == 1  # no retries attempted
    assert sleeps == []


def test_retry_sync_exhausts_attempts_and_reraises_last_error():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise DeepSeekHTTPError(500, f"fail-{calls['n']}")

    sleeps = []
    with pytest.raises(DeepSeekHTTPError) as exc_info:
        retry_sync(fn, attempts=3, base_delay=0.1, sleep=sleeps.append)

    assert calls["n"] == 3
    assert "fail-3" in str(exc_info.value)  # last exception, not the first
    assert len(sleeps) == 2


def test_retry_sync_backoff_schedule_is_exponential_and_capped():
    def fn():
        raise DeepSeekHTTPError(500, "x")

    sleeps = []
    with pytest.raises(DeepSeekHTTPError):
        retry_sync(
            fn, attempts=4, base_delay=1.0, max_delay=3.0, jitter=0.0,
            sleep=sleeps.append,
        )

    # base*2**0, base*2**1, base*2**2 capped at max_delay=3.0
    assert sleeps == pytest.approx([1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    "status,expected",
    [(429, True), (500, True), (502, True), (503, True), (504, True),
     (400, False), (401, False), (403, False), (404, False), (422, False)],
)
def test_is_retryable_deepseek_http_error_by_status(status, expected):
    assert is_retryable(DeepSeekHTTPError(status, "body")) is expected


def test_is_retryable_generic_transient_errors():
    assert is_retryable(asyncio.TimeoutError()) is True
    assert is_retryable(TimeoutError()) is True
    assert is_retryable(ConnectionError()) is True


def test_is_retryable_unknown_exception_is_false():
    assert is_retryable(Boom("whatever")) is False


async def test_retry_async_recovers_after_transient_failure():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        if calls["n"] < 2:
            raise DeepSeekHTTPError(500, "flaky")
        return "async-ok"

    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    result = await retry_async(fn, attempts=2, base_delay=0.05, sleep=fake_sleep)

    assert result == "async-ok"
    assert calls["n"] == 2
    assert len(sleeps) == 1


async def test_retry_async_non_retryable_raises_immediately():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise DeepSeekHTTPError(401, "unauthorized")

    async def fake_sleep(delay):
        raise AssertionError("should not sleep on non-retryable error")

    with pytest.raises(DeepSeekHTTPError):
        await retry_async(fn, attempts=3, base_delay=0.05, sleep=fake_sleep)

    assert calls["n"] == 1
