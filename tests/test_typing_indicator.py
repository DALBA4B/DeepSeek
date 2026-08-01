# tests/test_typing_indicator.py
"""
Tests for the "typing" indicator helpers in utils.py.

Telegram clears the typing bubble ~5s after sendChatAction, while a full
generation takes 10-30s. These tests pin the behaviour that keeps the bubble
alive for the whole wait: repeat while working, stop immediately after, and
never let an indicator failure break the reply.
"""

import asyncio

import pytest

import utils
from utils import keep_typing, start_typing, stop_typing


class FakeBot:
    """Records every send_chat_action call."""

    def __init__(self, fail_times: int = 0) -> None:
        self.calls: list = []
        self._fail_times = fail_times

    async def send_chat_action(self, chat_id: int, action: str) -> None:
        self.calls.append((chat_id, action))
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError("network blip")


@pytest.fixture
def fast_refresh(monkeypatch):
    """Shrink the refresh interval so tests don't wait real seconds."""
    monkeypatch.setattr(utils, "_TYPING_REFRESH_SECONDS", 0.01)
    return 0.01


@pytest.mark.asyncio
async def test_sends_immediately_without_waiting(fast_refresh):
    """The bubble must appear at once, not one refresh interval later."""
    bot = FakeBot()
    async with keep_typing(bot, chat_id=42):
        await asyncio.sleep(0)  # let the task start, but no refresh yet
    assert bot.calls, "typing should be sent before the first sleep"
    assert bot.calls[0] == (42, "typing")


@pytest.mark.asyncio
async def test_repeats_while_work_is_slow(fast_refresh):
    """A long generation should produce several refreshes, not one."""
    bot = FakeBot()
    async with keep_typing(bot, chat_id=1):
        await asyncio.sleep(0.1)
    # 0.1s at a 0.01s interval — expect many, assert conservatively to stay
    # robust on a loaded CI machine.
    assert len(bot.calls) >= 3


@pytest.mark.asyncio
async def test_stops_after_block_exits(fast_refresh):
    """No further refreshes once the reply has been sent."""
    bot = FakeBot()
    async with keep_typing(bot, chat_id=1):
        await asyncio.sleep(0.03)
    count_at_exit = len(bot.calls)
    await asyncio.sleep(0.05)
    assert len(bot.calls) == count_at_exit, "indicator kept running after exit"


@pytest.mark.asyncio
async def test_stops_when_block_raises(fast_refresh):
    """An exception mid-generation must not leak the background task."""
    bot = FakeBot()
    with pytest.raises(ValueError):
        async with keep_typing(bot, chat_id=1):
            await asyncio.sleep(0.02)
            raise ValueError("generation failed")
    count_at_exit = len(bot.calls)
    await asyncio.sleep(0.05)
    assert len(bot.calls) == count_at_exit


@pytest.mark.asyncio
async def test_survives_send_failures(fast_refresh):
    """A failing sendChatAction must not kill the loop or the caller."""
    bot = FakeBot(fail_times=2)
    async with keep_typing(bot, chat_id=1):
        await asyncio.sleep(0.06)
    # First two raised; the loop must have kept going past them.
    assert len(bot.calls) > 2


@pytest.mark.asyncio
async def test_start_and_stop_are_separable(fast_refresh):
    """
    main.py starts typing inside a callback and stops it after sending, so the
    two halves must work independently of a `with` block.
    """
    bot = FakeBot()
    task = start_typing(bot, chat_id=7)
    await asyncio.sleep(0.03)
    await stop_typing(task)
    count = len(bot.calls)
    assert count >= 2
    await asyncio.sleep(0.03)
    assert len(bot.calls) == count
    assert task.cancelled() or task.done()


@pytest.mark.asyncio
async def test_stop_typing_tolerates_none():
    """Bot stayed silent -> no task was ever started."""
    await stop_typing(None)


@pytest.mark.asyncio
async def test_gives_up_after_max_seconds(monkeypatch):
    """The loop must not pulse at Telegram forever if generation hangs."""
    monkeypatch.setattr(utils, "_TYPING_REFRESH_SECONDS", 0.01)
    monkeypatch.setattr(utils, "_TYPING_MAX_SECONDS", 0.0)
    bot = FakeBot()
    task = start_typing(bot, chat_id=1)
    await asyncio.wait_for(task, timeout=1.0)
    assert len(bot.calls) == 1, "should send once, then hit the deadline"
