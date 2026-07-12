# tests/test_responder_reaction_repeat.py
"""
Integration test: Responder._send_reaction() actually consults
RecentReactionTracker and falls back to text once an emoji is on a maxed-out
streak, EXCEPT when the user explicitly asked for a reaction (their call,
not ours to override). Message.set_reaction/reply_text are faked.
"""

from unittest.mock import AsyncMock

import pytest

import responder as responder_module
from responder import Responder


class _FakeMessage:
    def __init__(self, text=""):
        self.text = text
        self.set_reaction = AsyncMock()
        self.reply_text = AsyncMock()


@pytest.mark.asyncio
async def test_reaction_falls_back_to_text_after_maxed_streak(bot_config, monkeypatch):
    # Force the ambiguous 50/50 branch to always pick "use reaction" so the
    # anti-repeat guard is the only thing that can flip it to text.
    monkeypatch.setattr(responder_module.random, "choice", lambda seq: True)

    responder = Responder(bot_config)
    message = _FakeMessage(text="ору с этого")  # no REACTION_KEYWORDS/TEXT_KEYWORDS

    for _ in range(4):
        ok = await responder._send_reaction(message, "🤣")
        assert ok is True
        message.set_reaction.assert_awaited()
        message.set_reaction.reset_mock()

    # 5th identical reaction in a row must fall back to text instead.
    ok = await responder._send_reaction(message, "🤣")
    assert ok is True
    message.set_reaction.assert_not_awaited()
    message.reply_text.assert_awaited_once_with("🤣")


@pytest.mark.asyncio
async def test_explicit_reaction_request_bypasses_repeat_guard(bot_config):
    responder = Responder(bot_config)
    message = _FakeMessage(text="поставь реакцию")  # matches REACTION_KEYWORDS

    for _ in range(6):
        ok = await responder._send_reaction(message, "🤣")
        assert ok is True
        message.set_reaction.assert_awaited()
        message.set_reaction.reset_mock()
