# tests/test_responder_silent_reaction.py
"""
Unit tests for Responder.send_silent_reaction() — sets a Telegram reaction
with no text fallback on failure (unlike _send_reaction()'s ambiguous
reaction-vs-text heuristic). Message.set_reaction is faked, no real
Telegram/network calls.
"""

from unittest.mock import AsyncMock

import pytest

from responder import Responder


class _FakeMessage:
    def __init__(self, set_reaction=None):
        self.set_reaction = set_reaction or AsyncMock()


@pytest.mark.asyncio
async def test_send_silent_reaction_success(bot_config):
    responder = Responder(bot_config)
    message = _FakeMessage()

    ok = await responder.send_silent_reaction(message, "😂")

    assert ok is True
    message.set_reaction.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_silent_reaction_failure_returns_false_no_text_sent(bot_config):
    responder = Responder(bot_config)
    message = _FakeMessage(set_reaction=AsyncMock(side_effect=RuntimeError("boom")))

    ok = await responder.send_silent_reaction(message, "😂")

    # Must NOT fall back to sending text — the fake message has no
    # reply_text at all, so any text-fallback attempt would raise here.
    assert ok is False
