# tests/test_silent_reaction.py
"""
Unit tests for the "react instead of full silence" path (grade 0):
brain._silent_reaction_pool() and analyze_and_respond()'s decision to
return a SILENT_REACT: sentinel (comma-joined candidate pool) instead of
None. Pure logic — the DeepSeek classifier call is monkeypatched, no real
network involved.
"""

import pytest

import brain as brain_module
from brain import Brain, SILENT_REACT_PREFIX, _silent_reaction_pool
from conversation_analyzer import ClassificationResult


def test_silent_reaction_pool_joke_uses_joke_set():
    assert set(_silent_reaction_pool("joke")) == {"🤣", "😁", "🎉", "😱"}


def test_silent_reaction_pool_tease_uses_tease_set():
    assert set(_silent_reaction_pool("tease")) == {"👀", "🤨", "😈"}


def test_silent_reaction_pool_unknown_situation_uses_default():
    assert set(_silent_reaction_pool("nonexistent")) == {"👍", "🤣", "🔥"}


@pytest.mark.asyncio
async def test_grade_zero_with_probability_one_returns_silent_react(bot_config, monkeypatch):
    bot_config.silent_reaction_probability = 1.0

    async def fake_classify(self, msg_ctx, recent_messages):
        return ClassificationResult(
            grade=0, needs_memory=False, rag_query=None, reason="flood", situation="joke"
        )

    monkeypatch.setattr(
        brain_module.ConversationAnalyzer, "classify", fake_classify, raising=True
    )

    b = Brain(bot_config)
    result = await b.analyze_and_respond(
        message_text="хаха ору", author="Вася", recent_messages=[]
    )

    assert result is not None
    assert result.startswith(SILENT_REACT_PREFIX)
    pool = result[len(SILENT_REACT_PREFIX):].split(",")
    assert set(pool) == {"🤣", "😁", "🎉", "😱"}


@pytest.mark.asyncio
async def test_grade_zero_with_probability_zero_stays_fully_silent(bot_config, monkeypatch):
    bot_config.silent_reaction_probability = 0.0

    async def fake_classify(self, msg_ctx, recent_messages):
        return ClassificationResult(
            grade=0, needs_memory=False, rag_query=None, reason="flood", situation="joke"
        )

    monkeypatch.setattr(
        brain_module.ConversationAnalyzer, "classify", fake_classify, raising=True
    )

    b = Brain(bot_config)
    result = await b.analyze_and_respond(
        message_text="хаха ору", author="Вася", recent_messages=[]
    )

    assert result is None


@pytest.mark.asyncio
async def test_grade_zero_silent_reaction_disabled_in_text_only_mode(bot_config, monkeypatch):
    bot_config.text_only_mode = True
    bot_config.silent_reaction_probability = 1.0

    async def fake_classify(self, msg_ctx, recent_messages):
        return ClassificationResult(
            grade=0, needs_memory=False, rag_query=None, reason="flood", situation="joke"
        )

    monkeypatch.setattr(
        brain_module.ConversationAnalyzer, "classify", fake_classify, raising=True
    )

    b = Brain(bot_config)
    result = await b.analyze_and_respond(
        message_text="хаха ору", author="Вася", recent_messages=[]
    )

    assert result is None
