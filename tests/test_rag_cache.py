# tests/test_rag_cache.py
"""
Tests for rag_cache.py and its Brain integration.

The cache is deliberately conservative: a wrong reuse silently feeds the bot
an unrelated context, so every ambiguous case must be a miss. The Brain test
also proves the actual saving — a follow-up in the same thread must not call
LightRAG twice.
"""

import pytest

import llm_health
from brain import Brain
from conversation_analyzer import ClassificationResult, ConversationAnalyzer
from rag_cache import RagCache


def _put(cache, **overrides):
    kwargs = dict(
        chat_id=100,
        author="Вася",
        user_id=1,
        message_text="что известно про Darklin?",
        reply_text=None,
        query="Darklin семья",
        facts="FACTS",
        now=1000.0,
    )
    kwargs.update(overrides)
    cache.put(**kwargs)


def _get(cache, **overrides):
    kwargs = dict(
        chat_id=100,
        author="Вася",
        user_id=1,
        message_text="а что по семье?",
        reply_text=None,
        query="Darklin семья кто такой",
        now=1010.0,
    )
    kwargs.update(overrides)
    return cache.get(**kwargs)


class TestRagCache:
    def test_same_author_overlapping_query_is_a_hit(self):
        cache = RagCache()
        _put(cache)
        assert _get(cache) == "FACTS"
        assert cache.hits == 1

    def test_different_author_without_thread_is_a_miss(self):
        cache = RagCache()
        _put(cache)
        assert _get(cache, author="Петя", user_id=2) is None

    def test_reply_to_the_cached_message_is_a_hit(self):
        cache = RagCache()
        _put(cache)
        assert _get(
            cache, author="Петя", user_id=2,
            reply_text="что известно про Darklin?",
        ) == "FACTS"

    def test_both_messages_replying_to_the_same_parent_is_a_hit(self):
        cache = RagCache()
        _put(cache, reply_text="родительское сообщение")
        assert _get(
            cache, author="Петя", user_id=2,
            reply_text="родительское сообщение",
        ) == "FACTS"

    def test_expired_entry_is_a_miss_and_is_evicted(self):
        cache = RagCache(ttl_seconds=120)
        _put(cache, now=1000.0)
        assert _get(cache, now=1120.1) is None
        assert len(cache) == 0

    def test_weak_overlap_is_a_miss(self):
        cache = RagCache()
        _put(cache, query="Darklin семья")
        # Only "Darklin" is shared, and it is everywhere in the graph — not
        # enough to assume the follow-up is about the same facts.
        assert _get(cache, query="Darklin работа") is None

    def test_exact_one_word_query_is_a_hit(self):
        cache = RagCache()
        _put(cache, query="кунипас")
        assert _get(cache, query="кунипас") == "FACTS"

    def test_different_chats_do_not_share_entries(self):
        cache = RagCache()
        _put(cache)
        assert _get(cache, chat_id=200) is None

    def test_empty_facts_are_not_cached(self):
        cache = RagCache()
        _put(cache, facts="")
        assert _get(cache) is None
        assert len(cache) == 0

    def test_put_replaces_the_previous_entry(self):
        cache = RagCache()
        _put(cache, query="Darklin семья", facts="OLD")
        _put(cache, query="сборка ПК", facts="NEW")
        assert _get(cache, query="Darklin семья") is None
        assert _get(cache, query="сборка ПК") == "NEW"

    def test_invalidate_drops_the_chat(self):
        cache = RagCache()
        _put(cache)
        cache.invalidate(100)
        assert _get(cache) is None


class _CountingRagClient:
    query_mode = "mix"
    query_top_k = 20
    last_error = None

    def __init__(self):
        self.calls = 0

    async def retrieve(self, query):
        self.calls += 1
        return "FACTS: " + "x" * 600


def _fake_classifier(query="Darklin семья"):
    async def fake_classify(self, msg_ctx, recent_messages):
        return ClassificationResult(
            grade=2, needs_memory=True, rag_query=query, reason="test"
        )

    return fake_classify


async def _ask(brain, **overrides):
    kwargs = dict(
        message_text="что известно про семью Darklin?",
        author="Вася",
        recent_messages=[],
        user_id=1,
        chat_id=100,
    )
    kwargs.update(overrides)
    return await brain.analyze_and_respond(**kwargs)


@pytest.mark.asyncio
async def test_brain_reuses_context_for_a_follow_up(bot_config, monkeypatch):
    monkeypatch.setattr(llm_health, "_exhausted_until", 0.0)
    monkeypatch.setattr(ConversationAnalyzer, "classify", _fake_classifier())

    b = Brain(bot_config)
    rag = _CountingRagClient()
    b._rag_client = rag
    seen = []

    def fake_generate(**kwargs):
        seen.append(kwargs["rag_facts"])
        return "ok"

    monkeypatch.setattr(b, "generate_response", fake_generate)

    await _ask(b)
    await _ask(b, message_text="а что ещё про семью?")

    assert rag.calls == 1
    assert seen[0] == seen[1]
    assert b._rag_cache.hits == 1


@pytest.mark.asyncio
async def test_brain_does_not_reuse_across_chats(bot_config, monkeypatch):
    monkeypatch.setattr(llm_health, "_exhausted_until", 0.0)
    monkeypatch.setattr(ConversationAnalyzer, "classify", _fake_classifier())

    b = Brain(bot_config)
    rag = _CountingRagClient()
    b._rag_client = rag
    monkeypatch.setattr(b, "generate_response", lambda **kwargs: "ok")

    await _ask(b)
    await _ask(b, chat_id=200)

    assert rag.calls == 2


@pytest.mark.asyncio
async def test_brain_does_not_reuse_for_another_author(bot_config, monkeypatch):
    monkeypatch.setattr(llm_health, "_exhausted_until", 0.0)
    monkeypatch.setattr(ConversationAnalyzer, "classify", _fake_classifier())

    b = Brain(bot_config)
    rag = _CountingRagClient()
    b._rag_client = rag
    monkeypatch.setattr(b, "generate_response", lambda **kwargs: "ok")

    await _ask(b)
    await _ask(b, author="Петя", user_id=2, message_text="а что по семье?")

    assert rag.calls == 2


@pytest.mark.asyncio
async def test_cache_can_be_disabled(bot_config, monkeypatch):
    monkeypatch.setattr(llm_health, "_exhausted_until", 0.0)
    monkeypatch.setattr(ConversationAnalyzer, "classify", _fake_classifier())
    bot_config.rag_cache_enabled = False

    b = Brain(bot_config)
    rag = _CountingRagClient()
    b._rag_client = rag
    monkeypatch.setattr(b, "generate_response", lambda **kwargs: "ok")

    await _ask(b)
    await _ask(b, message_text="а что ещё про семью?")

    assert rag.calls == 2
    assert b._rag_cache is None


@pytest.mark.asyncio
async def test_short_context_is_not_cached(bot_config, monkeypatch):
    monkeypatch.setattr(llm_health, "_exhausted_until", 0.0)
    monkeypatch.setattr(ConversationAnalyzer, "classify", _fake_classifier())

    class _ShortRagClient:
        query_mode = "mix"
        query_top_k = 20
        last_error = None

        def __init__(self):
            self.calls = 0

        async def retrieve(self, query):
            self.calls += 1
            return "Knowledge Graph Data"

    b = Brain(bot_config)
    rag = _ShortRagClient()
    b._rag_client = rag
    monkeypatch.setattr(b, "generate_response", lambda **kwargs: "ok")

    await _ask(b)
    await _ask(b, message_text="а что ещё про семью?")

    assert rag.calls == 2
    assert len(b._rag_cache) == 0
