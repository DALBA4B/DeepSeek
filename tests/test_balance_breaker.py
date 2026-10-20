# tests/test_balance_breaker.py
"""
Pipeline tests for the DeepSeek 402 breaker:
* armed breaker → analyze_and_respond() skips classification AND LightRAG;
* a 402 from the classifier or the generator arms the breaker.
"""

from types import SimpleNamespace

import pytest

import brain as brain_module
import llm_health
from brain import Brain
from prompts import FALLBACK_RESPONSES
from retry import DeepSeekHTTPError


class _FakeRagClient:
    def __init__(self):
        self.calls = 0

    async def retrieve(self, query):
        self.calls += 1
        return "факты"


@pytest.mark.asyncio
async def test_armed_breaker_skips_classification_and_retrieval(bot_config, monkeypatch):
    monkeypatch.setattr(llm_health, "_exhausted_until", float("inf"))

    async def explode(self, msg_ctx, recent_messages):
        raise AssertionError("classify must not run while the balance is exhausted")

    monkeypatch.setattr(brain_module.ConversationAnalyzer, "classify", explode)

    b = Brain(bot_config)
    rag = _FakeRagClient()
    b._rag_client = rag

    result = await b.analyze_and_respond(
        message_text="что известно про Рому?", author="Вася", recent_messages=[]
    )

    assert result is None
    assert rag.calls == 0


@pytest.mark.asyncio
async def test_classifier_402_arms_the_breaker(monkeypatch):
    monkeypatch.setattr(llm_health, "_exhausted_until", 0.0)

    async def fake_call_llm(self, prompt, max_tokens):
        raise DeepSeekHTTPError(402, "Insufficient Balance")

    monkeypatch.setattr(brain_module.ConversationAnalyzer, "_call_llm", fake_call_llm)

    analyzer = brain_module.ConversationAnalyzer(api_key="test")
    msg_ctx = brain_module._MessageContext(
        author="Вася",
        text="привет",
        reply_text=None,
        bot_name="Дип Сик",
        bot_responded_recently=False,
    )
    result = await analyzer.classify(msg_ctx, [])

    assert result.from_fallback is True
    assert llm_health.balance_exhausted()


def test_generator_402_arms_the_breaker(bot_config, monkeypatch):
    monkeypatch.setattr(llm_health, "_exhausted_until", 0.0)

    class Fake402(Exception):
        status_code = 402

    class FakeCompletions:
        def create(self, **kwargs):
            raise Fake402(
                'Error code: 402 - {"error":{"message":"Insufficient Balance"}}'
            )

    class FakeClient:
        chat = SimpleNamespace(completions=FakeCompletions())

    b = Brain(bot_config)
    b.client = FakeClient()

    answer = b.generate_response(message_text="привет", context="", grade=2)

    assert answer == FALLBACK_RESPONSES["api_error"]
    assert llm_health.balance_exhausted()
