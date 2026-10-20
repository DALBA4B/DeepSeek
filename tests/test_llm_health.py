# tests/test_llm_health.py
"""
Tests for llm_health.py — the DeepSeek "balance exhausted" (HTTP 402)
circuit breaker.

The breaker exists because of 2026-10-14: DeepSeek ran out of balance and
the bot kept classifying (fallback heuristics), querying LightRAG and
failing at generation for every message. Any 402 now pauses the pipeline.
"""

import llm_health
from retry import DeepSeekHTTPError


def test_classifier_style_402_is_detected():
    assert llm_health.is_insufficient_balance(
        DeepSeekHTTPError(402, "Insufficient Balance")
    )


def test_generator_style_402_is_detected_by_status_code():
    class FakeOpenAIStatusError(Exception):
        status_code = 402

    exc = FakeOpenAIStatusError(
        "Error code: 402 - {'error': {'message': 'Insufficient Balance'}}"
    )
    assert llm_health.is_insufficient_balance(exc)


def test_sdk_errors_are_detected_by_message_when_status_is_missing():
    assert llm_health.is_insufficient_balance(
        Exception('Error code: 402 - {"error":{"message":"Insufficient Balance"}}')
    )


def test_other_errors_are_not_mistaken_for_balance_errors():
    assert not llm_health.is_insufficient_balance(DeepSeekHTTPError(429, "rate limited"))
    assert not llm_health.is_insufficient_balance(TimeoutError("timed out"))
    assert not llm_health.is_insufficient_balance(Exception("boom"))


def test_mark_arms_the_cooldown_and_it_expires_by_time(monkeypatch):
    monkeypatch.setattr(llm_health, "_exhausted_until", 0.0)
    assert not llm_health.balance_exhausted(now=1000.0)

    llm_health.mark_balance_exhausted(now=1000.0)
    assert llm_health.balance_exhausted(now=1000.0)
    assert llm_health.balance_exhausted(now=1000.0 + llm_health.cooldown_seconds() - 1)
    assert not llm_health.balance_exhausted(now=1000.0 + llm_health.cooldown_seconds())
    assert llm_health.seconds_left(now=1000.0) == llm_health.cooldown_seconds()


def test_seconds_left_never_goes_negative(monkeypatch):
    monkeypatch.setattr(llm_health, "_exhausted_until", 0.0)
    assert llm_health.seconds_left(now=12345.0) == 0.0
