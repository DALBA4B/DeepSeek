# tests/test_conversation_analyzer.py
"""
Unit tests for conversation_analyzer.py — the classify() integration tests
mock _call_llm directly (no aiohttp/real network involved anywhere here).
"""

import json


from conversation_analyzer import ConversationAnalyzer, ClassificationResult, _MessageContext
from retry import DeepSeekHTTPError


def _ctx(text, bot_name="Дип Сик", reply_text=None, bot_responded_recently=False, author="Tima"):
    return _MessageContext(
        author=author,
        text=text,
        reply_text=reply_text,
        bot_name=bot_name,
        bot_responded_recently=bot_responded_recently,
    )


# ── _is_directly_addressed ──────────────────────────────────────────────

def test_is_directly_addressed_full_name():
    assert ConversationAnalyzer._is_directly_addressed(_ctx("дип сик привет")) is True


def test_is_directly_addressed_short_form():
    assert ConversationAnalyzer._is_directly_addressed(_ctx("дип как дела")) is True


def test_is_directly_addressed_absent():
    assert ConversationAnalyzer._is_directly_addressed(_ctx("я люблю борщ")) is False


# ── _is_attack_on_bot ────────────────────────────────────────────────────

def test_is_attack_on_bot_named_and_insulted():
    assert ConversationAnalyzer._is_attack_on_bot(_ctx("дип ты тупой")) is True


def test_is_attack_on_bot_insult_without_addressing_bot():
    # No name mention, bot didn't just reply -> not aimed at the bot.
    ctx = _ctx("вот реально дурачек", bot_responded_recently=False)
    assert ConversationAnalyzer._is_attack_on_bot(ctx) is False


def test_is_attack_on_bot_followup_after_bot_replied():
    # No name mention, but the bot just answered -> still aimed at the bot.
    ctx = _ctx("вот реально дурачек", bot_responded_recently=True)
    assert ConversationAnalyzer._is_attack_on_bot(ctx) is True


def test_is_attack_on_bot_named_but_not_insulting():
    assert ConversationAnalyzer._is_attack_on_bot(_ctx("дип привет как дела")) is False


# ── _parse_response ──────────────────────────────────────────────────────

def test_parse_response_clean_json():
    raw = json.dumps({
        "grade": 2, "needs_memory": False, "rag_query": None,
        "situation": "joke", "reason": "just kidding",
    })
    result = ConversationAnalyzer._parse_response(raw)
    assert result == ClassificationResult(
        grade=2, needs_memory=False, rag_query=None,
        reason="just kidding", from_fallback=False, situation="joke",
    )


def test_parse_response_strips_markdown_fences():
    raw = "```json\n" + json.dumps({"grade": 1, "situation": "tease"}) + "\n```"
    result = ConversationAnalyzer._parse_response(raw)
    assert result is not None
    assert result.grade == 1
    assert result.situation == "tease"


def test_parse_response_malformed_json_returns_none():
    assert ConversationAnalyzer._parse_response("not json at all {") is None


def test_parse_response_clamps_grade_and_defaults_situation():
    raw = json.dumps({"grade": 99, "situation": "not-a-real-situation"})
    result = ConversationAnalyzer._parse_response(raw)
    assert result is not None
    assert result.grade == 3  # clamped to max
    assert result.situation == "casual"  # unknown -> default


def test_parse_response_empty_rag_query_becomes_none():
    raw = json.dumps({"grade": 2, "rag_query": "   "})
    result = ConversationAnalyzer._parse_response(raw)
    assert result.rag_query is None


# ── _fallback_classify ───────────────────────────────────────────────────

def test_fallback_classify_plain_short_message():
    analyzer = ConversationAnalyzer(api_key="x")
    result = analyzer._fallback_classify(_ctx("ok"))
    assert result.from_fallback is True
    assert result.grade == 0


def test_fallback_classify_direct_address():
    analyzer = ConversationAnalyzer(api_key="x")
    result = analyzer._fallback_classify(_ctx("дип расскажи анекдот"))
    assert result.grade == 2
    assert result.needs_memory is True


def test_fallback_classify_attack():
    analyzer = ConversationAnalyzer(api_key="x")
    result = analyzer._fallback_classify(_ctx("дип ты тупой"))
    assert result.situation == "defend"
    assert result.grade == 2


def test_fallback_classify_question():
    analyzer = ConversationAnalyzer(api_key="x")
    result = analyzer._fallback_classify(_ctx("а какой сегодня день недели вообще?"))
    assert result.situation == "help"
    assert result.grade >= 1


# ── classify() integration (mocked _call_llm, no real network) ──────────

async def test_classify_retries_once_then_succeeds(monkeypatch):
    analyzer = ConversationAnalyzer(api_key="x", max_attempts=2, retry_base_delay=0.01)

    calls = {"n": 0}

    async def fake_call_llm(prompt, max_tokens):
        calls["n"] += 1
        if calls["n"] == 1:
            raise DeepSeekHTTPError(503, "temporarily down")
        return json.dumps({"grade": 2, "situation": "help", "reason": "ok"})

    monkeypatch.setattr(analyzer, "_call_llm", fake_call_llm)

    result = await analyzer.classify(_ctx("объясни мне физику"), [])

    assert calls["n"] == 2
    assert result.from_fallback is False
    assert result.situation == "help"


async def test_classify_exhausts_retries_and_falls_back(monkeypatch):
    analyzer = ConversationAnalyzer(api_key="x", max_attempts=2, retry_base_delay=0.01)

    async def always_fails(prompt, max_tokens):
        raise DeepSeekHTTPError(500, "still down")

    monkeypatch.setattr(analyzer, "_call_llm", always_fails)

    result = await analyzer.classify(_ctx("дип расскажи анекдот"), [])

    assert result.from_fallback is True
    assert result.grade == 2  # direct-address heuristic still kicks in


async def test_classify_direct_address_bumps_low_grade(monkeypatch):
    analyzer = ConversationAnalyzer(api_key="x")

    async def fake_call_llm(prompt, max_tokens):
        return json.dumps({"grade": 0, "situation": "casual", "reason": "meh"})

    monkeypatch.setattr(analyzer, "_call_llm", fake_call_llm)

    result = await analyzer.classify(_ctx("дип привет"), [])

    assert result.grade == 2  # bumped from 0 because bot was addressed


async def test_classify_keyword_attack_overrides_llm_situation(monkeypatch):
    analyzer = ConversationAnalyzer(api_key="x")

    async def fake_call_llm(prompt, max_tokens):
        # LLM softens the insult to "casual" — the keyword floor should win.
        return json.dumps({"grade": 1, "situation": "casual", "reason": "meh"})

    monkeypatch.setattr(analyzer, "_call_llm", fake_call_llm)

    result = await analyzer.classify(_ctx("дип ты тупой"), [])

    assert result.situation == "defend"
    assert result.grade >= 2
