# tests/test_memory_badge.py
"""Memory badge: 🧠✅ is appended iff LightRAG facts were used — text only."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from brain import Brain


def _brain_with_canned_answer(bot_config, text: str) -> Brain:
    brain = Brain(config=bot_config)
    canned = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=None,
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = canned
    brain.client = mock_client
    return brain


def _generate(brain: Brain, rag_facts: str):
    return brain.generate_response(
        message_text="привет",
        context="Tima: привет",
        rag_facts=rag_facts,
        grade=2,
        situation="casual",
    )


def test_badge_appended_when_facts_used(bot_config):
    brain = _brain_with_canned_answer(bot_config, "Привет! Как дела?")
    assert _generate(brain, "факты из памяти").endswith("\n\n🧠✅")


def test_no_badge_without_facts(bot_config):
    brain = _brain_with_canned_answer(bot_config, "Привет! Как дела?")
    answer = _generate(brain, "")
    assert answer == "Привет! Как дела?"
    assert "🧠" not in answer


def test_no_badge_on_media_directive(bot_config):
    brain = _brain_with_canned_answer(bot_config, "STICKER:happy:123")
    answer = _generate(brain, "факты из памяти")
    assert answer is None or "🧠" not in answer
