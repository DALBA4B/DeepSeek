# tests/test_context_prompt.py
"""
Тесты сборки промпта с фактами из памяти.

Дублирование ловится именно здесь: раньше brain.py приклеивал факты к
контексту, а get_context_prompt вставлял их же вторым блоком — простыня на
40 тысяч знаков уезжала в модель дважды, и за неё платили дважды. Ни один
тест этого не видел, потому что оба места по отдельности работали правильно.
"""
import pytest

from prompts import get_context_prompt

FACTS = "Максим играет в Доту. Кирилл сдал B1 на 100/100."
CONTEXT = "Tima: чё как\nМаксим Тян: да норм"
MESSAGE = "во что максим играет"


def test_facts_appear_exactly_once():
    prompt = get_context_prompt(CONTEXT, MESSAGE, FACTS, "")
    assert prompt.count(FACTS) == 1


def test_facts_come_after_the_chat_history():
    """Модель следует тому, что прочитала последним."""
    prompt = get_context_prompt(CONTEXT, MESSAGE, FACTS, "")
    assert prompt.index(CONTEXT) < prompt.index(FACTS)


def test_message_stays_last():
    """Вопрос должен остаться в конце, иначе факты его перекрывают."""
    prompt = get_context_prompt(CONTEXT, MESSAGE, FACTS, "")
    assert prompt.index(FACTS) < prompt.rindex(MESSAGE)


def test_media_hint_goes_after_everything():
    prompt = get_context_prompt(CONTEXT, MESSAGE, FACTS, "можно стикер")
    assert prompt.rindex("можно стикер") > prompt.index(FACTS)


def test_block_tells_the_model_what_to_do_with_facts():
    """Блок — инструкция, а не просто заголовок над данными."""
    prompt = get_context_prompt(CONTEXT, MESSAGE, FACTS, "")
    assert "ТВОЯ ПАМЯТЬ О ЧАТЕ" in prompt
    assert "именам" in prompt


def test_header_matches_the_system_prompt():
    """
    Системный промпт ссылается на блок по имени. Разъедутся — инструкция
    будет указывать на несуществующий блок, и никто не заметит.
    """
    from prompts import get_system_prompt

    prompt = get_context_prompt(CONTEXT, MESSAGE, FACTS, "")
    system = get_system_prompt("Дип Сик", [], text_only_mode=True)
    assert "ТВОЯ ПАМЯТЬ О ЧАТЕ" in system
    assert "ТВОЯ ПАМЯТЬ О ЧАТЕ" in prompt


def test_no_knowledge_block_without_facts():
    prompt = get_context_prompt(CONTEXT, MESSAGE, "", "")
    assert "ТВОЯ ПАМЯТЬ О ЧАТЕ" not in prompt
    assert CONTEXT in prompt
    assert MESSAGE in prompt


def test_prompt_size_tracks_facts_size():
    """Промпт с фактами не должен быть вдвое больше самих фактов."""
    facts = "Ф" * 10000
    prompt = get_context_prompt(CONTEXT, MESSAGE, facts, "")
    assert len(prompt) < len(facts) * 1.5
