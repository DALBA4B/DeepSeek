# tests/test_prompts.py
"""Unit tests for prompts.get_name_variations() — pure string logic."""

import pytest

from prompts import get_name_variations


def test_single_word_name_matches_full_form():
    variants = get_name_variations("Вася")
    assert "вася" in variants


def test_multi_word_name_includes_full_and_short_forms():
    variants = get_name_variations("Дип Сик")
    assert "дип сик" in variants  # full form
    assert "дипсик" in variants  # glued
    assert "дип-сик" in variants  # dash-separated
    assert "дип" in variants  # short first-word form


def test_multi_word_name_short_form_matches_in_message():
    variants = get_name_variations("Дип Сик")
    text = "дип привет как дела"
    assert any(v in text for v in variants)


def test_short_first_word_under_three_chars_not_added():
    # "Ии Бот" -> first word "ии" is < 3 chars, must not be added on its own
    variants = get_name_variations("Ии Бот")
    assert "ии" not in variants


def test_empty_bot_name_returns_no_variants():
    assert get_name_variations("") == []
    assert get_name_variations(None) == []


def test_single_word_name_has_no_short_form_risk():
    # Single-word names rely solely on the full form — no separate short
    # form is synthesized (there's nothing to shorten), so no extra
    # substring-collision risk is introduced beyond the full name itself.
    variants = get_name_variations("Вася")
    assert variants == ["вася"]
