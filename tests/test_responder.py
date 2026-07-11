# tests/test_responder.py
"""Unit tests for responder.ResponseParser — pure string parsing, no network."""

from models import ResponseType
from responder import ResponseParser


def test_plain_text_passes_through_unchanged():
    result = ResponseParser.parse("привет как дела", text_only_mode=True)
    assert result.response_type == ResponseType.TEXT
    assert result.content == "привет как дела"


def test_giphy_prefix_blocked_in_text_only_mode():
    result = ResponseParser.parse("GIPHY:funny dance", text_only_mode=True)
    assert result.response_type == ResponseType.TEXT
    assert result.content == ""


def test_giphy_prefix_honored_when_not_text_only():
    result = ResponseParser.parse("GIPHY:funny dance", text_only_mode=False)
    assert result.response_type == ResponseType.GIF
    assert result.content == "funny dance"


def test_sticker_prefix_blocked_in_text_only_mode():
    result = ResponseParser.parse("STICKER:happy", text_only_mode=True)
    assert result.response_type == ResponseType.TEXT
    assert result.content == ""


def test_sticker_prefix_honored_when_not_text_only():
    result = ResponseParser.parse("STICKER:happy", text_only_mode=False)
    assert result.response_type == ResponseType.STICKER
    assert result.content == "happy"


def test_react_prefix_converted_to_text_in_text_only_mode():
    result = ResponseParser.parse("REACT:💀", text_only_mode=True)
    assert result.response_type == ResponseType.TEXT
    assert result.content == "💀"


def test_react_prefix_honored_when_not_text_only():
    result = ResponseParser.parse("REACT:💀", text_only_mode=False)
    assert result.response_type == ResponseType.REACTION
    assert result.content == "💀"


def test_prefix_matching_is_case_insensitive():
    result = ResponseParser.parse("giphy:funny cat", text_only_mode=False)
    assert result.response_type == ResponseType.GIF
    assert result.content == "funny cat"


def test_leading_whitespace_is_stripped_before_prefix_check():
    result = ResponseParser.parse("   GIPHY:funny cat  ", text_only_mode=False)
    assert result.response_type == ResponseType.GIF
    assert result.content == "funny cat"
