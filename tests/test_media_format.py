# tests/test_media_format.py
"""
Unit tests for brain._detect_media_request / _build_media_hint — the
code-enforced GIF/sticker format control (Проблема №2: гифка/стикер вместо
ответа, когда не просили, и наоборот). Pure logic, no network.
"""

from brain import _build_media_hint, _detect_media_request


def test_detect_media_request_gif_keyword():
    assert _detect_media_request("скинь гифку про котов") == "gif"


def test_detect_media_request_sticker_keyword():
    assert _detect_media_request("дай стикер повеселее") == "sticker"


def test_detect_media_request_none_for_plain_message():
    assert _detect_media_request("как дела сегодня?") is None


def test_detect_media_request_is_case_insensitive():
    assert _detect_media_request("ГИФКУ ДАВАЙ") == "gif"


def test_build_media_hint_explicit_gif_request_ignores_probability():
    # Even with probability=0, an explicit request must still be honored.
    hint = _build_media_hint("скинь гифку", media_probability=0.0)
    assert "GIPHY" in hint


def test_build_media_hint_explicit_sticker_request_ignores_probability():
    hint = _build_media_hint("дай стикер", media_probability=0.0)
    assert "STICKER" in hint


def test_build_media_hint_no_request_and_zero_probability_is_empty():
    hint = _build_media_hint("привет как сам", media_probability=0.0)
    assert hint == ""


def test_build_media_hint_no_request_but_probability_one_always_hints(monkeypatch):
    # random.random() < 1.0 is always true, so some media hint must fire.
    hint = _build_media_hint("привет как сам", media_probability=1.0)
    assert hint != ""
    assert any(fmt in hint for fmt in ("GIPHY", "STICKER", "REACT"))
