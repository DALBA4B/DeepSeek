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
    # Even with both probabilities=0, an explicit request must still be honored.
    hint, allowed = _build_media_hint("скинь гифку", media_probability=0.0, gif_probability=0.0)
    assert "GIPHY" in hint
    assert allowed


def test_build_media_hint_explicit_sticker_request_ignores_probability():
    hint, allowed = _build_media_hint("дай стикер", media_probability=0.0, gif_probability=0.0)
    assert "STICKER" in hint
    assert allowed


def test_build_media_hint_no_request_and_zero_probabilities_forbids_media():
    # No hint text is empty any more: media is forbidden via an explicit
    # negative instruction, so check the flag rather than the string.
    hint, allowed = _build_media_hint("привет как сам", media_probability=0.0, gif_probability=0.0)
    assert not allowed
    assert hint


def test_build_media_hint_no_request_but_gif_probability_one_always_gif():
    # roll < gif_probability is always true when gif_probability=1.0.
    hint, allowed = _build_media_hint("привет как сам", media_probability=0.0, gif_probability=1.0)
    assert "GIPHY" in hint
    assert allowed


def test_build_media_hint_no_request_but_media_probability_one_gives_sticker_or_react():
    # gif_probability=0 so the roll always lands in the media_probability band.
    hint, allowed = _build_media_hint("привет как сам", media_probability=1.0, gif_probability=0.0)
    assert allowed
    assert any(fmt in hint for fmt in ("STICKER", "REACT"))
    assert "GIPHY" not in hint
