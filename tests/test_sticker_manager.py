# tests/test_sticker_manager.py
"""
Unit tests for responder.StickerManager.get_file_id — emotion-aware sticker
selection from a loaded pack (matches by the pack's own per-sticker emoji
instead of picking a fully random sticker). Pure logic, no Telegram calls —
load_sticker_set() is bypassed by writing directly into _all_stickers.
"""

from responder import StickerManager


def test_matching_emotion_returns_only_matching_stickers():
    mgr = StickerManager()
    mgr._all_stickers = [
        ("id_happy_1", "😄"),
        ("id_happy_2", "😊"),
        ("id_sad_1", "😢"),
    ]

    for _ in range(10):
        assert mgr.get_file_id("happy") in ("id_happy_1", "id_happy_2")


def test_no_matching_emoji_falls_back_to_any_sticker_in_pack():
    mgr = StickerManager()
    mgr._all_stickers = [
        ("id_1", "🐱"),  # not mapped to any known emotion
        ("id_2", "🐶"),
    ]

    assert mgr.get_file_id("happy") in ("id_1", "id_2")


def test_empty_pack_falls_back_to_manual_mapping():
    mgr = StickerManager()
    mgr._all_stickers = []

    assert mgr.get_file_id("happy") == StickerManager.DEFAULT_STICKERS["happy"]


def test_empty_pack_unknown_emotion_returns_none():
    mgr = StickerManager()
    mgr._all_stickers = []

    assert mgr.get_file_id("unknown_emotion") is None


def test_emotion_matching_is_case_and_whitespace_insensitive():
    mgr = StickerManager()
    mgr._all_stickers = [("id_happy", "😄")]

    assert mgr.get_file_id("  HAPPY  ") == "id_happy"
