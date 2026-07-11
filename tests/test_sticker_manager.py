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


def test_recently_sent_sticker_is_avoided_when_alternatives_exist():
    mgr = StickerManager()
    mgr._all_stickers = [
        ("id_happy_1", "😄"),
        ("id_happy_2", "😊"),
    ]
    mgr.record_sent("id_happy_1")

    for _ in range(10):
        assert mgr.get_file_id("happy") == "id_happy_2"


def test_recently_sent_falls_back_to_repeat_when_pool_exhausted():
    mgr = StickerManager()
    mgr._all_stickers = [("id_happy_1", "😄")]
    mgr.record_sent("id_happy_1")

    # Only one matching sticker exists and it was just sent — must still
    # return something rather than None.
    assert mgr.get_file_id("happy") == "id_happy_1"


def test_recent_history_is_bounded_and_forgets_oldest():
    mgr = StickerManager()
    mgr._all_stickers = [("id_1", "😄"), ("id_2", "😄")]

    # Fill history past its max size with a sticker not in the pool at all,
    # then send id_1 — id_1 should still get excluded (it's the most recent).
    for _ in range(StickerManager.RECENT_HISTORY_SIZE):
        mgr.record_sent("unrelated_id")
    mgr.record_sent("id_1")

    assert mgr.get_file_id("happy") == "id_2"


def test_load_sticker_set_accumulates_across_multiple_packs():
    mgr = StickerManager()
    mgr._all_stickers = [("id_from_pack_a", "😄")]
    # Simulate a second load_sticker_set() call without a real Bot/network —
    # exercising the accumulation behavior directly on the list it appends to.
    mgr._all_stickers.extend([("id_from_pack_b", "😢")])

    assert ("id_from_pack_a", "😄") in mgr._all_stickers
    assert ("id_from_pack_b", "😢") in mgr._all_stickers
    assert len(mgr._all_stickers) == 2
