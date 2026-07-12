# tests/test_recent_reaction_tracker.py
"""
Unit tests for responder.RecentReactionTracker — blocks the same reaction
emoji from firing a 5th time in a row (minimal anti-spam guard). Pure logic,
no network.
"""

from responder import RecentReactionTracker


def test_fresh_tracker_never_avoids():
    tracker = RecentReactionTracker()
    assert tracker.should_avoid("👍") is False


def test_streak_below_max_is_not_avoided():
    tracker = RecentReactionTracker(max_streak=4)
    for _ in range(3):
        tracker.record("👍")
    # 3 in a row so far — a 4th is still fine (not yet at the max streak).
    assert tracker.should_avoid("👍") is False


def test_streak_at_max_is_avoided():
    tracker = RecentReactionTracker(max_streak=4)
    for _ in range(4):
        tracker.record("👍")
    # 4 in a row already — a 5th identical one should be avoided.
    assert tracker.should_avoid("👍") is True


def test_different_emoji_resets_streak():
    tracker = RecentReactionTracker(max_streak=4)
    for _ in range(4):
        tracker.record("👍")
    assert tracker.should_avoid("👍") is True

    tracker.record("😂")
    assert tracker.should_avoid("👍") is False
    assert tracker.should_avoid("😂") is False


def test_pick_non_repeating_avoids_maxed_out_candidate():
    tracker = RecentReactionTracker(max_streak=4)
    for _ in range(4):
        tracker.record("👍")

    for _ in range(20):
        assert tracker.pick_non_repeating(["👍", "😂"]) == "😂"


def test_pick_non_repeating_falls_back_to_repeat_if_no_alternative():
    tracker = RecentReactionTracker(max_streak=4)
    for _ in range(4):
        tracker.record("👍")

    # Only maxed-out candidate available — must still return something.
    assert tracker.pick_non_repeating(["👍"]) == "👍"
