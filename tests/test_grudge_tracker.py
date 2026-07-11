# tests/test_grudge_tracker.py
"""
Unit tests for brain.GrudgeTracker — pure in-RAM logic, no network.

GrudgeTracker.record_attack()/grudge_level() call datetime.now(timezone.utc)
directly, so tests that need to simulate "time passing" monkeypatch
brain.datetime with a small fake clock instead of sleeping for real.
"""

from datetime import datetime, timedelta, timezone

import brain as brain_module
from brain import GrudgeTracker


class _FakeDateTime(datetime):
    """datetime subclass whose now() is controlled by tests."""

    _now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls._now


def _set_now(monkeypatch, dt: datetime) -> None:
    _FakeDateTime._now = dt
    monkeypatch.setattr(brain_module, "datetime", _FakeDateTime)


def test_first_attack_has_no_prior_grudge():
    tracker = GrudgeTracker()
    key = (111, 222)

    # Before any attack, grudge_level is 0.
    assert tracker.grudge_level(key) == 0

    tracker.record_attack(key)
    # grudge_level counts PRIOR attacks — the one just recorded now counts.
    assert tracker.grudge_level(key) == 1


def test_repeated_attacks_within_window_escalate(monkeypatch):
    tracker = GrudgeTracker(grudge_window_sec=1800)
    key = (111, 222)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    _set_now(monkeypatch, base)
    tracker.record_attack(key)
    assert tracker.grudge_level(key) == 1

    _set_now(monkeypatch, base + timedelta(seconds=30))
    tracker.record_attack(key)
    assert tracker.grudge_level(key) == 2

    _set_now(monkeypatch, base + timedelta(seconds=60))
    tracker.record_attack(key)
    assert tracker.grudge_level(key) == 3


def test_attack_outside_window_is_not_counted(monkeypatch):
    tracker = GrudgeTracker(grudge_window_sec=60)
    key = (111, 222)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    _set_now(monkeypatch, base)
    tracker.record_attack(key)

    # Well past the 60s window — old attack should no longer count.
    _set_now(monkeypatch, base + timedelta(seconds=120))
    assert tracker.grudge_level(key) == 0


def test_window_boundary_is_inclusive(monkeypatch):
    tracker = GrudgeTracker(grudge_window_sec=60)
    key = (111, 222)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    _set_now(monkeypatch, base)
    tracker.record_attack(key)

    # Exactly at the edge of the window (cutoff = now - window).
    _set_now(monkeypatch, base + timedelta(seconds=60))
    assert tracker.grudge_level(key) == 1


def test_different_keys_are_independent():
    tracker = GrudgeTracker()
    user_a = (111, 222)
    user_b = (111, 333)  # same chat, different user
    user_c = (999, 222)  # different chat, same user id

    tracker.record_attack(user_a)
    tracker.record_attack(user_a)

    assert tracker.grudge_level(user_a) == 2
    assert tracker.grudge_level(user_b) == 0
    assert tracker.grudge_level(user_c) == 0
