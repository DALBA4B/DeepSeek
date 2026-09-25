# tests/test_power_switch.py
"""
Unit tests for bot_state.BotState — the /on, /off switch.

Persistence is a plain local JSON file, covered via tmp_path. No network.
"""

from bot_state import BotState


def test_defaults_to_enabled_on_first_run(tmp_path):
    state = BotState(file_path=str(tmp_path / "s.json"))
    assert state.is_enabled() is True


def test_set_enabled_persists_to_file(tmp_path):
    path = str(tmp_path / "s.json")

    state = BotState(file_path=path)
    state.set_enabled(False)
    assert state.is_enabled() is False

    # A fresh instance (simulated restart) reads back "off".
    reloaded = BotState(file_path=path)
    assert reloaded.is_enabled() is False


def test_set_enabled_idempotent(tmp_path):
    path = str(tmp_path / "s.json")
    state = BotState(file_path=path)
    state.set_enabled(True)  # already True — no-op, must not break anything
    assert state.is_enabled() is True


def test_missing_file_after_redeploy_means_enabled(tmp_path):
    # Railway wipes the filesystem on redeploy: no file → bot comes back on.
    state = BotState(file_path=str(tmp_path / "gone.json"))
    assert state.is_enabled() is True


def test_corrupt_file_falls_back_to_enabled(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("not json at all", encoding="utf-8")
    state = BotState(file_path=str(path))
    assert state.is_enabled() is True


def test_toggle_back_on(tmp_path):
    path = str(tmp_path / "s.json")
    state = BotState(file_path=path)
    state.set_enabled(False)
    state.set_enabled(True)
    assert BotState(file_path=path).is_enabled() is True
