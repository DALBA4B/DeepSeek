# tests/test_import_history.py
"""Multi-chat identity rules for tools/import_history.py.

Two chats can cover the same calendar dates. file_source must stay unique
(otherwise LightRAG returns 409 and silently keeps the old document), and the
progress file must not be shared (otherwise dates already done in one chat are
skipped in the other).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.import_history import (  # noqa: E402
    resolve_progress_path,
    validate_source_prefix,
)


def test_default_prefix_keeps_historical_progress_file():
    assert resolve_progress_path(None, "telegram_import") == "import_progress.json"


def test_custom_prefix_gets_its_own_progress_file():
    assert (
        resolve_progress_path(None, "telegram_import_other")
        == "import_progress_telegram_import_other.json"
    )


def test_explicit_progress_always_wins():
    assert (
        resolve_progress_path("custom_progress.json", "telegram_import_other")
        == "custom_progress.json"
    )


def test_source_prefix_accepts_letters_digits_underscore_dash():
    assert validate_source_prefix("telegram_import_2021-2022") == (
        "telegram_import_2021-2022"
    )


@pytest.mark.parametrize("bad", ["", "чат", "a b", "../other", "a/b", None])
def test_source_prefix_rejects_unsafe_values(bad):
    with pytest.raises(ValueError):
        validate_source_prefix(bad)
