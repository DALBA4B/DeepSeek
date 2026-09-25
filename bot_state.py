# bot_state.py
"""
Persistent on/off switch for the bot (/on, /off commands).

State lives in a local JSON file. On Railway the container filesystem is
ephemeral, so after a redeploy the bot comes back enabled (default) — that is
an accepted trade-off: /off is usually flipped while the bot is up, and a
fresh deploy means "bot is back" anyway.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_FILE_PATH = "bot_state.json"


class BotState:
    """
    Tracks whether the bot is enabled (responds to chat) or powered off.

    A failure to persist never raises — the flag still flips in RAM so the
    command doesn't appear to do nothing; the worst case is the state not
    surviving the next restart, which is logged.
    """

    def __init__(self, file_path: str = DEFAULT_FILE_PATH):
        """
        Load the persisted state.

        Args:
            file_path: Local JSON state file path.
        """
        self._file_path = file_path
        self._enabled: bool = self._load()
        logger.info(
            "BotState initialized: %s",
            "enabled" if self._enabled else "OFF",
        )

    def is_enabled(self) -> bool:
        """True when the bot should read and react to chat messages."""
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Flip the switch and persist it (best effort)."""
        if self._enabled == enabled:
            return
        self._enabled = enabled
        self._persist()
        logger.info("Bot switched %s via chat command", "ON" if enabled else "OFF")

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _load(self) -> bool:
        """Read the flag from the local file; default to enabled."""
        try:
            if os.path.exists(self._file_path):
                with open(self._file_path, "r", encoding="utf-8") as f:
                    return bool(json.load(f).get("enabled", True))
        except Exception as e:
            logger.warning("Could not read bot state file: %s", e)
        return True

    def _persist(self) -> None:
        """Write the flag to the local file."""
        try:
            with open(self._file_path, "w", encoding="utf-8") as f:
                json.dump({"enabled": self._enabled}, f)
        except Exception as e:
            logger.error("Failed to persist bot state file: %s", e)
