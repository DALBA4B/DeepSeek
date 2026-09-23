# bot_state.py
"""
Persistent on/off switch for the bot (/on, /off commands).

The whole point of the switch is that the bot must stay off across Railway
restarts — Railway containers are ephemeral, so RAM-only state would mean the
bot "resurrects" on every redeploy. State therefore lives in a Firestore
document (`bot_state/state`) whenever Firebase is configured, with a local
JSON file as a fallback (and as a read-through cache when Firestore is
unreachable at startup).
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Firestore location of the state document.
_COLLECTION = "bot_state"
_DOCUMENT = "state"

DEFAULT_FILE_PATH = "bot_state.json"


class BotState:
    """
    Tracks whether the bot is enabled (responds to chat) or powered off.

    Persistence priority: Firestore first (survives Railway restarts), local
    file second. A failure to persist never raises — the flag still flips in
    RAM so the command doesn't appear to do nothing; the worst case is the
    state not surviving the next restart, which is logged.
    """

    def __init__(self, firebase_db=None, file_path: str = DEFAULT_FILE_PATH):
        """
        Load the persisted state.

        Args:
            firebase_db: Optional Firestore client (from FirebaseStorage).
            file_path:   Local JSON fallback/cache path.
        """
        self._db = firebase_db
        self._file_path = file_path
        self._enabled: bool = self._load()
        logger.info(
            "BotState initialized: %s (source: %s)",
            "enabled" if self._enabled else "OFF",
            "firestore" if self._db else "local file",
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
        """Read the flag from Firestore, falling back to the local file."""
        if self._db is not None:
            try:
                doc = self._db.collection(_COLLECTION).document(_DOCUMENT).get()
                if doc.exists:
                    return bool(doc.to_dict().get("enabled", True))
            except Exception as e:
                logger.warning("Could not read bot state from Firestore: %s", e)

        # Local file fallback / first run default.
        try:
            if os.path.exists(self._file_path):
                with open(self._file_path, "r", encoding="utf-8") as f:
                    return bool(json.load(f).get("enabled", True))
        except Exception as e:
            logger.warning("Could not read bot state file: %s", e)
        return True

    def _persist(self) -> None:
        """Write the flag to Firestore (if available) and the local file."""
        if self._db is not None:
            try:
                self._db.collection(_COLLECTION).document(_DOCUMENT).set(
                    {"enabled": self._enabled}
                )
            except Exception as e:
                logger.error("Failed to persist bot state to Firestore: %s", e)
        try:
            with open(self._file_path, "w", encoding="utf-8") as f:
                json.dump({"enabled": self._enabled}, f)
        except Exception as e:
            logger.error("Failed to persist bot state file: %s", e)
