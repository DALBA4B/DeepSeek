# tests/conftest.py
"""Shared pytest fixtures — no test here talks to a real network."""

import pytest

from models import BotConfig


@pytest.fixture
def bot_config() -> BotConfig:
    """A minimal BotConfig with fake credentials, safe for unit tests."""
    return BotConfig(
        telegram_token="test-telegram-token",
        deepseek_api_key="test-deepseek-key",
        giphy_api_key="test-giphy-key",
        firebase_cred_path="test-firebase-cred.json",
        bot_name="Дип Сик",
    )
