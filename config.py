# config.py
"""
Configuration file for DeepSeek Telegram Bot.
Loads all API keys, tokens, and settings from environment variables (.env file).
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Telegram Bot
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN not found in environment variables. Check .env file.")

# DeepSeek API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in environment variables. Check .env file.")

# Giphy API
GIPHY_API_KEY = os.getenv("GIPHY_API_KEY")
if not GIPHY_API_KEY:
    raise ValueError("GIPHY_API_KEY not found in environment variables. Check .env file.")

# Firebase
FIREBASE_CRED_PATH = os.getenv("FIREBASE_CRED_PATH")
if not FIREBASE_CRED_PATH:
    raise ValueError("FIREBASE_CRED_PATH not found in environment variables. Check .env file.")

# Bot Settings
BOT_NAME = os.getenv("BOT_NAME", "Вася")
CHAT_ID = os.getenv("CHAT_ID")  # Optional: group chat ID to limit bot scope
if CHAT_ID:
    try:
        CHAT_ID = int(CHAT_ID)
    except ValueError:
        CHAT_ID = None

# Memory Settings
SHORT_MEMORY_LIMIT = 30  # Max messages in short-term memory
CONTEXT_MESSAGES_COUNT = 20  # Messages to include in DeepSeek context

# DeepSeek API Settings
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_MAX_TOKENS = 150
DEEPSEEK_TEMPERATURE = 0.8

# Response Probability
RANDOM_RESPONSE_PROBABILITY = 0.1  # 10% chance to respond randomly

# Giphy API Settings
GIPHY_API_URL = "https://api.giphy.com/v1/gifs/search"
GIPHY_LIMIT = 10
GIPHY_RATING = "pg-13"

# Available emojis for reactions
AVAILABLE_EMOJIS = [
    "👍", "👎", "❤️", "🔥", "🥰", "👏", "😂", "🤔", "🤯", "😱",
    "🤬", "😢", "🎉", "🤩", "🤮", "💩", "🙏", "👌", "🕊️", "🤡",
    "🥱", "🥴", "😍", "💯", "🤣", "⚡", "🍌", "🏆", "💔", "🤨",
    "😐", "🍓", "💋", "🖕", "😈", "😴", "😭", "🤓", "👻", "🎃",
    "🙈", "😇", "😨", "🤝", "✍️", "🤗", "🎅", "🎄", "☃️", "💅",
    "🤪", "🗿", "🆒", "💘", "🙉", "🦄", "😘", "💊", "🙊", "😎",
    "👾", "🤷", "😡"
]

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
