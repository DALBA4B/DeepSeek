# models.py
"""
Data models for the DeepSeek Telegram bot.
Uses dataclasses for type safety and validation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class ResponseType(Enum):
    """Types of responses the bot can send."""
    TEXT = "text"
    REACTION = "reaction"
    GIF = "gif"
    STICKER = "sticker"


@dataclass
class ChatMessage:
    """
    Represents a message in the chat.
    
    Attributes:
        user_id: Telegram user ID
        username: Display name of the user
        text: Message content
        message_id: Telegram message ID
        timestamp: When the message was received
    """
    user_id: int
    username: str
    text: str
    message_id: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for Firebase storage."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "text": self.text,
            "message_id": self.message_id,
            "timestamp": self.timestamp
        }

    def to_context_line(self) -> str:
        """Format message for context string."""
        return f"{self.username}: {self.text}"


@dataclass
class ParsedResponse:
    """
    Parsed response from DeepSeek.
    
    Attributes:
        response_type: Type of response (text, reaction, gif, sticker)
        content: The actual content (text, emoji, search query, or emotion)
    """
    response_type: ResponseType
    content: str


@dataclass
class BotConfig:
    """
    Bot configuration loaded from environment.
    
    Centralizes all configuration with type hints and defaults.
    """
    # Required API keys
    telegram_token: str
    deepseek_api_key: str
    giphy_api_key: str
    firebase_cred_path: str
    
    # Optional API keys
    gemini_api_key: Optional[str] = None
    
    # Bot settings
    bot_name: str = "Вася"
    chat_id: Optional[int] = None
    
    # Memory settings
    short_memory_limit: int = 30
    context_messages_count: int = 20
    
    # DeepSeek settings
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    deepseek_max_tokens: int = 150
    deepseek_temperature: float = 1.0
    
    # Response settings
    random_response_probability: float = 0.1
    
    # Giphy settings
    giphy_api_url: str = "https://api.giphy.com/v1/gifs/search"
    giphy_limit: int = 10
    giphy_rating: str = "pg-13"
    
    # Scheduler settings
    nightly_analysis_hour: int = 3
    nightly_analysis_minute: int = 0
    timezone: str = "Europe/Kiev"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class UserInfo:
    """
    User information stored in Firebase.
    
    Attributes:
        user_id: Telegram user ID
        username: Display name
        last_seen: Last activity timestamp
    """
    user_id: int
    username: str
    last_seen: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary for Firebase storage."""
        return {
            "username": self.username,
            "last_seen": self.last_seen
        }
