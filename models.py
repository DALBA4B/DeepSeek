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


class RequestComplexity(Enum):
    """Complexity levels for incoming requests."""
    SIMPLE = "simple"      # Quick answers: yes/no, reactions, short jokes
    NORMAL = "normal"      # Regular conversation
    COMPLEX = "complex"    # Explanations, stories, plans


class InterestStatus(Enum):
    """Status of user interests (likes or dislikes)."""
    LIKES = "likes"
    DISLIKES = "dislikes"


@dataclass
class InterestEntry:
    """
    A single interest entry with history tracking.
    Allows versioning of interest changes (user changes mind).
    """
    name: str
    status: InterestStatus
    added_at: datetime = field(default_factory=datetime.now)
    current: bool = True  # Is this the current status for this interest?
    
    def to_dict(self) -> dict:
        """Convert to dictionary for Firebase storage."""
        return {
            "name": self.name,
            "status": self.status.value,
            "added_at": self.added_at.isoformat(),
            "current": self.current
        }


@dataclass
class TokenRange:
    """Token range for dynamic response length."""
    min_tokens: int
    max_tokens: int
    
    @classmethod
    def for_complexity(cls, complexity: RequestComplexity) -> 'TokenRange':
        """Get token range for a given complexity level."""
        ranges = {
            RequestComplexity.SIMPLE: cls(80, 200),
            RequestComplexity.NORMAL: cls(150, 400),
            RequestComplexity.COMPLEX: cls(300, 800),
        }
        return ranges.get(complexity, cls(100, 300))
    
    def random_value(self) -> int:
        """Get a random token count within the range."""
        import random
        return random.randint(self.min_tokens, self.max_tokens)


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
    timestamp: datetime = field(default_factory=lambda: datetime.now())

    def to_dict(self) -> dict:
        """Convert to dictionary for Firebase storage."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "text": self.text,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "date": self.timestamp.date().isoformat()  # Add date field for Firebase queries
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
    
    # Bot settings
    bot_name: str = "Вася"
    chat_id: Optional[int] = None
    
    # Memory settings
    short_memory_limit: int = 30
    context_messages_count: int = 35
    
    # DeepSeek settings
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    deepseek_max_tokens: int = 150
    deepseek_temperature: float = 1.0
    
    # Response settings
    random_response_probability: float = 0.1
    use_smart_respond: bool = False  # Use AI to decide whether to respond (costs API calls)
    
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
    last_seen: datetime = field(default_factory=lambda: datetime.now())

    def to_dict(self) -> dict:
        """Convert to dictionary for Firebase storage."""
        return {
            "username": self.username,
            "last_seen": self.last_seen
        }
