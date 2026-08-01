# models.py
"""
Data models for the DeepSeek Telegram bot.
Uses dataclasses for type safety and validation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from utils import get_now


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
    added_at: datetime = field(default_factory=lambda: get_now("UTC"))
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
        reply_to_text: If this is a reply, the text of the message being replied to
        chat_id: Telegram chat ID (to distinguish data from different groups)
    """
    user_id: int
    username: str
    text: str
    message_id: int
    timestamp: datetime = field(default_factory=lambda: get_now("UTC"))
    reply_to_text: Optional[str] = None
    chat_id: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for Firebase storage."""
        data = {
            "user_id": self.user_id,
            "username": self.username,
            "text": self.text,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "date": self.timestamp.date().isoformat()  # date field for Firebase queries
        }
        if self.reply_to_text:
            data["reply_to_text"] = self.reply_to_text
        if self.chat_id is not None:
            data["chat_id"] = self.chat_id
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        """
        Build a ChatMessage from a stored dict (e.g. a Firestore document).

        Tolerant by design: ignores extra fields (like the `date` query helper
        stored alongside the message) and normalizes the timestamp whether it
        was stored as an ISO string or as a native Firestore datetime. This is
        the inverse of `to_dict` and avoids the KeyError/TypeError that
        `ChatMessage(**data)` hit on the extra `date` key.

        Args:
            data: Dict from storage

        Returns:
            A ChatMessage instance
        """
        ts = data.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                ts = get_now("UTC")
        elif not isinstance(ts, datetime):
            # Missing or unexpected type -> fall back to now rather than crash
            ts = get_now("UTC")

        return cls(
            user_id=data.get("user_id", 0),
            username=data.get("username", "Unknown"),
            text=data.get("text", ""),
            message_id=data.get("message_id", 0),
            timestamp=ts,
            reply_to_text=data.get("reply_to_text"),
            chat_id=data.get("chat_id"),
        )

    def to_context_line(self) -> str:
        """
        Format message for context string.

        Inline the reply context so both the classifier and the response
        generator see what a message is reacting to — this is what lets the
        bot treat "согласен" + reply to "я люблю киев" as one coherent unit.
        """
        if self.reply_to_text:
            quote = self.reply_to_text.strip().replace("\n", " ")
            if len(quote) > 80:
                quote = quote[:77] + "..."
            return f"{self.username} (reply на «{quote}»): {self.text}"
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

    # Optional API keys (V2 features such as LightRAG embeddings)
    openai_api_key: Optional[str] = None
    
    # Bot settings
    bot_name: str = "Вася"
    chat_id: Optional[int] = None
    
    # Memory settings
    short_memory_limit: int = 30
    context_messages_count: int = 20
    
    # DeepSeek settings
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_temperature: float = 1.0
    deepseek_timeout: float = 30.0          # per-attempt request timeout
    deepseek_max_attempts: int = 3          # main answer generation
    deepseek_retry_base_delay: float = 0.5
    classifier_model: str = "deepseek-v4-flash"  # fast model, runs on every message
    classifier_max_attempts: int = 2        # fast classifier (every message)
    classifier_retry_base_delay: float = 0.4

    # Response settings
    random_response_probability: float = 0.1
    use_smart_respond: bool = False  # Use AI to decide whether to respond (costs API calls)
    text_only_mode: bool = False  # Only text responses (no stickers, GIFs, reactions)
    media_response_probability: float = 0.05  # chance of an unprompted sticker/reaction reply
    gif_response_probability: float = 0.015  # chance of an unprompted GIF reply (rarer — network round-trip)
    silent_reaction_probability: float = 0.03  # chance of a reaction-only "ack" instead of full silence on grade 0
    
    # Giphy settings
    giphy_api_url: str = "https://api.giphy.com/v1/gifs/search"
    giphy_limit: int = 10
    giphy_rating: str = "pg-13"
    
    # Scheduler settings
    nightly_analysis_hour: int = 3
    nightly_analysis_minute: int = 0
    timezone: str = "Europe/Kiev"

    # LightRAG (long-term knowledge base about people / facts in the chat)
    # LightRAG is deployed as a separate service; the bot only talks to it
    # over HTTP. LLM + embedding models are configured ON the LightRAG side.
    lightrag_enabled: bool = True
    lightrag_api_url: str = ""          # e.g. https://xxx.up.railway.app
    lightrag_api_user: str = ""         # AUTH_ACCOUNTS user
    lightrag_api_password: str = ""     # AUTH_ACCOUNTS password
    lightrag_query_mode: str = "mix"    # mix / hybrid / local / global / naive
    # Seeds for the two vector lookups (entities and relations); the graph walk
    # then expands from them. 5 was too few once the graph passed ~7000 nodes —
    # the whole answer hung on five guessed seeds. Raising it costs nothing at
    # query time because lightrag_query_max_tokens still caps what comes back.
    lightrag_query_top_k: int = 20
    # Cap on the retrieved context. Without it LightRAG returned 48-90k chars
    # per query and all of it went straight into the prompt, so a single
    # question with memory cost ~20k tokens. 14000 was measured as the lowest
    # budget that still keeps the answer-bearing details (10000 already dropped
    # "Кирилл — взлом систем" and "Пан — C1 польский"); 0 disables the cap.
    lightrag_query_max_tokens: int = 14000
    lightrag_query_timeout: float = 35.0
    lightrag_insert_timeout: float = 15.0

    # Nightly ingest: how often and how the chat is grouped before insert
    rag_ingest_enabled: bool = True
    rag_ingest_every_n_days: int = 1     # 1 = every night, 3-7 = less often
    rag_ingest_timeblock_minutes: int = 15  # group messages into N-min blocks
    rag_ingest_max_per_block: int = 30   # hard cap on messages per block

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Run the test suite once at startup and log the result. Purely
    # informational: the outcome is never allowed to stop the bot, because a
    # red test on Railway would otherwise turn into a restart loop.
    selftest_on_startup: bool = False


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
    last_seen: datetime = field(default_factory=lambda: get_now("UTC"))

    def to_dict(self) -> dict:
        """Convert to dictionary for Firebase storage."""
        return {
            "username": self.username,
            "last_seen": self.last_seen
        }
