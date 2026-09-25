# memory.py
"""
Memory management for the DeepSeek Telegram bot.
Handles short-term (in-RAM) memory and a per-day daily log.
Now includes bot's own responses in short-term memory.

Long-term storage (Firestore) was removed: the bot runs with RAM-only memory
plus LightRAG for durable knowledge, so there is no storage backend here.
"""

import logging
from collections import deque
from datetime import datetime
from typing import List, Optional, Deque

from models import ChatMessage, BotConfig
from utils import get_now, to_aware

logger = logging.getLogger(__name__)


class RecentResponseTracker:
    """
    Tracks recent bot responses to avoid repetition.
    Stores last N emojis, gifs queries, and text snippets.
    """
    
    def __init__(self, max_items: int = 10):
        """
        Initialize tracker.
        
        Args:
            max_items: Maximum items to track per category
        """
        self._emojis: Deque[str] = deque(maxlen=max_items)
        self._gifs: Deque[str] = deque(maxlen=max_items)
        self._texts: Deque[str] = deque(maxlen=max_items)
        self._all_responses: Deque[str] = deque(maxlen=max_items * 2)
    
    def add_response(self, response_type: str, content: str) -> None:
        """
        Add a response to the tracker.
        
        Args:
            response_type: Type of response (text, reaction, gif, sticker)
            content: The response content
        """
        self._all_responses.append(content)
        
        if response_type == "reaction":
            self._emojis.append(content)
        elif response_type == "gif":
            self._gifs.append(content.lower())
        elif response_type == "text":
            # Store first 50 chars for comparison
            self._texts.append(content[:50].lower())
    
    def get_avoid_list(self) -> List[str]:
        """Get list of recent responses to avoid."""
        return list(self._all_responses)


class Memory:
    """
    Manages bot memory:
    - Short-term: Python deque (fast, limited to N messages)
    - Daily log: all messages of the current day in RAM (used by RAG ingest)

    Bot's own responses are stored in short-term memory only.
    """

    # Special user ID for bot's own messages
    BOT_USER_ID = -1

    def __init__(self, config: BotConfig):
        """
        Initialize memory with configuration.

        Args:
            config: Bot configuration
        """
        self.config = config
        self._short_term: Deque[ChatMessage] = deque(maxlen=config.short_memory_limit)
        self._bot_name = config.bot_name

        # Daily log for nightly analysis (stores all messages for the current day)
        self._daily_log: List[ChatMessage] = []

        logger.info(
            f"Memory initialized: short-term limit={config.short_memory_limit}, "
            f"daily-log=enabled"
        )

    def add_message(
        self,
        user_id: int,
        username: str,
        text: str,
        message_id: int,
        reply_to_text: Optional[str] = None,
        chat_id: Optional[int] = None,
    ) -> ChatMessage:
        """
        Add a new message to short-term memory and the daily log.

        Args:
            user_id: Telegram user ID
            username: Username or first name
            text: Message text
            message_id: Telegram message ID
            reply_to_text: If this message is a reply, the text of the replied-to
                message. Kept on the ChatMessage so it shows up in context lines
                and in the nightly RAG ingest (reply context stays with its block).
            chat_id: Telegram chat ID (identifies which group the message is from).

        Returns:
            Created ChatMessage instance
        """
        # Create message object (timezone-aware, in the configured timezone)
        now = get_now(self.config.timezone)
        message = ChatMessage(
            user_id=user_id,
            username=username,
            text=text,
            message_id=message_id,
            timestamp=now,
            reply_to_text=reply_to_text,
            chat_id=chat_id,
        )

        # Add to short-term memory (deque auto-trims to maxlen)
        self._short_term.append(message)
        
        # Add to daily log ONLY if message is from today (in configured timezone).
        # This prevents counter corruption after bot restart/redeploy, and keeps
        # "today" stable across servers that may run in UTC (Railway/Render).
        message_date = to_aware(message.timestamp, self.config.timezone).date()
        today = now.date()
        
        if message_date == today:
            self._daily_log.append(message)
            logger.debug(f"Added to daily log: {username}")
        else:
            logger.debug(f"Message from different day ({message_date}), skipping daily log")

        logger.info(f"Message added - {username}: {text[:50]}")
        return message

    def add_bot_response(self, text: str, message_id: int = 0) -> ChatMessage:
        """
        Add bot's own response to short-term memory only.
        This allows the bot to see what it said previously.

        Args:
            text: Bot's response text
            message_id: Telegram message ID (optional)

        Returns:
            Created ChatMessage instance
        """
        return self.add_message(
            user_id=self.BOT_USER_ID,
            username=self._bot_name,
            text=text,
            message_id=message_id,
        )

    def get_recent(self, count: Optional[int] = None) -> List[ChatMessage]:
        """
        Get the most recent messages from short-term memory.

        Args:
            count: Number of messages to retrieve (defaults to context_messages_count)

        Returns:
            List of ChatMessage objects
        """
        if count is None:
            count = self.config.context_messages_count
        # Convert deque to list for slicing (deque doesn't support slice indexing)
        messages_list = list(self._short_term)
        return messages_list[-count:] if messages_list else []

    def get_recent_context_lines(self) -> List[str]:
        """
        Return recent messages already formatted as context lines.

        This is what the V2 classifier and brain want: a list of
        "Name: text" (with reply context inlined) strings, not one blob.

        Returns:
            List of formatted context strings (newest last), empty if none.
        """
        recent = self.get_recent()
        return [msg.to_context_line() for msg in recent] if recent else []

    def get_messages_for_period(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        chat_id_filter: Optional[int] = None,
    ) -> List[ChatMessage]:
        """
        Fetch all messages in a time window for the nightly RAG ingest.

        Source: daily_log (RAM only — empty after a restart). **Bot messages
        are included** so the ingest block preserves conversation context (e.g.
        bot asked "нравится ли тебе кока кола?" and people replied "да" — both
        sides matter).

        Args:
            since: Inclusive lower bound (timezone-aware recommended). If None,
                no lower bound.
            until: Exclusive upper bound. If None, now.
            chat_id_filter: If set, only return messages from this chat.

        Returns:
            Chronologically ordered list of ChatMessage in the window.
        """
        tz = self.config.timezone
        if until is None:
            until = get_now(tz)

        messages: List[ChatMessage] = []

        for msg in self._daily_log:
            ts = to_aware(msg.timestamp, tz)
            if since is not None and ts < since:
                continue
            if ts >= until:
                continue
            if chat_id_filter is not None and msg.chat_id != chat_id_filter:
                continue
            messages.append(msg)

        messages.sort(key=lambda m: m.timestamp)
        return messages

    def bot_responded_recently(self, within_last_n: int = 3) -> bool:
        """
        Check if the bot responded within the last N messages.
        Used for conversation continuation without name mention.
        
        Args:
            within_last_n: Number of messages to look back
            
        Returns:
            True if bot responded recently
        """
        recent = list(self._short_term)[-within_last_n:]
        return any(msg.user_id == self.BOT_USER_ID for msg in recent)
