# memory.py
"""
Memory management for the DeepSeek Telegram bot.
Handles both short-term (in-RAM) and long-term (Firebase) memory.
Now includes bot's own responses in short-term memory.
"""

import logging
from collections import deque
from datetime import datetime
from typing import List, Optional, Deque
from abc import ABC, abstractmethod
import json

import firebase_admin
from firebase_admin import credentials, firestore

from models import ChatMessage, UserInfo, BotConfig
from prompts import FALLBACK_RESPONSES

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
    
    def is_recently_used(self, response_type: str, content: str) -> bool:
        """
        Check if a response was recently used.
        
        Args:
            response_type: Type of response
            content: Content to check
            
        Returns:
            True if recently used
        """
        if response_type == "reaction":
            return content in self._emojis
        elif response_type == "gif":
            return content.lower() in self._gifs
        elif response_type == "text":
            return content[:50].lower() in self._texts
        return False
    
    def get_avoid_list(self) -> List[str]:
        """Get list of recent responses to avoid."""
        return list(self._all_responses)
    
    def get_recent_emojis(self) -> List[str]:
        """Get recently used emojis."""
        return list(self._emojis)
    
    def clear(self) -> None:
        """Clear all tracked responses."""
        self._emojis.clear()
        self._gifs.clear()
        self._texts.clear()
        self._all_responses.clear()


class MemoryStorage(ABC):
    """Abstract base class for memory storage backends."""
    
    @abstractmethod
    def save_message(self, message: ChatMessage) -> None:
        """Save a message to storage."""
        pass
    
    @abstractmethod
    def update_user(self, user: UserInfo) -> None:
        """Update user information in storage."""
        pass
    
    @abstractmethod
    def get_client(self):
        """Get the underlying database client."""
        pass


class FirebaseStorage(MemoryStorage):
    """Firebase Firestore storage backend."""
    
    def __init__(self, cred_path: str):
        """
        Initialize Firebase connection.
        
        Args:
            cred_path: Path to Firebase credentials JSON file or JSON string
        """
        try:
            if not firebase_admin._apps:
                # Check if cred_path is JSON string (starts with '{') or file path
                if cred_path.strip().startswith('{'):
                    # It's a JSON string, parse it
                    cred_dict = json.loads(cred_path)
                    cred = credentials.Certificate(cred_dict)
                else:
                    # It's a file path
                    cred = credentials.Certificate(cred_path)
                
                firebase_admin.initialize_app(cred)
            
            self.db = firestore.client()
            logger.info("Firebase storage initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            raise
    
    def save_message(self, message: ChatMessage) -> None:
        """
        Save message to Firebase messages collection.
        
        Args:
            message: ChatMessage to save
        """
        try:
            self.db.collection('messages').add(message.to_dict())
            logger.debug(f"Message saved to Firebase: {message.text[:50]}")
        except Exception as e:
            logger.error(f"Error saving message to Firebase: {e}")
    
    def update_user(self, user: UserInfo) -> None:
        """
        Update user info in Firebase users collection.
        
        Args:
            user: UserInfo to update
        """
        try:
            self.db.collection('users').document(str(user.user_id)).set(
                user.to_dict(),
                merge=True
            )
            logger.debug(f"User updated in Firebase: {user.username}")
        except Exception as e:
            logger.error(f"Error updating user in Firebase: {e}")
    
    def get_client(self):
        """Get Firebase client for other modules."""
        return self.db


class Memory:
    """
    Manages bot memory with two tiers:
    - Short-term: Python list (fast, limited to N messages)
    - Long-term: Firebase Firestore (persistent, user messages only)
    
    Bot's own responses are stored in short-term memory only.
    """

    # Special user ID for bot's own messages
    BOT_USER_ID = -1

    def __init__(self, config: BotConfig, storage: Optional[MemoryStorage] = None):
        """
        Initialize memory with configuration.
        
        Args:
            config: Bot configuration
            storage: Optional storage backend (defaults to Firebase)
        """
        self.config = config
        self._short_term: Deque[ChatMessage] = deque(maxlen=config.short_memory_limit)
        self._bot_name = config.bot_name
        
        # Daily log for nightly analysis (stores all messages for the current day)
        self._daily_log: List[ChatMessage] = []
        
        # Initialize storage backend
        if storage is not None:
            self._storage = storage
        else:
            try:
                self._storage = FirebaseStorage(config.firebase_cred_path)
            except Exception as e:
                logger.warning(f"Firebase unavailable, running without long-term memory: {e}")
                self._storage = None
        
        logger.info(
            f"Memory initialized: short-term limit={config.short_memory_limit}, "
            f"long-term={'enabled' if self._storage else 'disabled'}, "
            f"daily-log=enabled"
        )

    @property
    def short_term_memory(self) -> Deque[ChatMessage]:
        """Get short-term memory (for backward compatibility)."""
        return self._short_term

    @property
    def storage(self) -> Optional[MemoryStorage]:
        """Get storage backend for other modules."""
        return self._storage

    def add_message(
        self,
        user_id: int,
        username: str,
        text: str,
        message_id: int,
        save_to_firebase: bool = True
    ) -> ChatMessage:
        """
        Add a new message to both short-term and long-term memory.

        Args:
            user_id: Telegram user ID
            username: Username or first name
            text: Message text
            message_id: Telegram message ID
            save_to_firebase: Whether to save to long-term storage (default: True)
            
        Returns:
            Created ChatMessage instance
        """
        # Create message object
        message = ChatMessage(
            user_id=user_id,
            username=username,
            text=text,
            message_id=message_id,
            timestamp=datetime.now()
        )

        # Add to short-term memory (deque auto-trims to maxlen)
        self._short_term.append(message)
        
        # Add to daily log ONLY if message is from today
        # This prevents counter corruption after bot restart/redeploy
        message_date = message.timestamp.date()
        today = datetime.now().date()
        
        if message_date == today:
            self._daily_log.append(message)
            logger.debug(f"Added to daily log: {username}")
        else:
            logger.debug(f"Message from different day ({message_date}), skipping daily log")

        # Save to long-term storage (only for user messages, not bot responses)
        if save_to_firebase and self._storage:
            self._storage.save_message(message)
            
            # Update user info
            user = UserInfo(
                user_id=user_id,
                username=username,
                last_seen=datetime.now()
            )
            self._storage.update_user(user)

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
            save_to_firebase=False  # Don't save bot responses to Firebase
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

    def get_context(self) -> str:
        """
        Format recent messages as context string for DeepSeek API.
        Includes both user messages and bot's own responses.

        Returns:
            Formatted string like "User1: message text\\nBot: response\\n..."
        """
        recent = self.get_recent()
        
        if not recent:
            return FALLBACK_RESPONSES["no_context"]

        context_lines = [msg.to_context_line() for msg in recent]
        return "\n".join(context_lines)

    def clear_short_memory(self) -> None:
        """Clear short-term memory (useful for testing)."""
        self._short_term.clear()
        logger.info("Short-term memory cleared")
    
    def get_message_count(self) -> int:
        """Get current number of messages in short-term memory."""
        return len(self._short_term)
    
    def get_user_messages_today(self, user_id: int) -> List[ChatMessage]:
        """
        Get all messages from a specific user for the current day.
        Retrieves from _daily_log ensuring full day context.
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            List of user's messages
        """
        if hasattr(self, '_daily_log'):
            return [msg for msg in self._daily_log if msg.user_id == user_id]
        return [msg for msg in self._short_term if msg.user_id == user_id]
    
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
    
    def get_last_bot_response(self) -> Optional[str]:
        """
        Get the last response from the bot.
        
        Returns:
            Last bot response text or None
        """
        for msg in reversed(list(self._short_term)):
            if msg.user_id == self.BOT_USER_ID:
                return msg.text
        return None

    def get_daily_log(self) -> List[ChatMessage]:
        """
        Get all messages from daily log.
        
        Returns:
            List of all messages from today
        """
        if hasattr(self, '_daily_log'):
            return list(self._daily_log)
        return []

    def clear_daily_log(self) -> None:
        """
        Clear daily log of messages older than 24 hours (or since last midnight).
        Used to prevent RAM from growing indefinitely.
        """
        if hasattr(self, '_daily_log'):
            # Keep messages from current day (since midnight)
            today_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Filter in place
            self._daily_log = [msg for msg in self._daily_log if msg.timestamp >= today_midnight]
            
            self._last_log_clear = datetime.now()
            logger.info(f"Daily log pruned. Retained {len(self._daily_log)} messages.")
