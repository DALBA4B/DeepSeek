# memory.py
"""
Memory management for the DeepSeek Telegram bot.
Handles both short-term (in-RAM) and long-term (Firebase) memory.
Now includes bot's own responses in short-term memory.
"""

import logging
import re
from collections import deque
from datetime import datetime
from typing import List, Optional, Deque
from abc import ABC, abstractmethod
import json

import firebase_admin
from firebase_admin import credentials, firestore

from models import ChatMessage, UserInfo, BotConfig
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


def _redact_credentials(text: str, secret: str) -> str:
    """
    Strip a credential blob out of an error message.

    Firebase errors quote the offending value in full, so a malformed
    FIREBASE_CRED_JSON would otherwise publish the service-account private key
    into the log stream. Both the whole blob and any PEM body inside it are
    replaced; the surrounding message is kept so the error stays diagnosable.
    """
    cleaned = text
    if secret:
        cleaned = cleaned.replace(secret, "<credentials redacted>")
        # The value may also appear with its outer braces stripped.
        inner = secret.strip().lstrip("{").rstrip("}")
        if len(inner) > 40:
            cleaned = cleaned.replace(inner, "<credentials redacted>")
    cleaned = re.sub(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        "<private key redacted>",
        cleaned,
        flags=re.DOTALL,
    )
    # Belt and braces: the blob may reach us reformatted (re-quoted, escaped,
    # partially clipped) so an exact-match replace misses it. Any surviving
    # service-account field name means secret material is still in the string.
    if re.search(r'private_key|private_key_id|"type"\s*:\s*"service_account', cleaned):
        return "<service account credentials redacted>"
    return cleaned


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
            # Never log the exception verbatim: when the credential blob is
            # malformed, the libraries echo the whole value back — private key
            # included — straight into the logs.
            logger.error(
                "Failed to initialize Firebase: %s: %s",
                type(e).__name__, _redact_credentials(str(e), cred_path),
            )
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
                logger.warning(
                    "Firebase unavailable, running without long-term memory: %s: %s",
                    type(e).__name__,
                    _redact_credentials(str(e), config.firebase_cred_path or ""),
                )
                self._storage = None
        
        logger.info(
            f"Memory initialized: short-term limit={config.short_memory_limit}, "
            f"long-term={'enabled' if self._storage else 'disabled'}, "
            f"daily-log=enabled"
        )

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
        save_to_firebase: bool = True,
        reply_to_text: Optional[str] = None,
        chat_id: Optional[int] = None,
    ) -> ChatMessage:
        """
        Add a new message to both short-term and long-term memory.

        Args:
            user_id: Telegram user ID
            username: Username or first name
            text: Message text
            message_id: Telegram message ID
            save_to_firebase: Whether to save to long-term storage (default: True)
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

        # Save to long-term storage (only for user messages, not bot responses)
        if save_to_firebase and self._storage:
            self._storage.save_message(message)
            
            # Update user info
            user = UserInfo(
                user_id=user_id,
                username=username,
                last_seen=now
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

        Source priority: Firebase (durable across restarts) → daily_log (RAM
        fallback when Firebase is unavailable). **Bot messages are included** so
        the ingest block preserves conversation context (e.g. bot asked "нравится
        ли тебе кока кола?" and people replied "да" — both sides matter).

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

        # Try Firebase first (survives restarts/redeploys)
        if self._storage is not None:
            try:
                db = self._storage.get_client()
                if db is not None:
                    # Server-side filtering by date keeps the query fast and cheap
                    # even when the collection grows to thousands of documents.
                    since_date = since.date() if since is not None else None
                    until_date = until.date() if until is not None else None

                    query = db.collection("messages")

                    if since_date is not None:
                        query = query.where("date", ">=", since_date.isoformat())
                    if until_date is not None:
                        query = query.where("date", "<=", until_date.isoformat())
                    if chat_id_filter is not None:
                        query = query.where("chat_id", "==", chat_id_filter)

                    docs = list(query.stream())
                    for doc in docs:
                        msg = ChatMessage.from_dict(doc.to_dict())
                        # Re-filter in Python for sub-day precision and chat_id
                        # (older docs may not have chat_id at all).
                        ts = to_aware(msg.timestamp, tz)
                        if since is not None and ts < since:
                            continue
                        if ts >= until:
                            continue
                        if chat_id_filter is not None and msg.chat_id != chat_id_filter:
                            continue
                        messages.append(msg)
                    if messages:
                        logger.info(
                            "Firebase returned %d messages in the ingest window",
                            len(messages),
                        )
                    messages.sort(key=lambda m: m.timestamp)
                    return messages
            except Exception as e:
                logger.warning(f"Firebase query for period failed, falling back to daily_log: {e}")

        # Fallback: daily_log (RAM only — empty after a restart)
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
