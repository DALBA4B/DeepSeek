# memory.py
"""
Memory management for the DeepSeek Telegram bot.
Handles both short-term (in-RAM) and long-term (Firebase) memory.
"""

import logging
from datetime import datetime
from typing import List, Dict, Optional

import firebase_admin
from firebase_admin import credentials, firestore

import config

logger = logging.getLogger(__name__)


class Memory:
    """
    Manages bot memory with two tiers:
    - Short-term: Python list (fast, limited to 30 messages)
    - Long-term: Firebase Firestore (persistent)
    """

    def __init__(self):
        """Initialize Firebase connection and short-term memory list."""
        try:
            # Initialize Firebase if not already done
            if not firebase_admin._apps:
                cred = credentials.Certificate(config.FIREBASE_CRED_PATH)
                firebase_admin.initialize_app(cred)
            
            self.db = firestore.client()
            self.short_term_memory: List[Dict] = []
            self.short_memory_limit = config.SHORT_MEMORY_LIMIT
            
            logger.info("Memory initialized: Firebase connected, short-term memory ready")
        except Exception as e:
            logger.error(f"Failed to initialize Memory: {e}")
            self.db = None
            raise

    def add_message(
        self,
        user_id: int,
        username: str,
        text: str,
        message_id: int
    ) -> None:
        """
        Add a new message to both short-term and long-term memory.

        Args:
            user_id: Telegram user ID
            username: Username or first name
            text: Message text
            message_id: Telegram message ID
        """
        try:
            # Create message object
            message = {
                "user_id": user_id,
                "username": username,
                "text": text,
                "message_id": message_id,
                "timestamp": datetime.now()
            }

            # Add to short-term memory (in-RAM list)
            self.short_term_memory.append(message)
            
            # Remove oldest message if limit exceeded
            if len(self.short_term_memory) > self.short_memory_limit:
                self.short_term_memory.pop(0)

            # Save to Firebase long-term memory
            if self.db:
                self.db.collection('messages').add({
                    'user_id': user_id,
                    'username': username,
                    'text': text,
                    'message_id': message_id,
                    'timestamp': datetime.now()
                })

                # Update user info
                self.db.collection('users').document(str(user_id)).set({
                    'username': username,
                    'last_seen': datetime.now()
                }, merge=True)

                logger.info(f"Message saved - {username}: {text[:50]}")
            
        except Exception as e:
            logger.error(f"Error adding message to memory: {e}")

    def get_recent(self, count: int = 20) -> List[Dict]:
        """
        Get the most recent messages from short-term memory.

        Args:
            count: Number of messages to retrieve

        Returns:
            List of message dictionaries
        """
        return self.short_term_memory[-count:] if self.short_term_memory else []

    def get_context(self) -> str:
        """
        Format recent messages as context string for DeepSeek API.

        Returns:
            Formatted string like "User1: message text\nUser2: message text\n..."
        """
        recent = self.get_recent(config.CONTEXT_MESSAGES_COUNT)
        
        if not recent:
            return "Нет предыдущих сообщений в чате."

        context_lines = []
        for msg in recent:
            context_lines.append(f"{msg['username']}: {msg['text']}")
        
        return "\n".join(context_lines)

    def clear_short_memory(self) -> None:
        """Clear short-term memory (useful for testing)."""
        self.short_term_memory.clear()
        logger.info("Short-term memory cleared")
