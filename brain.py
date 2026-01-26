# brain.py
"""
Brain module for the DeepSeek Telegram bot.
Handles decision-making and response generation using DeepSeek API.
Now integrates with knowledge graphs for personalized responses.
"""

import logging
import random
from typing import List, Optional

from openai import OpenAI

from models import BotConfig, ChatMessage
from prompts import get_system_prompt, get_context_prompt, BOT_NAME_VARIATIONS, FALLBACK_RESPONSES
from knowledge_graph import KnowledgeGraphManager, TopicDetector

logger = logging.getLogger(__name__)


class Brain:
    """
    AI logic for the bot using DeepSeek API.
    Makes decisions about when to respond and generates responses.
    Integrates with knowledge graphs for personalized context.
    """

    def __init__(
        self, 
        config: BotConfig, 
        available_stickers: Optional[List[str]] = None,
        knowledge_manager: Optional[KnowledgeGraphManager] = None
    ):
        """
        Initialize DeepSeek API client.
        
        Args:
            config: Bot configuration
            available_stickers: List of available sticker emotions for the prompt
            knowledge_manager: Optional KnowledgeGraphManager for personalized context
        """
        self.config = config
        self._available_stickers = available_stickers or ["happy", "sad", "laugh", "cool", "think", "wtf"]
        self._knowledge_manager = knowledge_manager
        self._topic_detector = TopicDetector()
        
        try:
            self.client = OpenAI(
                api_key=config.deepseek_api_key,
                base_url=config.deepseek_base_url
            )
            self._system_prompt = get_system_prompt(
                config.bot_name, 
                self._available_stickers
            )
            logger.info("Brain initialized: DeepSeek client ready")
        except Exception as e:
            logger.error(f"Failed to initialize Brain: {e}")
            raise

    def set_knowledge_manager(self, manager: KnowledgeGraphManager) -> None:
        """
        Set the knowledge graph manager.
        
        Args:
            manager: KnowledgeGraphManager instance
        """
        self._knowledge_manager = manager
        logger.info("Knowledge manager set for Brain")

    def should_respond(
        self, 
        message_text: str, 
        recent_messages: Optional[List[ChatMessage]] = None
    ) -> bool:
        """
        Determine if the bot should respond to a message.
        
        Conditions (in order of priority):
        1. Message mentions bot name or similar variations (case-insensitive)
        2. Message contains question mark "?"
        3. Random chance based on config probability
        
        Args:
            message_text: The received message text
            recent_messages: List of recent messages (for future context-aware decisions)
        
        Returns:
            True if bot should respond, False otherwise
        """
        message_lower = message_text.lower()
        
        # Check if bot name variations are mentioned
        for variation in BOT_NAME_VARIATIONS:
            if variation in message_lower:
                logger.info(f"Should respond: bot name '{variation}' mentioned")
                return True

        # Check if message contains question mark
        if "?" in message_text:
            logger.info("Should respond: question mark detected")
            return True

        # Random chance
        random_value = random.random()
        if random_value < self.config.random_response_probability:
            logger.info(f"Should respond: random chance ({random_value:.2%})")
            return True

        logger.debug(f"Should not respond to: {message_text[:50]}")
        return False

    def generate_response(
        self, 
        message_text: str, 
        context: str,
        user_id: Optional[int] = None,
        username: Optional[str] = None
    ) -> str:
        """
        Generate a response using DeepSeek API.
        Includes personalized context from knowledge graph if available.
        
        Args:
            message_text: The current message to respond to
            context: Formatted recent messages as context
            user_id: Optional user ID for personalized context
            username: Optional username for personalized context
        
        Returns:
            Generated response text (may contain REACT:, GIPHY:, STICKER: prefixes)
        """
        try:
            # Build enhanced context with knowledge graph
            enhanced_context = context
            
            if self._knowledge_manager and user_id:
                # Get personalized context from knowledge graph
                kg_context = self._knowledge_manager.get_relevant_context_for_message(
                    user_id=user_id,
                    message_text=message_text,
                    username=username or "Unknown"
                )
                
                if kg_context:
                    enhanced_context = f"ИНФОРМАЦИЯ О ПОЛЬЗОВАТЕЛЯХ:\n{kg_context}\n\nПОСЛЕДНИЕ СООБЩЕНИЯ:\n{context}"
                    logger.debug(f"Added knowledge graph context for user {user_id}")
            
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": get_context_prompt(enhanced_context, message_text)}
            ]

            response = self.client.chat.completions.create(
                model=self.config.deepseek_model,
                messages=messages,
                max_tokens=self.config.deepseek_max_tokens,
                temperature=self.config.deepseek_temperature
            )

            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated response: {answer[:50]}")
            return answer

        except Exception as e:
            logger.error(f"Error generating response from DeepSeek: {e}")
            return FALLBACK_RESPONSES["api_error"]

    def update_system_prompt(self, new_prompt: str) -> None:
        """
        Update the system prompt dynamically.
        
        Args:
            new_prompt: New system prompt to use
        """
        self._system_prompt = new_prompt
        logger.info("System prompt updated")

    @property
    def available_stickers(self) -> List[str]:
        """Get list of available sticker emotions."""
        return self._available_stickers
    
    @property
    def knowledge_manager(self) -> Optional[KnowledgeGraphManager]:
        """Get the knowledge graph manager."""
        return self._knowledge_manager
