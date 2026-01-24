# brain.py
"""
Brain module for the DeepSeek Telegram bot.
Handles decision-making and response generation using DeepSeek API.
"""

import logging
import random
from typing import Optional

from openai import OpenAI

import config

logger = logging.getLogger(__name__)


class Brain:
    """
    AI logic for the bot using DeepSeek API.
    Makes decisions about when to respond and generates responses.
    """

    SYSTEM_PROMPT = f"""Ты {config.BOT_NAME} - дружелюбный бот-участник группового чата. Ты говоришь как обычный человек с друзьями.

Характер:
- Дружелюбный, немного саркастичный
- Интересуешься разговорами в чате
- Любишь шутки и мемы
- Маленькими буквами, можно без точек в конце
- Можешь использовать сленг (кек, лол, имхо, норм, збс и тп)
- Никаких списков и перечислений
- Никогда не используй фразы "как AI я..." или "я не могу..."

Форматы ответа (выбери ОДИН):
1. Обычный текст для нормального ответа
2. "REACT:<эмодзи>" только для реакции (без текста после)
3. "GIPHY:<запрос на английском>" для гифки (без текста после)
4. "STICKER:<эмоция>" для стикера (без текста после)

Доступные стикеры: happy, sad, laugh, cool, think, wtf

Примеры правильных ответов:
- "кто за пиццу сегодня?" → "я за"
- "посмотрите какую машину увидел" [фото] → "REACT:🔥"
- "блин уронил телефон в унитаз" → "GIPHY:facepalm"
- "сдал экзамен на отлично!" → "STICKER:cool"
- "че думаете про новый фильм?" → "не смотрел ещё, он зашёл?"
- "согласны?" → "REACT:👍"

ВАЖНО:
- Пиши КОРОТКО (1-2 предложения достаточно)
- Не более 3 предложений в любом случае
- Не повторяй один и тот же ответ
"""

    def __init__(self):
        """Initialize DeepSeek API client."""
        try:
            self.client = OpenAI(
                api_key=config.DEEPSEEK_API_KEY,
                base_url=config.DEEPSEEK_BASE_URL
            )
            logger.info("Brain initialized: DeepSeek client ready")
        except Exception as e:
            logger.error(f"Failed to initialize Brain: {e}")
            raise

    def should_respond(self, message_text: str, recent_messages: list = None) -> bool:
        """
        Determine if the bot should respond to a message.
        
        Conditions:
        1. Message mentions bot name (case-insensitive)
        2. Message contains question mark "?"
        3. Random 10% chance on any message
        
        Args:
            message_text: The received message text
            recent_messages: List of recent messages (for context, unused for now)
        
        Returns:
            True if bot should respond, False otherwise
        """
        # Check if bot name is mentioned (case-insensitive)
        if config.BOT_NAME.lower() in message_text.lower():
            logger.info(f"Should respond: bot name mentioned in '{message_text[:50]}'")
            return True

        # Check if message contains question mark
        if "?" in message_text:
            logger.info(f"Should respond: question mark in '{message_text[:50]}'")
            return True

        # Random 10% chance on any message
        random_chance = random.random()
        if random_chance < config.RANDOM_RESPONSE_PROBABILITY:
            logger.info(f"Should respond: random chance ({random_chance:.2%}) in '{message_text[:50]}'")
            return True

        return False

    def generate_response(self, message_text: str, context: str) -> str:
        """
        Generate a response using DeepSeek API.
        
        Args:
            message_text: The current message to respond to
            context: Formatted recent messages as context
        
        Returns:
            Generated response text (may contain REACT:, GIPHY:, STICKER: prefixes)
        """
        try:
            # Prepare messages for API
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"""Последние сообщения в чате:
{context}

Новое сообщение для ответа: {message_text}"""
                }
            ]

            # Call DeepSeek API
            response = self.client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=messages,
                max_tokens=config.DEEPSEEK_MAX_TOKENS,
                temperature=config.DEEPSEEK_TEMPERATURE
            )

            # Extract response text
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated response: {answer[:50]}")
            return answer

        except Exception as e:
            logger.error(f"Error generating response from DeepSeek: {e}")
            return "Бабки закончились так что ответов больше не будет."  # Fallback response
