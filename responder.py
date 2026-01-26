# responder.py
"""
Response handler for the DeepSeek Telegram bot.
Processes and sends responses in different formats: text, reaction, GIF, sticker.
"""

import logging
import random
from typing import Dict, Optional

import aiohttp
from telegram import Bot, Message, ReactionTypeEmoji
from telegram.error import TelegramError

from models import BotConfig, ParsedResponse, ResponseType

logger = logging.getLogger(__name__)


class ResponseParser:
    """Parses DeepSeek responses to determine type and content."""
    
    # Response prefixes
    PREFIX_GIPHY = "GIPHY:"
    PREFIX_REACT = "REACT:"
    PREFIX_STICKER = "STICKER:"
    
    @classmethod
    def parse(cls, response_text: str) -> ParsedResponse:
        """
        Parse the response to determine its type and content.
        
        Args:
            response_text: The response from DeepSeek (may be in special format)
        
        Returns:
            ParsedResponse with type and content
        """
        text = response_text.strip()

        if text.startswith(cls.PREFIX_GIPHY):
            content = text[len(cls.PREFIX_GIPHY):].strip()
            logger.info(f"Parsed GIPHY response: {content}")
            return ParsedResponse(ResponseType.GIF, content)

        if text.startswith(cls.PREFIX_REACT):
            content = text[len(cls.PREFIX_REACT):].strip()
            logger.info(f"Parsed REACT response: {content}")
            return ParsedResponse(ResponseType.REACTION, content)

        if text.startswith(cls.PREFIX_STICKER):
            content = text[len(cls.PREFIX_STICKER):].strip().lower()
            logger.info(f"Parsed STICKER response: {content}")
            return ParsedResponse(ResponseType.STICKER, content)

        logger.info(f"Parsed TEXT response: {text[:50]}")
        return ParsedResponse(ResponseType.TEXT, text)


class GiphyClient:
    """Async client for Giphy API."""
    
    def __init__(self, config: BotConfig):
        """
        Initialize Giphy client.
        
        Args:
            config: Bot configuration with Giphy settings
        """
        self.api_key = config.giphy_api_key
        self.api_url = config.giphy_api_url
        self.limit = config.giphy_limit
        self.rating = config.giphy_rating
    
    async def search(self, query: str) -> Optional[str]:
        """
        Search for a GIF and return a random result URL.
        
        Args:
            query: Search query
            
        Returns:
            GIF URL or None if not found
        """
        params = {
            'api_key': self.api_key,
            'q': query,
            'limit': self.limit,
            'rating': self.rating
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_url, params=params, timeout=5) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    if not data.get('data'):
                        logger.warning(f"No GIFs found for query: {query}")
                        return None
                    
                    gif = random.choice(data['data'])
                    return gif['images']['original']['url']
                    
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching from Giphy API: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Giphy search: {e}")
            return None


class StickerManager:
    """Manages sticker file IDs and sending."""
    
    # Default sticker mapping (emotion -> file_id)
    DEFAULT_STICKERS: Dict[str, str] = {
        'happy': 'CAACAgIAAxkBAAEQUVxpdIeyvxepv5LBpDDNIWszpN8JJQAC85oAAgRqgUshcX0t9I5SSDgE',
        'sad': '',
        'laugh': '',
        'cool': '',
        'think': '',
        'wtf': ''
    }
    
    def __init__(self, custom_stickers: Optional[Dict[str, str]] = None):
        """
        Initialize sticker manager.
        
        Args:
            custom_stickers: Optional custom sticker mapping to merge with defaults
        """
        self._stickers = self.DEFAULT_STICKERS.copy()
        if custom_stickers:
            self._stickers.update(custom_stickers)
    
    def get_file_id(self, emotion: str) -> Optional[str]:
        """
        Get sticker file_id for an emotion.
        
        Args:
            emotion: Emotion name (e.g., 'happy', 'sad')
            
        Returns:
            File ID or None if not found/empty
        """
        file_id = self._stickers.get(emotion.lower().strip(), '')
        return file_id if file_id else None
    
    def add_sticker(self, emotion: str, file_id: str) -> None:
        """
        Add or update a sticker mapping.
        
        Args:
            emotion: Emotion name
            file_id: Telegram sticker file ID
        """
        self._stickers[emotion.lower().strip()] = file_id
        logger.info(f"Sticker added: {emotion} -> {file_id[:20]}...")
    
    @property
    def available_emotions(self) -> list:
        """Get list of emotions with available stickers."""
        return [k for k, v in self._stickers.items() if v]


class Responder:
    """
    Handles sending different types of responses to Telegram.
    Supports: text, reactions, GIFs, and stickers.
    """
    
    # Keywords for intelligent reaction/text choice
    REACTION_KEYWORDS = ['реакц', 'поставь', 'на сообщ', 'ткни', 'set reaction', 'put reaction']
    TEXT_KEYWORDS = ['напиши', 'скинь', 'отправь', 'в чат', 'send', 'write', 'text']

    def __init__(
        self, 
        config: BotConfig,
        giphy_client: Optional[GiphyClient] = None,
        sticker_manager: Optional[StickerManager] = None
    ):
        """
        Initialize responder with dependencies.
        
        Args:
            config: Bot configuration
            giphy_client: Optional Giphy client (created if not provided)
            sticker_manager: Optional sticker manager (created if not provided)
        """
        self.config = config
        self._giphy = giphy_client or GiphyClient(config)
        self._stickers = sticker_manager or StickerManager()
        logger.info("Responder initialized")

    async def send_response(
        self,
        message: Message,
        response_text: str,
        bot: Bot
    ) -> bool:
        """
        Send response to chat based on its type.
        
        Args:
            message: Original Telegram message object
            response_text: Response text (may contain special prefixes)
            bot: Telegram bot instance
            
        Returns:
            True if response was sent successfully
        """
        parsed = ResponseParser.parse(response_text)

        try:
            if parsed.response_type == ResponseType.TEXT:
                return await self._send_text(message, parsed.content)
            elif parsed.response_type == ResponseType.REACTION:
                return await self._send_reaction(message, parsed.content)
            elif parsed.response_type == ResponseType.GIF:
                return await self._send_gif(message, parsed.content, bot)
            elif parsed.response_type == ResponseType.STICKER:
                return await self._send_sticker(message, parsed.content, bot)
            else:
                logger.warning(f"Unknown response type: {parsed.response_type}")
                return False
        except Exception as e:
            logger.error(f"Error sending response: {e}")
            return False

    async def _send_text(self, message: Message, text: str) -> bool:
        """
        Send a text message reply.
        
        Args:
            message: Message to reply to
            text: Text to send
            
        Returns:
            True if sent successfully
        """
        try:
            await message.reply_text(text)
            logger.info(f"Text response sent: {text[:50]}")
            return True
        except TelegramError as e:
            logger.error(f"Error sending text message: {e}")
            return False

    async def _send_reaction(self, message: Message, emoji: str) -> bool:
        """
        Smartly chooses between reaction or text based on user intent.
        
        Args:
            message: Message to react to
            emoji: Emoji to use
            
        Returns:
            True if sent successfully
        """
        user_text = (message.text or "").lower()
        
        # Determine intent
        wants_reaction = any(kw in user_text for kw in self.REACTION_KEYWORDS)
        wants_text = any(kw in user_text for kw in self.TEXT_KEYWORDS)

        # Decide action
        if wants_reaction and not wants_text:
            use_reaction = True
        elif wants_text and not wants_reaction:
            use_reaction = False
        else:
            # Ambiguous - use 50/50 chance
            use_reaction = random.choice([True, False])

        if use_reaction:
            try:
                await message.set_reaction(
                    reaction=[ReactionTypeEmoji(emoji=emoji)],
                    is_big=False
                )
                logger.info(f"Reaction set: {emoji}")
                return True
            except Exception as e:
                logger.warning(f"Reaction failed: {e}. Falling back to text.")

        # Fallback to text
        return await self._send_text(message, emoji)

    async def _send_gif(self, message: Message, search_query: str, bot: Bot) -> bool:
        """
        Search Giphy API and send a random GIF.
        Falls back to text if GIF not found.
        
        Args:
            message: Message to reply to
            search_query: Search query for Giphy
            bot: Bot instance
            
        Returns:
            True if sent successfully
        """
        gif_url = await self._giphy.search(search_query)
        
        if gif_url:
            try:
                await bot.send_animation(
                    chat_id=message.chat_id,
                    animation=gif_url
                )
                logger.info(f"GIF sent for query: {search_query}")
                return True
            except TelegramError as e:
                logger.error(f"Error sending animation: {e}")
        
        # Fallback to text
        logger.info(f"GIF fallback to text: {search_query}")
        return await self._send_text(message, search_query)

    async def _send_sticker(self, message: Message, emotion: str, bot: Bot) -> bool:
        """
        Send a sticker based on emotion.
        Falls back to text if sticker file_id not available.
        
        Args:
            message: Message to reply to
            emotion: Emotion name (key in sticker map)
            bot: Bot instance
            
        Returns:
            True if sent successfully
        """
        file_id = self._stickers.get_file_id(emotion)

        if file_id:
            try:
                await bot.send_sticker(
                    chat_id=message.chat_id,
                    sticker=file_id
                )
                logger.info(f"Sticker sent for emotion: {emotion}")
                return True
            except TelegramError as e:
                logger.error(f"Error sending sticker: {e}")

        # Fallback to text
        logger.info(f"Sticker fallback to text: {emotion}")
        return await self._send_text(message, emotion)

    @property
    def sticker_manager(self) -> StickerManager:
        """Get the sticker manager for adding custom stickers."""
        return self._stickers
