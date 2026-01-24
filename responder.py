# responder.py
"""
Response handler for the DeepSeek Telegram bot.
Processes and sends responses in different formats: text, reaction, GIF, sticker.
"""

import logging
import random
from typing import Tuple, Optional

import requests
from telegram import Bot, Message
from telegram.error import TelegramError

import config

logger = logging.getLogger(__name__)


class Responder:
    """
    Handles sending different types of responses to Telegram.
    Supports: text, reactions, GIFs, and stickers.
    """

    # Mapping of emotions to sticker file IDs
    STICKER_MAP = {
        'happy': 'CAACAgIAAxkBAAEQUVxpdIeyvxepv5LBpDDNIWszpN8JJQAC85oAAgRqgUshcX0t9I5SSDgE',
        'sad': '',
        'laugh': '',
        'cool': '',
        'think': '',
        'wtf': ''
    }

    @staticmethod
    def parse_response(response_text: str) -> Tuple[str, str]:
        """
        Parse the response to determine its type and content.
        
        Args:
            response_text: The response from DeepSeek (may be in special format)
        
        Returns:
            Tuple of (response_type, content)
            Types: 'text', 'reaction', 'gif', 'sticker'
        """
        response_text = response_text.strip()

        # Check for GIPHY format
        if response_text.startswith("GIPHY:"):
            search_query = response_text[6:].strip()
            logger.info(f"Parsed GIPHY response: {search_query}")
            return "gif", search_query

        # Check for REACT format
        if response_text.startswith("REACT:"):
            emoji = response_text[6:].strip()
            logger.info(f"Parsed REACT response: {emoji}")
            return "reaction", emoji

        # Check for STICKER format
        if response_text.startswith("STICKER:"):
            emotion = response_text[8:].strip().lower()
            logger.info(f"Parsed STICKER response: {emotion}")
            return "sticker", emotion

        # Default to text
        logger.info(f"Parsed TEXT response: {response_text[:50]}")
        return "text", response_text

    @staticmethod
    async def send_response(
        message: Message,
        response_text: str,
        bot: Bot
    ) -> None:
        """
        Send response to chat based on its type.
        
        Args:
            message: Original Telegram message object
            response_text: Response text (may contain special prefixes)
            bot: Telegram bot instance
        """
        response_type, content = Responder.parse_response(response_text)

        try:
            if response_type == "text":
                await Responder._send_text(message, content, bot)
            elif response_type == "reaction":
                await Responder._send_reaction(message, content, bot)
            elif response_type == "gif":
                await Responder._send_gif(message, content, bot)
            elif response_type == "sticker":
                await Responder._send_sticker(message, content, bot)
        except Exception as e:
            logger.error(f"Error sending response: {e}")

    @staticmethod
    async def _send_text(message: Message, text: str, bot: Bot) -> None:
        """
        Send a text message reply.
        
        Args:
            message: Message to reply to
            text: Text to send
            bot: Bot instance
        """
        try:
            await message.reply_text(text)
            logger.info(f"Text response sent: {text[:50]}")
        except TelegramError as e:
            logger.error(f"Error sending text message: {e}")

    @staticmethod
    async def _send_reaction(message: Message, emoji: str, bot: Bot) -> None:
        """
        Send a reaction emoji to the message.
        Falls back to text if reaction fails.
        
        Args:
            message: Message to react to
            emoji: Emoji to use as reaction
            bot: Bot instance
        """
        try:
            # Try to set reaction
            reaction = [{"type": "emoji", "emoji": emoji}]
            await bot.set_message_reaction(
                chat_id=message.chat_id,
                message_id=message.message_id,
                reaction=reaction
            )
            logger.info(f"Reaction sent: {emoji}")
        except TelegramError as e:
            logger.warning(f"Error setting reaction, falling back to text: {e}")
            # Fallback: send emoji as text
            try:
                await message.reply_text(emoji)
                logger.info(f"Emoji sent as text fallback: {emoji}")
            except TelegramError as e2:
                logger.error(f"Error sending emoji fallback: {e2}")

    @staticmethod
    async def _send_gif(message: Message, search_query: str, bot: Bot) -> None:
        """
        Search Giphy API and send a random GIF.
        Falls back to text if GIF not found.
        
        Args:
            message: Message to reply to
            search_query: Search query for Giphy
            bot: Bot instance
        """
        try:
            # Search Giphy
            params = {
                'api_key': config.GIPHY_API_KEY,
                'q': search_query,
                'limit': config.GIPHY_LIMIT,
                'rating': config.GIPHY_RATING
            }
            response = requests.get(config.GIPHY_API_URL, params=params, timeout=5)
            response.raise_for_status()

            data = response.json()
            
            if not data.get('data'):
                logger.warning(f"No GIFs found for query: {search_query}")
                # Fallback: send search query as text
                await message.reply_text(search_query)
                return

            # Pick random GIF
            gif = random.choice(data['data'])
            gif_url = gif['images']['original']['url']

            # Send animation
            await bot.send_animation(
                chat_id=message.chat_id,
                animation=gif_url
            )
            logger.info(f"GIF sent for query: {search_query}")

        except requests.RequestException as e:
            logger.error(f"Error fetching from Giphy API: {e}")
            # Fallback: send search query as text
            try:
                await message.reply_text(search_query)
            except TelegramError as e2:
                logger.error(f"Error sending fallback text: {e2}")
        except TelegramError as e:
            logger.error(f"Error sending animation to Telegram: {e}")

    @staticmethod
    async def _send_sticker(message: Message, emotion: str, bot: Bot) -> None:
        """
        Send a sticker based on emotion.
        Falls back to text if sticker file_id not available.
        
        Args:
            message: Message to reply to
            emotion: Emotion name (key in STICKER_MAP)
            bot: Bot instance
        """
        try:
            emotion = emotion.lower().strip()
            file_id = Responder.STICKER_MAP.get(emotion, '')

            if file_id:
                # Send sticker if file_id exists
                await bot.send_sticker(
                    chat_id=message.chat_id,
                    sticker=file_id
                )
                logger.info(f"Sticker sent for emotion: {emotion}")
            else:
                # Fallback: send emotion name as text
                logger.warning(f"No sticker file_id for emotion: {emotion}")
                await message.reply_text(emotion)
                logger.info(f"Emotion sent as text fallback: {emotion}")

        except TelegramError as e:
            logger.error(f"Error sending sticker: {e}")
            # Final fallback: send emotion as text
            try:
                await message.reply_text(emotion)
            except TelegramError as e2:
                logger.error(f"Error sending emotion fallback: {e2}")
