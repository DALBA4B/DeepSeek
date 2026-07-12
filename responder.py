# responder.py
"""
Response handler for the DeepSeek Telegram bot.
Processes and sends responses in different formats: text, reaction, GIF, sticker.
"""

import logging
import random
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import ClientTimeout
from telegram import Bot, Message, ReactionTypeEmoji
from telegram.error import TelegramError

from models import BotConfig, ParsedResponse, ResponseType
from prompts import ALLOWED_REACTION_EMOJIS

logger = logging.getLogger(__name__)

# Fast membership check — see prompts.ALLOWED_REACTION_EMOJIS for why this
# whitelist exists (Telegram rejects anything outside its own reaction set
# with REACTION_INVALID).
_ALLOWED_REACTIONS = frozenset(ALLOWED_REACTION_EMOJIS)


# Alternative queries for GIF search retry (increases success rate from ~70% to ~85%)
# When primary query fails, bot tries these alternatives before falling back to text
GIF_ALTERNATIVE_QUERIES: Dict[str, List[str]] = {
    # Gaming
    "дота": ["gaming", "video game", "esports"],
    "dota": ["gaming", "video game", "esports"],
    "кс": ["gaming", "fps", "counter strike"],
    "cs": ["gaming", "fps", "counter strike"],
    "лол": ["gaming", "moba", "league"],
    "лига": ["gaming", "moba", "league"],
    
    # Sports
    "футбол": ["sports", "soccer", "football"],
    "football": ["soccer", "sports", "kick"],
    "баскетбол": ["basketball", "sports", "court"],
    "волейбол": ["volleyball", "sports", "ball"],
    "теннис": ["tennis", "sports", "racket"],
    
    # Food
    "пицца": ["pizza", "food", "slice"],
    "суши": ["sushi", "food", "japanese"],
    "бургер": ["burger", "food", "fast food"],
    "кофе": ["coffee", "cafe", "latte"],
    "пиво": ["beer", "drink", "alcohol"],
    
    # Emotions/Actions
    "смешно": ["funny", "laugh", "comedy"],
    "грустно": ["sad", "cry", "sadness"],
    "злой": ["angry", "rage", "mad"],
    "спать": ["sleep", "tired", "nap"],
    "танец": ["dance", "party", "music"],
    "плачу": ["cry", "tears", "sad"],
    "ржу": ["laugh", "funny", "comedy"],
    
    # Generic
    "реакция": ["reaction", "response", "emotion"],
    "гиф": ["gif", "animation", "funny"],
    "gif": ["animation", "funny", "video"],
}


class ResponseParser:
    """Parses DeepSeek responses to determine type and content."""

    # Response prefixes
    PREFIX_GIPHY = "GIPHY:"
    PREFIX_REACT = "REACT:"
    PREFIX_STICKER = "STICKER:"

    @classmethod
    def parse(cls, response_text: str, text_only_mode: bool = True) -> ParsedResponse:
        """
        Parse the response to determine its type and content.
        Case-insensitive check for prefixes.

        In text_only_mode: strips any prefixes and always returns TEXT.

        Args:
            response_text: The response from DeepSeek (may be in special format)
            text_only_mode: If True, force all responses to TEXT

        Returns:
            ParsedResponse with type and content
        """
        text = response_text.strip()
        text_upper = text.upper()

        if text_upper.startswith(cls.PREFIX_GIPHY):
            content = text[len(cls.PREFIX_GIPHY):].strip()
            if text_only_mode:
                # GIPHY query is not meaningful text — return empty to skip
                logger.info(f"TEXT_ONLY_MODE: blocked GIPHY response: {content}")
                return ParsedResponse(ResponseType.TEXT, "")
            logger.info(f"Parsed GIPHY response: {content}")
            return ParsedResponse(ResponseType.GIF, content)

        if text_upper.startswith(cls.PREFIX_REACT):
            content = text[len(cls.PREFIX_REACT):].strip()
            if text_only_mode:
                # Emoji is fine as text message
                logger.info(f"TEXT_ONLY_MODE: converted REACT to text: {content}")
                return ParsedResponse(ResponseType.TEXT, content)
            logger.info(f"Parsed REACT response: {content}")
            return ParsedResponse(ResponseType.REACTION, content)

        if text_upper.startswith(cls.PREFIX_STICKER):
            content = text[len(cls.PREFIX_STICKER):].strip().lower()
            if text_only_mode:
                # Sticker emotion is not meaningful text — return empty to skip
                logger.info(f"TEXT_ONLY_MODE: blocked STICKER response: {content}")
                return ParsedResponse(ResponseType.TEXT, "")
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
                async with session.get(self.api_url, params=params, timeout=ClientTimeout(total=5)) as response:
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

    # Default sticker mapping (emotion -> file_id) - kept as backup
    DEFAULT_STICKERS: Dict[str, str] = {
        'happy': 'CAACAgIAAxkBAAEQUVxpdIeyvxepv5LBpDDNIWszpN8JJQAC85oAAgRqgUshcX0t9I5SSDgE',
    }

    # Which sticker-pack emoji(s) count as a match for each bot emotion name.
    # Telegram sticker sets carry one associated emoji per sticker
    # (sticker.emoji) — this lets us pick a sticker that actually matches
    # the emotion the model asked for, instead of a purely random one.
    EMOTION_EMOJIS: Dict[str, List[str]] = {
        'happy': ['😄', '😊', '🙂', '😁', '😀'],
        'sad': ['😢', '😞', '🙁', '😔'],
        'laugh': ['😂', '🤣'],
        'cool': ['😎'],
        'think': ['🤔'],
        'wtf': ['🤨', '😳', '😲', '🫤'],
    }

    # How many recently-sent stickers to avoid repeating (see get_file_id).
    RECENT_HISTORY_SIZE = 10

    def __init__(self, custom_stickers: Optional[Dict[str, str]] = None):
        """
        Initialize sticker manager.
        """
        self._stickers = self.DEFAULT_STICKERS.copy()
        if custom_stickers:
            self._stickers.update(custom_stickers)

        # (file_id, emoji) pairs across every loaded pack — load_sticker_set
        # can be called multiple times (multiple packs) and accumulates here.
        self._all_stickers: List[Tuple[str, str]] = []

        # Most recently sent sticker file_ids, oldest first — excluded from
        # selection in get_file_id() so the bot doesn't spam the same sticker.
        self._recently_sent: Deque[str] = deque(maxlen=self.RECENT_HISTORY_SIZE)

    async def load_sticker_set(self, bot: Bot, set_name: str) -> None:
        """
        Load all stickers from a sticker set, keeping each sticker's
        associated emoji so replies can be matched by emotion. Can be called
        multiple times with different pack names — stickers accumulate
        across packs instead of replacing the previous pack.

        Args:
            bot: Telegram Bot instance
            set_name: Name of the sticker set (e.g. 'userpack...')
        """
        try:
            sticker_set = await bot.get_sticker_set(set_name)

            new_stickers = [
                (sticker.file_id, sticker.emoji or "") for sticker in sticker_set.stickers
            ]
            self._all_stickers.extend(new_stickers)
            logger.info(
                "Loaded %d stickers from set '%s' (total across all packs: %d)",
                len(new_stickers), set_name, len(self._all_stickers),
            )

        except Exception as e:
            logger.error(f"Failed to load sticker set '{set_name}': {e}")

    def get_file_id(self, emotion: str) -> Optional[str]:
        """
        Get a sticker matching `emotion`, avoiding recent repeats. If a full
        set is loaded, prefer a sticker whose pack emoji matches the emotion;
        fall back to any sticker if nothing matches. Otherwise try the
        manual emotion->file_id mapping.

        Recently-sent stickers (see record_sent) are excluded from the
        candidate pool where possible — but a small pool exhausted by recent
        history still returns something rather than nothing (repeats are
        better than no sticker at all).
        """
        if self._all_stickers:
            wanted_emojis = self.EMOTION_EMOJIS.get(emotion.lower().strip(), [])
            matches = [
                file_id for file_id, emoji in self._all_stickers if emoji in wanted_emojis
            ]
            # No sticker in any loaded pack carries that emoji — any sticker
            # beats none.
            pool = matches if matches else [file_id for file_id, _ in self._all_stickers]

            fresh = [file_id for file_id in pool if file_id not in self._recently_sent]
            return random.choice(fresh if fresh else pool)

        # Fallback to manual mapping
        file_id = self._stickers.get(emotion.lower().strip(), '')
        return file_id if file_id else None

    def record_sent(self, file_id: str) -> None:
        """Note that `file_id` was just sent, so get_file_id() avoids repeating it."""
        self._recently_sent.append(file_id)


class RecentReactionTracker:
    """
    Minimal repeat guard for reactions: blocks the SAME emoji from being used
    a 5th time in a row, without otherwise interfering (occasional repeats —
    e.g. 👍 a couple messages apart — are normal chat behavior; only a long
    identical streak looks robotic).

    Process-local, in-RAM — same tradeoff as GrudgeTracker/StickerManager's
    recent-history: this is a light UX guard, not data worth persisting.
    Global across chats (matches the rest of the codebase's single-chat
    assumption — see PROJECT_ANALYSIS_V2.md "Баг №7").
    """

    def __init__(self, max_streak: int = 4):
        self._max_streak = max_streak
        self._last_emoji: Optional[str] = None
        self._streak = 0

    def should_avoid(self, emoji: str) -> bool:
        """True if using `emoji` again would extend an already-maxed streak."""
        return emoji == self._last_emoji and self._streak >= self._max_streak

    def record(self, emoji: str) -> None:
        """Call once per reaction actually sent/set."""
        if emoji == self._last_emoji:
            self._streak += 1
        else:
            self._last_emoji = emoji
            self._streak = 1

    def pick_non_repeating(self, candidates: List[str]) -> str:
        """
        Pick a random emoji from `candidates`, preferring ones that wouldn't
        extend a maxed-out streak. Falls back to a repeat if every candidate
        would (better a repeat than nothing / no reaction at all).
        """
        fresh = [c for c in candidates if not self.should_avoid(c)]
        return random.choice(fresh if fresh else candidates)


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
        self._reactions = RecentReactionTracker()
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
        parsed = ResponseParser.parse(response_text, text_only_mode=self.config.text_only_mode)

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
        if not text:
            logger.info("Empty text response, skipping send")
            return False
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

        # Minimal anti-spam guard: don't let the SAME emoji fire a 5th time
        # in a row (the model picks freely and has no memory of its own).
        # An explicit "поставь реакцию" request still wins — repeats there
        # are the user's call, not ours to second-guess.
        if use_reaction and not wants_reaction and self._reactions.should_avoid(emoji):
            use_reaction = False

        # The model is told to only use ALLOWED_REACTION_EMOJIS, but if it
        # picks something outside that set anyway, skip the doomed API call
        # entirely (Telegram would reject it with REACTION_INVALID) and go
        # straight to text.
        if use_reaction and emoji not in _ALLOWED_REACTIONS:
            logger.info(f"'{emoji}' isn't a valid Telegram reaction, sending as text instead")
            use_reaction = False

        if use_reaction:
            try:
                await message.set_reaction(
                    reaction=[ReactionTypeEmoji(emoji=emoji)],
                    is_big=False
                )
                self._reactions.record(emoji)
                logger.info(f"Reaction set: {emoji}")
                return True
            except Exception as e:
                logger.warning(f"Reaction failed: {e}. Falling back to text.")

        # Fallback to text
        return await self._send_text(message, emoji)

    async def send_silent_reaction(self, message: Message, emoji: str) -> bool:
        """
        Set a Telegram reaction with NO text fallback.

        Used when the bot chose to react instead of replying at all (grade 0
        — "don't butt into the conversation"). Unlike _send_reaction(), this
        never falls back to a text message: if setting the reaction fails,
        the bot simply stays silent, which was the entire point of choosing
        this path over a normal reply.

        Args:
            message: Message to react to
            emoji: Emoji to use

        Returns:
            True if the reaction was set successfully
        """
        if emoji not in _ALLOWED_REACTIONS:
            logger.warning(f"'{emoji}' isn't a valid Telegram reaction, staying silent")
            return False

        try:
            await message.set_reaction(
                reaction=[ReactionTypeEmoji(emoji=emoji)],
                is_big=False,
            )
            self._reactions.record(emoji)
            logger.info(f"Silent reaction set: {emoji}")
            return True
        except Exception as e:
            logger.warning(f"Silent reaction failed, staying silent: {e}")
            return False

    async def send_silent_reaction_from_pool(self, message: Message, candidates: List[str]) -> bool:
        """
        Like send_silent_reaction(), but picks the actual emoji here from a
        pool of situation-appropriate candidates (see brain._silent_reaction_pool),
        avoiding one that's already on a maxed-out repeat streak.

        Args:
            message: Message to react to
            candidates: Candidate emoji, any of which fits the situation

        Returns:
            True if the reaction was set successfully
        """
        if not candidates:
            return False
        emoji = self._reactions.pick_non_repeating(candidates)
        return await self.send_silent_reaction(message, emoji)

    async def _send_gif(self, message: Message, search_query: str, bot: Bot) -> bool:
        """
        Search Giphy API and send a random GIF.
        Retries with alternative queries if first fails.
        Falls back to text action if all attempts fail.
        
        Args:
            message: Message to reply to
            search_query: Search query for Giphy
            bot: Bot instance
            
        Returns:
            True if sent successfully
        """
        # Try primary query first
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
        
        # Try alternative queries if primary failed
        alt_queries = GIF_ALTERNATIVE_QUERIES.get(search_query.lower(), [])
        
        for alt_query in alt_queries:
            logger.debug(f"Retrying with alternative query: {alt_query}")
            gif_url = await self._giphy.search(alt_query)
            
            if gif_url:
                try:
                    await bot.send_animation(
                        chat_id=message.chat_id,
                        animation=gif_url
                    )
                    logger.info(f"GIF sent with alt query: {alt_query} (original: {search_query})")
                    return True
                except TelegramError as e:
                    logger.warning(f"Error sending animation with alt query {alt_query}: {e}")
                    continue
        
        # Fallback to natural text if all GIF attempts failed
        fallback_phrases = [
            "чет гифка не грузится(",
            "не нашел подходящую гифку, но представь что тут смешно",
            "гифки сломались, но я всё равно ржу",
            "🤷‍♂️ не нашел гифку",
            "лан, без гифки обойдемся"
        ]
        fallback_text = random.choice(fallback_phrases)
        logger.info(f"GIF fallback to text: {fallback_text}")
        return await self._send_text(message, fallback_text)

    async def _send_sticker(self, message: Message, emotion: str, bot: Bot) -> bool:
        """
        Send a sticker based on emotion.
        Falls back to emoji or text action if sticker file_id not available.
        
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
                self._stickers.record_sent(file_id)
                logger.info(f"Sticker sent for emotion: {emotion}")
                return True
            except TelegramError as e:
                logger.error(f"Error sending sticker: {e}")

        # Fallback to emoji or text action
        emoji_map = {
            'happy': '😄',
            'sad': '😢',
            'laugh': '😂',
            'cool': '😎',
            'think': '🤔',
            'wtf': '🤨'
        }
        
        fallback_text = emoji_map.get(emotion.lower(), f"*стикер: {emotion}*")
        logger.info(f"Sticker fallback to text: {fallback_text}")
        return await self._send_text(message, fallback_text)

    @property
    def sticker_manager(self) -> StickerManager:
        """Get the sticker manager for adding custom stickers."""
        return self._stickers
