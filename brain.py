# brain.py
"""
Brain module for the DeepSeek Telegram bot.
Handles decision-making and response generation using DeepSeek API.
Integrates with LightRAG for long-term knowledge about chat participants.
"""

import asyncio
import logging
import random
from typing import List, Optional

from openai import OpenAI

from models import BotConfig, ChatMessage, ParsedResponse, RequestComplexity, TokenRange
from prompts import get_system_prompt, get_context_prompt, get_name_variations, CONTINUATION_TRIGGERS, FALLBACK_RESPONSES
from rag_client import RagClient, RagClientError
from conversation_analyzer import ConversationAnalyzer, ClassificationResult, _MessageContext

logger = logging.getLogger(__name__)


class RequestClassifier:
    """
    Classifies message complexity to determine appropriate response length.
    Uses keyword matching and pattern detection.
    """

    SIMPLE_KEYWORDS = [
        'да?', 'нет?', 'ок?', 'норм?', 'реакц', 'поставь', 'лайк',
        'согласен', 'да/нет', 'коротко', 'одним словом', 'быстро',
        'ещё', 'еще', 'another', 'more', 'продолжай',
        'шутка', 'анекдот', 'шутку', 'прикол', 'рассмеши', 'рофл',
        'как дела', 'чо делаешь', 'что делаешь', 'как сам', 'чо как',
        'ну', 'ага', 'понял', 'спс', 'благодарю', 'ок', 'окей',
        'круто', 'класс', 'топ', 'огонь', 'красава',
    ]

    COMPLEX_KEYWORDS = [
        'расскажи', 'объясни', 'почему', 'как работает', 'подробно',
        'план', 'список', 'пошагово', 'детально', 'разбери',
        'история', 'напиши текст', 'сочини', 'придумай историю',
        'что думаешь о', 'мнение', 'проанализируй', 'сравни',
        'помоги', 'посоветуй', 'как сделать', 'как мне', 'подскажи',
        'научи', 'покажи как', 'объясни как',
    ]

    @classmethod
    def classify(cls, message: str) -> RequestComplexity:
        msg_lower = message.lower()
        for keyword in cls.SIMPLE_KEYWORDS:
            if keyword in msg_lower:
                return RequestComplexity.SIMPLE
        for keyword in cls.COMPLEX_KEYWORDS:
            if keyword in msg_lower:
                return RequestComplexity.COMPLEX
        if len(message) > 100 and '?' in message:
            return RequestComplexity.COMPLEX
        if len(message) < 30 and '?' not in message:
            return RequestComplexity.SIMPLE
        return RequestComplexity.NORMAL


class Brain:
    """
    AI logic for the bot using DeepSeek API.
    Makes decisions about when/how to respond and generates responses.
    Integrates with LightRAG for long-term knowledge about people.
    """

    def __init__(
        self,
        config: BotConfig,
        available_stickers: Optional[List[str]] = None,
        rag_client: Optional[RagClient] = None,
    ):
        self.config = config
        self._available_stickers = available_stickers or [
            "happy", "sad", "laugh", "cool", "think", "wtf"
        ]
        self._rag_client = rag_client
        self._name_variations = get_name_variations(config.bot_name)

        # ConversationAnalyzer for grade 0-3 + memory decisions
        self._analyzer = ConversationAnalyzer(
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
            model="deepseek-chat",  # fast model for classification
            temperature=0.3,
        )

        try:
            self.client = OpenAI(
                api_key=config.deepseek_api_key,
                base_url=config.deepseek_base_url,
            )
            self._system_prompt = get_system_prompt(
                config.bot_name,
                self._available_stickers,
                text_only_mode=config.text_only_mode,
            )
            logger.info("Brain initialized: DeepSeek client ready")
        except Exception as e:
            logger.error(f"Failed to initialize Brain: {e}")
            raise

    # ------------------------------------------------------------------ #
    # Public: new V2 flow
    # ------------------------------------------------------------------ #
    async def analyze_and_respond(
        self,
        message_text: str,
        author: str,
        recent_messages: List[str],
        reply_text: Optional[str] = None,
        bot_responded_recently: bool = False,
        avoid_responses: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        V2 main entry point. One-stop: classify → fetch memory → generate.

        Args:
            message_text: The incoming message.
            author: Who sent it (display name).
            recent_messages: Last N formatted context lines ("Name: text").
            reply_text: If the message is a reply, the text being replied to.
            bot_responded_recently: Whether the bot answered in the last few msgs.
            avoid_responses: Recent bot responses to avoid repeating.

        Returns:
            Generated response text, or None if grade==0 (skip).
        """
        # Step 1: Classify
        msg_ctx = _MessageContext(
            author=author,
            text=message_text,
            reply_text=reply_text,
            bot_name=self.config.bot_name,
            bot_responded_recently=bot_responded_recently,
        )
        result = await self._analyzer.classify(msg_ctx, recent_messages)

        logger.info(
            "Classification: grade=%d needs_memory=%s query=%r reason=%r fallback=%s",
            result.grade,
            result.needs_memory,
            result.rag_query,
            result.reason,
            result.from_fallback,
        )

        # Step 2: Skip if grade == 0
        if result.grade == 0:
            return None

        # Step 3: Fetch facts from LightRAG if needed
        rag_facts = ""
        if result.needs_memory and self._rag_client and result.rag_query:
            rag_facts = await self._rag_client.retrieve(result.rag_query)
            if rag_facts:
                logger.info("LightRAG facts retrieved (%d chars)", len(rag_facts))
            else:
                logger.debug("LightRAG returned no facts for query: %r", result.rag_query)

        # Step 4: Generate response with grade-aware parameters.
        # generate_response() uses the synchronous OpenAI SDK, so run it in a
        # thread to avoid blocking the event loop while DeepSeek thinks
        # (otherwise the bot freezes — same class of bug as the old analyzer).
        context = "\n".join(recent_messages)
        return await asyncio.to_thread(
            self.generate_response,
            message_text=message_text,
            context=context,
            rag_facts=rag_facts,
            grade=result.grade,
            avoid_responses=avoid_responses,
        )

    # ------------------------------------------------------------------ #
    # Public: response generation
    # ------------------------------------------------------------------ #
    def generate_response(
        self,
        message_text: str,
        context: str,
        rag_facts: str = "",
        grade: int = 2,
        avoid_responses: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a response using DeepSeek API.

        Args:
            message_text: The current message to respond to.
            context: Formatted recent messages as context.
            rag_facts: Optional facts from LightRAG knowledge base.
            grade: Response depth (0=skip, 1=react, 2=normal, 3=deep).
            avoid_responses: Optional list of recent responses to avoid.

        Returns:
            Generated response text (may contain REACT:/GIPHY:/STICKER: prefixes).
        """
        try:
            # Dynamic tokens based on grade
            token_range = self._tokens_for_grade(grade)
            dynamic_max_tokens = token_range.random_value()

            # Temperature: lower for deep answers, higher for reactions
            temp_map = {0: 1.0, 1: 1.2, 2: 1.0, 3: 0.8}
            dynamic_temperature = temp_map.get(grade, self.config.deepseek_temperature)

            # Enhance context with rag facts
            enhanced_context = context
            if rag_facts:
                enhanced_context = f"ЗНАНИЯ О ЛЮДЯХ:\n{rag_facts}\n\n{context}"

            # Add avoid list if provided
            if avoid_responses:
                avoid_str = ", ".join(avoid_responses[-5:])
                enhanced_context = (
                    f"НЕ ИСПОЛЬЗУЙ ЭТИ ОТВЕТЫ: {avoid_str}\n\n{enhanced_context}"
                )

            messages = [
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": get_context_prompt(
                        enhanced_context, message_text, rag_facts
                    ),
                },
            ]

            response = self.client.chat.completions.create(
                model=self.config.deepseek_model,
                messages=messages,
                max_tokens=dynamic_max_tokens,
                temperature=dynamic_temperature,
                extra_body={"thinking": {"type": "disabled"}},
            )

            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated response (grade={grade}): {answer[:60]}")
            return answer

        except Exception as e:
            logger.error(f"Error generating response from DeepSeek: {e}")
            return FALLBACK_RESPONSES["api_error"]

    # ------------------------------------------------------------------ #
    # Public: legacy methods (kept for backward compatibility)
    # ------------------------------------------------------------------ #
    def should_respond(
        self,
        message_text: str,
        bot_responded_recently: bool = False,
    ) -> bool:
        """Legacy heuristic check. Prefer analyze_and_respond() for V2."""
        message_lower = message_text.lower()

        for variation in self._name_variations:
            if variation in message_lower:
                return True

        if bot_responded_recently:
            for trigger in CONTINUATION_TRIGGERS:
                if trigger in message_lower:
                    return True

        if "?" in message_text:
            return True

        return random.random() < self.config.random_response_probability

    async def smart_should_respond(
        self,
        message_text: str,
        context: str,
        bot_responded_recently: bool = False,
    ) -> bool:
        """Legacy AI-based respond check. Prefer analyze_and_respond() for V2."""
        message_lower = message_text.lower()
        for variation in self._name_variations:
            if variation in message_lower:
                return True

        try:
            decision_prompt = f"""Ты участник группового чата. Реши — отвечать или нет.

Контекст:
{context[-400:]}

Новое сообщение: "{message_text}"

Ты отвечал недавно: {"да" if bot_responded_recently else "нет"}

Ответь ТОЛЬКО "да" или "нет".

Отвечай "да" если:
- Есть вопрос к чату
- Можешь добавить что-то интересное или смешное
- Тебя как будто спрашивают или ждут реакции

Отвечай "нет" если:
- Междометие без смысла
- Ты только что отвечал
- Люди болтают между собой и ты не в теме"""

            response = self.client.chat.completions.create(
                model=self.config.deepseek_model,
                messages=[{"role": "user", "content": decision_prompt}],
                max_tokens=3,
                temperature=0.7,
            )

            answer = response.choices[0].message.content.strip().lower()
            return "да" in answer or "yes" in answer

        except Exception as e:
            logger.warning(f"Smart respond failed, falling back: {e}")
            return random.random() < self.config.random_response_probability

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tokens_for_grade(grade: int) -> TokenRange:
        """Map grade 0-3 to token budget."""
        ranges = {
            0: TokenRange(0, 50),        # skip (shouldn't reach generate)
            1: TokenRange(20, 150),       # short reaction
            2: TokenRange(150, 400),      # normal answer
            3: TokenRange(300, 800),      # deep, detailed
        }
        return ranges.get(grade, TokenRange(150, 400))

    def update_system_prompt(self, new_prompt: str) -> None:
        self._system_prompt = new_prompt
        logger.info("System prompt updated")

    @property
    def available_stickers(self) -> List[str]:
        return self._available_stickers
