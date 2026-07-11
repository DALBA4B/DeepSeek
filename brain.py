# brain.py
"""
Brain module for the DeepSeek Telegram bot.
Handles decision-making and response generation using DeepSeek API.
Integrates with LightRAG for long-term knowledge about chat participants.
"""

import asyncio
import logging
import random
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, Hashable, List, Optional, Tuple

from openai import OpenAI

from models import BotConfig, ChatMessage, ParsedResponse, RequestComplexity, TokenRange
from prompts import (
    get_system_prompt,
    get_system_prompt_for_situation,
    get_context_prompt,
    get_name_variations,
    CONTINUATION_TRIGGERS,
    FALLBACK_RESPONSES,
    GIF_REQUEST_KEYWORDS,
    STICKER_REQUEST_KEYWORDS,
)
from rag_client import RagClient, RagClientError
from conversation_analyzer import ConversationAnalyzer, ClassificationResult, _MessageContext
from retry import retry_sync

logger = logging.getLogger(__name__)

# Situations the classifier can emit — see conversation_analyzer._SITUATIONS.
_SITUATIONS = ("joke", "help", "casual", "tease", "defend")


class GrudgeTracker:
    """
    Lightweight in-RAM tracker for "who recently attacked the bot".

    Process-local only (no Firebase) — resets on redeploy/restart, the same
    tradeoff already accepted by RecentResponseTracker (memory.py) and the
    daily_log RAM fallback. This only needs to survive one running process:
    it decides how hot a "defend" reply should be (repeat offender within
    the window → escalated tone). It never gates WHETHER the bot defends —
    that call is trusted to the classifier's own per-message judgement.

    Keyed by an arbitrary hashable (this module uses (chat_id, user_id) so a
    grudge doesn't leak across chats or users).
    """

    def __init__(self, grudge_window_sec: int = 1800):
        self._grudge_window = timedelta(seconds=grudge_window_sec)
        self._attacks: Dict[Hashable, Deque[datetime]] = defaultdict(lambda: deque(maxlen=20))

    def record_attack(self, key: Hashable) -> None:
        """Note that `key` just attacked the bot (called once per attack)."""
        self._attacks[key].append(datetime.now(timezone.utc))

    def grudge_level(self, key: Hashable) -> int:
        """How many attacks from `key` fall within the grudge window (excluding the current one)."""
        if key not in self._attacks:
            return 0
        cutoff = datetime.now(timezone.utc) - self._grudge_window
        return sum(1 for ts in self._attacks[key] if ts >= cutoff)


class RagUsageStats:
    """
    Process-local counters for how much LightRAG memory actually gets used.

    Answers the question from /ragstats: "как часто система обращается к
    памяти и насколько это влияет на ответ?" — not persisted (resets on
    restart), same tradeoff as GrudgeTracker: this is an observability aid,
    not data that needs to survive a redeploy.
    """

    def __init__(self) -> None:
        self.total_classified = 0
        self.needs_memory = 0
        self.facts_retrieved = 0

    def record_classification(self, needs_memory: bool) -> None:
        """Call once per classify() result."""
        self.total_classified += 1
        if needs_memory:
            self.needs_memory += 1

    def record_retrieval(self, facts_found: bool) -> None:
        """Call once per RagClient.retrieve() attempt (needs_memory==True)."""
        if facts_found:
            self.facts_retrieved += 1

    def as_dict(self) -> dict:
        hit_rate = (
            round(self.facts_retrieved / self.needs_memory * 100, 1)
            if self.needs_memory
            else 0.0
        )
        return {
            "total_classified": self.total_classified,
            "needs_memory": self.needs_memory,
            "facts_retrieved": self.facts_retrieved,
            "hit_rate_pct": hit_rate,
        }


def _detect_media_request(message_text: str) -> Optional[str]:
    """
    Keyword-based check: did the user explicitly ask for a GIF or sticker?
    Used to force compliance (see _build_media_hint) instead of hoping the
    model's own judgement lines up with what was actually asked for.
    """
    text_lower = message_text.lower()
    if any(kw in text_lower for kw in GIF_REQUEST_KEYWORDS):
        return "gif"
    if any(kw in text_lower for kw in STICKER_REQUEST_KEYWORDS):
        return "sticker"
    return None


# Prefix for a "react instead of full silence" result from analyze_and_respond
# (see _pick_silent_reaction_emoji) — main.py checks for this specifically and
# routes it to Responder.send_silent_reaction() instead of the normal
# text/GIPHY/STICKER/REACT pipeline (which has its own reaction-vs-text
# fallback heuristic that would defeat the point of "don't butt in").
SILENT_REACT_PREFIX = "SILENT_REACT:"

_SILENT_REACTION_EMOJIS = {
    "joke": ["😂", "💀", "🔥"],
    "tease": ["😏", "👀"],
}
_DEFAULT_SILENT_REACTION_EMOJIS = ["👍", "😂"]


def _pick_silent_reaction_emoji(situation: str) -> str:
    """Pick a reaction emoji matching the situation, for the grade==0 ack."""
    candidates = _SILENT_REACTION_EMOJIS.get(situation, _DEFAULT_SILENT_REACTION_EMOJIS)
    return random.choice(candidates)


_RANDOM_NON_GIF_HINTS = (
    "можно один раз ответить стикером (STICKER:<эмоция>), если это в тему",
    "можно один раз ответить реакцией-эмодзи (REACT:<эмодзи>), если это в тему",
)
_RANDOM_GIF_HINT = "можно один раз ответить гифкой (GIPHY:<запрос>), если это в тему"


def _build_media_hint(
    message_text: str, media_probability: float, gif_probability: float
) -> str:
    """
    Decide, in CODE (not left to the model's own judgement), whether this
    turn is allowed to use GIPHY:/REACT:/STICKER: instead of plain text.

    This is the fix for "гифка/стикер вместо ответа, когда не просили, и
    текст, когда просили" (Проблема №2): an explicit request from the user
    always wins and is force-instructed; otherwise there's a small,
    independently-tuned chance of an unprompted media reply — `gif_probability`
    for GIPHY (kept lower: a GIF is a network round-trip that can fail/be
    slow) and `media_probability` for sticker/reaction. Returns "" when none
    of these apply — the base prompt already tells the model "no format line
    this turn = plain text", so an empty hint means text.
    """
    explicit = _detect_media_request(message_text)
    if explicit == "gif":
        return "пользователь явно просит гифку — ответь строго GIPHY:<короткий запрос на английском>, без текста до/после"
    if explicit == "sticker":
        return "пользователь явно просит стикер — ответь строго STICKER:<эмоция>, без текста до/после"

    roll = random.random()
    if roll < gif_probability:
        return _RANDOM_GIF_HINT
    if roll < gif_probability + media_probability:
        return random.choice(_RANDOM_NON_GIF_HINTS)

    return ""


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
            max_attempts=config.classifier_max_attempts,
            retry_base_delay=config.classifier_retry_base_delay,
        )

        try:
            self.client = OpenAI(
                api_key=config.deepseek_api_key,
                base_url=config.deepseek_base_url,
                timeout=config.deepseek_timeout,
            )
            # Default/legacy blended prompt — used as a fallback for unknown
            # situations. Kept immutable after init (see update_system_prompt
            # note below): Brain is a single instance shared across all
            # concurrently-handled messages, so per-message prompt swapping
            # must never mutate shared state.
            self._system_prompt = get_system_prompt(
                config.bot_name,
                self._available_stickers,
                text_only_mode=config.text_only_mode,
            )
            # One prompt per situation, built once — plain dict lookup at
            # request time, no shared mutable state. Fixes "не понимал в
            # каком стиле ответить": the model gets exactly one role instead
            # of five to average over. "defend" here is grudge_level=0; a
            # higher grudge_level rebuilds a hotter prompt on the fly in
            # generate_response() (still just a local variable, not stored).
            self._situation_prompts: Dict[str, str] = {
                situation: get_system_prompt_for_situation(
                    config.bot_name,
                    self._available_stickers,
                    situation=situation,
                    text_only_mode=config.text_only_mode,
                )
                for situation in _SITUATIONS
            }
            self._grudge = GrudgeTracker()
            self._rag_usage = RagUsageStats()
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
        user_id: Optional[int] = None,
        chat_id: Optional[int] = None,
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
            user_id: Telegram numeric user id — used to key grudge/cooldown
                tracking for "defend" situations. Optional so callers that
                don't have it (or tests) still work; grudge escalation is
                simply skipped when absent.
            chat_id: Telegram chat id — combined with user_id so a grudge in
                one chat doesn't leak into another.

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
        self._rag_usage.record_classification(result.needs_memory)

        logger.info(
            "Classification: grade=%d needs_memory=%s query=%r situation=%s reason=%r fallback=%s",
            result.grade,
            result.needs_memory,
            result.rag_query,
            result.situation,
            result.reason,
            result.from_fallback,
        )

        # Step 2: Skip if grade == 0 — but occasionally just react instead of
        # staying fully silent (e.g. someone told a joke elsewhere in the
        # conversation, not aimed at the bot — a reaction acknowledges it
        # without butting in with a full reply). No DeepSeek call needed.
        if result.grade == 0:
            if (
                not self.config.text_only_mode
                and random.random() < self.config.silent_reaction_probability
            ):
                emoji = _pick_silent_reaction_emoji(result.situation)
                logger.info(
                    "Silent reaction instead of full silence: %s (situation=%s)",
                    emoji, result.situation,
                )
                return f"{SILENT_REACT_PREFIX}{emoji}"
            return None

        # Step 2b: Grudge escalation for "defend" situations. We trust the
        # classifier's own per-message situation call here (it already
        # reliably detects an ongoing attack even without a fresh name
        # mention or keyword hit) — grudge_level only controls HOW hot the
        # comeback is, it never suppresses/downgrades a defend the model
        # already decided on.
        situation = result.situation
        grudge_level = 0
        if situation == "defend" and user_id is not None:
            key = (chat_id, user_id)
            grudge_level = self._grudge.grudge_level(key)
            self._grudge.record_attack(key)
            logger.info("Defend triggered for %s: grudge_level=%d", key, grudge_level)

        # Step 3: Fetch facts from LightRAG if needed
        rag_facts = ""
        if result.needs_memory and self._rag_client and result.rag_query:
            rag_facts = await self._rag_client.retrieve(result.rag_query)
            self._rag_usage.record_retrieval(bool(rag_facts))
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
            situation=situation,
            grudge_level=grudge_level,
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
        situation: str = "casual",
        grudge_level: int = 0,
    ) -> str:
        """
        Generate a response using DeepSeek API.

        Args:
            message_text: The current message to respond to.
            context: Formatted recent messages as context.
            rag_facts: Optional facts from LightRAG knowledge base.
            grade: Response depth (0=skip, 1=react, 2=normal, 3=deep).
            avoid_responses: Optional list of recent responses to avoid.
            situation: One of joke|help|casual|tease|defend — picks which
                pre-built system prompt to use (see _situation_prompts).
            grudge_level: For situation="defend" only — >=1 rebuilds a
                hotter, escalated prompt (repeat offender), computed
                method-locally so no shared state is mutated.

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
            if situation == "defend":
                dynamic_temperature = 1.15

            # Pick the system prompt for this situation. grudge_level > 0
            # needs a freshly-built escalated prompt (method-local variable —
            # never cached/mutated on self, since Brain is shared across
            # concurrently-handled messages).
            if situation == "defend" and grudge_level > 0:
                system_prompt = get_system_prompt_for_situation(
                    self.config.bot_name,
                    self._available_stickers,
                    situation="defend",
                    text_only_mode=self.config.text_only_mode,
                    grudge_level=grudge_level,
                )
            else:
                system_prompt = self._situation_prompts.get(situation, self._system_prompt)

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

            # Code-enforced format control (Проблема №2 fix): the model does
            # NOT get to freely pick GIPHY/REACT/STICKER anymore — either the
            # user explicitly asked for one, or a small random roll allows
            # it this turn. text_only_mode skips this entirely (would be
            # stripped by ResponseParser anyway).
            media_hint = ""
            if not self.config.text_only_mode:
                media_hint = _build_media_hint(
                    message_text,
                    self.config.media_response_probability,
                    self.config.gif_response_probability,
                )

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": get_context_prompt(
                        enhanced_context, message_text, rag_facts, media_hint
                    ),
                },
            ]

            response = retry_sync(
                lambda: self.client.chat.completions.create(
                    model=self.config.deepseek_model,
                    messages=messages,
                    max_tokens=dynamic_max_tokens,
                    temperature=dynamic_temperature,
                    extra_body={"thinking": {"type": "disabled"}},
                ),
                attempts=self.config.deepseek_max_attempts,
                base_delay=self.config.deepseek_retry_base_delay,
                max_delay=4.0,
                jitter=0.3,
            )

            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated response (grade={grade}, situation={situation}): {answer[:60]}")
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

    def grudge_level_for(self, chat_id: Optional[int], user_id: int) -> int:
        """Current grudge level for (chat_id, user_id) — used by /mood."""
        return self._grudge.grudge_level((chat_id, user_id))

    def get_rag_usage_stats(self) -> dict:
        """Process-local LightRAG usage counters — used by /ragstats."""
        return self._rag_usage.as_dict()

    @property
    def available_stickers(self) -> List[str]:
        return self._available_stickers
