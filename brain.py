# brain.py
"""
Brain module for the DeepSeek Telegram bot.
Handles decision-making and response generation using DeepSeek API.
Integrates with LightRAG for long-term knowledge about chat participants.
"""

import asyncio
import logging
import random
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Deque, Dict, Hashable, List, Optional, Tuple

from openai import OpenAI

from models import BotConfig, TokenRange
from prompts import (
    get_system_prompt,
    get_system_prompt_for_situation,
    get_context_prompt,
    FALLBACK_RESPONSES,
    GIF_REQUEST_KEYWORDS,
    STICKER_REQUEST_KEYWORDS,
)
from rag_client import RagClient
from conversation_analyzer import ConversationAnalyzer, _MessageContext
from retry import retry_sync

logger = logging.getLogger(__name__)
# Verbose per-step RAG pipeline tracing (classify/retrieve/prompt/generate).
# Silent by default — enabled via RAG_DEBUG_LOGGING (see main.setup_logging).
_rag_debug = logging.getLogger("ragdebug")

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
# (see _silent_reaction_pool) — main.py checks for this specifically and
# routes it to Responder.send_silent_reaction_from_pool() instead of the
# normal text/GIPHY/STICKER/REACT pipeline (which has its own reaction-vs-text
# fallback heuristic that would defeat the point of "don't butt in").
#
# The full candidate pool (comma-joined) is sent, not one pre-picked emoji —
# Responder makes the final choice there, since it's the one place that
# actually knows what was recently sent and can avoid repeating it (see
# Responder._reactions / RecentReactionTracker).
SILENT_REACT_PREFIX = "SILENT_REACT:"


# Only emoji from prompts.ALLOWED_REACTION_EMOJIS — Telegram rejects
# anything else with REACTION_INVALID (see prompts.py for details). Notably
# 😂 and 💀 are NOT in Telegram's set (🤣 is the closest valid equivalent).
_SILENT_REACTION_EMOJIS = {
    "joke": ["🤣", "😁", "🎉", "😱"],
    "tease": ["👀", "🤨", "😈"],
    "casual": ["👍", "🫡", "🤝"],
}
_DEFAULT_SILENT_REACTION_EMOJIS = ["👍", "🤣", "🔥"]


def _silent_reaction_pool(situation: str) -> List[str]:
    """Candidate reaction emoji matching the situation, for the grade==0 ack."""
    return _SILENT_REACTION_EMOJIS.get(situation, _DEFAULT_SILENT_REACTION_EMOJIS)


_RANDOM_NON_GIF_HINTS = (
    "можно один раз ответить стикером (STICKER:<эмоция>), если это в тему",
    "можно один раз ответить реакцией-эмодзи (REACT:<эмодзи>), если это в тему",
)
_RANDOM_GIF_HINT = "можно один раз ответить гифкой (GIPHY:<запрос>), если это в тему"
_NO_MEDIA_HINT = (
    "в этот раз — только обычный текст, без GIPHY:/REACT:/STICKER:, "
    "даже если тема шутливая"
)


def _build_media_hint(
    message_text: str, media_probability: float, gif_probability: float
) -> Tuple[str, bool]:
    """
    Decide, in CODE (not left to the model's own judgement), whether this
    turn is allowed to use GIPHY:/REACT:/STICKER: instead of plain text.

    This is the fix for "гифка/стикер вместо ответа, когда не просили, и
    текст, когда просили" (Проблема №2): an explicit request from the user
    always wins and is force-instructed; otherwise there's a small,
    independently-tuned chance of an unprompted media reply — `gif_probability`
    for GIPHY (kept lower: a GIF is a network round-trip that can fail/be
    slow) and `media_probability` for sticker/reaction.

    Returns (hint, allowed). When not allowed, the hint is now an explicit
    negative instruction placed right next to the message (not just relying
    on the general rule stated once at the top of the system prompt, which
    the model can drift away from) — see the caller for the code-level
    guard that also rejects a media reply if the model ignores this anyway.
    """
    explicit = _detect_media_request(message_text)
    if explicit == "gif":
        return (
            "пользователь явно просит гифку — ответь строго GIPHY:<короткий запрос на английском>, без текста до/после",
            True,
        )
    if explicit == "sticker":
        return (
            "пользователь явно просит стикер — ответь строго STICKER:<эмоция>, без текста до/после",
            True,
        )

    roll = random.random()
    if roll < gif_probability:
        return (_RANDOM_GIF_HINT, True)
    if roll < gif_probability + media_probability:
        return (random.choice(_RANDOM_NON_GIF_HINTS), True)

    return (_NO_MEDIA_HINT, False)


def _is_media_response(text: str) -> bool:
    """True if the model's answer used GIPHY:/REACT:/STICKER: instead of plain text."""
    prefix = text.strip().upper()
    return prefix.startswith(("GIPHY:", "REACT:", "STICKER:"))


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

        # ConversationAnalyzer for grade 0-3 + memory decisions
        self._analyzer = ConversationAnalyzer(
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
            model=config.classifier_model,  # fast model for classification
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
            # situations. Kept immutable after init: Brain is a single
            # instance shared across all concurrently-handled messages, so
            # per-message prompt swapping must never mutate shared state.
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
        on_decided_to_respond: Optional[Callable[[], Awaitable[None]]] = None,
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
            on_decided_to_respond: Awaited once classification says the bot is
                actually going to answer (grade > 0). The caller uses it to
                start the "typing" indicator — showing it before classification
                would tell the chat the bot is typing on every single message,
                including the ones it then silently ignores.

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
        _rag_debug.info(
            "RAG-DEBUG [1/4 classify] message=%r author=%r reply_to=%r bot_responded_recently=%s",
            message_text, author, reply_text, bot_responded_recently,
        )
        t0 = time.monotonic()
        result = await self._analyzer.classify(msg_ctx, recent_messages)
        classify_ms = (time.monotonic() - t0) * 1000
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
        _rag_debug.info(
            "RAG-DEBUG [1/4 classify] done in %.0fms -> needs_memory=%s rag_query=%r "
            "(решение почему: %r, from_fallback=%s)",
            classify_ms, result.needs_memory, result.rag_query, result.reason, result.from_fallback,
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
                pool = _silent_reaction_pool(result.situation)
                logger.info(
                    "Silent reaction instead of full silence: pool=%s (situation=%s)",
                    pool, result.situation,
                )
                return f"{SILENT_REACT_PREFIX}{','.join(pool)}"
            return None

        # Throttle grade=1 — short reactions fire only ~30% of the time to
        # avoid the bot chiming in on every casual message.
        if result.grade == 1 and random.random() >= self.config.grade1_response_probability:
            return None

        # From here on the bot is committed to producing a real answer, so this
        # is the first honest moment to show "typing". A failing indicator must
        # never cost us the reply — it's cosmetic.
        if on_decided_to_respond is not None:
            try:
                await on_decided_to_respond()
            except Exception as e:
                logger.debug("typing indicator callback failed: %s", e)

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
            _rag_debug.info(
                "RAG-DEBUG [2/4 retrieve] querying LightRAG: query=%r mode=%s top_k=%s",
                result.rag_query, self._rag_client.query_mode, self._rag_client.query_top_k,
            )
            t1 = time.monotonic()
            rag_facts = await self._rag_client.retrieve(result.rag_query)
            retrieve_ms = (time.monotonic() - t1) * 1000
            self._rag_usage.record_retrieval(bool(rag_facts))
            if rag_facts:
                logger.info(
                    "LightRAG facts retrieved (%d chars)", len(rag_facts)
                )
                _rag_debug.info(
                    "RAG-DEBUG [2/4 retrieve] done in %.0fms -> %d chars returned:\n%s",
                    retrieve_ms, len(rag_facts), rag_facts,
                )
            else:
                # A memory miss is never fatal — but it must be visible. The bot
                # is about to answer as if it remembers nothing, and silently
                # degrading here is exactly how a broken/slow LightRAG goes
                # unnoticed for days. Separate the two causes: a failed request
                # (timeout/network/HTTP) is an infrastructure problem, an empty
                # result is a retrieval-quality problem.
                rag_error = getattr(self._rag_client, "last_error", None)
                if rag_error:
                    logger.warning(
                        "Отвечаю БЕЗ памяти: LightRAG не ответил за %.0fms (query=%r): %s",
                        retrieve_ms, result.rag_query, rag_error,
                    )
                else:
                    logger.warning(
                        "Отвечаю БЕЗ памяти: LightRAG ничего не нашёл за %.0fms (query=%r)",
                        retrieve_ms, result.rag_query,
                    )
                _rag_debug.info(
                    "RAG-DEBUG [2/4 retrieve] done in %.0fms -> ничего не найдено (%s)",
                    retrieve_ms, rag_error or "пустой результат, LightRAG ответил",
                )
        elif result.needs_memory:
            _rag_debug.info(
                "RAG-DEBUG [2/4 retrieve] skipped: needs_memory=True но rag_client=%s rag_query=%r "
                "(нет клиента или классификатор не дал запрос)",
                bool(self._rag_client), result.rag_query,
            )

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

            # rag_facts is NOT spliced in here — get_context_prompt() places it
            # in its own block. Doing both put the same 40k-character blob in
            # the prompt twice (measured: exactly 2x on every query), so every
            # message with memory was billed double for nothing.
            enhanced_context = context

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
            media_allowed = False
            if not self.config.text_only_mode:
                media_hint, media_allowed = _build_media_hint(
                    message_text,
                    self.config.media_response_probability,
                    self.config.gif_response_probability,
                )

            user_prompt = get_context_prompt(
                enhanced_context, message_text, rag_facts, media_hint
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            _rag_debug.info(
                "RAG-DEBUG [3/4 prompt] situation=%s grade=%d grudge_level=%d "
                "rag_facts_included=%s media_hint=%r max_tokens=%d temperature=%.2f",
                situation, grade, grudge_level, bool(rag_facts), media_hint,
                dynamic_max_tokens, dynamic_temperature,
            )
            _rag_debug.info(
                "RAG-DEBUG [3/4 prompt] system_prompt (%d chars):\n%s",
                len(system_prompt), system_prompt,
            )
            _rag_debug.info(
                "RAG-DEBUG [3/4 prompt] user_prompt (%d chars):\n%s",
                len(user_prompt), user_prompt,
            )

            t2 = time.monotonic()
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
            generate_ms = (time.monotonic() - t2) * 1000

            answer = response.choices[0].message.content.strip()
            usage = getattr(response, "usage", None)
            if usage is not None:
                _rag_debug.info(
                    "RAG-DEBUG [4/4 generate] done in %.0fms -> tokens: prompt=%s completion=%s total=%s",
                    generate_ms,
                    getattr(usage, "prompt_tokens", "?"),
                    getattr(usage, "completion_tokens", "?"),
                    getattr(usage, "total_tokens", "?"),
                )
            else:
                _rag_debug.info(
                    "RAG-DEBUG [4/4 generate] done in %.0fms -> tokens: (usage не вернулся)",
                    generate_ms,
                )
            # Code-level guard (Проблема №2, part 2): the hint above is just an
            # instruction — the model can still ignore it. If it used
            # GIPHY:/REACT:/STICKER: on a turn that wasn't allowed, don't ship
            # that to the user; ask once more for plain text, and fall back to
            # silence (None) rather than send a broken/mismatched reply if it
            # still refuses.
            if not media_allowed and _is_media_response(answer):
                logger.warning(
                    "Model used %s despite media not being allowed this turn "
                    "(message=%r) — retrying as text-only",
                    answer.split(":", 1)[0], message_text,
                )
                retry_messages = messages + [
                    {"role": "assistant", "content": answer},
                    {
                        "role": "user",
                        "content": (
                            "Это нельзя было отвечать гифкой/стикером/реакцией — "
                            "ответь на исходный вопрос обычным текстом."
                        ),
                    },
                ]
                try:
                    retry_response = retry_sync(
                        lambda: self.client.chat.completions.create(
                            model=self.config.deepseek_model,
                            messages=retry_messages,
                            max_tokens=dynamic_max_tokens,
                            temperature=dynamic_temperature,
                            extra_body={"thinking": {"type": "disabled"}},
                        ),
                        attempts=self.config.deepseek_max_attempts,
                        base_delay=self.config.deepseek_retry_base_delay,
                        max_delay=4.0,
                        jitter=0.3,
                    )
                    retry_answer = retry_response.choices[0].message.content.strip()
                except Exception as e:
                    logger.error(f"Retry after blocked media response failed: {e}")
                    return None

                if _is_media_response(retry_answer):
                    logger.warning(
                        "Retry still returned %s — staying silent this turn",
                        retry_answer.split(":", 1)[0],
                    )
                    return None
                answer = retry_answer

            logger.info(f"Generated response (grade={grade}, situation={situation}): {answer[:60]}")
            return answer

        except Exception as e:
            logger.error(f"Error generating response from DeepSeek: {e}")
            return FALLBACK_RESPONSES["api_error"]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tokens_for_grade(grade: int) -> TokenRange:
        """Map grade 0-3 to token budget."""
        # Max values have headroom: at grade 2 max=700 ~10% of replies were
        # cut off mid-word (measured on 2026 chat history).
        ranges = {
            0: TokenRange(0, 50),        # skip (shouldn't reach generate)
            1: TokenRange(20, 150),       # short reaction
            2: TokenRange(150, 900),      # normal answer
            3: TokenRange(400, 1800),     # deep, detailed
        }
        return ranges.get(grade, TokenRange(150, 400))

    async def summarize_person(self, name: str, raw_facts: str) -> Optional[str]:
        """
        Turn a raw LightRAG context blob into a readable summary of a person.

        /profile used to print the retrieval result verbatim, which meant the
        chat got JSON entity records with `<SEP>` separators. The facts are
        there — they just need a model pass to become a paragraph. Returns None
        on failure so the caller can fall back to the raw text.

        Args:
            name: Person the profile was requested for.
            raw_facts: Context blob as returned by RagClient.retrieve().
        """
        prompt = (
            f"Ниже — сырые данные из базы знаний про человека по имени {name}.\n"
            f"Перескажи их живым текстом: кто это, чем занимается, что любит, "
            f"характерные привычки и запомнившиеся истории.\n"
            f"Только то, что есть в данных — не придумывай. Если данных мало, "
            f"так и скажи. Без JSON, без служебных пометок вроде <SEP>, "
            f"до 900 символов.\n\n{raw_facts}"
        )
        try:
            response = await asyncio.to_thread(
                lambda: self.client.chat.completions.create(
                    model=self.config.deepseek_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=600,
                    temperature=0.4,
                    extra_body={"thinking": {"type": "disabled"}},
                )
            )
            return (response.choices[0].message.content or "").strip() or None
        except Exception as e:
            logger.warning("summarize_person failed for %r: %s", name, e)
            return None

    def grudge_level_for(self, chat_id: Optional[int], user_id: int) -> int:
        """Current grudge level for (chat_id, user_id) — used by /mood."""
        return self._grudge.grudge_level((chat_id, user_id))

    def get_rag_usage_stats(self) -> dict:
        """Process-local LightRAG usage counters — used by /ragstats."""
        return self._rag_usage.as_dict()

    @property
    def available_stickers(self) -> List[str]:
        return self._available_stickers
