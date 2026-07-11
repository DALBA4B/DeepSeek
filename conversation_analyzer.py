# conversation_analyzer.py
"""
Fast single-call message classifier for the bot.

Replaces the old binary should_respond / smart_should_respond with one
unified DeepSeek call that returns:

    grade:           0–3  (skip / react / normal / deep)
    needs_memory:    bool (should we query LightRAG before answering?)
    rag_query:       str or None (what to search for in the knowledge base)

Design principles
----------------
* ONE DeepSeek call replaces both should_respond + memory decision.
* Uses the FAST model (deepseek-chat), not the thinking one — must be instant.
* Structured JSON output (DeepSeek supports response_format=json).
* On any failure → safe fallback via lightweight heuristics (bot never blocks).
* Fully async (aiohttp) so the event loop is never blocked.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import List, Optional

import aiohttp

from prompts import get_name_variations, INSULT_KEYWORDS

logger = logging.getLogger(__name__)

# ── Request token budget for the classifier ──────────────────────────
_CLASSIFIER_MAX_TOKENS = 220

# ── Valid situation values (unknown/missing → "casual") ──────────────
_SITUATIONS = {"joke", "help", "casual", "tease", "defend"}

# ── Prompt template ────────────────────────────────────────────────
_CLASSIFIER_PROMPT = """\
Ты — анализатор группового чата. Бота зовут {bot_name}. Оцени сообщение и верни JSON.

Контекст (последние сообщения):
{context}

Текущее сообщение от {author}: {message}
{reply_line}

Верни СТРОГО JSON (без markdown, без комментариев):
{{
  "grade": <0-3>,
  "needs_memory": <true/false>,
  "rag_query": <строка или null>,
  "situation": "<joke|help|casual|tease|defend>",
  "reason": "<кратко почему>"
}}

grade:
  0 — не отвечать (флуд, междометие, не к боту, неинтересно)
  1 — короткая реакция (стикер, гифка, смайл, 1-3 слова)
  2 — обычный содержательный ответ
  3 — развёрнутый ответ, требует вникания

needs_memory — true если:
  grade == 3
  упоминается конкретный человек из чата и его прошлое/факты
  есть слова-триггеры памяти: "помнишь", "вчера", "тот раз", "ты знал что"
  кто-то рассказывает о себе (факт worth remembering)

rag_query — если needs_memory, краткий поисковый запрос для базы знаний \
(например "Максим Dota игры"). Если needs_memory=false — null.

situation — в каком стиле отвечать:
  joke   — человек шутит / мем / абсурд, поддержать веселье
  help   — реальный вопрос или просьба о помощи, объяснить что-то
  casual — обычный разговор, мнение, болтовня, поддержка
  tease  — уместен лёгкий дружеский подкол
  defend — на {bot_name} наезжают, оскорбляют, провоцируют — агрессия направлена на бота

reason — одно предложение почему такой grade."""


# ── Data class ────────────────────────────────────────────────────────
@dataclass
class ClassificationResult:
    """
    Output of a single classify() call.
    """
    grade: int          # 0 = skip, 1 = react, 2 = normal, 3 = deep
    needs_memory: bool
    rag_query: Optional[str]
    reason: str
    from_fallback: bool = False  # True when structured parse failed
    situation: str = "casual"    # joke | help | casual | tease | defend


@dataclass
class _MessageContext:
    """Minimal info the classifier needs about a message."""
    author: str
    text: str
    reply_text: Optional[str]
    bot_name: str
    bot_responded_recently: bool


# ── The classifier itself ────────────────────────────────────────────
class ConversationAnalyzer:
    """
    Fast message classifier powered by a single DeepSeek call.

    Attributes:
        api_key:     DeepSeek API key
        base_url:    DeepSeek API base URL
        model:       Fast model name (e.g. deepseek-chat)
        temperature:  Low for deterministic classification
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.3,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def classify(
        self,
        message_context: _MessageContext,
        recent_messages: List[str],  # already formatted as "Name: text"
        max_tokens: int = _CLASSIFIER_MAX_TOKENS,
    ) -> ClassificationResult:
        """
        Classify a single incoming message.

        Args:
            message_context: Who sent what, with optional reply info.
            recent_messages: Last N formatted context lines.
            max_tokens:      Token budget for the classifier response.

        Returns:
            ClassificationResult with grade 0-3, memory flag, and optional
            rag_query. On failure returns a safe heuristic fallback.
        """
        # Quick heuristic pre-check: bot name → always at least grade 2,
        # even if the LLM (which now also sees {bot_name} in the prompt)
        # disagrees. Belt-and-suspenders against misclassifying direct address.
        directly_addressed = self._is_directly_addressed(message_context)
        is_attack = self._is_attack_on_bot(message_context)

        prompt = self._build_prompt(message_context, recent_messages)

        try:
            raw = await self._call_llm(prompt, max_tokens)
            result = self._parse_response(raw)
            if result is not None:
                if directly_addressed and result.grade < 2:
                    result.grade = 2
                if is_attack:
                    # Hard override: a keyword-confirmed insult aimed at the
                    # bot always gets the defend prompt, even if the (safety
                    # tuned) LLM softened it to something else.
                    result.situation = "defend"
                    result.grade = max(result.grade, 2)
                return result
        except Exception as e:
            logger.warning(f"Classifier LLM call failed: {e}")

        # Fallback: lightweight heuristics
        return self._fallback_classify(message_context)

    # ------------------------------------------------------------------ #
    # Internal: prompt building
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_prompt(
        msg: _MessageContext,
        recent_messages: List[str],
    ) -> str:
        # Keep context compact — last N messages
        context_block = "\n".join(recent_messages[-10:]) if recent_messages else "(пусто)"
        reply_line = (
            f'Ответ на сообщение: "{msg.reply_text}"'
            if msg.reply_text
            else "Ответ на сообщение: нет"
        )
        return _CLASSIFIER_PROMPT.format(
            bot_name=msg.bot_name,
            context=context_block,
            author=msg.author,
            message=msg.text,
            reply_line=reply_line,
        )

    # ------------------------------------------------------------------ #
    # Internal: LLM call (async, single request)
    # ------------------------------------------------------------------ #
    async def _call_llm(self, prompt: str, max_tokens: int) -> str:
        """
        Fire a single chat completion request to DeepSeek.
        Returns the raw assistant message text.
        """
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=8.0),
            ) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"DeepSeek {resp.status}: {text[:200]}")
                data = json.loads(text)
                return data["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------ #
    # Internal: JSON parsing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_response(raw: str) -> Optional[ClassificationResult]:
        """
        Parse the JSON returned by the classifier LLM.
        Returns None if parsing fails (caller should fall back).
        """
        try:
            # Strip potential markdown code fences
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            data = json.loads(cleaned)

            grade = int(data.get("grade", 0))
            grade = max(0, min(3, grade))  # clamp

            needs_memory = bool(data.get("needs_memory", False))
            rag_query = data.get("rag_query")
            if isinstance(rag_query, str) and not rag_query.strip():
                rag_query = None

            reason = str(data.get("reason", ""))[:100]

            situation = str(data.get("situation", "casual")).strip().lower()
            if situation not in _SITUATIONS:
                situation = "casual"

            return ClassificationResult(
                grade=grade,
                needs_memory=needs_memory,
                rag_query=rag_query,
                reason=reason,
                from_fallback=False,
                situation=situation,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Classifier JSON parse failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Internal: heuristic fallback (no LLM, no network)
    # ------------------------------------------------------------------ #
    def _fallback_classify(self, msg: _MessageContext) -> ClassificationResult:
        """
        Cheap rule-based fallback when the LLM call fails.
        Ensures the bot still makes reasonable decisions.
        """
        text_lower = msg.text.lower()
        grade = 0
        needs_memory = False
        situation = "casual"

        if self._is_attack_on_bot(msg):
            # No network, but a keyword-confirmed insult still deserves a
            # real defend reply rather than silence/casual chat.
            grade = 2
            situation = "defend"
        # Direct address → always respond
        elif self._is_directly_addressed(msg):
            grade = 2
            needs_memory = True
        # Question → respond
        elif "?" in msg.text:
            grade = 2 if len(msg.text) > 30 else 1
            situation = "help"
        # Very short → skip
        elif len(msg.text) < 10:
            grade = 0
        # Bot responded recently + continuation trigger → react
        elif msg.bot_responded_recently and any(
            t in text_lower for t in ("ещё", "еще", "ну да", "ну нет", "ага")
        ):
            grade = 1

        return ClassificationResult(
            grade=grade,
            needs_memory=needs_memory,
            rag_query=msg.author if needs_memory else None,
            reason="(fallback heuristic)",
            from_fallback=True,
            situation=situation,
        )

    # ------------------------------------------------------------------ #
    # Internal: helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_directly_addressed(msg: _MessageContext) -> bool:
        """Check if the message directly addresses the bot."""
        text_lower = msg.text.lower()
        return any(variant in text_lower for variant in get_name_variations(msg.bot_name))

    @staticmethod
    def _is_attack_on_bot(msg: _MessageContext) -> bool:
        """
        Directionality + hostility: the message is aimed at the bot AND
        contains an insult/provocation stem. Used as a hard override on top
        of the LLM's own "situation" judgement — DeepSeek's safety tuning
        tends to soften genuine insults, so this keyword floor guarantees a
        real attack still gets the defend prompt (see PROJECT_ANALYSIS_V2.md
        Причина 3.3).

        Directionality is satisfied either by an explicit name mention in
        THIS message, or by the bot having just replied (an ongoing back-
        and-forth doesn't repeat the bot's name on every line — "дип ты
        тупой" followed by a bare "дурачек" two seconds later is still
        aimed at the bot, not a new addressee).
        """
        text_lower = msg.text.lower()
        aimed_at_bot = (
            ConversationAnalyzer._is_directly_addressed(msg)
            or msg.bot_responded_recently
        )
        if not aimed_at_bot:
            return False
        return any(kw in text_lower for kw in INSULT_KEYWORDS)
