# main.py
"""
Main entry point for the DeepSeek Telegram bot.

Phase B (LightRAG): the answer pipeline now goes through
Brain.analyze_and_respond() (classify grade 0-3 → fetch LightRAG facts →
generate), and a nightly RagIngestTask feeds the day's chat into LightRAG.
"""

import asyncio
import logging
import os
import signal
import subprocess
import sys
from logging.handlers import RotatingFileHandler
from typing import List, Optional

from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes

from config import get_config, ConfigError
from models import BotConfig
from memory import Memory, RecentResponseTracker
from brain import Brain, SILENT_REACT_PREFIX
from responder import Responder, ResponseParser
from rag_client import RagClient
from rag_ingestor import RagIngestor
from night_analyzator import TaskScheduler, RagIngestTask
from utils import keep_typing, start_typing, stop_typing

logger = logging.getLogger(__name__)

# Silence verbose logging from external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram.ext._application").setLevel(logging.WARNING)

# Sticker packs the bot picks from (combined). Add more short names here to
# give the bot a wider variety — find a pack's exact short name (Telegram's
# in-app display name doesn't always match) with tools/sticker_finder.py.
STICKER_PACKS = [
    "userpack7845974_by_stickrubot",
    "aye3_by_APT_bot",
    "unwrsa_by_stickrubot",
    "userpack7251755_by_stickrubot",
    "kggqqwk_by_stickrubot"
]


def build_rag_client(config: BotConfig) -> Optional[RagClient]:
    """
    Construct a RagClient from config, or return None when LightRAG is off /
    not configured. Returning None is intentional: Brain and the ingestor both
    tolerate a missing client and degrade to "answer without long-term memory".
    """
    if not config.lightrag_enabled:
        logger.info("LightRAG disabled by config (LIGHTRAG_ENABLED=false)")
        return None
    if not config.lightrag_api_url:
        logger.warning("LightRAG enabled but LIGHTRAG_API_URL is empty — running without long-term memory")
        return None
    return RagClient(
        base_url=config.lightrag_api_url,
        username=config.lightrag_api_user,
        password=config.lightrag_api_password,
        query_mode=config.lightrag_query_mode,
        query_top_k=config.lightrag_query_top_k,
        query_max_tokens=config.lightrag_query_max_tokens,
        query_timeout=config.lightrag_query_timeout,
        insert_timeout=config.lightrag_insert_timeout,
        context_desc_cap=config.lightrag_context_desc_cap,
        context_max_chunks=config.lightrag_context_max_chunks,
    )


class DeepSeekBot:
    """
    Main bot class that orchestrates all components.
    Uses dependency injection for testability.
    """

    def __init__(
        self,
        config: BotConfig,
        memory: Optional[Memory] = None,
        brain: Optional[Brain] = None,
        responder: Optional[Responder] = None,
        rag_client: Optional[RagClient] = None,
        scheduler: Optional[TaskScheduler] = None,
    ):
        """
        Initialize bot with configuration and optional dependencies.

        Args:
            config: Bot configuration
            memory: Optional Memory instance
            brain: Optional Brain instance
            responder: Optional Responder instance
            rag_client: Optional RagClient instance (long-term knowledge base)
            scheduler: Optional TaskScheduler instance
        """
        self.config = config

        # Memory first (needed by RAG ingestor)
        self.memory = memory or Memory(config)

        # LightRAG client (may be None if disabled/unconfigured)
        self.rag_client = rag_client or build_rag_client(config)

        # Brain — V2 pipeline (classify → RAG facts → generate).
        # rag_client is optional; Brain handles None gracefully.
        self.brain = brain or Brain(
            config,
            available_stickers=["happy", "sad", "laugh", "cool", "think", "wtf"],
            rag_client=self.rag_client,
        )

        # Responder (unchanged — parses TEXT/GIPHY/REACT/STICKER)
        self.responder = responder or Responder(config)

        # Scheduler uses the configured timezone (fixes bug #2: run hour was
        # hard-coded; the hour/minute now come from config too).
        self.scheduler = scheduler or TaskScheduler(timezone=config.timezone)

        # Nightly RAG ingest (replaces the old knowledge-graph nightly run)
        self.rag_ingestor: Optional[RagIngestor] = None
        self.rag_task: Optional[RagIngestTask] = None
        self._setup_rag_ingest()

        self._app: Optional[Application] = None
        self._running = False

        # Track recent responses to discourage repetition
        self._response_tracker = RecentResponseTracker(max_items=10)

        logger.info("DeepSeekBot initialized (Phase B: LightRAG pipeline)")

    def _run_selftest(self) -> None:
        """
        Run pytest once at startup and log the summary.

        Deliberately advisory-only: pytest runs in a separate process and the
        exit code is only logged, never acted on. A broken test must not keep
        the bot from starting (on Railway that would mean a restart loop) — the
        point is that a failure is visible in the logs afterwards.
        """
        if not self.config.selftest_on_startup:
            return

        logger.info("Self-test: running pytest (result is logged, never fatal)...")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--no-header"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=True,
                text=True,
                timeout=120,
            )
        except FileNotFoundError:
            logger.warning("Self-test skipped: pytest is not installed")
            return
        except subprocess.TimeoutExpired:
            logger.warning("Self-test timed out after 120s — skipping")
            return
        except Exception as e:
            logger.warning("Self-test could not be started: %s", e)
            return

        # pytest's own last line is the summary ("102 passed in 3.78s").
        tail = [l for l in (proc.stdout or "").strip().splitlines() if l.strip()]
        summary = tail[-1] if tail else "(no output)"
        if proc.returncode == 0:
            logger.info("Self-test OK: %s", summary)
        else:
            logger.warning("Self-test FAILED (exit %d): %s", proc.returncode, summary)
            for line in tail:
                if line.startswith("FAILED") or line.startswith("ERROR"):
                    logger.warning("Self-test: %s", line)

    def _setup_rag_ingest(self) -> None:
        """Wire up the nightly RAG ingest task if LightRAG is configured."""
        if not self.config.rag_ingest_enabled:
            logger.info("RAG nightly ingest disabled by config (RAG_INGEST_ENABLED=false)")
            return
        if self.rag_client is None:
            logger.info("RAG nightly ingest skipped: no RagClient configured")
            return

        firebase_db = self.memory.storage.get_client() if self.memory.storage else None
        self.rag_ingestor = RagIngestor(
            rag_client=self.rag_client,
            memory=self.memory,
            firebase_db=firebase_db,
            config=self.config,
        )
        self.rag_task = RagIngestTask(
            ingestor=self.rag_ingestor,
            run_hour=self.config.nightly_analysis_hour,
            run_minute=self.config.nightly_analysis_minute,
            timezone=self.config.timezone,
            every_n_days=self.config.rag_ingest_every_n_days,
        )
        self.rag_task.register(self.scheduler)
        logger.info(
            "RAG ingest task registered for %02d:%02d (every %d day(s))",
            self.config.nightly_analysis_hour,
            self.config.nightly_analysis_minute,
            self.config.rag_ingest_every_n_days,
        )

    # ------------------------------------------------------------------ #
    # Message handling (V2 pipeline)
    # ------------------------------------------------------------------ #
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle an incoming group-chat message through the V2 pipeline."""
        message = update.message
        if not message:
            return

        # Ignore messages from the bot itself and from other bots
        if message.from_user.id == context.bot.id:
            return
        if message.from_user.is_bot:
            logger.debug(f"Ignoring message from another bot: {message.from_user.username}")
            return

        # Chat filter
        if self.config.chat_id and message.chat_id != self.config.chat_id:
            logger.debug(f"Ignoring message from chat {message.chat_id} (not in allowed chat)")
            return

        user_id = message.from_user.id
        username = message.from_user.first_name or message.from_user.username or "Unknown"
        text = message.text or ""
        message_id = message.message_id

        try:
            if not text:
                logger.debug("Message has no text, ignoring")
                return

            logger.info(f"Received message from {username} (ID: {user_id}): {text[:50]}")

            # Extract reply context (if this message is a reply to another)
            reply_text = ""
            if message.reply_to_message and message.reply_to_message.text:
                reply_text = message.reply_to_message.text

            # Save to memory (short-term + daily_log + Firebase), with reply context
            self.memory.add_message(
                user_id, username, text, message_id,
                reply_to_text=reply_text or None,
                chat_id=message.chat_id,
            )

            bot_was_recent = self.memory.bot_responded_recently(within_last_n=3)

            # "typing" is deliberately NOT shown yet: at this point we don't
            # know whether the bot will answer at all. It's started from the
            # callback below, once classification has decided it will, and runs
            # until the reply is actually sent — a single send_chat_action only
            # holds the bubble ~5s, far less than a full generation takes.
            typing_task: Optional[asyncio.Task] = None

            async def show_typing() -> None:
                nonlocal typing_task
                if typing_task is None:
                    typing_task = start_typing(context.bot, message.chat_id)

            try:
                # V2: single call decides grade 0-3, whether memory is needed, and
                # generates the answer. Returns None when grade == 0 (stay silent).
                recent_lines: List[str] = self.memory.get_recent_context_lines()
                response = await self.brain.analyze_and_respond(
                    message_text=text,
                    author=username,
                    recent_messages=recent_lines,
                    reply_text=reply_text or None,
                    bot_responded_recently=bot_was_recent,
                    avoid_responses=self._response_tracker.get_avoid_list(),
                    user_id=user_id,
                    chat_id=message.chat_id,
                    on_decided_to_respond=show_typing,
                )

                if response is None:
                    logger.debug("Bot decided to stay silent (grade 0)")
                    return

                # Reaction-only ack (grade 0 but chose not to be fully silent) —
                # bypasses the normal text/GIPHY/STICKER/REACT pipeline entirely,
                # no message history tracking, since this isn't really a "reply".
                if response.startswith(SILENT_REACT_PREFIX):
                    candidates = response[len(SILENT_REACT_PREFIX):].split(",")
                    await self.responder.send_silent_reaction_from_pool(message, candidates)
                    return

                logger.info(f"Generated response: {response[:50]}")

                # Send whatever the brain produced (text / GIPHY: / REACT: / STICKER:)
                success = await self.responder.send_response(message, response, context.bot)
            finally:
                # Stop the indicator on every exit path: silence, reaction,
                # successful send, or an exception mid-generation.
                await stop_typing(typing_task)

            if success:
                parsed = ResponseParser.parse(response, text_only_mode=self.config.text_only_mode)
                self.memory.add_bot_response(text=parsed.content, message_id=0)
                self._response_tracker.add_response(parsed.response_type.value, parsed.content)

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)

    # ------------------------------------------------------------------ #
    # Commands: RAG (Phase B)
    # ------------------------------------------------------------------ #
    async def _cmd_ragstats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /ragstats: LightRAG connectivity + last ingest summary."""
        if not update.effective_chat:
            return
        chat_id = update.effective_chat.id

        if self.rag_client is None:
            await context.bot.send_message(
                chat_id=chat_id, text="ℹ️ LightRAG не настроен (LIGHTRAG_ENABLED=false или нет URL)"
            )
            return

        ok = await self.rag_client.health()
        status = "🟢 онлайн" if ok else "🔴 недоступен"

        stats_line = ""
        if self.rag_ingestor is not None:
            stats = self.rag_ingestor.get_last_stats()
            last_ts = self.rag_ingestor.get_last_ingest_timestamp()
            if stats and last_ts:
                stats_line = (
                    f"\n\n🕒 Последняя индексация: {last_ts.strftime('%Y-%m-%d %H:%M')}"
                    f"\n📝 Сообщений: {stats.get('messages', 0)}"
                    f"\n📦 Блоков: {stats.get('blocks', 0)}"
                    f"\n⬆️ Добавлено: {stats.get('inserted', 0)}"
                )
            else:
                stats_line = "\n\n🕒 Индексаций ещё не было"

        # How often the bot actually reaches for memory, and how often
        # LightRAG has something to say (process-local since last restart).
        usage = self.brain.get_rag_usage_stats()
        usage_line = (
            f"\n\n🧠 Обращений к памяти: {usage['needs_memory']}/{usage['total_classified']} сообщений"
            f"\n✅ Найдено фактов: {usage['facts_retrieved']} ({usage['hit_rate_pct']}%)"
        )

        await context.bot.send_message(
            chat_id=chat_id,
            text=f"📊 LightRAG: {status}\n🔗 URL: {self.config.lightrag_api_url}{stats_line}{usage_line}",
        )

    async def _cmd_ragnow(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /ragnow: manually trigger a RAG ingest for the last 24h."""
        if not update.effective_chat:
            return
        chat_id = update.effective_chat.id

        if self.rag_ingestor is None:
            await context.bot.send_message(chat_id=chat_id, text="ℹ️ RAG-индексация не настроена")
            return

        await context.bot.send_message(chat_id=chat_id, text="⏳ Запускаю индексацию последних 24ч...")

        try:
            stats = await self.rag_ingestor.ingest()
        except Exception as e:
            logger.error(f"Manual RAG ingest failed: {e}", exc_info=True)
            await context.bot.send_message(chat_id=chat_id, text=f"❌ Ошибка: {str(e)[:100]}")
            return

        if stats.get("skipped") == "no_messages":
            await context.bot.send_message(chat_id=chat_id, text="📭 Нет новых сообщений для индексации")
            return

        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"✅ Готово!\n"
                f"📝 Сообщений: {stats.get('messages', 0)}\n"
                f"📦 Блоков: {stats.get('blocks', 0)}\n"
                f"⬆️ Добавлено: {stats.get('inserted', 0)}"
            ),
        )

    async def _cmd_profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle /profile <имя>: ask LightRAG for facts about a person and show
        a compact summary. Useful for debugging what the bot "remembers".
        """
        if not update.effective_chat:
            return
        chat_id = update.effective_chat.id

        if self.rag_client is None:
            await context.bot.send_message(chat_id=chat_id, text="ℹ️ LightRAG не настроен")
            return

        # Parse target name from command args or from a replied-to message
        args = context.args
        target = " ".join(args).strip() if args else ""
        if not target and update.message and update.message.reply_to_message:
            ru = update.message.reply_to_message.from_user
            target = ru.first_name or ru.username or ""
        if not target:
            await context.bot.send_message(
                chat_id=chat_id, text="Использование: /profile <имя> (или ответь на сообщение человека)"
            )
            return

        # LightRAG retrieval plus a summarization call — comfortably longer than
        # Telegram's ~5s action window, so hold the indicator for both.
        async with keep_typing(context.bot, chat_id):
            facts = await self.rag_client.retrieve(
                f"факты, интересы и привычки человека по имени {target}"
            )
            if not facts:
                await context.bot.send_message(
                    chat_id=chat_id, text=f"🤷 Ничего не знаю про «{target}». Возможно, ещё не проиндексировано."
                )
                return

            # The retrieval result is a machine-readable context blob (JSON entity
            # records, <SEP>-joined descriptions). Send it through the model so the
            # chat gets prose; fall back to the trimmed blob if that call fails.
            summary = await self.brain.summarize_person(target, facts)

        if not summary:
            summary = facts.strip()
            if len(summary) > 1500:
                summary = summary[:1497] + "..."

        await context.bot.send_message(
            chat_id=chat_id,
            text=f"📊 Что я знаю про {target}:\n\n{summary}",
        )

    async def _cmd_mood(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle /mood: show the bot's current grudge level towards a person
        (reply to their message to target them). Reads GrudgeTracker state —
        purely in-RAM, resets on restart, escalates "Ответочка" tone only.
        """
        if not update.effective_chat or not update.message:
            return
        chat_id = update.effective_chat.id

        if not update.message.reply_to_message or not update.message.reply_to_message.from_user:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Ответь командой /mood на чьё-нибудь сообщение, чтобы узнать уровень обиды на него.",
            )
            return

        target = update.message.reply_to_message.from_user
        target_name = target.first_name or target.username or "этот человек"
        level = self.brain.grudge_level_for(chat_id, target.id)

        if level == 0:
            mood_text = "🙂 спокоен, обид нет"
        elif level <= 2:
            mood_text = f"😐 слегка задет (уровень {level})"
        else:
            mood_text = f"🔥 реально обижен (уровень {level}) — ответочка будет жёстче"

        await context.bot.send_message(
            chat_id=chat_id, text=f"Настроение насчёт {target_name}: {mood_text}"
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown handlers."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self._running = False
            self.scheduler.stop()
            if self._app:
                self._app.stop_running()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def _startup_handler(self, app: Application) -> None:
        """Called when the Application starts: start scheduler, configure bot."""
        logger.info("Startup handler called - starting scheduler...")
        await self.scheduler.start()

        # Give the RAG task a bot handle so it can post ingest reports
        if self.rag_task and self.config.chat_id:
            self.rag_task.set_bot(bot=app.bot, chat_id=self.config.chat_id)
            logger.info("RAG ingest task bot configured")

        # Load sticker packs (still used by the responder). Multiple packs are
        # supported — load_sticker_set() accumulates across calls instead of
        # replacing, so the bot picks from all of them combined. Find a pack's
        # exact name (Telegram short names don't always match what's shown in
        # the app) with tools/sticker_finder.py.
        for sticker_pack_name in STICKER_PACKS:
            try:
                logger.info(f"Loading sticker pack '{sticker_pack_name}'...")
                if hasattr(self.responder, 'sticker_manager'):
                    await self.responder.sticker_manager.load_sticker_set(app.bot, sticker_pack_name)
            except Exception as e:
                logger.error(f"Failed to load sticker pack '{sticker_pack_name}': {e}")

        logger.info("Message handler registered")
        logger.info("=" * 50)
        self._running = True

    async def _shutdown_handler(self, app: Application) -> None:
        """Called when the Application shuts down."""
        logger.info("Shutdown handler called - stopping scheduler...")
        self._running = False
        self.scheduler.stop()
        logger.info("Bot shutdown complete")

    def run(self) -> None:
        """Initialize and start the bot. Blocks until the bot is stopped."""
        try:
            logger.info("=" * 50)
            logger.info("Starting DeepSeek Telegram Bot")
            logger.info(f"Bot name: {self.config.bot_name}")
            logger.info(f"Chat filter: {self.config.chat_id or 'All chats'}")
            logger.info(f"LightRAG: {'on' if self.rag_client else 'off'}")
            logger.info("=" * 50)

            self._run_selftest()

            self._setup_signal_handlers()

            logger.info("Creating Telegram Application...")
            self._app = Application.builder().token(self.config.telegram_token).build()

            # Text messages (exclude commands)
            self._app.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
            )

            # Commands
            self._app.add_handler(CommandHandler("ragstats", self._cmd_ragstats))
            self._app.add_handler(CommandHandler("ragnow", self._cmd_ragnow))
            self._app.add_handler(CommandHandler("profile", self._cmd_profile))
            self._app.add_handler(CommandHandler("mood", self._cmd_mood))

            self._app.post_init = self._startup_handler
            self._app.post_shutdown = self._shutdown_handler

            logger.info("Starting polling...")
            self._app.run_polling(allowed_updates=Update.ALL_TYPES)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user (KeyboardInterrupt)")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            raise


def setup_logging(config: BotConfig) -> None:
    """
    Configure logging based on config settings.

    Always logs to console. Also logs to bot.log (rotated, UTF-8) when
    LOG_TO_FILE=true — lets something else tail/Read the file directly
    without needing to be attached to the process's own stdout.
    """
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if os.getenv("LOG_TO_FILE", "false").lower() in ("true", "1", "yes"):
        file_handler = RotatingFileHandler(
            "bot.log", maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        handlers.append(file_handler)

    logging.basicConfig(
        format=config.log_format,
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )

    # Verbose per-step RAG pipeline tracing (classify/retrieve/prompt/generate
    # payloads and timings) — off by default, since it dumps full prompts and
    # facts on every message. Enable with RAG_DEBUG_LOGGING=true when actually
    # debugging the memory pipeline.
    rag_debug_enabled = os.getenv("RAG_DEBUG_LOGGING", "false").lower() in ("true", "1", "yes")
    logging.getLogger("ragdebug").setLevel(
        logging.INFO if rag_debug_enabled else logging.WARNING
    )


def main() -> None:
    """Main entry point: load configuration and start the bot."""
    try:
        config = get_config()
        setup_logging(config)
        bot = DeepSeekBot(config)
        bot.run()

    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
