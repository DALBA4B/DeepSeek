# scheduler.py
"""
Task scheduler for the DeepSeek Telegram bot.
Handles scheduled tasks like nightly DeepSeek analysis.
"""

import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Callable, Optional, Awaitable
import pytz

from utils import get_now

logger = logging.getLogger(__name__)


class TaskScheduler:
    """
    Simple task scheduler for running periodic tasks.
    Designed for running nightly analysis at 3:00 AM.
    """
    
    def __init__(self, timezone: str = "Europe/Kiev"):
        """
        Initialize scheduler.
        
        Args:
            timezone: Timezone for scheduling (default: Europe/Kiev)
        """
        self._timezone = pytz.timezone(timezone)
        self._tasks: dict = {}
        self._running = False
        self._task_handle: Optional[asyncio.Task] = None
        logger.info(f"TaskScheduler initialized with timezone: {timezone}")
    
    def schedule_daily(
        self, 
        name: str, 
        hour: int, 
        minute: int, 
        callback: Callable[[], Awaitable[None]]
    ) -> None:
        """
        Schedule a task to run daily at a specific time.
        
        Args:
            name: Task name for logging
            hour: Hour to run (0-23)
            minute: Minute to run (0-59)
            callback: Async function to call
        """
        self._tasks[name] = {
            "time": time(hour, minute),
            "callback": callback,
            "last_run": None
        }
        logger.info(f"Scheduled task '{name}' for {hour:02d}:{minute:02d} daily")
    
    def _get_next_run_time(self, target_time: time) -> datetime:
        """
        Calculate the next run time for a daily task.
        
        Args:
            target_time: Target time of day
            
        Returns:
            Next datetime to run the task
        """
        now = datetime.now(self._timezone)
        target = self._timezone.localize(
            datetime.combine(now.date(), target_time)
        )
        
        # If target time has passed today, schedule for tomorrow
        if target <= now:
            target += timedelta(days=1)
        
        return target
    
    async def _run_scheduler(self) -> None:
        """Main scheduler loop."""
        logger.info("Scheduler loop started")
        
        while self._running:
            now = datetime.now(self._timezone)
            
            for name, task in self._tasks.items():
                target_time = task["time"]
                last_run = task["last_run"]
                
                # Check if it's time to run
                current_time = now.time()
                
                # Check if we're within 1 minute of target time
                target_minutes = target_time.hour * 60 + target_time.minute
                current_minutes = current_time.hour * 60 + current_time.minute
                
                if abs(target_minutes - current_minutes) <= 1:
                    # Check if we haven't run today
                    if last_run is None or last_run.date() < now.date():
                        logger.info(f"Running scheduled task: {name}")
                        try:
                            await task["callback"]()
                            task["last_run"] = now
                            logger.info(f"Task '{name}' completed successfully")
                        except Exception as e:
                            logger.error(f"Error running task '{name}': {e}")
            
            # Sleep for 30 seconds before checking again
            await asyncio.sleep(30)
    
    async def start(self) -> None:
        """Start the scheduler in the background."""
        if self._running:
            logger.warning("Scheduler already running")
            return
        
        self._running = True
        self._task_handle = asyncio.create_task(self._run_scheduler())
        logger.info("Scheduler started")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task_handle:
            self._task_handle.cancel()
            self._task_handle = None
        logger.info("Scheduler stopped")
    
    async def run_task_now(self, name: str) -> bool:
        """
        Manually run a scheduled task immediately.
        
        Args:
            name: Task name to run
            
        Returns:
            True if task ran successfully
        """
        if name not in self._tasks:
            logger.error(f"Task '{name}' not found")
            return False
        
        try:
            logger.info(f"Manually running task: {name}")
            await self._tasks[name]["callback"]()
            self._tasks[name]["last_run"] = datetime.now(self._timezone)
            logger.info(f"Task '{name}' completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error running task '{name}': {e}")
            return False


class RagIngestTask:
    """
    Nightly task that feeds the day's chat into LightRAG via RagIngestor.

    This is the Phase B replacement for the old knowledge-graph nightly run.
    It is fully async (RagIngestor uses aiohttp internally), so unlike the
    legacy analyzer it never blocks the bot's event loop. Every-N-days
    cadence is handled in should_run_today(), so RAG_INGEST_EVERY_N_DAYS=3
    only actually ingests every third night while still registering daily.
    """

    def __init__(
        self,
        ingestor,
        run_hour: int = 3,
        run_minute: int = 0,
        timezone: str = "UTC",
        every_n_days: int = 1,
    ):
        """
        Initialize the nightly RAG ingest task.

        Args:
            ingestor:     RagIngestor instance (does the real work).
            run_hour:     Hour to run (default 3 AM).
            run_minute:   Minute to run (default 0).
            timezone:     IANA timezone name for report timestamps.
            every_n_days: Only ingest every Nth day (1 = nightly). Decided by
                          counting days since the persisted last ingest.
        """
        self._ingestor = ingestor
        self.run_hour = run_hour
        self.run_minute = run_minute
        self._timezone = timezone
        self._every_n_days = max(1, every_n_days)
        self._bot = None
        self._chat_id = None

    def set_bot(self, bot, chat_id: int) -> None:
        """Attach the Telegram bot so ingest reports can be sent to the chat."""
        self._bot = bot
        self._chat_id = chat_id
        logger.info(f"RAG ingest task configured to report to chat {chat_id}")

    async def _report(self, text: str) -> None:
        """Best-effort send a status message to the chat."""
        if not (self._bot and self._chat_id):
            return
        try:
            await self._bot.send_message(chat_id=self._chat_id, text=text)
        except Exception as e:
            logger.error(f"Failed to send RAG report: {e}")

    def should_run_today(self) -> bool:
        """
        Decide whether today is an ingest day, based on the persisted cursor.

        With every_n_days=3, we skip days that fall inside the window since the
        last successful ingest. On the first run (no cursor) we always run.
        """
        if self._every_n_days <= 1:
            return True
        last = self._ingestor.get_last_ingest_timestamp()
        if last is None:
            return True
        now = get_now(self._timezone)
        return (now.date() - last.date()).days >= self._every_n_days

    async def run(self) -> None:
        """Run the nightly RAG ingest and report results to the chat."""
        if not self.should_run_today():
            logger.info("RAG ingest: skipping today (every_n_days=%d)", self._every_n_days)
            return

        logger.info("Starting nightly RAG ingest task")
        started_at = get_now(self._timezone).strftime("%H:%M:%S")
        await self._report(f"🌙 RAG-индексация началась в {started_at}\n⏳ Обрабатываю чат...")

        try:
            stats = await self._ingestor.ingest()
        except Exception as e:
            logger.error(f"RAG ingest task failed: {e}", exc_info=True)
            await self._report(f"❌ Ошибка индексации: {str(e)[:100]}")
            return

        if stats.get("skipped") == "no_messages":
            await self._report("📭 Нет новых сообщений для индексации")
            return

        report = (
            f"✅ RAG-индексация завершена!\n"
            f"📝 Сообщений: {stats.get('messages', 0)}\n"
            f"📦 Блоков: {stats.get('blocks', 0)}\n"
            f"⬆️ Добавлено: {stats.get('inserted', 0)}"
        )
        if stats.get("failed"):
            report += f"\n⚠️ Ошибок: {stats['failed']}"
        await self._report(report)

    def register(self, scheduler: "TaskScheduler") -> None:
        """Register this task with a scheduler."""
        scheduler.schedule_daily(
            name="rag_ingest",
            hour=self.run_hour,
            minute=self.run_minute,
            callback=self.run,
        )
        logger.info(f"RAG ingest registered for {self.run_hour:02d}:{self.run_minute:02d}")
