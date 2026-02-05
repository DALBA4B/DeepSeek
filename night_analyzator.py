# scheduler.py
"""
Task scheduler for the DeepSeek Telegram bot.
Handles scheduled tasks like nightly DeepSeek analysis.
"""

import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Callable, Optional, Awaitable, Any
import pytz

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


class NightlyAnalysisTask:
    """
    Nightly analysis task that runs DeepSeek analyzer.
    """
    
    def __init__(
        self, 
        deepseek_analyzer,
        message_collector,
        memory: Optional[Any] = None,
        knowledge_manager: Optional[Any] = None,
        run_hour: int = 3,
        run_minute: int = 0
    ):
        """
        Initialize nightly analysis task.
        
        Args:
            deepseek_analyzer: DeepSeekAnalyzer instance
            message_collector: DailyMessageCollector instance
            memory: Memory instance (optional) for cleanup
            knowledge_manager: KnowledgeGraphManager instance (optional) for cache clearing
            run_hour: Hour to run (default: 3 AM)
            run_minute: Minute to run (default: 0)
        """
        self._analyzer = deepseek_analyzer
        self._collector = message_collector
        self._memory = memory
        self._knowledge_manager = knowledge_manager
        self.run_hour = run_hour
        self.run_minute = run_minute
    
    async def run(self) -> None:
        """Run the nightly analysis and clear knowledge graph cache."""
        logger.info("Starting nightly analysis task")
        
        try:
            # Collect yesterday's messages
            messages_by_user = await self._collector.get_yesterday_messages()
            
            if messages_by_user:
                # Run analysis
                results = await self._analyzer.run_nightly_analysis(messages_by_user)
                logger.info(f"Nightly analysis complete. Updated {len(results)} user profiles.")
            else:
                logger.info("No messages to analyze from yesterday")
            
            # Clear knowledge graph cache to ensure fresh data on next use
            if self._knowledge_manager:
                self._knowledge_manager.clear_cache()
                logger.info("Knowledge graph cache cleared after nightly analysis")
            
            # Prune daily log in memory (remove analyzed messages)
            if self._memory:
                self._memory.clear_daily_log()
                logger.info("Daily log cleared after analysis")
                
        except Exception as e:
            logger.error(f"Error in nightly analysis task: {e}")
    
    def register(self, scheduler: TaskScheduler) -> None:
        """
        Register this task with a scheduler.
        
        Args:
            scheduler: TaskScheduler instance
        """
        scheduler.schedule_daily(
            name="nightly_analysis",
            hour=self.run_hour,
            minute=self.run_minute,
            callback=self.run
        )
        logger.info(f"Nightly analysis registered for {self.run_hour:02d}:{self.run_minute:02d}")
