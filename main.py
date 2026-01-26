# main.py
"""
Main entry point for the DeepSeek Telegram bot.
Initializes all components including knowledge graphs and scheduler.
"""

import logging
import signal
import sys
from typing import Optional

from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

from config import get_config, ConfigError
from models import BotConfig
from memory import Memory
from brain import Brain
from responder import Responder, ResponseParser
from knowledge_graph import KnowledgeGraphManager
from scheduler import TaskScheduler, NightlyAnalysisTask

logger = logging.getLogger(__name__)


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
        knowledge_manager: Optional[KnowledgeGraphManager] = None,
        scheduler: Optional[TaskScheduler] = None
    ):
        """
        Initialize bot with configuration and optional dependencies.
        
        Args:
            config: Bot configuration
            memory: Optional Memory instance
            brain: Optional Brain instance
            responder: Optional Responder instance
            knowledge_manager: Optional KnowledgeGraphManager instance
            scheduler: Optional TaskScheduler instance
        """
        self.config = config
        
        # Initialize memory first (needed for knowledge manager)
        self.memory = memory or Memory(config)
        
        # Initialize knowledge graph manager
        firebase_db = self.memory.storage.get_client() if self.memory.storage else None
        self.knowledge_manager = knowledge_manager or KnowledgeGraphManager(firebase_db)
        
        # Initialize brain with knowledge manager
        self.brain = brain or Brain(
            config, 
            available_stickers=["happy", "sad", "laugh", "cool", "think", "wtf"],
            knowledge_manager=self.knowledge_manager
        )
        
        # Initialize responder
        self.responder = responder or Responder(config)
        
        # Initialize scheduler
        self.scheduler = scheduler or TaskScheduler(timezone="Europe/Kiev")
        
        # Setup nightly analysis if Gemini API key is available
        self._setup_nightly_analysis()
        
        self._app: Optional[Application] = None
        self._running = False
        
        logger.info("DeepSeekBot initialized with all components")

    def _setup_nightly_analysis(self) -> None:
        """Setup nightly Gemini analysis task if API key is available."""
        gemini_api_key = getattr(self.config, 'gemini_api_key', None)
        
        if not gemini_api_key:
            logger.info("Gemini API key not configured, nightly analysis disabled")
            return
        
        try:
            from gemini_analyzer import GeminiAnalyzer, DailyMessageCollector
            
            firebase_db = self.memory.storage.get_client() if self.memory.storage else None
            
            if not firebase_db:
                logger.warning("Firebase not available, nightly analysis disabled")
                return
            
            analyzer = GeminiAnalyzer(gemini_api_key, self.knowledge_manager)
            collector = DailyMessageCollector(firebase_db)
            
            nightly_task = NightlyAnalysisTask(
                gemini_analyzer=analyzer,
                message_collector=collector,
                run_hour=3,
                run_minute=0
            )
            nightly_task.register(self.scheduler)
            
            logger.info("Nightly analysis task configured for 3:00 AM")
            
        except ImportError as e:
            logger.warning(f"Could not setup nightly analysis: {e}")
        except Exception as e:
            logger.error(f"Error setting up nightly analysis: {e}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle incoming messages from the group chat.
        
        Flow:
        1. Validate message and check filters
        2. Save message to memory
        3. Decide if bot should respond
        4. Generate and send response
        5. Save bot's response to short-term memory
        
        Args:
            update: Telegram update object
            context: Bot context
        """
        message = update.message
        
        # Ignore messages without text
        if not message or not message.text:
            return

        # Ignore messages from the bot itself
        if message.from_user.id == context.bot.id:
            logger.debug("Ignoring message from bot itself")
            return

        # Ignore messages from other bots
        if message.from_user.is_bot:
            logger.debug(f"Ignoring message from another bot: {message.from_user.username}")
            return

        # Check chat_id filter if configured
        if self.config.chat_id and message.chat_id != self.config.chat_id:
            logger.debug(f"Ignoring message from chat {message.chat_id} (not in allowed chat)")
            return

        user_id = message.from_user.id
        username = message.from_user.first_name or message.from_user.username or "Unknown"
        text = message.text
        message_id = message.message_id

        try:
            logger.info(f"Received message from {username} (ID: {user_id}): {text[:50]}")

            # Save message to memory
            self.memory.add_message(user_id, username, text, message_id)

            # Check if bot should respond
            recent = self.memory.get_recent()
            if not self.brain.should_respond(text, recent):
                logger.debug("Bot decided not to respond to this message")
                return

            logger.info(f"Bot will respond to: {text[:50]}")

            # Get context for DeepSeek
            context_str = self.memory.get_context()

            # Generate response with personalized context
            response = self.brain.generate_response(
                text, 
                context_str,
                user_id=user_id,
                username=username
            )
            logger.info(f"Generated response: {response[:50]}")

            # Send response
            success = await self.responder.send_response(message, response, context.bot)
            
            # Save bot's response to short-term memory (so bot can see what it said)
            if success:
                # Parse response to get actual content (without REACT:, GIPHY:, etc.)
                parsed = ResponseParser.parse(response)
                self.memory.add_bot_response(
                    text=parsed.content,
                    message_id=0  # Bot responses don't have message IDs in memory
                )
                logger.debug("Bot response saved to short-term memory")

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)

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
        """
        Called when the Application starts up (async context with event loop).
        Initializes scheduler and handlers here since we now have a running event loop.
        """
        logger.info("Startup handler called - starting scheduler...")
        await self.scheduler.start()
        logger.info("Message handler registered")
        logger.info("Bot is running... Press Ctrl+C to stop")
        logger.info("=" * 50)
        self._running = True

    async def _shutdown_handler(self, app: Application) -> None:
        """Called when the Application shuts down."""
        logger.info("Shutdown handler called - stopping scheduler...")
        self._running = False
        self.scheduler.stop()
        logger.info("Bot shutdown complete")

    def run(self) -> None:
        """
        Initialize and start the bot.
        Blocks until the bot is stopped.
        """
        try:
            logger.info("=" * 50)
            logger.info("Starting DeepSeek Telegram Bot")
            logger.info(f"Bot name: {self.config.bot_name}")
            logger.info(f"Chat filter: {self.config.chat_id or 'All chats'}")
            logger.info("=" * 50)

            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()

            # Create application
            logger.info("Creating Telegram Application...")
            self._app = Application.builder().token(self.config.telegram_token).build()

            # Add message handler for all text messages
            message_handler = MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self.handle_message
            )
            self._app.add_handler(message_handler)

            # Register startup and shutdown handlers
            self._app.post_init = self._startup_handler
            self._app.post_shutdown = self._shutdown_handler

            logger.info("Starting polling...")
            
            # Start polling (this runs the event loop)
            self._app.run_polling(allowed_updates=Update.ALL_TYPES)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user (KeyboardInterrupt)")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            raise


def setup_logging(config: BotConfig) -> None:
    """
    Configure logging based on config settings.
    
    Args:
        config: Bot configuration
    """
    logging.basicConfig(
        format=config.log_format,
        level=getattr(logging, config.log_level.upper(), logging.INFO)
    )


def main() -> None:
    """
    Main entry point.
    Loads configuration and starts the bot.
    """
    try:
        # Load configuration
        config = get_config()
        
        # Setup logging
        setup_logging(config)
        
        # Create and run bot
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
