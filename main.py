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
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes

from config import get_config, ConfigError
from models import BotConfig
from memory import Memory, RecentResponseTracker
from brain import Brain
from otvetcik import Responder, ResponseParser
from graph_memory import KnowledgeGraphManager
from night_analyzator import TaskScheduler, NightlyAnalysisTask

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
        
        # Initialize DeepSeek analyzer (will be set in _setup_nightly_analysis if available)
        self.deepseek_analyzer = None
        
        # Setup nightly analysis if DeepSeek API key is available
        self._setup_nightly_analysis()
        
        self._app: Optional[Application] = None
        self._running = False
        
        # Initialize response tracker for anti-repeat functionality
        self._response_tracker = RecentResponseTracker(max_items=10)
        
        logger.info("DeepSeekBot initialized with all components")

    def _setup_nightly_analysis(self) -> None:
        """Setup nightly DeepSeek analysis task if API key is available."""
        deepseek_api_key = self.config.deepseek_api_key
        
        if not deepseek_api_key:
            logger.info("DeepSeek API key not configured, nightly analysis disabled")
            return
        
        try:
            from deepseek_analyzer import DeepSeekAnalyzer, DailyMessageCollector
            
            firebase_db = self.memory.storage.get_client() if self.memory.storage else None
            
            # Check if any data source is available
            if not firebase_db:
                logger.info("Firebase not available, using RAM memory for nightly analysis")
            
            analyzer = DeepSeekAnalyzer(deepseek_api_key, self.knowledge_manager)
            
            # Save analyzer for /analyze command
            self.deepseek_analyzer = analyzer
            
            # Pass memory to collector for fallback/primary source
            collector = DailyMessageCollector(firebase_db, memory=self.memory)
            
            # Pass memory to task for nightly cleanup
            nightly_task = NightlyAnalysisTask(
                deepseek_analyzer=analyzer,
                message_collector=collector,
                memory=self.memory,
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
        """
        message = update.message
        
        # Check if message exists
        if not message:
            return

        # DEBUG: Log ALL message types
        logger.info(f"=== MESSAGE RECEIVED ===")
        logger.info(f"has text: {bool(message.text)}, text: {message.text[:30] if message.text else 'None'}")
        logger.info(f"has photo: {bool(message.photo)}")
        logger.info(f"has video: {bool(message.video)}")
        logger.info(f"has caption: {bool(message.caption)}, caption: {message.caption[:30] if message.caption else 'None'}")
        logger.info(f"has document: {bool(message.document)}")
        logger.info(f"======================")

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
        text = message.text or ""
        message_id = message.message_id

        try:
            # Process text messages only
            if not text:
                logger.debug("Message has no text, ignoring")
                return

            logger.info(f"Received text message from {username} (ID: {user_id}): {text[:50]}")

            # Save message to memory
            self.memory.add_message(user_id, username, text, message_id)

            # Check if bot should respond (with conversation continuation support)
            bot_was_recent = self.memory.bot_responded_recently(within_last_n=3)
            
            if not self.brain.should_respond(text, bot_responded_recently=bot_was_recent):
                logger.debug("Bot decided not to respond to this message")
                return

            logger.info(f"Bot will respond to: {text[:50]}")

            # Get context for DeepSeek
            context_str = self.memory.get_context()

            # Generate response with personalized context and avoid list
            response = self.brain.generate_response(
                text, 
                context_str,
                user_id=user_id,
                username=username,
                avoid_responses=self._response_tracker.get_avoid_list()
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
                
                # Track response for anti-repeat
                self._response_tracker.add_response(parsed.response_type.value, parsed.content)
                
                logger.debug("Bot response saved to short-term memory")

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)

    async def _cmd_daily_log(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle /daily_log command to show all messages from today.
        Useful for debugging and verifying daily memory works.
        """
        if not update.effective_chat or not update.message:
            return
        
        chat_id = update.effective_chat.id
        
        try:
            daily_messages = self.memory.get_daily_log()
            
            if not daily_messages:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="📋 Daily log is empty"
                )
                return
            
            # Group by user
            users_data = {}
            for msg in daily_messages:
                if msg.user_id not in users_data:
                    users_data[msg.user_id] = {"username": msg.username, "count": 0, "messages": []}
                users_data[msg.user_id]["count"] += 1
                users_data[msg.user_id]["messages"].append(f"[{msg.timestamp.strftime('%H:%M')}] {msg.text[:50]}")
            
            # Format output
            lines = [f"📋 Daily Log ({len(daily_messages)} messages total)\n"]
            
            for uid, data in users_data.items():
                user_type = "🤖 Bot" if uid == -1 else f"👤 {data['username']}"
                lines.append(f"\n{user_type}: {data['count']} messages")
                for msg in data['messages'][:3]:  # Show first 3 messages
                    lines.append(f"  • {msg}")
                if len(data['messages']) > 3:
                    lines.append(f"  ... and {len(data['messages']) - 3} more")
            
            text = "\n".join(lines)
            await context.bot.send_message(chat_id=chat_id, text=text)
            
        except Exception as e:
            logger.error(f"Error in daily_log command: {e}", exc_info=True)
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"❌ Error: {str(e)[:100]}"
            )

    async def _cmd_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle /analyze command to manually trigger DeepSeek analysis.
        Shows detailed analysis results for all users in daily log.
        """
        if not update.effective_chat or not update.message:
            return
        
        chat_id = update.effective_chat.id
        
        try:
            # Get daily messages from memory
            daily_messages = self.memory.get_daily_log()
            
            if not daily_messages:
                logger.info("No messages to analyze")
                return
            
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"🔄 Analyzing {len(daily_messages)} messages from today...\n⏳ This may take a moment..."
            )
            
            # Group messages by user (exclude bot messages - user_id == -1)
            users_data: dict = {}
            for msg in daily_messages:
                # Skip bot's own messages
                if msg.user_id == -1:
                    continue
                    
                if msg.user_id not in users_data:
                    users_data[msg.user_id] = {"username": msg.username, "messages": []}
                users_data[msg.user_id]["messages"].append(msg)
            
            # Analyze each user
            results = []
            detailed_results = []
            from deepseek_analyzer import DeepSeekAnalyzer
            
            logger.info(f"DeepSeek analyzer available: {hasattr(self, 'deepseek_analyzer') and self.deepseek_analyzer is not None}")
            
            if hasattr(self, 'deepseek_analyzer') and self.deepseek_analyzer is not None:
                analyzer = self.deepseek_analyzer
                for uid, data in users_data.items():
                    logger.info(f"Analyzing {len(data['messages'])} messages for {data['username']} (ID: {uid})")
                    graph = await analyzer.analyze_user_messages(
                        user_id=uid,
                        username=data['username'],
                        messages=data['messages']
                    )
                    if graph:
                        # Get new facts count from graph (analyzer.new_facts is not available, so check via graph comparison)
                        results.append(f"✅ {data['username']}: {len(data['messages'])} msgs")
                        detailed_results.append(self._format_analysis_details(data['username'], graph, show_only_new=True))
                    else:
                        results.append(f"⚠️ {data['username']}: Analysis failed")
                
                result_text = "\n".join(results)
                
                # Send summary
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"✅ Analysis complete!\n\n{result_text}\n\n📈 Knowledge graphs updated."
                )
                
                # Send detailed results for each user
                if detailed_results:
                    for detail_text in detailed_results:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=detail_text,
                            parse_mode="HTML"
                        )
                
                logger.info(f"Analysis complete. Results:\n{result_text}")
            else:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="❌ DeepSeek analyzer not available"
                )
        
        except Exception as e:
            logger.error(f"Error in analyze command: {e}", exc_info=True)
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"❌ Analysis error: {str(e)[:100]}"
            )

    def _format_analysis_details(self, username: str, graph, show_only_new: bool = False) -> str:
        """
        Format knowledge graph into readable message - only facts and interests.
        
        Args:
            username: User's username
            graph: UserKnowledgeGraph object
            show_only_new: If True, show only newly discovered facts
            
        Returns:
            Formatted HTML string with analysis details
        """
        lines = [f"<b>📊 {username}</b>"]
        
        # Quick facts - show only new ones if requested
        facts_to_show = getattr(graph, 'new_facts', graph.quick_facts) if show_only_new else graph.quick_facts
        
        if facts_to_show:
            for fact in facts_to_show:
                lines.append(f"  • {fact}")
        
        # All interests by category (no arbitrary limits)
        if graph.interests:
            for category, items in graph.interests.items():
                if items:
                    item_names = ", ".join(items.keys())
                    lines.append(f"  <b>{category}:</b> {item_names}")
        
        return "\n".join(lines)

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
        
        # Load sticker pack
        try:
            sticker_pack_name = "userpack7845974bystickrubot"
            logger.info(f"Loading sticker pack '{sticker_pack_name}'...")
            
            # Access sticker manager from responder
            if hasattr(self.responder, 'sticker_manager'):
                await self.responder.sticker_manager.load_sticker_set(app.bot, sticker_pack_name)
                logger.info("Sticker pack loaded successfully")
            else:
                logger.warning("Responder does not have sticker_manager attribute")
                
        except Exception as e:
            logger.error(f"Failed to load sticker pack: {e}")
            
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

            # Add message handler for text messages only
            message_handler = MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self.handle_message
            )
            self._app.add_handler(message_handler)
            
            # Add /analyze command handler
            analyze_handler = CommandHandler("analyze", self._cmd_analyze)
            self._app.add_handler(analyze_handler)
            
            # Add /daily_log command handler
            daily_log_handler = CommandHandler("daily_log", self._cmd_daily_log)
            self._app.add_handler(daily_log_handler)

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
