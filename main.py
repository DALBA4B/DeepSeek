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
        
        # Initialize Gemini analyzer (will be set in _setup_nightly_analysis if available)
        self.gemini_analyzer = None
        
        # Setup nightly analysis if Gemini API key is available
        self._setup_nightly_analysis()
        
        self._app: Optional[Application] = None
        self._running = False
        
        # Initialize response tracker for anti-repeat functionality
        self._response_tracker = RecentResponseTracker(max_items=10)
        
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
            
            # Check if any data source is available
            if not firebase_db:
                logger.info("Firebase not available, using RAM memory for nightly analysis")
            
            analyzer = GeminiAnalyzer(gemini_api_key, self.knowledge_manager)
            
            # Save analyzer for /analyze command
            self.gemini_analyzer = analyzer
            
            # Pass memory to collector for fallback/primary source
            collector = DailyMessageCollector(firebase_db, memory=self.memory)
            
            # Pass memory to task for nightly cleanup
            nightly_task = NightlyAnalysisTask(
                gemini_analyzer=analyzer,
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

            # Check if bot should respond (with conversation continuation support)
            recent = self.memory.get_recent()
            bot_was_recent = self.memory.bot_responded_recently(within_last_n=3)
            
            if not self.brain.should_respond(text, recent, bot_responded_recently=bot_was_recent):
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
                avoid_responses=self._response_tracker.get_avoid_list() if hasattr(self, '_response_tracker') else None
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
                if hasattr(self, '_response_tracker'):
                    self._response_tracker.add_response(parsed.response_type.value, parsed.content)
                
                logger.debug("Bot response saved to short-term memory")

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)

    async def _cmd_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle /analyze command to manually trigger Gemini analysis.
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
            
            # Group messages by user
            users_data: dict = {}
            for msg in daily_messages:
                if msg.user_id not in users_data:
                    users_data[msg.user_id] = {"username": msg.username, "messages": []}
                users_data[msg.user_id]["messages"].append(msg)
            
            # Analyze each user
            results = []
            detailed_results = []
            from gemini_analyzer import GeminiAnalyzer
            
            logger.info(f"Gemini analyzer available: {hasattr(self, 'gemini_analyzer') and self.gemini_analyzer is not None}")
            
            if hasattr(self, 'gemini_analyzer') and self.gemini_analyzer is not None:
                analyzer = self.gemini_analyzer
                for uid, data in users_data.items():
                    logger.info(f"Analyzing {len(data['messages'])} messages for {data['username']} (ID: {uid})")
                    graph = await analyzer.analyze_user_messages(
                        user_id=uid,
                        username=data['username'],
                        messages=data['messages']
                    )
                    if graph:
                        results.append(f"✅ {data['username']}: {len(data['messages'])} msgs")
                        detailed_results.append(self._format_analysis_details(data['username'], graph))
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
                    text="❌ Gemini analyzer not available"
                )
        
        except Exception as e:
            logger.error(f"Error in analyze command: {e}", exc_info=True)
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"❌ Analysis error: {str(e)[:100]}"
            )

    def _format_analysis_details(self, username: str, graph) -> str:
        """
        Format knowledge graph into readable message.
        
        Args:
            username: User's username
            graph: UserKnowledgeGraph object
            
        Returns:
            Formatted HTML string with analysis details
        """
        lines = [f"<b>📊 Analysis for {username}</b>"]
        
        # Quick facts
        if graph.quick_facts:
            lines.append(f"\n<b>💡 Quick Facts ({len(graph.quick_facts)}):</b>")
            for fact in graph.quick_facts[:5]:  # Show top 5
                lines.append(f"  • {fact[:80]}")
            if len(graph.quick_facts) > 5:
                lines.append(f"  ... and {len(graph.quick_facts) - 5} more")
        
        # Interests by category
        if graph.interests:
            lines.append(f"\n<b>🎯 Interests ({len(graph.interests)} categories):</b>")
            for category, items in graph.interests.items():
                if items:
                    item_names = ", ".join(list(items.keys())[:3])
                    lines.append(f"  <b>{category}:</b> {item_names}")
                    if len(items) > 3:
                        lines.append(f"    ({len(items) - 3} more in {category})")
        
        # Personal info
        if graph.personal:
            lines.append(f"\n<b>👤 Personal Info:</b>")
            for key, value in list(graph.personal.items())[:3]:
                value_str = str(value)[:60] if value else "Unknown"
                lines.append(f"  • {key}: {value_str}")
            if len(graph.personal) > 3:
                lines.append(f"  ... and {len(graph.personal) - 3} more attributes")
        
        # Social info
        if graph.social:
            lines.append(f"\n<b>👥 Social:</b>")
            if "friends_mentioned" in graph.social and graph.social["friends_mentioned"]:
                friends = graph.social["friends_mentioned"]
                lines.append(f"  Friends: {', '.join(friends[:3])}")
                if len(friends) > 3:
                    lines.append(f"  ({len(friends) - 3} more friends)")
            for key in ["relationship_status", "family"]:
                if key in graph.social and graph.social[key]:
                    lines.append(f"  • {key}: {graph.social[key]}")
        
        # Patterns
        if graph.active_hours or graph.typical_topics:
            lines.append(f"\n<b>📈 Patterns:</b>")
            if graph.active_hours:
                hours_str = f"{min(graph.active_hours)}:00 - {max(graph.active_hours)}:00" if len(graph.active_hours) > 1 else f"{graph.active_hours[0]}:00"
                lines.append(f"  Active: {hours_str}")
            if graph.typical_topics:
                topics = ", ".join(graph.typical_topics[:3])
                lines.append(f"  Topics: {topics}")
        
        lines.append(f"\n⏰ <i>Updated: {graph.updated_at.strftime('%H:%M:%S')}</i>")
        
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

            # Add message handler for all text messages
            message_handler = MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self.handle_message
            )
            self._app.add_handler(message_handler)
            
            # Add /analyze command handler
            analyze_handler = CommandHandler("analyze", self._cmd_analyze)
            self._app.add_handler(analyze_handler)

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
