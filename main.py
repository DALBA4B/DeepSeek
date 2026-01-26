# main.py
"""
Main entry point for the DeepSeek Telegram bot.
Initializes all components and runs the bot with dependency injection.
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
from responder import Responder

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
        responder: Optional[Responder] = None
    ):
        """
        Initialize bot with configuration and optional dependencies.
        
        Args:
            config: Bot configuration
            memory: Optional Memory instance (created if not provided)
            brain: Optional Brain instance (created if not provided)
            responder: Optional Responder instance (created if not provided)
        """
        self.config = config
        
        # Initialize components with dependency injection
        self.memory = memory or Memory(config)
        self.brain = brain or Brain(config, available_stickers=["happy", "sad", "laugh", "cool", "think", "wtf"])
        self.responder = responder or Responder(config)
        
        self._app: Optional[Application] = None
        self._running = False
        
        logger.info("DeepSeekBot initialized with all components")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle incoming messages from the group chat.
        
        Flow:
        1. Validate message and check filters
        2. Save message to memory
        3. Decide if bot should respond
        4. Generate and send response
        
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

            # Generate response
            response = self.brain.generate_response(text, context_str)
            logger.info(f"Generated response: {response[:50]}")

            # Send response
            await self.responder.send_response(message, response, context.bot)

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown handlers."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self._running = False
            if self._app:
                self._app.stop_running()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

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

            logger.info("Message handler registered")
            logger.info("Bot is running... Press Ctrl+C to stop")
            logger.info("=" * 50)

            self._running = True
            
            # Start polling
            self._app.run_polling(allowed_updates=Update.ALL_TYPES)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user (KeyboardInterrupt)")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            raise
        finally:
            self._running = False
            logger.info("Bot shutdown complete")


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
