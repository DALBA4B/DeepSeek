# main.py
"""
Main entry point for the DeepSeek Telegram bot.
Initializes all components and runs the bot.
"""

import logging
from typing import Optional

from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

import config
from memory import Memory
from brain import Brain
from responder import Responder

# Configure logging
logging.basicConfig(
    format=config.LOG_FORMAT,
    level=config.LOG_LEVEL
)
logger = logging.getLogger(__name__)

# Global instances (will be initialized in main())
memory: Optional[Memory] = None
brain: Optional[Brain] = None
responder: Optional[Responder] = None


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle incoming messages from the group chat.
    
    Flow:
    1. Check if message is from bot or another bot
    2. Save message to memory (short-term + long-term)
    3. Decide if bot should respond using should_respond()
    4. Generate response using generate_response()
    5. Send response using send_response()
    
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

    user_id = message.from_user.id
    username = message.from_user.first_name or message.from_user.username or "Unknown"
    text = message.text
    message_id = message.message_id

    try:
        # Log received message
        logger.info(f"Received message from {username} (ID: {user_id}): {text[:50]}")

        # Save message to memory
        memory.add_message(user_id, username, text, message_id)

        # Check if bot should respond
        recent = memory.get_recent()
        if not brain.should_respond(text, recent):
            logger.debug("Bot decided not to respond to this message")
            return

        logger.info(f"Bot will respond to: {text[:50]}")

        # Get context for DeepSeek
        context_str = memory.get_context()

        # Generate response
        response = brain.generate_response(text, context_str)
        logger.info(f"Generated response: {response[:50]}")

        # Send response
        await responder.send_response(message, response, context.bot)

    except Exception as e:
        logger.error(f"Error handling message: {e}", exc_info=True)


def main() -> None:
    """
    Initialize and start the bot.
    """
    global memory, brain, responder

    try:
        logger.info("="*50)
        logger.info("Starting DeepSeek Telegram Bot")
        logger.info("="*50)

        # Initialize components
        logger.info("Initializing Memory...")
        memory = Memory()
        
        logger.info("Initializing Brain...")
        brain = Brain()
        
        logger.info("Initializing Responder...")
        responder = Responder()

        # Create application
        logger.info("Creating Telegram Application...")
        app = Application.builder().token(config.TELEGRAM_TOKEN).build()

        # Add message handler for all text messages
        message_handler = MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        )
        app.add_handler(message_handler)

        logger.info("Message handler registered")
        logger.info("Bot is running... Press Ctrl+C to stop")
        logger.info("="*50)

        # Start polling (Application manages its own event loop)
        app.run_polling()

    except KeyboardInterrupt:
        logger.info("Bot stopped by user (KeyboardInterrupt)")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
