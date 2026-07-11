# sticker_finder.py
"""
Temporary standalone utility — NOT part of the bot pipeline.

Run this, then have anyone send any sticker from the pack you want the name
of into a chat the bot is in. The script prints the sticker's pack name
(set_name) to the console — that's the exact string main.py needs for
Responder's load_sticker_set() call. Stop with Ctrl+C once you have it.

Usage:
    python sticker_finder.py
"""

import logging

from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

from config import get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


async def print_sticker_set_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    sticker = update.message.sticker
    if not sticker:
        return
    print(f"\n>>> Sticker pack name: {sticker.set_name!r}\n")
    logger.info("Sticker from %s: set_name=%s", sticker.file_id, sticker.set_name)


def main() -> None:
    config = get_config()
    app = Application.builder().token(config.telegram_token).build()
    app.add_handler(MessageHandler(filters.Sticker.ALL, print_sticker_set_name))

    print("Listening for stickers... send any sticker from the pack in a chat with the bot.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
