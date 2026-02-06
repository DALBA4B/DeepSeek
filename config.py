# config.py
"""
Configuration management for DeepSeek Telegram Bot.
Loads settings from environment variables with validation and type safety.
"""

import os
import logging
from typing import Optional, List

from dotenv import load_dotenv

from models import BotConfig

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


def _get_required_env(key: str) -> str:
    """
    Get required environment variable or raise ConfigError.
    
    Args:
        key: Environment variable name
        
    Returns:
        Value of the environment variable
        
    Raises:
        ConfigError: If variable is not set
    """
    value = os.getenv(key)
    if not value:
        raise ConfigError(f"{key} not found in environment variables. Check .env file.")
    return value


def _get_optional_int(key: str, default: Optional[int] = None) -> Optional[int]:
    """
    Get optional integer environment variable.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        
    Returns:
        Integer value or default
    """
    value = os.getenv(key)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid integer value for {key}: {value}, using default: {default}")
        return default


def _get_optional_float(key: str, default: float) -> float:
    """
    Get optional float environment variable.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        
    Returns:
        Float value or default
    """
    value = os.getenv(key)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Invalid float value for {key}: {value}, using default: {default}")
        return default


def _get_firebase_credentials() -> Optional[str]:
    """
    Get Firebase credentials from environment.
    Tries FIREBASE_CRED_JSON first (for Railway), then FIREBASE_CRED_PATH (for local dev).
    Returns None if neither is available.
    
    Returns:
        Firebase credentials path or None
    """
    # Priority 1: FIREBASE_CRED_JSON (full JSON string - for Railway)
    firebase_json = os.getenv("FIREBASE_CRED_JSON")
    if firebase_json:
        logger.info("Using FIREBASE_CRED_JSON from environment")
        return firebase_json
    
    # Priority 2: FIREBASE_CRED_PATH (file path - for local development)
    firebase_path = os.getenv("FIREBASE_CRED_PATH")
    if firebase_path:
        if os.path.exists(firebase_path):
            logger.info(f"Using FIREBASE_CRED_PATH: {firebase_path}")
            return firebase_path
        else:
            logger.warning(f"FIREBASE_CRED_PATH specified but file not found: {firebase_path}")
    
    # No Firebase credentials found
    logger.warning("No Firebase credentials found (FIREBASE_CRED_JSON or FIREBASE_CRED_PATH)")
    return None


def load_config() -> BotConfig:
    """
    Load and validate all configuration from environment.
    
    Returns:
        BotConfig instance with all settings
        
    Raises:
        ConfigError: If required configuration is missing
    """
    return BotConfig(
        # Required API keys
        telegram_token=_get_required_env("TELEGRAM_TOKEN"),
        deepseek_api_key=_get_required_env("DEEPSEEK_API_KEY"),
        giphy_api_key=_get_required_env("GIPHY_API_KEY"),
        firebase_cred_path=_get_firebase_credentials(),
        
        # Bot settings
        bot_name=os.getenv("BOT_NAME", "Вася"),
        chat_id=_get_optional_int("CHAT_ID"),
        
        # Memory settings
        # short_memory_limit: recent context for quick responses (RAM deque with N last messages)
        # All messages for the entire day are kept in daily_log (RAM) for DeepSeek analysis
        short_memory_limit=_get_optional_int("SHORT_MEMORY_LIMIT", 30) or 30,
        context_messages_count=_get_optional_int("CONTEXT_MESSAGES_COUNT", 20) or 20,
        
        # DeepSeek settings
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        deepseek_max_tokens=_get_optional_int("DEEPSEEK_MAX_TOKENS", 150) or 150,
        deepseek_temperature=_get_optional_float("DEEPSEEK_TEMPERATURE", 1.0),
        
        # Response settings
        random_response_probability=_get_optional_float("RANDOM_RESPONSE_PROBABILITY", 0.1),
        use_smart_respond=os.getenv("USE_SMART_RESPOND", "false").lower() in ("true", "1", "yes"),
        
        # Giphy settings
        giphy_api_url=os.getenv("GIPHY_API_URL", "https://api.giphy.com/v1/gifs/search"),
        giphy_limit=_get_optional_int("GIPHY_LIMIT", 10) or 10,
        giphy_rating=os.getenv("GIPHY_RATING", "pg-13"),
        
        # Scheduler settings
        nightly_analysis_hour=_get_optional_int("NIGHTLY_ANALYSIS_HOUR", 3) or 3,
        nightly_analysis_minute=_get_optional_int("NIGHTLY_ANALYSIS_MINUTE", 0) or 0,
        timezone=os.getenv("TIMEZONE", "Europe/Kiev"),
        
        # Logging
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )


# Singleton config instance (lazy loaded)
_config: Optional[BotConfig] = None


def get_config() -> BotConfig:
    """
    Get the singleton config instance.
    
    Returns:
        BotConfig instance
        
    Raises:
        ConfigError: If configuration is invalid
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config
