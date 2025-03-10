"""Global settings for the Rephysco LLM System.

This module provides configuration for logging, default parameters, and other
global settings.
"""

import logging
import os
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Default cache settings
DEFAULT_CACHE_DIR = os.path.expanduser("~/.rephysco/cache")
DEFAULT_CACHE_TTL = 86400  # 1 day in seconds
ENABLE_CACHING = True

# Default retry settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0
DEFAULT_BACKOFF_FACTOR = 2.0
ENABLE_RETRIES = True

# Default generation parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = None


class Config:
    """Global configuration that can be modified at runtime."""
    
    # Cache settings
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR
    cache_ttl: int = DEFAULT_CACHE_TTL
    enable_caching: bool = ENABLE_CACHING
    
    # Retry settings
    max_retries: int = DEFAULT_MAX_RETRIES
    base_delay: float = DEFAULT_BASE_DELAY
    max_delay: float = DEFAULT_MAX_DELAY
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR
    enable_retries: bool = ENABLE_RETRIES
    
    # Default generation parameters
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS
    
    @classmethod
    def update(cls, **kwargs):
        """Update configuration settings.
        
        Args:
            **kwargs: Configuration settings to update
        """
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                logging.warning(f"Unknown configuration setting: {key}")
    
    @classmethod
    def reset(cls):
        """Reset all configuration settings to their default values."""
        cls.cache_dir = DEFAULT_CACHE_DIR
        cls.cache_ttl = DEFAULT_CACHE_TTL
        cls.enable_caching = ENABLE_CACHING
        cls.max_retries = DEFAULT_MAX_RETRIES
        cls.base_delay = DEFAULT_BASE_DELAY
        cls.max_delay = DEFAULT_MAX_DELAY
        cls.backoff_factor = DEFAULT_BACKOFF_FACTOR
        cls.enable_retries = ENABLE_RETRIES
        cls.temperature = DEFAULT_TEMPERATURE
        cls.max_tokens = DEFAULT_MAX_TOKENS


# Create a global instance
config = Config()
