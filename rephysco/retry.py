"""Retry utilities for the Rephysco LLM System.

This module provides retry functionality for handling transient errors when
interacting with LLM providers.
"""

import asyncio
import logging
import random
from functools import wraps
from typing import Any, Callable, List, Type, TypeVar

from .errors import RateLimitError

# Type variable for the return type of the wrapped function
T = TypeVar("T")

logger = logging.getLogger(__name__)


def async_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: List[Type[Exception]] = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for retrying async functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Factor by which the delay increases with each retry
        jitter: Whether to add random jitter to the delay
        retryable_exceptions: List of exception types to retry on (defaults to RateLimitError)
        
    Returns:
        Decorator function
    """
    if retryable_exceptions is None:
        retryable_exceptions = [RateLimitError]
    
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            retries = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except tuple(retryable_exceptions) as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded: {e}")
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (backoff_factor ** (retries - 1)), max_delay)
                    
                    # Add jitter if enabled (±20%)
                    if jitter:
                        delay = delay * (0.8 + 0.4 * random.random())
                    
                    logger.warning(
                        f"Retry {retries}/{max_retries} after error: {e}. "
                        f"Waiting {delay:.2f} seconds..."
                    )
                    
                    await asyncio.sleep(delay)
        
        return wrapper
    
    return decorator
