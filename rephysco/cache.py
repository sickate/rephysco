"""Caching utilities for the Rephysco LLM System.

This module provides caching functionality to reduce API calls and improve performance.
"""

import hashlib
import json
import os
from typing import Any, Dict, Optional

import diskcache


class LLMCache:
    """Cache for LLM responses.
    
    This class provides a simple caching mechanism to avoid redundant API calls
    for identical requests.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, ttl: int = 86400):
        """Initialize the cache.
        
        Args:
            cache_dir: Directory to store the cache (defaults to ~/.rephysco/cache)
            ttl: Time-to-live for cache entries in seconds (default: 1 day)
        """
        if cache_dir is None:
            home_dir = os.path.expanduser("~")
            cache_dir = os.path.join(home_dir, ".rephysco", "cache")
        
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = diskcache.Cache(cache_dir)
        self.ttl = ttl
    
    def get_cache_key(self, request_data: Dict[str, Any]) -> str:
        """Generate a cache key for a request.
        
        Args:
            request_data: The request data to generate a key for
            
        Returns:
            A string cache key
        """
        # Sort the keys to ensure consistent hashing
        serialized = json.dumps(request_data, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()
    
    def get(self, request_data: Dict[str, Any]) -> Optional[Any]:
        """Get a cached response for a request.
        
        Args:
            request_data: The request data to look up
            
        Returns:
            The cached response, or None if not found
        """
        key = self.get_cache_key(request_data)
        return self.cache.get(key)
    
    def set(self, request_data: Dict[str, Any], response: Any) -> None:
        """Cache a response for a request.
        
        Args:
            request_data: The request data to cache
            response: The response to cache
        """
        key = self.get_cache_key(request_data)
        self.cache.set(key, response, expire=self.ttl)
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
    
    def close(self) -> None:
        """Close the cache."""
        self.cache.close()
