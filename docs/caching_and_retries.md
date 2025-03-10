# Caching and Retries

This document provides detailed information about the caching and retry functionality in Rephysco.

## Caching

Rephysco includes a disk-based caching system to reduce redundant API calls. This can significantly reduce costs and improve response times for frequently used prompts.

### How Caching Works

1. When a request is made, Rephysco generates a cache key based on:
   - The provider type
   - The model name
   - The messages (including system, user, and assistant messages)
   - The temperature and other generation parameters

2. Before making an API call, Rephysco checks if a response for this cache key exists in the cache.
   - If a cached response exists and is not expired, it is returned immediately without making an API call.
   - If no cached response exists, or if the cached response is expired, an API call is made and the response is cached.

3. Cached responses are stored on disk with a configurable time-to-live (TTL).

### Configuring Caching

Caching can be configured globally or per-client:

```python
from rephysco import LLMClient, ModelProvider, Config

# Global configuration
Config.update(
    cache_dir="~/.my_cache_dir",  # Custom cache directory
    cache_ttl=3600,               # Cache TTL in seconds (1 hour)
    enable_caching=True           # Enable caching globally
)

# Per-client configuration
client = LLMClient(
    provider=ModelProvider.OPENAI,
    enable_cache=True  # Enable caching for this client
)
```

### Caching Limitations

- Streaming responses are not cached
- Multimodal inputs (images) are included in the cache key, but the images themselves are not cached
- Very large responses may impact cache performance

## Retries

Rephysco includes an automatic retry mechanism for handling transient errors, such as rate limits. This helps improve reliability when working with LLM providers.

### How Retries Work

1. When a request fails with a retryable error (e.g., `RateLimitError`), Rephysco will automatically retry the request.
2. Retries use exponential backoff with jitter to avoid overwhelming the provider's API.
3. The retry mechanism is implemented using a decorator that wraps the `generate` and `chat` methods.

### Configuring Retries

Retries can be configured globally or per-client:

```python
from rephysco import LLMClient, ModelProvider, Config

# Global configuration
Config.update(
    max_retries=5,        # Maximum number of retry attempts
    base_delay=1.0,       # Initial delay in seconds
    max_delay=60.0,       # Maximum delay in seconds
    backoff_factor=2.0,   # Factor to increase delay after each retry
    enable_retries=True   # Enable retries globally
)

# Per-client configuration
client = LLMClient(
    provider=ModelProvider.OPENAI,
    enable_retries=True  # Enable retries for this client
)
```

### Retryable Errors

By default, only `RateLimitError` is considered retryable. This is because other errors (like authentication errors or content policy violations) are unlikely to be resolved by retrying.

## Best Practices

### Caching

- Enable caching for non-time-sensitive or deterministic requests
- Use a shorter TTL for time-sensitive information
- Consider disabling caching for highly dynamic content

### Retries

- Use retries for production applications to improve reliability
- Set reasonable retry limits to avoid excessive API calls
- Consider implementing additional error handling for non-retryable errors

## Example: Using Caching and Retries

```python
import asyncio
from rephysco import LLMClient, ModelProvider, Config

async def main():
    # Configure global settings
    Config.update(
        cache_ttl=3600,     # Cache responses for 1 hour
        max_retries=5,      # Retry up to 5 times
        base_delay=1.0,     # Start with 1 second delay
        backoff_factor=2.0  # Double the delay after each retry
    )
    
    # Create a client with caching and retries
    client = LLMClient(
        provider=ModelProvider.OPENAI,
        enable_cache=True,
        enable_retries=True
    )
    
    try:
        # This will be cached after the first call
        response = await client.generate(
            prompt="Explain quantum computing in simple terms."
        )
        print(f"Response: {response[:100]}...")
        
        # This will use the cached response
        cached_response = await client.generate(
            prompt="Explain quantum computing in simple terms."
        )
        print("Used cached response:", response == cached_response)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close the client to release resources
        client.close()

if __name__ == "__main__":
    asyncio.run(main()) 