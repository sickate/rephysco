# Implementing New Providers

This document provides a guide for implementing new LLM providers in the Rephysco library.

## Overview

Rephysco is designed to be extensible, allowing you to add support for new LLM providers as they become available. This guide will walk you through the process of implementing a new provider.

## Provider Implementation Steps

1. [Add Provider Configuration](#1-add-provider-configuration)
2. [Create Provider Class](#2-create-provider-class)
3. [Update Factory Function](#3-update-factory-function)
4. [Update Exports](#4-update-exports)
5. [Test the Provider](#5-test-the-provider)

### 1. Add Provider Configuration

First, add the new provider's configuration to `rephsyco/config.py`:

```python
class NewProvider:
    # Define model names
    MODEL_1 = "model-1"
    MODEL_2 = "model-2"
    
    # Define API endpoints
    BASE_URL = "https://api.newprovider.com/v1"
    
    # Define API key environment variable
    API_KEY = os.getenv("NEW_PROVIDER_API_KEY")
```

### 2. Create Provider Class

Create a new file in the `rephsyco/providers` directory for your provider (e.g., `newprovider.py`):

```python
"""NewProvider implementation for the Rephysco LLM System."""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from ..config import NewProvider as NewProviderConfig
from ..errors import AuthenticationError, ProviderError, RateLimitError, ContentFilterError
from ..types import Message, ModelResponse, Role, StreamChunk
from .base import BaseProvider


class NewProviderProvider(BaseProvider):
    """Provider implementation for NewProvider API."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the NewProvider provider."""
        api_key = api_key or NewProviderConfig.API_KEY
        super().__init__(api_key=api_key, **kwargs)
        
        # Initialize the OpenAI client with NewProvider base URL
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=kwargs.get("base_url", NewProviderConfig.BASE_URL)
        )
        
        if not self.validate_api_key():
            raise AuthenticationError("NewProvider API key is required")
    
    @property
    def default_model(self) -> str:
        """Get the default model for NewProvider."""
        return NewProviderConfig.MODEL_1
    
    def format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for the NewProvider API."""
        formatted_messages = []
        
        for message in messages:
            # Format messages according to NewProvider's API requirements
            formatted_message = {
                "role": message.role.value,
                "content": message.content
            }
            
            # Add name if present (for function messages)
            if message.name:
                formatted_message["name"] = message.name
            
            # Handle multimodal content if supported
            if message.images and message.content is not None:
                # Implement multimodal formatting if supported
                pass
            
            formatted_messages.append(formatted_message)
        
        return formatted_messages
    
    async def generate(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[ModelResponse, AsyncGenerator[StreamChunk, None]]:
        """Generate a response from NewProvider."""
        model = model or self.default_model
        formatted_messages = self.format_messages(messages)
        
        # Prepare the request parameters
        params = {
            "model": model,
            "messages": formatted_messages,
            "temperature": temperature,
            "stream": stream
        }
        
        # Add max_tokens if specified
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in params:
                params[key] = value
        
        try:
            if stream:
                # Handle streaming response
                stream_response = await self.client.chat.completions.create(**params)
                return self._handle_streaming_response(stream_response)
            else:
                # Handle regular response
                response = await self.client.chat.completions.create(**params)
                return self.parse_response(response)
        except Exception as e:
            # Handle API errors
            error_message = str(e)
            
            if "authentication" in error_message.lower() or "api key" in error_message.lower():
                raise AuthenticationError(f"Authentication failed: {error_message}")
            elif "rate limit" in error_message.lower():
                raise RateLimitError(f"Rate limit exceeded: {error_message}")
            elif "content filter" in error_message.lower() or "content policy" in error_message.lower():
                raise ContentFilterError(f"Content filtered: {error_message}")
            else:
                raise ProviderError(f"NewProvider API error: {error_message}")
    
    async def _handle_streaming_response(self, response) -> AsyncGenerator[StreamChunk, None]:
        """Handle a streaming response from the NewProvider API."""
        async for chunk in response:
            yield self.parse_stream_chunk(chunk)
    
    def parse_response(self, response: ChatCompletion) -> ModelResponse:
        """Parse a response from the NewProvider API."""
        try:
            content = response.choices[0].message.content or ""
            model = response.model
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            finish_reason = response.choices[0].finish_reason
            
            return ModelResponse(
                content=content,
                model=model,
                usage=usage,
                finish_reason=finish_reason
            )
        except (AttributeError, IndexError) as e:
            raise ProviderError(f"Failed to parse NewProvider response: {e}")
    
    def parse_stream_chunk(self, chunk: ChatCompletionChunk) -> StreamChunk:
        """Parse a streaming chunk from the NewProvider API."""
        try:
            content = ""
            finish_reason = None
            model = chunk.model
            
            if chunk.choices and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                
                if choice.delta.content is not None:
                    content = choice.delta.content
                
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
            
            return StreamChunk(
                content=content,
                finish_reason=finish_reason,
                model=model
            )
        except (AttributeError, IndexError) as e:
            # Return an empty chunk on parsing errors
            return StreamChunk(content="")
```

### 3. Update Factory Function

Update the factory function in `rephsyco/providers/factory.py` to include your new provider:

```python
from ..types import ModelProvider
from .base import BaseProvider
from .openai import OpenAIProvider
from .aliyun import AliyunProvider
from .xai import XAIProvider
from .gemini import GeminiProvider
from .siliconflow import SiliconFlowProvider
from .newprovider import NewProviderProvider  # Import your new provider

def create_provider(
    provider_type: ModelProvider,
    api_key: Optional[str] = None,
    **kwargs
) -> BaseProvider:
    """Create a provider instance based on the provider type."""
    if provider_type == ModelProvider.OPENAI:
        return OpenAIProvider(api_key=api_key, **kwargs)
    elif provider_type == ModelProvider.ALIYUN:
        return AliyunProvider(api_key=api_key, **kwargs)
    elif provider_type == ModelProvider.XAI:
        return XAIProvider(api_key=api_key, **kwargs)
    elif provider_type == ModelProvider.GEMINI:
        return GeminiProvider(api_key=api_key, **kwargs)
    elif provider_type == ModelProvider.SILICON_FLOW:
        return SiliconFlowProvider(api_key=api_key, **kwargs)
    elif provider_type == ModelProvider.NEW_PROVIDER:  # Add your new provider
        return NewProviderProvider(api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")
```

### 4. Update Exports

Update the exports in `rephsyco/providers/__init__.py`:

```python
from .base import BaseProvider
from .factory import create_provider
from .openai import OpenAIProvider
from .aliyun import AliyunProvider
from .xai import XAIProvider
from .gemini import GeminiProvider
from .siliconflow import SiliconFlowProvider
from .newprovider import NewProviderProvider  # Import your new provider

__all__ = [
    "BaseProvider",
    "create_provider",
    "OpenAIProvider",
    "AliyunProvider",
    "XAIProvider",
    "GeminiProvider",
    "SiliconFlowProvider",
    "NewProviderProvider",  # Add your new provider
]
```

Also, update the `ModelProvider` enum in `rephsyco/types.py`:

```python
class ModelProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    ALIYUN = "aliyun"
    SILICON_FLOW = "silicon_flow"
    XAI = "xai"
    NEW_PROVIDER = "new_provider"  # Add your new provider
```

### 5. Test the Provider

Create a test script to verify that your provider works correctly:

```python
import asyncio
from rephsyco import LLMClient, ModelProvider
from rephsyco.config import NewProvider

async def test_new_provider():
    # Create a client with your new provider
    client = LLMClient(
        provider=ModelProvider.NEW_PROVIDER,
        model=NewProvider.MODEL_1
    )
    
    # Generate a response
    response = await client.generate(
        prompt="Explain quantum computing in simple terms."
    )
    
    print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(test_new_provider())
```

## Provider Implementation Considerations

When implementing a new provider, consider the following:

### 1. API Compatibility

Most providers offer OpenAI-compatible APIs, which makes implementation easier. If the provider has an OpenAI-compatible API, you can use the OpenAI SDK as shown in the example above.

If the provider has a different API format, you'll need to implement custom HTTP requests and response parsing.

### 2. Authentication

Different providers may have different authentication methods. Most use API keys, but some may use OAuth or other methods. Make sure to handle authentication correctly.

### 3. Error Handling

Each provider may have different error codes and messages. Make sure to handle errors appropriately and map them to Rephysco's error types. The following error types are available:

- `AuthenticationError`: For authentication failures
- `RateLimitError`: For rate limit exceeded errors (these will be automatically retried)
- `ContentFilterError`: For content policy violations
- `InvalidRequestError`: For invalid request parameters
- `ServiceUnavailableError`: For service unavailability
- `ProviderError`: For general provider errors

### 4. Multimodal Support

If the provider supports multimodal inputs (e.g., images), make sure to implement the appropriate message formatting.

### 5. Streaming Support

If the provider supports streaming responses, make sure to implement the appropriate streaming handling.

### 6. Caching Compatibility

Ensure your provider implementation works with the caching system. The cache key is generated based on the provider, model, messages, and other parameters. Make sure your implementation returns consistent responses for identical inputs.

### 7. Retry Compatibility

Ensure your provider implementation properly raises `RateLimitError` when rate limits are exceeded, so the retry mechanism can work correctly.

## Example: Implementing a Provider with a Custom API

If the provider doesn't have an OpenAI-compatible API, you'll need to implement custom HTTP requests and response parsing. Here's an example:

```python
async def generate(
    self,
    messages: List[Message],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    **kwargs
) -> Union[ModelResponse, AsyncGenerator[StreamChunk, None]]:
    """Generate a response from CustomProvider."""
    model = model or self.default_model
    formatted_messages = self.format_messages(messages)
    
    # Prepare the request payload for the custom API
    payload = {
        "model_id": model,
        "inputs": formatted_messages,
        "parameters": {
            "temperature": temperature,
            "stream": stream
        }
    }
    
    # Add max_tokens if specified
    if max_tokens is not None:
        payload["parameters"]["max_tokens"] = max_tokens
    
    # Add any additional parameters
    for key, value in kwargs.items():
        if key not in payload["parameters"]:
            payload["parameters"][key] = value
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}"
    }
    
    # Make the API request
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{self.base_url}/generate",
            headers=headers,
            json=payload
        ) as response:
            if response.status != 200:
                # Handle errors
                error_data = await response.json()
                error_message = error_data.get("error", {}).get("message", "Unknown error")
                
                if response.status == 401:
                    raise AuthenticationError(f"Authentication failed: {error_message}")
                elif response.status == 429:
                    raise RateLimitError(f"Rate limit exceeded: {error_message}")
                else:
                    raise ProviderError(f"CustomProvider API error ({response.status}): {error_message}")
            
            if stream:
                return self._handle_custom_streaming_response(response)
            else:
                response_data = await response.json()
                return self.parse_custom_response(response_data)
```

## Conclusion

By following these steps, you can implement support for new LLM providers in the Rephysco library. This allows you to leverage the unified interface and infrastructure handling of Rephysco with any LLM provider. 