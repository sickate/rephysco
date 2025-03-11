"""Aliyun provider implementation for the Rephysco LLM System.

This module implements the Aliyun provider, which supports models like Qwen-Max, Qwen-Plus, etc.
"""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage

from ..config import Aliyun as AliyunConfig
from ..errors import AuthenticationError, ProviderError, RateLimitError, ContentFilterError
from ..types import Message, ModelResponse, Role, StreamChunk
from .base import BaseProvider


class AliyunProvider(BaseProvider):
    """Provider implementation for Aliyun API (DashScope)."""
    
    # Models that require streaming mode
    STREAMING_ONLY_MODELS = [AliyunConfig.QWEN_OMNI]
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the Aliyun provider.
        
        Args:
            api_key: Aliyun API key (defaults to environment variable)
            **kwargs: Additional configuration options
        """
        api_key = api_key or AliyunConfig.API_KEY
        super().__init__(api_key=api_key, **kwargs)
        
        # Initialize the OpenAI client with Aliyun base URL
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=kwargs.get("base_url", AliyunConfig.BASE_URL)
        )
        
        if not self.validate_api_key():
            raise AuthenticationError("Aliyun API key is required")
    
    @property
    def default_model(self) -> str:
        """Get the default model for Aliyun."""
        return AliyunConfig.QWEN_MAX
    
    def format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for the Aliyun API.
        
        Args:
            messages: List of messages in the conversation
            
        Returns:
            List of formatted message dictionaries
        """
        formatted_messages = []
        
        for message in messages:
            # Handle multimodal content (images)
            if message.images and message.content is not None:
                # Aliyun requires a specific format for multimodal content
                content_list = []
                
                # Add text content
                if message.content:
                    content_list.append({
                        "type": "text",
                        "text": message.content
                    })
                
                # Add image content
                for image_url in message.images:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    })
                
                formatted_message = {
                    "role": message.role.value,
                    "content": content_list
                }
            else:
                # Regular text message
                formatted_message = {
                    "role": message.role.value,
                    "content": message.content
                }
            
            # Add name if present (for function messages)
            if message.name:
                formatted_message["name"] = message.name
            
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
        """Generate a response from Aliyun.
        
        Args:
            messages: List of messages in the conversation
            model: The specific model to use (defaults to default_model)
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional Aliyun-specific parameters
            
        Returns:
            Either a ModelResponse object or an AsyncGenerator yielding StreamChunks
        """
        model = model or self.default_model
        formatted_messages = self.format_messages(messages)
        
        # Check if the model requires streaming mode
        if model in self.STREAMING_ONLY_MODELS and not stream:
            stream = True
        
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
                raise ProviderError(f"Aliyun API error: {error_message}")
    
    async def _handle_streaming_response(self, response) -> AsyncGenerator[StreamChunk, None]:
        """Handle a streaming response from the Aliyun API.
        
        Args:
            response: The streaming response from Aliyun
            
        Yields:
            StreamChunk objects
        """
        async for chunk in response:
            yield self.parse_stream_chunk(chunk)
    
    def parse_response(self, response: ChatCompletion) -> ModelResponse:
        """Parse a response from the Aliyun API.
        
        Args:
            response: The response from Aliyun
            
        Returns:
            A ModelResponse object
        """
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
            raise ProviderError(f"Failed to parse Aliyun response: {e}")
    
    def parse_stream_chunk(self, chunk: ChatCompletionChunk) -> StreamChunk:
        """Parse a streaming chunk from the Aliyun API.
        
        Args:
            chunk: The chunk from Aliyun
            
        Returns:
            A StreamChunk object
        """
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