"""Gemini provider implementation for the Rephysco LLM System.

This module implements the Gemini provider, which supports models like Gemini-1.5-Pro, Gemini-2.0-Flash, etc.
"""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage

from ..config import Gemini as GeminiConfig
from ..errors import AuthenticationError, ProviderError, RateLimitError, ContentFilterError
from ..types import Message, ModelResponse, Role, StreamChunk
from .base import BaseProvider


class GeminiProvider(BaseProvider):
    """Provider implementation for Google's Gemini API."""
    
    # Models with vision capabilities
    VISION_MODELS = [GeminiConfig.GEMINI_1_5_PRO]
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the Gemini provider.
        
        Args:
            api_key: Gemini API key (defaults to environment variable)
            **kwargs: Additional configuration options
        """
        api_key = api_key or GeminiConfig.API_KEY
        super().__init__(api_key=api_key, **kwargs)
        
        # Initialize the OpenAI client with Gemini base URL
        # Google AI Studio provides an OpenAI-compatible endpoint
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=kwargs.get("base_url", GeminiConfig.BASE_URL)
        )
        
        # Store the current model (used for multimodal formatting)
        self._current_model = self.default_model
        
        if not self.validate_api_key():
            raise AuthenticationError("Gemini API key is required")
    
    @property
    def default_model(self) -> str:
        """Get the default model for Gemini."""
        return GeminiConfig.GEMINI_1_5_PRO
    
    def format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for the Gemini API.
        
        Args:
            messages: List of messages in the conversation
            
        Returns:
            List of formatted message dictionaries
        """
        formatted_messages = []
        
        for message in messages:
            # Check if this is a multimodal message with images
            if message.images and message.content is not None:
                # Check if we're using a vision model
                if self._current_model in self.VISION_MODELS:
                    # Format for vision models
                    content_parts = []
                    
                    # Add text content
                    if message.content:
                        content_parts.append({
                            "type": "text",
                            "text": message.content
                        })
                    
                    # Add image content
                    for image_url in message.images:
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        })
                    
                    formatted_message = {
                        "role": message.role.value,
                        "content": content_parts
                    }
                else:
                    # For non-vision models, just use the text content
                    formatted_message = {
                        "role": message.role.value,
                        "content": message.content
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
        """Generate a response from Gemini.
        
        Args:
            messages: List of messages in the conversation
            model: The specific model to use (defaults to default_model)
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional Gemini-specific parameters
            
        Returns:
            Either a ModelResponse object or an AsyncGenerator yielding StreamChunks
        """
        model = model or self.default_model
        
        # Store the current model for use in format_messages
        self._current_model = model
        
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
                raise ProviderError(f"Gemini API error: {error_message}")
    
    async def _handle_streaming_response(self, response) -> AsyncGenerator[StreamChunk, None]:
        """Handle a streaming response from the Gemini API.
        
        Args:
            response: The streaming response from Gemini
            
        Yields:
            StreamChunk objects
        """
        async for chunk in response:
            yield self.parse_stream_chunk(chunk)
    
    def parse_response(self, response: ChatCompletion) -> ModelResponse:
        """Parse a response from the Gemini API.
        
        Args:
            response: The response from Gemini
            
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
            raise ProviderError(f"Failed to parse Gemini response: {e}")
    
    def parse_stream_chunk(self, chunk: ChatCompletionChunk) -> StreamChunk:
        """Parse a streaming chunk from the Gemini API.
        
        Args:
            chunk: The chunk from Gemini
            
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