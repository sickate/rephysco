"""Unified client for interacting with LLM providers.

This module provides a high-level client for interacting with various LLM providers
through a consistent interface, handling common functionality like API key management,
default model selection, and conversation history.
"""

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import pendulum

from .cache import LLMCache
from .errors import InvalidRequestError, RateLimitError
from .conversation import Conversation
from .providers import create_provider
from .retry import async_retry
from .settings import Config
from .types import Message, ModelProvider, ModelResponse, Role, StreamChunk


class LLMClient:
    """Main client for interacting with LLM providers.
    
    This class provides a unified interface for sending messages to different LLM
    providers, handling conversation history, and managing provider-specific details.
    """
    
    def __init__(
        self,
        provider: ModelProvider,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        enable_cache: bool = None,
        enable_retries: bool = None,
        **kwargs
    ):
        """Initialize the LLM client.
        
        Args:
            provider: The LLM provider to use
            model: The model to use (provider-specific)
            api_key: API key for the provider
            enable_cache: Whether to enable caching (defaults to Config.enable_caching)
            enable_retries: Whether to enable retries (defaults to Config.enable_retries)
            **kwargs: Additional provider-specific arguments
        """
        self.provider_type = provider
        self.provider = create_provider(provider, api_key=api_key, **kwargs)
        self.model = model or self.provider.default_model
        self.conversation = Conversation()
        
        # Initialize cache if enabled
        self.enable_cache = Config.enable_caching if enable_cache is None else enable_cache
        self.cache = LLMCache(Config.cache_dir, Config.cache_ttl) if self.enable_cache else None
        
        # Set retry configuration
        self.enable_retries = Config.enable_retries if enable_retries is None else enable_retries
    
    @async_retry(
        max_retries=Config.max_retries,
        base_delay=Config.base_delay,
        max_delay=Config.max_delay,
        backoff_factor=Config.backoff_factor,
        retryable_exceptions=[RateLimitError]
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate a response to a prompt.
        
        Args:
            prompt: The prompt to generate a response for
            system_prompt: Optional system prompt to set the context
            model: The model to use (overrides the client's model)
            temperature: Temperature for generation
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Either a string response or an AsyncGenerator of string chunks
        """
        messages = []
        
        if system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=system_prompt))
            
        messages.append(Message(role=Role.USER, content=prompt))
        
        # Check cache for non-streaming requests
        if self.enable_cache and not stream:
            cache_key = {
                "provider": self.provider_type.value,
                "model": model or self.model,
                "messages": [msg.to_dict() for msg in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
            
            cached_response = self.cache.get(cache_key)
            if cached_response:
                return cached_response
        
        response = await self.provider.generate(
            messages=messages,
            model=model or self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs
        )
        
        if stream:
            async def stream_content():
                async for chunk in response:
                    yield chunk.content
            
            return stream_content()
        else:
            content = response.content
            
            # Cache the response
            if self.enable_cache:
                self.cache.set(cache_key, content)
                
            return content
    
    @async_retry(
        max_retries=Config.max_retries,
        base_delay=Config.base_delay,
        max_delay=Config.max_delay,
        backoff_factor=Config.backoff_factor,
        retryable_exceptions=[RateLimitError]
    )
    async def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        images: Optional[List[str]] = None,
        clear_history: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Send a message in a conversation and get a response.
        
        Args:
            message: The message to send
            system_prompt: Optional system prompt to set the context
            model: The model to use (overrides the client's model)
            temperature: Temperature for generation
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            images: Optional list of image URLs or base64-encoded images
            clear_history: Whether to clear the conversation history
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Either a string response or an AsyncGenerator of string chunks
        """
        if clear_history:
            self.conversation.clear()
            
        if system_prompt and (not self.conversation.has_system_prompt() or clear_history):
            self.conversation.set_system_prompt(system_prompt)
            
        self.conversation.add_user_message(message, images=images)
        
        # Check cache for non-streaming requests
        if self.enable_cache and not stream:
            cache_key = {
                "provider": self.provider_type.value,
                "model": model or self.model,
                "messages": [msg.to_dict() for msg in self.conversation.messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
            
            cached_response = self.cache.get(cache_key)
            if cached_response:
                self.conversation.add_assistant_message(cached_response)
                return cached_response
        
        response = await self.provider.generate(
            messages=self.conversation.messages,
            model=model or self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs
        )
        
        if stream:
            content_buffer = []
            
            async def stream_content():
                async for chunk in response:
                    content_buffer.append(chunk.content)
                    yield chunk.content
                
                # Add the complete message to the conversation history
                full_response = "".join(content_buffer)
                self.conversation.add_assistant_message(full_response)
            
            return stream_content()
        else:
            content = response.content
            self.conversation.add_assistant_message(content)
            
            # Cache the response
            if self.enable_cache:
                self.cache.set(cache_key, content)
                
            return content
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation.clear()
    
    def get_history(self) -> List[Message]:
        """Get the conversation history.
        
        Returns:
            The list of messages in the conversation
        """
        return self.conversation.messages
    
    def close(self) -> None:
        """Close the client and release resources."""
        if self.cache:
            self.cache.close()
