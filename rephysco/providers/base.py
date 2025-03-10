"""
Base provider class for the Rephysco LLM System.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from ..types import Message, ModelResponse, StreamChunk


class BaseProvider(ABC):
    """Base class for all LLM providers."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the provider.
        
        Args:
            api_key: API key for the provider
            **kwargs: Additional provider-specific arguments
        """
        self.api_key = api_key
        self.kwargs = kwargs
    
    @property
    @abstractmethod
    def default_model(self) -> str:
        """Get the default model for this provider."""
        pass
    
    @abstractmethod
    def format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for the provider's API.
        
        Args:
            messages: List of messages to format
            
        Returns:
            Formatted messages ready for the provider's API
        """
        pass
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[ModelResponse, AsyncGenerator[StreamChunk, None]]:
        """Generate a response from the provider.
        
        Args:
            messages: List of messages to send to the provider
            model: Model to use for generation
            temperature: Temperature for generation
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Either a ModelResponse or an AsyncGenerator of StreamChunks
        """
        pass
    
    def validate_api_key(self) -> bool:
        """Validate that the API key is present.
        
        Returns:
            True if the API key is present, False otherwise
        """
        return self.api_key is not None and len(self.api_key) > 0
