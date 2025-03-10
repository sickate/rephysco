"""Factory function for creating provider instances.

This module provides a factory function for creating provider instances based on
the requested provider type.
"""

from typing import Optional

from ..types import ModelProvider
from .base import BaseProvider


def create_provider(
    provider_type: ModelProvider,
    api_key: Optional[str] = None,
    **kwargs
) -> BaseProvider:
    """Create a provider instance based on the provider type.
    
    Args:
        provider_type: Type of provider to create
        api_key: API key for the provider
        **kwargs: Additional provider-specific arguments
        
    Returns:
        An instance of the specified provider
        
    Raises:
        ValueError: If the provider type is not supported
        ImportError: If the provider module cannot be imported
    """
    if provider_type == ModelProvider.OPENAI:
        from .openai import OpenAIProvider
        return OpenAIProvider(api_key=api_key, **kwargs)
    elif provider_type == ModelProvider.ALIYUN:
        from .aliyun import AliyunProvider
        return AliyunProvider(api_key=api_key, **kwargs)
    elif provider_type == ModelProvider.XAI:
        from .xai import XAIProvider
        return XAIProvider(api_key=api_key, **kwargs)
    elif provider_type == ModelProvider.GEMINI:
        from .gemini import GeminiProvider
        return GeminiProvider(api_key=api_key, **kwargs)
    elif provider_type == ModelProvider.SILICON_FLOW:
        from .siliconflow import SiliconFlowProvider
        return SiliconFlowProvider(api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}") 