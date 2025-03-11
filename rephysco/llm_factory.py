from typing import Dict, Any, Optional

from llama_index.llms.openai import OpenAI as OpenAILLM
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import LLM

from rephysco.config import OpenAI, Aliyun, SiliconFlow, XAI
from rephysco.types import ModelProvider


class LLMFactory:
    """Factory for creating LlamaIndex LLMs based on provider and model."""
    
    # Default models for each provider
    DEFAULT_MODELS = {
        ModelProvider.OPENAI.value: OpenAI.GPT_4O,
        ModelProvider.ALIYUN.value: Aliyun.QWEN_OMNI,
        ModelProvider.SILICON_FLOW.value: SiliconFlow.DEEPSEEK_V3,
        ModelProvider.XAI.value: XAI.GROK_2_VISION,
    }

    @staticmethod
    def create_llm(
        provider: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ) -> LLM:
        """
        Create a LlamaIndex LLM based on provider and model.
        
        Args:
            provider: The model provider (from ModelProvider enum)
            model: The specific model name (if None, uses default for provider)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            api_key: API key (if not using default from config)
            base_url: Base URL (if not using default from config)
            **kwargs: Additional arguments to pass to the LLM constructor
            
        Returns:
            A configured LlamaIndex LLM instance
        """
        # Use default model if none provided
        if model is None:
            if provider in LLMFactory.DEFAULT_MODELS:
                model = LLMFactory.DEFAULT_MODELS[provider]
            else:
                raise ValueError(f"No default model available for provider: {provider}")
        
        # Set default values for provider-specific configurations
        if provider == ModelProvider.OPENAI.value:
            return LLMFactory._create_openai_llm(model, temperature, max_tokens, **kwargs)
        elif provider == ModelProvider.ALIYUN.value:
            if api_key is None:
                api_key = Aliyun.API_KEY
            if base_url is None:
                base_url = Aliyun.BASE_URL
            return LLMFactory._create_aliyun_llm(model, temperature, max_tokens, api_key, base_url, **kwargs)
        elif provider == ModelProvider.SILICON_FLOW.value:
            if api_key is None:
                api_key = SiliconFlow.API_KEY
            if base_url is None:
                base_url = SiliconFlow.BASE_URL
            return LLMFactory._create_silicon_flow_llm(model, temperature, max_tokens, api_key, base_url, **kwargs)
        elif provider == ModelProvider.XAI.value:
            if api_key is None:
                api_key = XAI.API_KEY
            if base_url is None:
                base_url = XAI.BASE_URL
            return LLMFactory._create_xai_llm(model, temperature, max_tokens, api_key, base_url, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def _create_openai_llm(
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> OpenAILLM:
        """Create an OpenAI LLM."""
        # Default configuration
        config = {
            "model": model,
            "temperature": temperature,
        }
        
        # Add max_tokens if provided
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
            
        # Handle model-specific requirements
        if model in [OpenAI.GPT_4O, OpenAI.GPT_4O_MINI]:
            # These models work well with streaming
            if "additional_kwargs" not in config:
                config["additional_kwargs"] = {}
            config["additional_kwargs"]["stream"] = True
            
        # Override with any provided kwargs
        config.update(kwargs)
        
        return OpenAILLM(**config)

    @staticmethod
    def _create_aliyun_llm(
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        api_key: str,
        base_url: str,
        **kwargs
    ) -> OpenAILike:
        """Create an Aliyun LLM using OpenAILike."""
        # Default configuration
        config = {
            "model": model,
            "temperature": temperature,
            "api_key": api_key,
            "api_base": base_url,
            "is_chat_model": True,
            "context_window": 8192,  # Reasonable default for most models
        }
        
        # Add max_tokens if provided
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        else:
            config["max_tokens"] = 1024  # Reasonable default
            
        # Enable streaming for all Aliyun models by default
        if "additional_kwargs" not in config:
            config["additional_kwargs"] = {}
        config["additional_kwargs"]["stream"] = True
            
        # Override with any provided kwargs
        config.update(kwargs)
        
        return OpenAILike(**config)

    @staticmethod
    def _create_silicon_flow_llm(
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        api_key: str,
        base_url: str,
        **kwargs
    ) -> OpenAILike:
        """Create a SiliconFlow LLM using OpenAILike."""
        # Default configuration
        config = {
            "model": model,
            "temperature": temperature,
            "api_key": api_key,
            "api_base": base_url,
            "is_chat_model": True,
            "context_window": 8192,  # Reasonable default
        }
        
        # Add max_tokens if provided
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        else:
            config["max_tokens"] = 1024  # Reasonable default
            
        # Enable streaming by default
        if "additional_kwargs" not in config:
            config["additional_kwargs"] = {}
        config["additional_kwargs"]["stream"] = True
            
        # Override with any provided kwargs
        config.update(kwargs)
        
        return OpenAILike(**config)

    @staticmethod
    def _create_xai_llm(
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        api_key: str,
        base_url: str,
        **kwargs
    ) -> OpenAILike:
        """Create an XAI LLM using OpenAILike."""
        # Default configuration
        config = {
            "model": model,
            "temperature": temperature,
            "api_key": api_key,
            "api_base": base_url,
            "is_chat_model": True,
            "context_window": 8192,  # Reasonable default
        }
        
        # Add max_tokens if provided
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        else:
            config["max_tokens"] = 1024  # Reasonable default
            
        # Enable streaming by default
        if "additional_kwargs" not in config:
            config["additional_kwargs"] = {}
        config["additional_kwargs"]["stream"] = True
            
        # Override with any provided kwargs
        config.update(kwargs)
        
        return OpenAILike(**config) 