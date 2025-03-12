from typing import Dict, Any, Optional

from llama_index.llms.openai import OpenAI as OpenAILLM
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import LLM

from rephysco.config import OpenAI, Aliyun, SiliconFlow, XAI, Gemini
from rephysco.types import ModelProvider


class LLMFactory:
    """Factory for creating LlamaIndex LLMs based on provider and model."""
    
    # Default models for each provider
    DEFAULT_MODELS = {
        ModelProvider.OPENAI.value: OpenAI.GPT_4O,
        ModelProvider.ALIYUN.value: Aliyun.QWEN_OMNI,
        ModelProvider.SILICON_FLOW.value: SiliconFlow.DEEPSEEK_V3,
        ModelProvider.XAI.value: XAI.GROK_2_VISION,
        ModelProvider.GEMINI.value: Gemini.GEMINI_2_0_FLASH,
    }
    
    # Provider-specific API keys and base URLs
    PROVIDER_CONFIGS = {
        ModelProvider.ALIYUN.value: {
            "api_key": Aliyun.API_KEY,
            "base_url": Aliyun.BASE_URL,
        },
        ModelProvider.SILICON_FLOW.value: {
            "api_key": SiliconFlow.API_KEY,
            "base_url": SiliconFlow.BASE_URL,
        },
        ModelProvider.XAI.value: {
            "api_key": XAI.API_KEY,
            "base_url": XAI.BASE_URL,
        },
        ModelProvider.GEMINI.value: {
            "api_key": Gemini.API_KEY,
            "base_url": Gemini.BASE_URL,
        },
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
        
        # Get provider-specific API key and base URL if not provided
        if provider in LLMFactory.PROVIDER_CONFIGS:
            if api_key is None:
                api_key = LLMFactory.PROVIDER_CONFIGS[provider]["api_key"]
            if base_url is None:
                base_url = LLMFactory.PROVIDER_CONFIGS[provider]["base_url"]
        
        # Create the LLM based on the provider
        if provider == ModelProvider.OPENAI.value:
            return LLMFactory._create_openai_llm(model, temperature, max_tokens, **kwargs)
        elif provider in [ModelProvider.ALIYUN.value, ModelProvider.SILICON_FLOW.value, ModelProvider.XAI.value, ModelProvider.GEMINI.value]:
            return LLMFactory._create_openai_like_llm(
                provider=provider,
                model=model, 
                temperature=temperature, 
                max_tokens=max_tokens, 
                api_key=api_key, 
                base_url=base_url, 
                **kwargs
            )
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
            
        # Handle streaming if requested
        if 'stream' in kwargs and kwargs['stream'] == True:
            if "additional_kwargs" not in config:
                config["additional_kwargs"] = {}
            config["additional_kwargs"]["stream"] = True
                   
        # Override with any provided kwargs
        config.update(kwargs)
        
        return OpenAILLM(**config)

    @staticmethod
    def _create_openai_like_llm(
        provider: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        api_key: str,
        base_url: str,
        **kwargs
    ) -> OpenAILike:
        """Create an OpenAI-like LLM for providers using compatible APIs."""
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

        # Handle streaming if requested
        if 'stream' in kwargs and kwargs['stream'] == True:
            if "additional_kwargs" not in config:
                config["additional_kwargs"] = {}
            config["additional_kwargs"]["stream"] = True
    
        # Provider-specific configurations
        if provider == ModelProvider.GEMINI.value:
            # Gemini-specific configurations if needed
            pass
        elif provider == ModelProvider.ALIYUN.value:
            # Aliyun-specific configurations if needed
            pass
        elif provider == ModelProvider.SILICON_FLOW.value:
            # SiliconFlow-specific configurations if needed
            pass
        elif provider == ModelProvider.XAI.value:
            # XAI-specific configurations if needed
            pass
            
        # Override with any provided kwargs
        config.update(kwargs)
        
        return OpenAILike(**config) 