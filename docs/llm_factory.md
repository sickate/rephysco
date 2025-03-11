# LLM Factory

The `LLMFactory` class provides a centralized way to create LLM instances for different providers with appropriate configurations.

## Overview

The LLM Factory implements the Factory pattern to create LLM instances with provider-specific and model-specific configurations. This simplifies the process of creating LLMs and ensures that all necessary settings are applied correctly.

## Usage

```python
from rephysco.llm_factory import LLMFactory
from rephysco.types import ModelProvider

# Create an LLM with default settings
llm = LLMFactory.create_llm(provider=ModelProvider.OPENAI.value)

# Create an LLM with custom settings
llm = LLMFactory.create_llm(
    provider=ModelProvider.ALIYUN.value,
    model="qwen-omni-turbo",
    temperature=0.8,
    max_tokens=2000
)
```

## Default Models

Each provider has a default model that will be used if no model is specified:

| Provider | Default Model |
|----------|---------------|
| OpenAI | GPT-4o |
| Aliyun | Qwen Omni |
| SiliconFlow | DeepSeek V2 |
| XAI | Claude 3 Opus |

## Provider-Specific Configuration

The factory automatically configures each LLM with the appropriate settings for its provider:

- **API Keys**: Loaded from environment variables or configuration
- **Base URLs**: Set to the correct endpoints
- **Streaming**: Enabled by default for all models
- **Context Windows**: Set to appropriate values for each model

## Implementation Details

The factory uses a set of private methods to create LLMs for each provider:

- `_create_openai_llm`: Creates an OpenAI LLM
- `_create_aliyun_llm`: Creates an Aliyun LLM
- `_create_silicon_flow_llm`: Creates a SiliconFlow LLM
- `_create_xai_llm`: Creates an XAI LLM

Each method applies the appropriate configuration for its provider and handles any provider-specific requirements.

## Extending the Factory

To add support for a new provider:

1. Add a new private method to create LLMs for the provider
2. Add the provider to the `DEFAULT_MODELS` dictionary
3. Update the `create_llm` method to call the new method for the provider