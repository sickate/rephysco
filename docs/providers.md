# Provider Documentation

This document provides detailed information about the different LLM providers supported by Rephysco, including their capabilities, limitations, and usage patterns.

## Overview

Rephysco supports the following LLM providers:

1. [OpenAI](#openai)
2. [Gemini](#gemini)
3. [Aliyun (DashScope)](#aliyun-dashscope)
4. [XAI (Grok)](#xai-grok)
5. [SiliconFlow](#siliconflow)

Each provider has its own strengths, limitations, and specific features. This document will help you choose the right provider for your use case.

## OpenAI

### Models

- **GPT-4o**: Latest multimodal model with strong reasoning and vision capabilities
- **GPT-4o-mini**: Smaller, faster version of GPT-4o
- **GPT-o1**: Advanced reasoning model
- **GPT-o3-mini**: Smaller, faster version of GPT-o1

### Features

- **Text Generation**: High-quality text generation with strong reasoning
- **Multimodal**: Can process and understand images
- **Streaming**: Supports streaming responses for real-time interaction
- **Function Calling**: Supports structured function calling

### API Details

OpenAI uses its native API format. Rephysco uses the official OpenAI Python SDK to interact with the API.

### Usage Example

```python
from rephsyco import LLMClient, ModelProvider
from rephsyco.config import OpenAI

client = LLMClient(
    provider=ModelProvider.OPENAI,
    model=OpenAI.GPT_4O
)

# Text generation
response = await client.generate(
    prompt="Explain quantum computing in simple terms."
)

# Multimodal
response = await client.chat(
    message="What's in this image?",
    images=["https://example.com/image.jpg"]
)

# Streaming
stream = await client.generate(
    prompt="Write a poem about AI.",
    stream=True
)
async for chunk in stream:
    print(chunk, end="")
```

## Gemini

### Models

- **GEMINI_1_5_PRO**: Advanced reasoning and multimodal model
- **GEMINI_2_0_FLASH**: Fast and accurate text generation
- **GEMINI_2_0_FLASH_LITE**: Faster and cheaper version of FLASH
- **GEMINI_1_5_FLASH_8B**: Fastest and cheapest model

### Features

- **Text Generation**: High-quality text generation with good reasoning
- **Multimodal**: GEMINI_1_5_PRO supports image understanding
- **Streaming**: Supports streaming responses
- **Function Calling**: Limited support for function calling

### Limitations

- Multimodal inputs may require specific formatting
- Some models may have more limited context windows

### API Details

Gemini provides an OpenAI-compatible API endpoint. Rephysco uses the OpenAI SDK to interact with this endpoint.

### Usage Example

```python
from rephsyco import LLMClient, ModelProvider
from rephsyco.config import Gemini

client = LLMClient(
    provider=ModelProvider.GEMINI,
    model=Gemini.GEMINI_1_5_PRO
)

# Text generation
response = await client.generate(
    prompt="Explain quantum computing in simple terms."
)

# Multimodal (only with GEMINI_1_5_PRO)
response = await client.chat(
    message="What's in this image?",
    images=["https://example.com/image.jpg"]
)
```

## Aliyun (DashScope)

### Models

- **QWEN_MAX**: Most powerful text generation model
- **QWEN_PLUS**: Balanced text generation model
- **QWEN_OMNI**: Multimodal model supporting text, image, audio, and video

### Features

- **Text Generation**: High-quality text generation with multilingual support
- **Multimodal**: QWEN_OMNI supports images, audio, and video
- **Streaming**: Supports streaming responses (required for QWEN_OMNI)

### Limitations

- QWEN_OMNI requires streaming mode
- Multimodal inputs must be provided as URLs
- May have different rate limits than other providers

### API Details

Aliyun provides an OpenAI-compatible API endpoint. Rephysco uses the OpenAI SDK to interact with this endpoint.

### Usage Example

```python
from rephsyco import LLMClient, ModelProvider
from rephsyco.config import Aliyun

# Text generation with Qwen-Max
client = LLMClient(
    provider=ModelProvider.ALIYUN,
    model=Aliyun.QWEN_MAX
)
response = await client.generate(
    prompt="Explain quantum computing in simple terms."
)

# Multimodal with Qwen-Omni (requires streaming)
client = LLMClient(
    provider=ModelProvider.ALIYUN,
    model=Aliyun.QWEN_OMNI
)
stream = await client.chat(
    message="What's in this image?",
    images=["https://example.com/image.jpg"],
    stream=True  # Required for QWEN_OMNI
)
async for chunk in stream:
    print(chunk, end="")
```

## XAI (Grok)

### Models

- **GROK_2**: Advanced text generation model
- **GROK_2_VISION**: Multimodal model with vision capabilities

### Features

- **Text Generation**: High-quality text generation with strong reasoning
- **Multimodal**: GROK_2_VISION supports image understanding
- **Streaming**: Supports streaming responses

### API Details

XAI provides an OpenAI-compatible API endpoint. Rephysco uses the OpenAI SDK to interact with this endpoint.

### Usage Example

```python
from rephsyco import LLMClient, ModelProvider
from rephsyco.config import XAI

# Text generation with Grok-2
client = LLMClient(
    provider=ModelProvider.XAI,
    model=XAI.GROK_2
)
response = await client.generate(
    prompt="Explain quantum computing in simple terms."
)

# Multimodal with Grok-2-Vision
client = LLMClient(
    provider=ModelProvider.XAI,
    model=XAI.GROK_2_VISION
)
response = await client.chat(
    message="What's in this image?",
    images=["https://example.com/image.jpg"]
)
```

## SiliconFlow

### Models

- **DEEPSEEK_V3**: Advanced text generation model
- **DEEPSEEK_R1**: Specialized reasoning model
- **DEEPSEEK_V3_PRO**: Premium version of DEEPSEEK_V3
- **DEEPSEEK_R1_PRO**: Premium version of DEEPSEEK_R1
- **QWEN2_5_72B_INSTRUCT**: Large instruction-tuned model

### Features

- **Text Generation**: High-quality text generation with strong reasoning
- **Streaming**: Supports streaming responses

### Limitations

- No multimodal support
- Focused on text-only interactions

### API Details

SiliconFlow provides an OpenAI-compatible API endpoint. Rephysco uses the OpenAI SDK to interact with this endpoint.

### Usage Example

```python
from rephsyco import LLMClient, ModelProvider
from rephsyco.config import SiliconFlow

client = LLMClient(
    provider=ModelProvider.SILICON_FLOW,
    model=SiliconFlow.DEEPSEEK_V3
)

# Text generation
response = await client.generate(
    prompt="Explain quantum computing in simple terms."
)

# Streaming
stream = await client.generate(
    prompt="Write a poem about AI.",
    stream=True
)
async for chunk in stream:
    print(chunk, end="")
```

## Provider Comparison

| Provider    | Text Quality | Reasoning | Multimodal | Streaming | Function Calling | Cost |
|-------------|--------------|-----------|------------|-----------|------------------|------|
| OpenAI      | Excellent    | Excellent | Yes        | Yes       | Yes              | High |
| Gemini      | Very Good    | Very Good | Limited    | Yes       | Limited          | Medium |
| Aliyun      | Very Good    | Good      | Yes        | Yes       | Limited          | Medium |
| XAI         | Very Good    | Very Good | Limited    | Yes       | Limited          | Medium |
| SiliconFlow | Very Good    | Excellent | No         | Yes       | Limited          | Low |

## Choosing the Right Provider

When choosing a provider, consider the following factors:

1. **Task Requirements**: What kind of tasks do you need to perform? Text generation, reasoning, multimodal understanding?
2. **Performance**: Which provider offers the best performance for your specific use case?
3. **Cost**: What is your budget? Some providers are more cost-effective than others.
4. **API Limits**: What are the rate limits and quotas for each provider?
5. **Special Features**: Do you need specific features like function calling or multimodal capabilities?

## Best Practices

1. **Start with OpenAI**: If you're new to LLMs, start with OpenAI as it has the most comprehensive features and documentation.
2. **Test Multiple Providers**: Different providers excel at different tasks. Test multiple providers to find the best fit for your use case.
3. **Use Streaming**: For real-time applications, use streaming responses to improve user experience.
4. **Handle Errors**: Implement proper error handling to deal with API errors, rate limits, and other issues.
5. **Monitor Usage**: Keep track of your API usage to avoid unexpected costs. 