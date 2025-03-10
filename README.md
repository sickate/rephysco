# Rephysco

Rephysco is a lightweight wrapper for interacting with various LLM providers. It provides a unified interface for making API calls to different LLM services, handling infrastructure concerns like caching, retries, and token counting.

## Philosophy

Rephysco is designed with the following principles in mind:

1. **Simplicity**: Provide a minimal, clean interface for LLM interactions
2. **Provider Agnostic**: Support multiple LLM providers with a unified API
3. **Infrastructure Handling**: Manage caching, retries, token counting, and error handling
4. **Conversation Management**: Track multi-turn conversations with different LLMs

## Core Components

- **LLMClient**: Main entry point for LLM interactions
- **Conversation**: Stateful tracker for conversations
- **Provider**: Implementations for various LLM providers (OpenAI, Gemini, etc.)
- **Cache**: Disk-based caching to reduce API calls
- **Retry**: Automatic retries with exponential backoff for transient errors

## Supported Providers

Rephysco supports the following LLM providers:

### OpenAI

- **Models**: GPT-4o, GPT-4o-mini, GPT-o1, GPT-o3-mini
- **Features**: Text generation, multimodal (images), streaming
- **Strengths**: High-quality responses, strong reasoning, multimodal capabilities
- **API Compatibility**: Native OpenAI API

### Gemini

- **Models**: Gemini-1.5-Pro, Gemini-2.0-Flash, Gemini-2.0-Flash-Lite, Gemini-1.5-Flash-8B
- **Features**: Text generation, multimodal (images), streaming
- **Strengths**: Fast responses, good reasoning, multimodal capabilities
- **API Compatibility**: OpenAI-compatible API
- **Limitations**: Multimodal inputs require base64-encoded images

### Aliyun (DashScope)

- **Models**: Qwen-Max, Qwen-Plus, Qwen-Omni
- **Features**: Text generation, multimodal (images, audio, video), streaming
- **Strengths**: Multilingual support, multimodal capabilities
- **API Compatibility**: OpenAI-compatible API
- **Limitations**: Qwen-Omni requires streaming mode

### XAI (Grok)

- **Models**: Grok-2, Grok-2-Vision
- **Features**: Text generation, multimodal (images), streaming
- **Strengths**: Strong reasoning, multimodal capabilities
- **API Compatibility**: OpenAI-compatible API

### SiliconFlow

- **Models**: DeepSeek-V3, DeepSeek-R1, DeepSeek-V3-Pro, DeepSeek-R1-Pro, Qwen2.5-72B-Instruct
- **Features**: Text generation, streaming
- **Strengths**: Strong reasoning, specialized models
- **API Compatibility**: OpenAI-compatible API
- **Limitations**: No multimodal support

## Features

- Unified API across providers
- Caching to reduce API calls
- Retries with exponential backoff
- Token counting
- Error handling
- Streaming support
- Multimodal capabilities (for supported providers)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rephysco.git
cd rephysco

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Configuration

Rephysco can be configured through environment variables or configuration files:

```bash
# Set API keys in environment variables
export OPENAI_API_KEY="your-api-key"
export GOOGLE_API_KEY="your-api-key"
export SILICON_FLOW_API_KEY="your-api-key"
export DASHSCOPE_API_KEY="your-api-key"
export XAI_API_KEY="your-api-key"
```

Or create a `.env` file in your project directory:

```
OPENAI_API_KEY=your-api-key
GOOGLE_API_KEY=your-api-key
SILICON_FLOW_API_KEY=your-api-key
DASHSCOPE_API_KEY=your-api-key
XAI_API_KEY=your-api-key
```

## Usage

### Basic Usage

```python
import asyncio
from rephysco import LLMClient, ModelProvider

async def main():
    # Create a client
    client = LLMClient(provider=ModelProvider.OPENAI)
    
    # Generate a response
    response = await client.generate(
        prompt="Explain quantum computing in simple terms."
    )
    
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

### Conversation Example

```python
import asyncio
from rephysco import LLMClient, ModelProvider

async def main():
    # Create a client
    client = LLMClient(provider=ModelProvider.OPENAI)
    
    # Start a conversation with a system prompt
    system_prompt = "You are a helpful assistant that provides concise answers."
    
    # First message
    response = await client.chat(
        message="What are the three laws of thermodynamics?",
        system_prompt=system_prompt
    )
    print(f"Assistant: {response}")
    
    # Follow-up question
    response = await client.chat(message="Which one is most relevant to refrigerators?")
    print(f"Assistant: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Streaming Responses

```python
import asyncio
from rephysco import LLMClient, ModelProvider

async def main():
    # Create a client
    client = LLMClient(provider=ModelProvider.OPENAI)
    
    # Generate a streaming response
    response_stream = await client.generate(
        prompt="Write a short poem about artificial intelligence.",
        stream=True
    )
    
    print("Assistant: ", end="")
    async for chunk in response_stream:
        print(chunk, end="")
    print()  # Add a newline at the end

if __name__ == "__main__":
    asyncio.run(main())
```

### Multimodal Input

```python
import asyncio
from rephysco import LLMClient, ModelProvider

async def main():
    # Create a client with a multimodal-capable provider
    client = LLMClient(provider=ModelProvider.OPENAI, model="gpt-4o")
    
    # Generate a response with an image
    response = await client.chat(
        message="What's in this image?",
        images=["https://example.com/image.jpg"]
    )
    
    print(f"Assistant: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Using Different Providers

```python
import asyncio
from rephysco import LLMClient, ModelProvider
from rephysco.config import Gemini, SiliconFlow, XAI, Aliyun

async def main():
    # OpenAI
    openai_client = LLMClient(provider=ModelProvider.OPENAI)
    
    # Gemini
    gemini_client = LLMClient(
        provider=ModelProvider.GEMINI,
        model=Gemini.GEMINI_1_5_PRO
    )
    
    # Aliyun
    aliyun_client = LLMClient(
        provider=ModelProvider.ALIYUN,
        model=Aliyun.QWEN_MAX
    )
    
    # XAI (Grok)
    xai_client = LLMClient(
        provider=ModelProvider.XAI,
        model=XAI.GROK_2
    )
    
    # SiliconFlow
    siliconflow_client = LLMClient(
        provider=ModelProvider.SILICON_FLOW,
        model=SiliconFlow.DEEPSEEK_V3
    )
    
    # Use any client with the same API
    response = await openai_client.generate(
        prompt="Explain quantum computing in simple terms."
    )
    print(f"OpenAI: {response[:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
```

### Caching and Retries

Rephysco includes built-in caching and retry functionality:

```python
import asyncio
from rephysco import LLMClient, ModelProvider, Config

async def main():
    # Configure global settings
    Config.update(
        cache_ttl=3600,  # Cache responses for 1 hour
        max_retries=5,   # Retry up to 5 times
        base_delay=1.0,  # Start with 1 second delay
        backoff_factor=2.0  # Double the delay after each retry
    )
    
    # Create a client with caching and retries
    client = LLMClient(
        provider=ModelProvider.OPENAI,
        enable_cache=True,
        enable_retries=True
    )
    
    # Generate a response (will be cached)
    response1 = await client.generate(
        prompt="Explain quantum computing in simple terms."
    )
    
    # This will use the cached response
    response2 = await client.generate(
        prompt="Explain quantum computing in simple terms."
    )
    
    print(f"Response: {response1[:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
```

## Running the CLI

Rephysco includes a command-line interface for interacting with LLMs. You can run it in two ways:

```bash
# Run the generate command
python -m rephysco generate --provider openai "Explain quantum computing in simple terms."

# Start an interactive chat session
python -m rephysco chat --provider openai --system "You are a helpful assistant."
```

### CLI Commands

```bash
# Generate a response
python -m rephysco generate --provider openai --model gpt-4o --temperature 0.7 "Explain quantum computing in simple terms."

# Start a chat session
python -m rephysco chat --provider openai --model gpt-4o --temperature 0.7 --system "You are a helpful assistant."
```

### Chat Mode Commands

In chat mode, you can use the following commands:

- `exit` or `quit`: End the session
- `system: <prompt>`: Set a new system prompt
- `temp: <value>`: Change the temperature
- `clear`: Clear the conversation history

## Architecture

Rephysco is designed as a lightweight wrapper around LLM providers:

1. **LLMClient**: Handles provider-specific details, authentication, and basic API calls
2. **Conversation**: Manages multi-turn conversations with methods for sending messages and retrieving history
3. **Provider Interface**: Defines a common interface that all providers implement
4. **Provider Implementations**: Provider-specific code that handles the details of each API
5. **Cache**: Disk-based caching to reduce redundant API calls
6. **Retry**: Automatic retries with exponential backoff for transient errors

The architecture leverages the fact that most providers have OpenAI-compatible APIs, allowing for a common base implementation with provider-specific overrides only where necessary.

## Development

To add a new provider:

1. Add the provider to the `ModelProvider` enum in `types.py`
2. Create a new provider implementation in the `providers` directory
3. Update the factory function in `factory.py` to create instances of the new provider

## Dependencies

- `openai`: Official OpenAI Python client
- `pendulum`: Modern Python datetime library
- `diskcache`: Disk-based cache implementation
- `rich`: Rich text and formatting in the terminal
- `click`: Command-line interface creation kit
- `aiohttp`: Asynchronous HTTP client/server framework
- `pydantic`: Data validation and settings management