# Agent Implementation

Rephysco includes a ReAct agent implementation with Google Search and other tools.

## Overview

The agent implementation uses LlamaIndex's ReActAgent to provide a conversational interface with access to tools like Google Search. This allows the agent to search for information and perform other actions to answer user queries.

## Usage

### Using the CLI

```bash
# Start an agent session with OpenAI
python -m rephysco agent -p openai

# Start an agent session with Aliyun
python -m rephysco agent -p aliyun -m qwen-omni-turbo
```

### Using the Agent Directly

```python
from rephysco.examples.agent_session import run_agent_session
from rephysco.types import ModelProvider

# Run an interactive agent session
run_agent_session(
    provider=ModelProvider.OPENAI.value,
    model=None,  # Use default model
    temperature=0.7,
    verbose=True
)
```

### Using the Standalone Script

```bash
# Run the agent with OpenAI
python -m rephysco.examples.run_agent -p openai

# Run the agent with custom settings
python -m rephysco.examples.run_agent -p aliyun -m qwen-omni-turbo -t 0.8 -v
```

## Available Tools

The agent has access to the following tools:

- **Google Search**: Search the web for information
- **Current Time**: Get the current date and time

## Implementation Details

The agent implementation is located in `rephysco/examples/agent_session.py`. It uses:

- **LLMFactory**: To create the appropriate LLM for the specified provider
- **ReActAgent**: From LlamaIndex to implement the agent's reasoning and action capabilities
- **Rich**: For terminal output formatting and streaming display

## Streaming Output

The agent uses Rich's `Live` display to show streaming output in real-time. This provides a smooth, interactive experience when using the agent.

## Extending the Agent

To add new tools to the agent:

1. Create a new tool using LlamaIndex's `FunctionTool` or another tool implementation
2. Add the tool to the list of tools in the `run_agent_session` function