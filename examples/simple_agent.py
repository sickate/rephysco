# This file contains the simple agent usage of the Borges LLM System.
# Including:
# - a simple agent that can answer questions and can search the web
# - a simple agent that can answer questions and can use tools
# We use LlamaIndex to build the simple agent.

"""
Example of a simple agent using LlamaIndex's ReActAgent with Google Search integration.

This example demonstrates how to create a ReActAgent that can search the web
and answer questions using the Rephysco LLM wrapper.
"""

import asyncio
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import pendulum
from dotenv import load_dotenv
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.tools.google import GoogleSearchToolSpec

from rephysco import LLMClient, ModelProvider
from rephysco.config import OpenAI as OpenAIConfig, GoogleSearch

# Load environment variables from .env file
load_dotenv()


# Ensure API keys are available
if not GoogleSearch.API_KEY or not GoogleSearch.CSE_ID:
    print("Error: Google API key or CSE ID not found in environment variables.")
    print("Please set GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID in your .env file.")
    sys.exit(1)

if not OpenAIConfig.API_KEY:
    print("Error: OpenAI API key not found in environment variables.")
    print("Please set OPENAI_API_KEY in your .env file.")
    sys.exit(1)


def get_current_time() -> str:
    """Get the current time.
    
    Returns:
        Current time as a string
    """
    now = pendulum.now()
    return f"The current time is {now.format('YYYY-MM-DD HH:mm:ss')}"


class RephyscoReActAgent:
    """A ReAct agent implementation using Rephysco and LlamaIndex.
    
    This agent uses LlamaIndex's ReActAgent with Google Search integration,
    powered by Rephysco's LLM wrapper.
    """
    
    def __init__(
        self,
        provider: ModelProvider = ModelProvider.OPENAI,
        model: Optional[str] = None,
        temperature: float = 0.7,
        verbose: bool = False
    ):
        """Initialize the ReAct agent.
        
        Args:
            provider: The LLM provider to use
            model: The model to use (provider-specific)
            temperature: Temperature for generation
            verbose: Whether to print verbose output
        """
        self.provider = provider
        self.model = model or OpenAIConfig.GPT_4O
        self.temperature = temperature
        self.verbose = verbose
        
        # Create LlamaIndex LLM wrapper
        self.llm = OpenAI(
            model=self.model,
            temperature=self.temperature,
            api_key=OpenAIConfig.API_KEY
        )
        
        # Create tools
        self.tools = self._create_tools()
        
        # Create ReActAgent
        self.agent = ReActAgent.from_tools(
            self.tools,
            llm=self.llm,
            verbose=self.verbose
        )
    
    def _create_tools(self) -> List[BaseTool]:
        """Create tools for the agent.
        
        Returns:
            List of tools
        """
        # Create Google Search tool
        google_search_tool = GoogleSearchToolSpec(
            key=GoogleSearch.API_KEY,
            engine=GoogleSearch.CSE_ID
        )
        
        # Create current time tool
        current_time_tool = FunctionTool.from_defaults(
            fn=get_current_time,
            name="get_current_time",
            description="Get the current time"
        )
        
        return google_search_tool.to_tool_list() + [current_time_tool]
    
    async def run(self, query: str) -> str:
        """Run the agent on a query.
        
        Args:
            query: The query to run
            
        Returns:
            The agent's response
        """
        if self.verbose:
            print(f"Query: {query}")
        
        # Run the agent
        response = await self.agent.aquery(query)
        
        if self.verbose:
            print(f"Response: {response.response}")
            # Sources might not be available in all response types
            if hasattr(response, 'sources') and response.sources:
                print(f"Sources: {response.sources}")
        
        return response.response


async def main():
    """Run the example."""
    # Create the agent
    agent = RephyscoReActAgent(
        provider=ModelProvider.OPENAI,
        model=OpenAIConfig.GPT_4O,
        verbose=True
    )
    
    # Example queries
    queries = [
        "What is the current time?",
        "What is the latest news about artificial intelligence?",
        "Who won the last Super Bowl and what was the score?",
        "What are the key features of Python 3.12?",
    ]
    
    # Run the agent on each query
    for query in queries:
        print("\n" + "=" * 50)
        print(f"Query: {query}")
        print("=" * 50)
        
        response = await agent.run(query)
        
        print("\nResponse:")
        print(response)
        print("=" * 50)
        
        # Wait a bit between queries to avoid rate limits
        await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
