# This file contains the simple agent usage of the Borges LLM System.
# Including:
# - a simple agent that can answer questions and can search the web
# - a simple agent that can answer questions and can use tools
# We use LlamaIndex to build the simple agent.

"""Simple agent example using the Rephysco LLM System.

This example demonstrates how to create a simple agent that can use tools to answer questions.
"""

import asyncio
import json
import os
from typing import Any, Callable, Dict, List, Optional

import pendulum
import requests
from rich.console import Console
from rich.markdown import Markdown

from rephsyco import LLMClient, Message, ModelProvider, Role


# Define some simple tools
def search_web(query: str) -> str:
    """Search the web for information.
    
    Args:
        query: The search query
        
    Returns:
        Search results as a string
    """
    # This is a mock implementation - in a real application, you would use a search API
    return f"Results for '{query}': Found information about {query} from multiple sources."


def get_current_weather(location: str) -> str:
    """Get the current weather for a location.
    
    Args:
        location: The location to get weather for
        
    Returns:
        Weather information as a string
    """
    # This is a mock implementation - in a real application, you would use a weather API
    return f"Weather in {location}: 72°F, Partly Cloudy, Humidity: 65%"


def get_current_time() -> str:
    """Get the current time.
    
    Returns:
        Current time as a string
    """
    now = pendulum.now()
    return f"The current time is {now.format('YYYY-MM-DD HH:mm:ss')}"


# Define the available tools
TOOLS = {
    "search_web": {
        "description": "Search the web for information",
        "parameters": {
            "query": {
                "type": "string",
                "description": "The search query"
            }
        },
        "function": search_web
    },
    "get_current_weather": {
        "description": "Get the current weather for a location",
        "parameters": {
            "location": {
                "type": "string",
                "description": "The location to get weather for"
            }
        },
        "function": get_current_weather
    },
    "get_current_time": {
        "description": "Get the current time",
        "parameters": {},
        "function": get_current_time
    }
}


class SimpleAgent:
    """A simple agent that can use tools to answer questions."""
    
    def __init__(self, provider: ModelProvider = ModelProvider.OPENAI):
        """Initialize the agent.
        
        Args:
            provider: The LLM provider to use
        """
        self.client = LLMClient(provider=provider)
        self.tools = TOOLS
        self.console = Console()
    
    async def run(self, query: str) -> str:
        """Run the agent on a query.
        
        Args:
            query: The user query
            
        Returns:
            The agent's response
        """
        # System prompt that instructs the model to use tools
        system_prompt = """You are a helpful assistant with access to the following tools:

{tools}

To use a tool, respond with a JSON object in the following format:
```json
{{
    "tool": "tool_name",
    "parameters": {{
        "param1": "value1",
        "param2": "value2"
    }}
}}
```

If you don't need to use a tool, just respond normally.
"""
        
        # Format the tools for the system prompt
        tools_description = ""
        for name, tool in self.tools.items():
            params = ", ".join([f"{param}: {details['description']}" 
                               for param, details in tool["parameters"].items()])
            tools_description += f"- {name}: {tool['description']}"
            if params:
                tools_description += f" (Parameters: {params})"
            tools_description += "\n"
        
        formatted_system_prompt = system_prompt.format(tools=tools_description)
        
        # Start the conversation
        self.console.print(f"[bold green]User:[/bold green] {query}")
        
        # First response from the assistant
        response = await self.client.chat(
            message=query,
            system_prompt=formatted_system_prompt
        )
        
        # Check if the response is a tool call
        tool_call = self._parse_tool_call(response)
        
        if tool_call:
            # Execute the tool
            tool_name = tool_call["tool"]
            parameters = tool_call["parameters"]
            
            self.console.print(f"[bold yellow]Agent is using tool:[/bold yellow] {tool_name}")
            
            # Call the tool function
            tool_function = self.tools[tool_name]["function"]
            tool_result = tool_function(**parameters)
            
            # Send the tool result back to the assistant
            tool_message = f"Tool result: {tool_result}"
            
            # Add the tool result to the conversation
            self.client.conversation_history.append(
                Message(role=Role.FUNCTION, content=tool_result, name=tool_name)
            )
            
            # Get the final response
            final_response = await self.client.chat(
                message="Please provide a final answer based on the tool results."
            )
            
            self.console.print(f"[bold purple]Assistant:[/bold purple]")
            self.console.print(Markdown(final_response))
            
            return final_response
        else:
            # The assistant responded directly
            self.console.print(f"[bold purple]Assistant:[/bold purple]")
            self.console.print(Markdown(response))
            
            return response
    
    def _parse_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse a tool call from the response.
        
        Args:
            response: The response from the assistant
            
        Returns:
            A dictionary with tool name and parameters, or None if no tool call
        """
        # Look for JSON blocks in the response
        if "```json" in response:
            try:
                # Extract the JSON block
                json_block = response.split("```json")[1].split("```")[0].strip()
                tool_call = json.loads(json_block)
                
                # Validate the tool call
                if "tool" in tool_call and tool_call["tool"] in self.tools:
                    return tool_call
            except (json.JSONDecodeError, IndexError):
                pass
        
        return None


async def main():
    """Run the agent example."""
    console = Console()
    console.print("[bold blue]Simple Agent Example[/bold blue]")
    
    agent = SimpleAgent()
    
    # Example queries that might use tools
    queries = [
        "What's the weather like in New York?",
        "What time is it right now?",
        "Can you search for information about quantum computing?"
    ]
    
    for query in queries:
        console.print("\n" + "-" * 50)
        await agent.run(query)


if __name__ == "__main__":
    asyncio.run(main())
