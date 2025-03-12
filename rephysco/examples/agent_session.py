"""
Agent session implementation for Rephysco.

This module provides the implementation for running an interactive agent session
with Google Search capability.
"""

import sys
from typing import Optional

import pendulum
from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text
from rich.live import Live
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.tools.google import GoogleSearchToolSpec

from rephysco.config import GoogleSearch
from rephysco.llm_factory import LLMFactory

console = Console()

def run_agent_session(provider: str, model: Optional[str], temperature: float, verbose: bool):
    """
    Run an interactive agent session with Google Search capability.
    
    Args:
        provider: The LLM provider to use
        model: The specific model to use (if None, uses provider default)
        temperature: Temperature for generation
        verbose: Whether to show verbose output
    """
    # Get the actual model name if not provided (using default)
    actual_model = model
    if actual_model is None and provider in LLMFactory.DEFAULT_MODELS:
        actual_model = LLMFactory.DEFAULT_MODELS[provider]
    
    console.print("[bold]Starting agent chat session with Google Search...[/bold]")
    console.print(f"[bold blue]Provider:[/bold blue] {provider}")
    console.print(f"[bold blue]Model:[/bold blue] {actual_model}")
    console.print(f"[bold blue]Temperature:[/bold blue] {temperature}")
    console.print("Type 'exit' or 'quit' to end the session.")
    console.print()
    
    # Create LlamaIndex LLM using our factory
    try:
        llm = LLMFactory.create_llm(
            provider=provider,
            model=model,
            temperature=temperature,
        )
    except Exception as e:
        console.print(f"[bold red]Error creating LLM:[/bold red] {str(e)}")
        return
    
    # Create Google Search tool
    google_search_tools = GoogleSearchToolSpec(
        key=GoogleSearch.API_KEY,
        engine=GoogleSearch.CSE_ID
    ).to_tool_list()
    
    # Create current time tool
    def get_current_time():
        """Get the current date and time."""
        return pendulum.now().format("YYYY-MM-DD HH:mm:ss")
    
    time_tool = FunctionTool.from_defaults(
        name="get_current_time",
        description="Get the current date and time",
        fn=get_current_time
    )
    
    # Create the agent
    agent = ReActAgent.from_tools(
        [*google_search_tools, time_tool],
        llm=llm,
        verbose=verbose
    )
    
    # Start the chat loop
    while True:
        user_input = Prompt.ask("\n[bold]You[/bold]")
        
        if user_input.lower() in ["exit", "quit"]:
            console.print("[bold blue]Agent session ended[/bold blue]")
            break
        
        console.print()
        with console.status("[bold green]Agent is thinking...[/bold green]"):
            try:
                # Run the agent with streaming
                response = agent.stream_chat(user_input)
                
                # Print the response
                console.print("\n[bold]Agent:[/bold]")
                if hasattr(response, 'response_gen'):
                    content = Text("")
                    with Live(content, refresh_per_second=10) as live:
                        for chunk in response.response_gen:
                            if hasattr(chunk, 'content'):
                                chunk_text = chunk.content
                            else:
                                chunk_text = str(chunk)
                            content.append(chunk_text)
                            live.update(content)
                else:
                    console.print(response.response)
                    console.print()
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        
        console.print()

if __name__ == "__main__":
    # Example usage when run directly
    from rephysco.types import ModelProvider
    
    run_agent_session(
        provider=ModelProvider.OPENAI.value,
        model=None,  # Use default model
        temperature=0.7,
        verbose=True
    ) 