#!/usr/bin/env python
"""Command-line interface for the Rephysco LLM System.

This module provides a command-line interface for interacting with LLMs through
the Rephysco library.
"""

import asyncio
import os
import sys
from typing import List, Optional, Dict, Any
from enum import Enum

import click
import pendulum
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.text import Text
from rich.live import Live
from dotenv import load_dotenv
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI as OpenAILLM
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.tools.google import GoogleSearchToolSpec

from .client import LLMClient
from .types import ModelProvider
from .config import Aliyun, OpenAI, GoogleSearch
from .llm_factory import LLMFactory
from .examples.agent_session import run_agent_session

console = Console()

@click.group()
def cli():
    """Rephysco - A lightweight wrapper for interacting with various LLM providers."""
    pass


@cli.command()
@click.argument("prompt")
@click.option("--provider", "-p", type=click.Choice([p.value for p in ModelProvider]), default="openai", help="LLM provider to use")
@click.option("--model", "-m", help="Model to use (provider-specific)")
@click.option("--system", "-s", help="System prompt to set the context")
@click.option("--temperature", "-t", type=float, default=0.7, help="Temperature for generation")
@click.option("--max-tokens", type=int, help="Maximum number of tokens to generate")
@click.option("--stream/--no-stream", default=True, help="Whether to stream the response")
def generate(
    prompt: str,
    provider: str,
    model: Optional[str],
    system: Optional[str],
    temperature: float,
    max_tokens: Optional[int],
    stream: bool,
):
    """Generate a response to a prompt."""
    async def _generate():
        client = LLMClient(
            provider=ModelProvider(provider),
            model=model,
        )
        
        if stream:
            console.print("[bold]Generating response...[/bold]")
            response_stream = await client.generate(
                prompt=prompt,
                system_prompt=system,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            
            console.print("\n[bold]Response:[/bold]")
            full_response = ""
            async for chunk in response_stream:
                full_response += chunk
                console.print(chunk, end="")
            console.print("\n")
            
            # Display as markdown for better formatting
            console.print(Markdown(full_response))
        else:
            console.print("[bold]Generating response...[/bold]")
            response = await client.generate(
                prompt=prompt,
                system_prompt=system,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            
            console.print("\n[bold]Response:[/bold]")
            console.print(Markdown(response))
    
    asyncio.run(_generate())


@cli.command()
@click.option("--provider", "-p", type=click.Choice([p.value for p in ModelProvider]), default="openai", help="LLM provider to use")
@click.option("--model", "-m", help="Model to use (provider-specific)")
@click.option("--system", "-s", help="System prompt to set the context")
@click.option("--temperature", "-t", type=float, default=0.7, help="Temperature for generation")
@click.option("--max-tokens", type=int, help="Maximum number of tokens to generate")
def chat(
    provider: str,
    model: Optional[str],
    system: Optional[str],
    temperature: float,
    max_tokens: Optional[int],
):
    """Start an interactive chat session."""
    async def _chat_session():
        client = LLMClient(
            provider=ModelProvider(provider),
            model=model,
        )
        
        console.print("[bold]Starting chat session...[/bold]")
        console.print("Type 'exit' or 'quit' to end the session.")
        console.print("Type 'system: <prompt>' to set a new system prompt.")
        console.print("Type 'temp: <value>' to change the temperature.")
        console.print("Type 'clear' to clear the conversation history.")
        console.print()
        
        if system:
            console.print(f"[bold]System:[/bold] {system}")
        
        current_temp = temperature
        
        while True:
            user_input = Prompt.ask("[bold]You[/bold]")
            
            if user_input.lower() in ["exit", "quit"]:
                break
            elif user_input.lower() == "clear":
                await client.chat("", clear_history=True)
                console.print("[bold]Conversation history cleared.[/bold]")
                continue
            elif user_input.lower().startswith("system:"):
                new_system = user_input[7:].strip()
                await client.chat("", system_prompt=new_system, clear_history=True)
                console.print(f"[bold]System prompt set:[/bold] {new_system}")
                continue
            elif user_input.lower().startswith("temp:"):
                try:
                    current_temp = float(user_input[5:].strip())
                    console.print(f"[bold]Temperature set to:[/bold] {current_temp}")
                except ValueError:
                    console.print("[bold red]Invalid temperature value.[/bold red]")
                continue
            
            response_stream = await client.chat(
                message=user_input,
                system_prompt=system,
                temperature=current_temp,
                max_tokens=max_tokens,
                stream=True,
            )
            
            console.print("[bold]Assistant:[/bold] ", end="")
            full_response = ""
            async for chunk in response_stream:
                full_response += chunk
                console.print(chunk, end="")
            console.print("\n")
    
    try:
        asyncio.run(_chat_session())
    except KeyboardInterrupt:
        console.print("\n[bold blue]Chat session ended[/bold blue]")


@cli.command()
@click.option("--provider", "-p", type=click.Choice([p.value for p in ModelProvider]),
    default=ModelProvider.ALIYUN.value, help="LLM provider to use"
)
@click.option("--model", "-m", default=None, help="Specific model to use (provider-dependent)")
@click.option("--temperature", "-t", type=float, default=0.7, help="Temperature for generation (0.0 to 1.0)")
@click.option("--verbose/--quiet", default=True, help="Show verbose output")
def agent(provider: str, model: str, temperature: float, verbose: bool):
    """
    Start an interactive agent session with Google Search capability.
    
    This command starts an interactive chat session with an AI agent that has
    access to Google Search and other tools. You can ask questions and the agent
    will respond, potentially using search to find information.
    """
    # Check for Google API keys
    if not GoogleSearch.API_KEY or not GoogleSearch.CSE_ID:
        console.print("[bold red]Error:[/bold red] Google API key or CSE ID not found in environment variables.")
        console.print("Please set GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID in your .env file.")
        return
            
    try:
        # Call the agent session implementation from examples
        run_agent_session(provider, model, temperature, verbose)
    except KeyboardInterrupt:
        console.print("\n[bold blue]Agent session ended[/bold blue]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
