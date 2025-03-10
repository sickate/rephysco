#!/usr/bin/env python
"""Command-line interface for the Rephysco LLM System.

This module provides a command-line interface for interacting with LLMs through
the Rephysco library.
"""

import asyncio
import os
import sys
from typing import Optional

import click
from rich.console import Console
from rich.markdown import Markdown

from .client import LLMClient
from .types import ModelProvider


console = Console()


@click.group()
def cli():
    """Rephysco CLI - Interact with LLMs from the command line."""
    pass


@cli.command()
@click.option(
    "--provider",
    type=click.Choice([p.value for p in ModelProvider]),
    default=ModelProvider.OPENAI.value,
    help="LLM provider to use"
)
@click.option(
    "--model",
    help="Specific model to use (provider-dependent)"
)
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    help="Temperature for generation (0.0 to 1.0)"
)
@click.option(
    "--system",
    help="System prompt to set context"
)
@click.option(
    "--max-tokens",
    type=int,
    help="Maximum tokens to generate"
)
@click.argument("prompt")
def generate(
    provider: str,
    model: Optional[str],
    temperature: float,
    system: Optional[str],
    max_tokens: Optional[int],
    prompt: str
):
    """Generate a response to a prompt."""
    async def _generate():
        # Create the client
        client = LLMClient(
            provider=ModelProvider(provider),
            model=model
        )
        
        # Generate the response
        console.print(f"[bold green]User:[/bold green] {prompt}")
        
        response = await client.generate(
            prompt=prompt,
            system_prompt=system,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        console.print(f"[bold purple]Assistant:[/bold purple]")
        console.print(Markdown(response))
    
    asyncio.run(_generate())


@cli.command()
@click.option(
    "--provider",
    type=click.Choice([p.value for p in ModelProvider]),
    default=ModelProvider.OPENAI.value,
    help="LLM provider to use"
)
@click.option(
    "--model",
    help="Specific model to use (provider-dependent)"
)
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    help="Temperature for generation (0.0 to 1.0)"
)
@click.option(
    "--system",
    help="System prompt to set context"
)
def chat(
    provider: str,
    model: Optional[str],
    temperature: float,
    system: Optional[str]
):
    """Start an interactive chat session."""
    async def _chat_session():
        # Create the client
        client = LLMClient(
            provider=ModelProvider(provider),
            model=model
        )
        
        console.print("[bold blue]Rephysco Chat[/bold blue]")
        console.print("Type 'exit' or 'quit' to end the session")
        console.print("Type 'system: <prompt>' to set a new system prompt")
        console.print("Type 'temp: <value>' to change the temperature")
        console.print("Type 'clear' to clear the conversation history")
        console.print("-" * 50)
        
        # Set initial system prompt if provided
        current_system_prompt = system
        current_temperature = temperature
        
        while True:
            # Get user input
            user_input = input("\n[User]: ")
            
            # Check for special commands
            if user_input.lower() in ["exit", "quit"]:
                console.print("[bold blue]Ending chat session[/bold blue]")
                break
            
            elif user_input.lower() == "clear":
                client.clear_history()
                console.print("[bold yellow]Conversation history cleared[/bold yellow]")
                continue
            
            elif user_input.lower().startswith("system:"):
                current_system_prompt = user_input[7:].strip()
                console.print(f"[bold yellow]System prompt set to:[/bold yellow] {current_system_prompt}")
                client.clear_history()  # Clear history when system prompt changes
                continue
            
            elif user_input.lower().startswith("temp:"):
                try:
                    current_temperature = float(user_input[5:].strip())
                    console.print(f"[bold yellow]Temperature set to:[/bold yellow] {current_temperature}")
                except ValueError:
                    console.print("[bold red]Invalid temperature value[/bold red]")
                continue
            
            # Generate response
            response = await client.chat(
                message=user_input,
                system_prompt=current_system_prompt,
                temperature=current_temperature
            )
            
            console.print("\n[Assistant]:")
            console.print(Markdown(response))
    
    try:
        asyncio.run(_chat_session())
    except KeyboardInterrupt:
        console.print("\n[bold blue]Chat session ended[/bold blue]")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
