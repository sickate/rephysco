#!/usr/bin/env python
"""Main entry point for the Rephysco LLM System.

This module allows the package to be run as a module:
python -m rephysco
"""

import asyncio
import os
import sys
from typing import List, Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

from .client import LLMClient
from .types import ModelProvider
from .cli import main


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
    async def _chat():
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
    
    asyncio.run(_chat())


if __name__ == "__main__":
    main()
