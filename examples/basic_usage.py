"""Basic usage examples for the Rephysco LLM System."""

import asyncio
from typing import Optional

import pendulum
from rich.console import Console
from rich.markdown import Markdown

from rephysco import LLMClient, ModelProvider


async def simple_completion():
    """Demonstrate a simple completion with OpenAI."""
    console = Console()
    console.print("[bold blue]Simple Completion Example[/bold blue]")
    
    # Create a client with OpenAI
    client = LLMClient(provider=ModelProvider.OPENAI)
    
    # Generate a response
    prompt = "Introduce yourself in 20 words"
    console.print(f"[bold green]User:[/bold green] {prompt}")
    
    response = await client.generate(prompt=prompt)
    console.print(f"[bold purple]Assistant:[/bold purple]")
    console.print(Markdown(response))


async def conversation_example():
    """Demonstrate a multi-turn conversation."""
    console = Console()
    console.print("\n[bold blue]Conversation Example[/bold blue]")
    
    # Create a client
    client = LLMClient(provider=ModelProvider.OPENAI)
    
    # Start a conversation with a system prompt
    system_prompt = "You are a helpful assistant that provides concise answers."
    
    # First message
    message = "What are the three laws of thermodynamics?"
    console.print(f"[bold green]User:[/bold green] {message}")
    
    response = await client.chat(
        message=message,
        system_prompt=system_prompt
    )
    console.print(f"[bold purple]Assistant:[/bold purple]")
    console.print(Markdown(response))
    
    # Follow-up question
    message = "Which one is most relevant to refrigerators?"
    console.print(f"[bold green]User:[/bold green] {message}")
    
    response = await client.chat(message=message)
    console.print(f"[bold purple]Assistant:[/bold purple]")
    console.print(Markdown(response))


async def streaming_example():
    """Demonstrate streaming responses."""
    console = Console()
    console.print("\n[bold blue]Streaming Example[/bold blue]")
    
    # Create a client
    client = LLMClient(provider=ModelProvider.OPENAI)
    
    # Generate a streaming response
    prompt = "Write a short poem about artificial intelligence."
    console.print(f"[bold green]User:[/bold green] {prompt}")
    
    console.print(f"[bold purple]Assistant:[/bold purple]", end="")
    
    response_stream = await client.generate(
        prompt=prompt,
        stream=True
    )
    
    async for chunk in response_stream:
        console.print(chunk, end="")
    
    console.print()  # Add a newline at the end


async def main():
    """Run all examples."""
    await simple_completion()
    await conversation_example()
    await streaming_example()


if __name__ == "__main__":
    asyncio.run(main())



