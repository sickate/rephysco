"""Multiple providers example using the Rephysco LLM System.

This example demonstrates how to use different LLM providers with the same interface.
"""

import asyncio
import os
import logging
from typing import List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.live import Live
from rich.text import Text

from rephysco import LLMClient, ModelProvider
from rephysco.config import Aliyun, XAI, Gemini, SiliconFlow
from examples.utils import save_response_to_file, encode_image_from_url

# Set up logging
logger = logging.getLogger(__name__)


async def compare_providers(prompt: str, system_prompt: Optional[str] = None):
    """Compare responses from different providers for the same prompt.
    
    Args:
        prompt: The prompt to send to all providers
        system_prompt: Optional system prompt to set context
    """
    console = Console()
    console.print(f"[bold blue]Comparing Provider Responses[/bold blue]")
    
    # List of providers to compare
    providers = [
        ModelProvider.OPENAI,
        ModelProvider.ALIYUN,
        ModelProvider.XAI,
        ModelProvider.GEMINI,
        ModelProvider.SILICON_FLOW,
    ]
    
    # Create a table for comparison
    table = Table(title=f"Responses to: {prompt}")
    table.add_column("Provider", style="cyan")
    table.add_column("Response", style="green")
    
    # Get responses from each provider
    for provider in providers:
        try:
            console.print(f"[bold]Querying {provider.value}...[/bold]")
            
            # Create a client for this provider
            client = LLMClient(provider=provider)
            
            # Generate a response
            response = await client.generate(
                prompt=prompt,
                system_prompt=system_prompt
            )
            
            # Add to the comparison table
            table.add_row(provider.value, response[:200] + "..." if len(response) > 200 else response)
            
            # Save the full response to a file
            save_response_to_file(response, f"{provider.value}_response.md")
            
        except Exception as e:
            logger.exception(f"Error with {provider.value}")
            console.print(f"[bold red]Error with {provider.value}: {str(e)}[/bold red]")
            table.add_row(provider.value, f"Error: {str(e)}")
    
    # Display the comparison table
    console.print(table)


async def provider_specific_features():
    """Demonstrate provider-specific features."""
    console = Console()
    console.print("\n[bold blue]Provider-Specific Features[/bold blue]")
    
    # Use a smaller, more reliable image for testing
    image_url = "https://images.unsplash.com/photo-1549479900-9ffbe7c483be"
    
    # Aliyun multimodal example
    console.print("\n[bold cyan]Aliyun Multimodal Example[/bold cyan]")
    try:
        # Use the multimodal-capable model from Aliyun
        client = LLMClient(provider=ModelProvider.ALIYUN, model=Aliyun.QWEN_OMNI)
        
        console.print(f"[bold green]Image:[/bold green] {image_url}")
        console.print(f"[bold green]Using model:[/bold green] {Aliyun.QWEN_OMNI}")
        console.print("[bold yellow]Note: This model requires streaming mode[/bold yellow]")
        
        # Use streaming mode as required by the model
        # Aliyun can't access this url so we need to encode the image
        image_base64 = encode_image_from_url(image_url)
        base64_url = f"data:image/jpeg;base64,{image_base64}"
        response_stream = await client.chat(
            message="What's in this image? Please describe it in detail.",
            images=[base64_url],
            stream=True  # Enable streaming as required by qwen-omni-turbo
        )
        
        # Display the streaming response
        console.print(f"[bold purple]Aliyun Response:[/bold purple]")
        
        # Collect the full response for saving to file
        full_response = ""
        
        # Create a live display for the streaming response
        with Live(Text(""), refresh_per_second=10) as live:
            async for chunk in response_stream:
                full_response += chunk
                live.update(Text(full_response))
        
        # Display the final response with Markdown formatting
        console.print(Markdown(full_response))
        
        # Save the full response to a file
        save_response_to_file(full_response, "aliyun_multimodal_response.md")
        
    except Exception as e:
        logger.exception("Error with Aliyun multimodal")
        console.print(f"[bold red]Error with Aliyun multimodal: {str(e)}[/bold red]")
    
    # XAI Grok-2-Vision example
    console.print("\n[bold cyan]XAI Grok-2-Vision Example[/bold cyan]")
    try:
        # Use the vision-capable model from XAI
        client = LLMClient(provider=ModelProvider.XAI, model=XAI.GROK_2_VISION)
        
        console.print(f"[bold green]Image:[/bold green] {image_url}")
        console.print(f"[bold green]Using model:[/bold green] {XAI.GROK_2_VISION}")
        
        response = await client.chat(
            message="Describe this image in detail.",
            images=[image_url]
        )
        
        console.print(f"[bold purple]XAI Response:[/bold purple]")
        console.print(Markdown(response))
        
    except Exception as e:
        logger.exception("Error with XAI multimodal")
        console.print(f"[bold red]Error with XAI multimodal: {str(e)}[/bold red]")
    
    # Gemini multimodal example
    console.print("\n[bold cyan]Gemini Multimodal Example[/bold cyan]")
    try:
        # Use the vision-capable model from Gemini
        client = LLMClient(provider=ModelProvider.GEMINI, model=Gemini.GEMINI_1_5_PRO)
        
        console.print(f"[bold green]Image:[/bold green] {image_url}")
        console.print(f"[bold green]Using model:[/bold green] {Gemini.GEMINI_1_5_PRO}")
        
        response = await client.chat(
            message="What do you see in this image?",
            images=[base64_url] # Gemini can only take base64 encoded images
        )
        
        console.print(f"[bold purple]Gemini Response:[/bold purple]")
        console.print(Markdown(response))
        
    except Exception as e:
        logger.exception("Error with Gemini multimodal")
        console.print(f"[bold red]Error with Gemini multimodal: {str(e)}[/bold red]")
    
    # SiliconFlow Reasoning example
    console.print("\n[bold cyan]SiliconFlow Example[/bold cyan]")
    try:
        # Use SiliconFlow model
        client = LLMClient(provider=ModelProvider.SILICON_FLOW, model=SiliconFlow.DEEPSEEK_R1)
        
        console.print(f"[bold green]Using model:[/bold green] {SiliconFlow.DEEPSEEK_R1}")
        
        response = await client.chat(
            message="Explain how neural networks work in simple terms."
        )
        
        console.print(f"[bold purple]SiliconFlow Response:[/bold purple]")
        console.print(Markdown(response))
        
    except Exception as e:
        logger.exception("Error with SiliconFlow")
        console.print(f"[bold red]Error with SiliconFlow: {str(e)}[/bold red]")


async def main():
    """Run the multiple providers examples."""
    # Compare providers on the same prompt
    # await compare_providers(
    #     prompt="Explain the concept of quantum entanglement in simple terms.",
    #     system_prompt="You are a helpful assistant that explains complex concepts in simple terms."
    # )
    
    # Demonstrate provider-specific features
    await provider_specific_features()


if __name__ == "__main__":
    asyncio.run(main()) 