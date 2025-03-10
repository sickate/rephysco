"""Multimodal example using the Rephysco LLM System.

This example demonstrates how to use the library with multimodal inputs (images).
"""

import asyncio
import os
from typing import List, Optional

from rich.console import Console
from rich.markdown import Markdown

from rephysco import LLMClient, ModelProvider
from rephysco.config import OpenAI, Aliyun, Gemini, SiliconFlow, XAI
from examples.utils import encode_image_from_url, encode_image_from_file, save_response_to_file


async def image_analysis(image_url: str):
    """Analyze an image using a multimodal model.
    
    Args:
        image_url: URL of the image to analyze
    """
    console = Console()
    console.print("[bold blue]Image Analysis Example[/bold blue]")
    
    # Create a client with a multimodal-capable provider
    client = LLMClient(provider=ModelProvider.OPENAI, model=OpenAI.GPT_4O_MINI)
    
    # Prompt for image analysis
    prompt = "What's in this image? Provide a detailed description."
    
    console.print(f"[bold green]User:[/bold green] {prompt}")
    console.print(f"[bold green]Image:[/bold green] {image_url}")
    
    # Generate a response with the image
    response = await client.chat(
        message=prompt,
        images=[image_url]
    )
    
    console.print(f"[bold purple]Assistant:[/bold purple]")
    console.print(Markdown(response))
    
    # Save the response to a file
    save_response_to_file(response, "image_analysis.md")


async def multiple_images_comparison(image_urls: List[str]):
    """Compare multiple images using a multimodal model.
    
    Args:
        image_urls: List of image URLs to compare
    """
    console = Console()
    console.print("\n[bold blue]Multiple Images Comparison Example[/bold blue]")
    
    # Create a client with a multimodal-capable provider
    client = LLMClient(provider=ModelProvider.OPENAI, model="gpt-4o")
    
    # Prompt for image comparison
    prompt = "Compare these images and tell me the similarities and differences between them."
    
    console.print(f"[bold green]User:[/bold green] {prompt}")
    for i, url in enumerate(image_urls, 1):
        console.print(f"[bold green]Image {i}:[/bold green] {url}")
    
    # Generate a response with multiple images
    response = await client.chat(
        message=prompt,
        images=image_urls
    )
    
    console.print(f"[bold purple]Assistant:[/bold purple]")
    console.print(Markdown(response))
    
    # Save the response to a file
    save_response_to_file(response, "image_comparison.md")


async def image_with_follow_up_questions(image_url: str):
    """Ask follow-up questions about an image in a conversation.
    
    Args:
        image_url: URL of the image to discuss
    """
    console = Console()
    console.print("\n[bold blue]Image with Follow-up Questions Example[/bold blue]")
    
    # Create a client with a multimodal-capable provider
    client = LLMClient(provider=ModelProvider.OPENAI, model="gpt-4o")
    
    # Initial prompt with image
    initial_prompt = "What's in this image?"
    
    console.print(f"[bold green]User:[/bold green] {initial_prompt}")
    console.print(f"[bold green]Image:[/bold green] {image_url}")
    
    # Generate initial response with the image
    response = await client.chat(
        message=initial_prompt,
        images=[image_url]
    )
    
    console.print(f"[bold purple]Assistant:[/bold purple]")
    console.print(Markdown(response))
    
    # Follow-up questions (without including the image again)
    follow_up_questions = [
        "What colors are most prominent in this image?",
        "What emotions does this image evoke?",
        "How would you describe the composition of this image?"
    ]
    
    for question in follow_up_questions:
        console.print(f"[bold green]User:[/bold green] {question}")
        
        response = await client.chat(message=question)
        
        console.print(f"[bold purple]Assistant:[/bold purple]")
        console.print(Markdown(response))
    
    # Save the conversation history
    history = client.get_history()
    conversation_text = "\n\n".join([
        f"**{msg.role.value.capitalize()}**: {msg.content}"
        for msg in history
    ])
    save_response_to_file(conversation_text, "image_conversation.md")


async def main():
    """Run the multimodal examples."""
    # Example image URLs (replace with actual URLs for testing)
    image_url = "https://images.unsplash.com/photo-1682687220063-4742bd7fd538"
    comparison_urls = [
        "https://images.unsplash.com/photo-1682687220063-4742bd7fd538",
        "https://images.unsplash.com/photo-1575936123452-b67c3203c357"
    ]
    
    # Run the examples
    await image_analysis(image_url)
    await multiple_images_comparison(comparison_urls)
    await image_with_follow_up_questions(image_url)


if __name__ == "__main__":
    asyncio.run(main()) 