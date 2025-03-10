"""Utility functions for the Rephysco LLM System examples.

This module provides helper functions for the examples, including:
- Token counting
- Image handling
- Logging
- Error handling
"""

import base64
import logging
import os
from typing import Optional

import pendulum
import requests
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm


# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("rephysco.examples")


def setup_logging(level: int = logging.INFO) -> None:
    """Set up logging with rich formatting.
    
    Args:
        level: The logging level to use
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )


def encode_image_from_url(url: str) -> str:
    """Encode an image from a URL as base64.
    
    Args:
        url: The URL of the image
        
    Returns:
        Base64-encoded image data
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get the total file size for the progress bar
    total_size = int(response.headers.get("content-length", 0))
    
    # Download with progress bar
    image_data = bytearray()
    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading image") as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            image_data.extend(chunk)
            pbar.update(len(chunk))
    
    return base64.b64encode(image_data).decode("utf-8")


def encode_image_from_file(file_path: str) -> str:
    """Encode an image from a file as base64.
    
    Args:
        file_path: The path to the image file
        
    Returns:
        Base64-encoded image data
    """
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def save_response_to_file(response: str, filename: Optional[str] = None) -> str:
    """Save a response to a file.
    
    Args:
        response: The response to save
        filename: The filename to use (defaults to timestamp)
        
    Returns:
        The path to the saved file
    """
    if filename is None:
        timestamp = pendulum.now().format("YYYYMMDD_HHmmss")
        filename = f"response_{timestamp}.md"
    
    # Create the outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    file_path = os.path.join("outputs", filename)
    
    with open(file_path, "w") as f:
        f.write(response)
    
    logger.info(f"Response saved to {file_path}")
    return file_path