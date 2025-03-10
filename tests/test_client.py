"""
Tests for the LLMClient class.
"""

import os
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from rephysco import LLMClient, ModelProvider
from rephysco.types import Message, ModelResponse, Role, StreamChunk
from rephysco.settings import Config


@pytest.fixture
def mock_provider():
    """Create a mock provider for testing."""
    provider = MagicMock()
    provider.default_model = "test-model"
    provider.generate = AsyncMock()
    return provider


@pytest.fixture
def client(mock_provider):
    """Create a client with a mock provider for testing."""
    with patch("rephysco.client.create_provider", return_value=mock_provider):
        # Disable caching and retries for testing
        with patch.object(Config, "enable_caching", False), patch.object(Config, "enable_retries", False):
            client = LLMClient(provider=ModelProvider.OPENAI)
            yield client


@pytest.mark.asyncio
async def test_generate(client, mock_provider):
    """Test the generate method."""
    # Setup
    mock_provider.generate.return_value = ModelResponse(
        content="Test response",
        model="test-model",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        finish_reason="stop"
    )
    
    # Execute
    response = await client.generate(
        prompt="Test prompt",
        system_prompt="Test system prompt"
    )
    
    # Verify
    assert response == "Test response"
    mock_provider.generate.assert_called_once()
    
    # Check that the messages were formatted correctly
    call_args = mock_provider.generate.call_args
    messages = call_args.kwargs.get("messages", [])
    assert len(messages) == 2
    assert messages[0].role == Role.SYSTEM
    assert messages[0].content == "Test system prompt"
    assert messages[1].role == Role.USER
    assert messages[1].content == "Test prompt"


@pytest.mark.asyncio
async def test_chat(client, mock_provider):
    """Test the chat method."""
    # Setup
    mock_provider.generate.return_value = ModelResponse(
        content="Test response",
        model="test-model",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        finish_reason="stop"
    )
    
    # Execute
    response = await client.chat(
        message="Test message",
        system_prompt="Test system prompt"
    )
    
    # Verify
    assert response == "Test response"
    mock_provider.generate.assert_called_once()
    
    # Check that the conversation was updated correctly
    assert len(client.conversation.messages) == 3
    assert client.conversation.messages[0].role == Role.SYSTEM
    assert client.conversation.messages[0].content == "Test system prompt"
    assert client.conversation.messages[1].role == Role.USER
    assert client.conversation.messages[1].content == "Test message"
    assert client.conversation.messages[2].role == Role.ASSISTANT
    assert client.conversation.messages[2].content == "Test response"


@pytest.mark.asyncio
async def test_generate_streaming(client, mock_provider):
    """Test the generate method with streaming."""
    # Setup
    async def mock_stream():
        chunks = [
            StreamChunk(content="Test ", model="test-model"),
            StreamChunk(content="response", model="test-model", finish_reason="stop")
        ]
        for chunk in chunks:
            yield chunk
    
    mock_provider.generate.return_value = mock_stream()
    
    # Execute
    response_stream = await client.generate(
        prompt="Test prompt",
        stream=True
    )
    
    # Verify
    chunks = []
    async for chunk in response_stream:
        chunks.append(chunk)
    
    assert chunks == ["Test ", "response"]
    mock_provider.generate.assert_called_once()


@pytest.mark.asyncio
async def test_chat_streaming(client, mock_provider):
    """Test the chat method with streaming."""
    # Setup
    async def mock_stream():
        chunks = [
            StreamChunk(content="Test ", model="test-model"),
            StreamChunk(content="response", model="test-model", finish_reason="stop")
        ]
        for chunk in chunks:
            yield chunk
    
    mock_provider.generate.return_value = mock_stream()
    
    # Execute
    response_stream = await client.chat(
        message="Test message",
        stream=True
    )
    
    # Verify
    chunks = []
    async for chunk in response_stream:
        chunks.append(chunk)
    
    assert chunks == ["Test ", "response"]
    mock_provider.generate.assert_called_once()
    
    # Check that the conversation was updated correctly
    assert len(client.conversation.messages) == 2
    assert client.conversation.messages[0].role == Role.USER
    assert client.conversation.messages[0].content == "Test message"
    assert client.conversation.messages[1].role == Role.ASSISTANT
    assert client.conversation.messages[1].content == "Test response" 