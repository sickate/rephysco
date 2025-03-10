"""
Conversation management for the Rephysco LLM System.
"""

from typing import List, Optional

from .types import Message, Role


class Conversation:
    """Manages a conversation with an LLM."""
    
    def __init__(self):
        """Initialize a new conversation."""
        self.messages: List[Message] = []
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation.
        
        Args:
            message: The message to add
        """
        self.messages.append(message)
    
    def add_user_message(self, content: str, images: Optional[List[str]] = None) -> None:
        """Add a user message to the conversation.
        
        Args:
            content: The content of the message
            images: Optional list of image URLs or base64-encoded images
        """
        self.add_message(Message(role=Role.USER, content=content, images=images))
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation.
        
        Args:
            content: The content of the message
        """
        self.add_message(Message(role=Role.ASSISTANT, content=content))
    
    def add_system_message(self, content: str) -> None:
        """Add a system message to the conversation.
        
        Args:
            content: The content of the message
        """
        self.add_message(Message(role=Role.SYSTEM, content=content))
    
    def set_system_prompt(self, content: str) -> None:
        """Set the system prompt for the conversation.
        
        If a system prompt already exists, it will be replaced.
        
        Args:
            content: The content of the system prompt
        """
        # Remove any existing system messages
        self.messages = [msg for msg in self.messages if msg.role != Role.SYSTEM]
        
        # Add the new system message at the beginning
        self.messages.insert(0, Message(role=Role.SYSTEM, content=content))
    
    def has_system_prompt(self) -> bool:
        """Check if the conversation has a system prompt.
        
        Returns:
            True if the conversation has a system prompt, False otherwise
        """
        return any(msg.role == Role.SYSTEM for msg in self.messages)
    
    def clear(self) -> None:
        """Clear the conversation history."""
        self.messages = []
    
    def get_last_user_message(self) -> Optional[Message]:
        """Get the last user message in the conversation.
        
        Returns:
            The last user message, or None if there are no user messages
        """
        for msg in reversed(self.messages):
            if msg.role == Role.USER:
                return msg
        return None
    
    def get_last_assistant_message(self) -> Optional[Message]:
        """Get the last assistant message in the conversation.
        
        Returns:
            The last assistant message, or None if there are no assistant messages
        """
        for msg in reversed(self.messages):
            if msg.role == Role.ASSISTANT:
                return msg
        return None 