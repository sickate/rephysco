"""
Type definitions for the Rephysco package.
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field


class Role(str, Enum):
    """Role of a message in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class ModelProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    ALIYUN = "aliyun"
    SILICON_FLOW = "silicon_flow"
    XAI = "xai"


class Message(BaseModel):
    """A message in a conversation."""
    role: Role
    content: Optional[str] = None
    name: Optional[str] = None
    images: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary.
        
        Returns:
            Dictionary representation of the message
        """
        result = {"role": self.role.value}
        
        if self.content is not None:
            result["content"] = self.content
            
        if self.name is not None:
            result["name"] = self.name
            
        if self.images is not None and len(self.images) > 0:
            result["images"] = self.images
            
        return result


class ModelResponse(BaseModel):
    """Response from an LLM model."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None


class StreamChunk(BaseModel):
    """A chunk of a streaming response."""
    content: str
    finish_reason: Optional[str] = None
    model: Optional[str] = None 