"""Configuration for LLM providers and models.

This module contains configuration settings for supported LLM providers,
including model types and API keys.
"""

import os
from dotenv import load_dotenv
from typing import Dict, Optional

# Load environment variables
load_dotenv()

class SiliconFlow:
    """SiliconFlow configuration."""
    # Models
    DEEPSEEK_V3 = "deepseek-v3"
    DEEPSEEK_R1 = "deepseek-r1"
    DEEPSEEK_V3_PRO = "deepseek-v3-pro"
    DEEPSEEK_R1_PRO = "deepseek-r1-pro"
    QWEN2_5_72B_INSTRUCT = "qwen2.5-72b-instruct"
    
    # API configuration
    API_KEY = os.getenv("SILICON_FLOW_API_KEY")
    BASE_URL = os.getenv("SILICON_FLOW_BASE_URL", "https://api.siliconflow.com/v1")

class Aliyun:
    DEEPSEEK_R1 = 'deepseek-r1'
    DEEPSEEK_V3 = 'deepseek-v3'
    QWEN_MAX = 'qwen-max-latest'
    QWEN_PLUS = 'qwen-plus'
    QWEN_OMNI = 'qwen-omni-turbo' # 通义千问全模态理解生成大模型，支持文本, 图像，语音，视频输入理解和混合输入理解，具备文本和语音同时流式生成能力，提供了4种自然对话音色

    # API endpoints
    HTTP_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
    BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    API_KEY = os.getenv("DASHSCOPE_API_KEY")

class XAI:
    GROK_2_VISION = 'grok-2-vision-1212'
    GROK_2 = 'grok-2-1212'

    BASE_URL = "https://api.x.ai/v1"
    API_KEY = os.getenv("XAI_API_KEY")

class Gemini:
    GEMINI_1_5_PRO = "gemini-1.5-pro" # reasoning and creativity
    GEMINI_2_0_FLASH = "gemini-2.0-flash" # fast and accurate
    GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite" # faster and cheaper
    GEMINI_1_5_FLASH_8B = "gemini-1.5-flash-8b" # fastest and cheapest

    API_KEY = os.getenv("GOOGLE_API_KEY")
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

class OpenAI:
    """OpenAI configuration."""
    # Models
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_O1 = "gpt-o1"
    GPT_O3_MINI = "gpt-o3-mini"
    
    # API configuration
    API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


class GoogleSearch:
    API_KEY = os.getenv("GOOGLE_API_KEY")
    CSE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
