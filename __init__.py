"""
LLM Wrapper - A unified interface for multiple LLM providers
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .src.gen_wrapper.llm_wrapper import LLMWrapper, LLMWrapperError
from .src.gen_wrapper.providers_config import get_provider_config, global_config

__all__ = [
    "LLMWrapper",
    "LLMWrapperError", 
    "get_provider_config",
    "global_config"
]