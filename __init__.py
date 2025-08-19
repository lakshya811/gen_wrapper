"""
LLM Wrapper - A unified interface for multiple LLM providers
"""

__version__ = "0.1.1"
__author__ = "Lakshya Kaushik"
__email__ = "lakshya.kaushik811@gmail.com"

from .src.gen_wrapper.llm_wrapper import LLMWrapper, LLMWrapperError
from .src.gen_wrapper.providers_config import get_provider_config, global_config

__all__ = [
    "LLMWrapper",
    "LLMWrapperError", 
    "get_provider_config",
    "global_config"
]