import pytest
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to Python path so we can import our modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Load environment variables
load_dotenv()

@pytest.fixture
def api_keys():
    """Fixture to provide API keys"""
    return {
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "groq": os.getenv("GROQ_API_KEY"),
        "fireworks": os.getenv("FIREWORKS_API_KEY"),
        "gemini": os.getenv("GOOGLE_API_KEY"),
        "azure_openai": {
            "key": os.getenv("AZURE_OPENAI_API_KEY"),
            "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT")
        }
    }

@pytest.fixture
def has_api_key():
    """Factory to check if provider has API key"""
    def _has_key(provider, api_keys):
        if provider == "azure_openai":
            return api_keys[provider]["key"] and api_keys[provider]["endpoint"]
        elif provider == "llama_qwen":
            return True  # Local server, no key needed
        else:
            return api_keys.get(provider) is not None
    return _has_key

@pytest.fixture(scope="session")
def test_config():
    """Global test configuration"""
    return {
        "max_retries": 3,
        "timeout": 30,
        "rate_limit_delay": 1.0,
        "test_message": "Hello, this is a test message",
        "max_tokens": 10
    }

@pytest.fixture
def mock_response():
    """Mock response for testing"""
    return {
        "message": "Hello! How can I help you today?",
        "error": False,
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25
        }
    }