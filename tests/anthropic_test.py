import pytest
from gen_wrapper.llm_wrapper import LLMWrapper, LLMWrapperError

class TestAnthropicClient:
    def test_list_models(self):
        """Test listing Anthropic models"""
        models = LLMWrapper.list_models("anthropic")
        assert "claude-3-opus-20240229" in models
        assert "claude-3-sonnet-20240229" in models
    
    def test_initialization_default_model(self, api_keys, has_api_key):
        """Test Anthropic initialization with default model"""
        if not has_api_key("anthropic", api_keys):
            pytest.skip("Anthropic API key not available")
        
        wrapper = LLMWrapper("anthropic")
        assert wrapper.provider_name == "anthropic"
        assert wrapper.model == "claude-3-opus-20240229"
    
    def test_provider_info(self, api_keys, has_api_key):
        """Test getting Anthropic provider info"""
        if not has_api_key("anthropic", api_keys):
            pytest.skip("Anthropic API key not available")
        
        wrapper = LLMWrapper("anthropic")
        info = wrapper.get_provider_info()
        
        assert info["provider"] == "anthropic"
        assert info["base_url"] == "https://api.anthropic.com/v1"
        assert info["langchain_support"] is True
    
    @pytest.mark.slow
    def test_simple_chat(self, api_keys, has_api_key):
        """Test Anthropic simple chat"""
        if not has_api_key("anthropic", api_keys):
            pytest.skip("Anthropic API key not available")
        
        wrapper = LLMWrapper("anthropic")
        response = wrapper.simple_chat("Say 'hello' only", max_tokens=5)
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.slow
    def test_system_message_handling(self, api_keys, has_api_key):
        """Test Anthropic system message handling"""
        if not has_api_key("anthropic", api_keys):
            pytest.skip("Anthropic API key not available")
        
        wrapper = LLMWrapper("anthropic")
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be very brief."},
            {"role": "user", "content": "What is AI?"}
        ]
        
        response = wrapper.chat(messages, max_tokens=20)
        
        assert "message" in response
        assert isinstance(response["message"], str)
        assert len(response["message"]) > 0