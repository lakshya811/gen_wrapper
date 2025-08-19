import pytest
from gen_wrapper.llm_wrapper import LLMWrapper, LLMWrapperError
from gen_wrapper.providers_config import get_provider_config, validate_all_configs, global_config

class TestBaseFunctionality:
    def test_list_all_providers(self):
        """Test listing all providers"""
        providers = LLMWrapper.list_providers()
        expected_providers = ["openai", "anthropic", "groq", "fireworks", "gemini", "azure_openai", "llama_qwen"]
        
        for provider in expected_providers:
            assert provider in providers
    
    def test_invalid_provider(self):
        """Test invalid provider raises error"""
        with pytest.raises(LLMWrapperError, match="Provider 'invalid_provider' not supported"):
            LLMWrapper("invalid_provider")
    
    def test_provider_config_access(self):
        """Test accessing provider configurations"""
        for provider in LLMWrapper.list_providers():
            config = get_provider_config(provider)
            assert config is not None
            assert hasattr(config, 'default_model')
            assert hasattr(config, 'base_url')
            assert hasattr(config, 'timeout')
    
    def test_list_models_for_invalid_provider(self):
        """Test listing models for invalid provider"""
        with pytest.raises(LLMWrapperError, match="Provider 'invalid_provider' not supported"):
            LLMWrapper.list_models("invalid_provider")
    
    def test_list_models_no_api_key(self):
        """Test listing models without API key (should return empty or fallback)"""
        # Test with a provider that requires API key
        models = LLMWrapper.list_models("openai")
        # Should return empty list or fallback models when no API key
        assert isinstance(models, list)
    
    def test_config_validation(self):
        """Test that all provider configs are valid"""
        validation_results = validate_all_configs()
        
        for provider, is_valid in validation_results.items():
            assert is_valid, f"Configuration for {provider} is invalid"
    
    def test_global_config_access(self):
        """Test global configuration access"""
        assert global_config.service_name == "llm-wrapper"
        assert global_config.environment in ["development", "staging", "production"]
        assert isinstance(global_config.prometheus_enabled, bool)
    
    def test_model_name_flexibility(self):
        """Test that any model name is accepted (no validation)"""
        # Should not raise error even with fake model names
        try:
            # This should succeed in initialization
            wrapper = LLMWrapper("openai", "definitely-fake-model")
            assert wrapper.model == "definitely-fake-model"
        except LLMWrapperError as e:
            # Only should fail on missing API key, not invalid model
            assert "API key" in str(e)
    
    def test_default_model_assignment(self):
        """Test default model assignment when none specified"""
        try:
            wrapper = LLMWrapper("groq")  # No model specified
            # Should use default model from config
            config = get_provider_config("groq")
            assert wrapper.model == config.default_model
        except LLMWrapperError as e:
            # Expected if no API key
            assert "API key" in str(e)
    
    def test_provider_info_structure(self):
        """Test provider info returns expected structure"""
        try:
            wrapper = LLMWrapper("groq")
            info = wrapper.get_provider_info()
            
            required_fields = ["provider", "model", "base_url", "langchain_support", "timeout", "max_retries"]
            for field in required_fields:
                assert field in info, f"Missing field: {field}"
                
        except LLMWrapperError as e:
            # Expected if no API key
            assert "API key" in str(e)
    
    def test_error_message_quality(self):
        """Test that error messages are helpful"""
        with pytest.raises(LLMWrapperError) as exc_info:
            LLMWrapper("nonexistent_provider")
        
        error_msg = str(exc_info.value)
        assert "not supported" in error_msg
        assert "Available:" in error_msg
    
    def test_case_sensitivity(self):
        """Test provider name case sensitivity"""
        with pytest.raises(LLMWrapperError):
            LLMWrapper("OPENAI")  # Should be lowercase
        
        with pytest.raises(LLMWrapperError):
            LLMWrapper("OpenAI")  # Should be lowercase
    
    def test_empty_model_name(self):
        """Test behavior with empty model name"""
        try:
            wrapper = LLMWrapper("groq", "")
            # Should fall back to default model
            config = get_provider_config("groq")
            assert wrapper.model == config.default_model
        except LLMWrapperError as e:
            # Expected if no API key
            assert "API key" in str(e)
    
    def test_none_model_name(self):
        """Test behavior with None model name"""
        try:
            wrapper = LLMWrapper("groq", None)
            # Should use default model
            config = get_provider_config("groq")
            assert wrapper.model == config.default_model
        except LLMWrapperError as e:
            # Expected if no API key
            assert "API key" in str(e)
    
    def test_message_validation(self):
        """Test that message format validation works"""
        try:
            wrapper = LLMWrapper("groq")
            
            # Should accept proper message format
            valid_messages = [{"role": "user", "content": "Hello"}]
            # This will likely fail due to no API key, but format should be accepted
            
            # Should reject invalid message format
            invalid_messages = "Just a string"
            with pytest.raises((LLMWrapperError, TypeError, ValueError)):
                wrapper.chat(invalid_messages)
                
        except LLMWrapperError as e:
            # Expected if no API key - that's fine for this test
            if "API key" not in str(e):
                # If it's not an API key error, re-raise
                raise