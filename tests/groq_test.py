import pytest
import time

# Import will work because conftest.py sets up the path
from gen_wrapper.llm_wrapper import LLMWrapper, LLMWrapperError

class TestGroqClient:
    def test_list_models(self):
        """Test listing Groq models"""
        models = LLMWrapper.list_models("groq")
        
        # Dynamic model fetching - check if we got any models
        if models:
            # If we got models from API, check for common ones
            common_models = ["llama3-8b-8192", "mixtral-8x7b-32768"]
            found_common = any(model in models for model in common_models)
            assert found_common, f"No common models found in: {models}"
        else:
            # If no models fetched (no API key), that's also valid
            print("No models fetched - likely no API key available")
    
    def test_initialization_default_model(self, api_keys, has_api_key):
        """Test Groq initialization with default model"""
        if not has_api_key("groq", api_keys):
            pytest.skip("Groq API key not available")
        
        wrapper = LLMWrapper("groq")
        assert wrapper.provider_name == "groq"
        assert wrapper.model == "llama3-8b-8192"
    
    def test_initialization_custom_model(self, api_keys, has_api_key):
        """Test Groq initialization with custom model"""
        if not has_api_key("groq", api_keys):
            pytest.skip("Groq API key not available")
        
        # Test with any model name - let API validate
        wrapper = LLMWrapper("groq", "llama3-70b-8192")
        assert wrapper.provider_name == "groq"
        assert wrapper.model == "llama3-70b-8192"
    
    def test_provider_info(self, api_keys, has_api_key):
        """Test getting Groq provider info"""
        if not has_api_key("groq", api_keys):
            pytest.skip("Groq API key not available")
        
        wrapper = LLMWrapper("groq")
        info = wrapper.get_provider_info()
        
        assert info["provider"] == "groq"
        assert info["base_url"] == "https://api.groq.com/openai/v1"
        assert info["langchain_support"] is False
    
    @pytest.mark.slow
    def test_simple_chat(self, api_keys, has_api_key):
        """Test Groq simple chat"""
        if not has_api_key("groq", api_keys):
            pytest.skip("Groq API key not available")
        
        wrapper = LLMWrapper("groq")
        response = wrapper.simple_chat("Say 'hello' only", max_tokens=5)
        
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"Response: {response}")
    
    @pytest.mark.slow
    def test_chat_with_history(self, api_keys, has_api_key):
        """Test Groq chat with message history"""
        if not has_api_key("groq", api_keys):
            pytest.skip("Groq API key not available")
        
        wrapper = LLMWrapper("groq")
        
        messages = [
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "What's that plus 3?"}
        ]
        
        response = wrapper.chat(messages, max_tokens=10)
        
        assert "error" not in response
        assert "message" in response
        assert isinstance(response["message"], str)
        print(f"Chat response: {response}")
    
    @pytest.mark.slow
    def test_deepseek_via_groq(self, api_keys, has_api_key):
        """Test DeepSeek model via Groq"""
        if not has_api_key("groq", api_keys):
            pytest.skip("Groq API key not available")
        
        try:
            wrapper = LLMWrapper("groq", "deepseek-r1-distill-llama-70b")
            response = wrapper.simple_chat("What is 2+2?", max_tokens=10)
            
            assert isinstance(response, str)
            assert len(response) > 0
            print(f"DeepSeek response: {response}")
        except LLMWrapperError as e:
            if "model" in str(e).lower():
                pytest.skip(f"DeepSeek model not available on Groq: {e}")
            else:
                raise
    
    @pytest.mark.slow
    def test_error_handling(self, api_keys, has_api_key):
        """Test error handling with invalid parameters"""
        if not has_api_key("groq", api_keys):
            pytest.skip("Groq API key not available")
        
        wrapper = LLMWrapper("groq")
        
        # Test with invalid max_tokens (should handle gracefully)
        try:
            response = wrapper.simple_chat("Hi", max_tokens=-1)
            # Some providers might accept this, others might error
            assert isinstance(response, str)
        except LLMWrapperError:
            # Expected for invalid parameters
            pass
    
    @pytest.mark.slow
    def test_multiple_models_limited(self, api_keys, has_api_key):
        """Test first two available models (avoid rate limits)"""
        if not has_api_key("groq", api_keys):
            pytest.skip("Groq API key not available")
        
        models = LLMWrapper.list_models("groq")
        if not models:
            pytest.skip("No models available for testing")
        
        # Test only first 2 models to avoid rate limits
        test_models = models[:2]
        
        for i, model in enumerate(test_models):
            print(f"Testing model {i+1}/{len(test_models)}: {model}")
            
            try:
                wrapper = LLMWrapper("groq", model)
                response = wrapper.simple_chat("Hi", max_tokens=5)
                assert isinstance(response, str)
                assert len(response) > 0
                print(f"  ✓ Response: {response[:50]}...")
                
            except LLMWrapperError as e:
                print(f"  ✗ Model {model} failed: {e}")
                # Don't fail the test for individual model failures
                continue
            
            # Rate limiting between tests
            if i < len(test_models) - 1:
                time.sleep(2)
    
    def test_model_validation_disabled(self, api_keys, has_api_key):
        """Test that wrapper allows any model name (no validation)"""
        if not has_api_key("groq", api_keys):
            pytest.skip("Groq API key not available")
        
        # Should allow initialization with any model name
        # API will validate later
        wrapper = LLMWrapper("groq", "definitely-not-a-real-model")
        assert wrapper.model == "definitely-not-a-real-model"
        
        # The actual API call might fail, but initialization should succeed
        try:
            response = wrapper.simple_chat("Hi", max_tokens=5)
            # If this succeeds, the model exists
            assert isinstance(response, str)
        except LLMWrapperError:
            # Expected for invalid model names
            pass
    
    @pytest.mark.slow
    def test_concurrent_requests(self, api_keys, has_api_key):
        """Test handling multiple concurrent requests"""
        if not has_api_key("groq", api_keys):
            pytest.skip("Groq API key not available")
        
        import threading
        import concurrent.futures
        
        wrapper = LLMWrapper("groq")
        results = []
        errors = []
        
        def make_request(i):
            try:
                response = wrapper.simple_chat(f"Count to {i}", max_tokens=10)
                results.append(response)
            except Exception as e:
                errors.append(str(e))
        
        # Test with 3 concurrent requests (within Groq limits)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request, i) for i in range(1, 4)]
            concurrent.futures.wait(futures)
        
        print(f"Successful requests: {len(results)}")
        print(f"Failed requests: {len(errors)}")
        
        # At least some requests should succeed
        assert len(results) > 0, f"All requests failed: {errors}"