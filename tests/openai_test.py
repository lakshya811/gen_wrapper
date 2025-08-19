import pytest
import os
from gen_wrapper.llm_wrapper import LLMWrapper, LLMWrapperError
import time
class TestOpenAIClient:
    
    def test_list_models(self):
        """Test listing available models"""
        models = LLMWrapper.list_models("openai")
        assert isinstance(models, list)
        assert len(models) > 0
        # Check for common OpenAI models
        model_names = [m.lower() for m in models]
        assert any("gpt-4" in m for m in model_names)
        print(f"Available OpenAI models: {models[:5]}...")  # Show first 5
    
    def test_initialization_default_model(self):
        """Test initialization with default model"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        wrapper = LLMWrapper("openai")
        assert wrapper.provider_name == "openai"
        assert wrapper.model == "gpt-4o-mini"  # Default model from config
        print(f"Initialized with default model: {wrapper.model}")
    
    def test_initialization_custom_model(self):
        """Test initialization with custom model"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        wrapper = LLMWrapper("openai", "gpt-4o")
        assert wrapper.provider_name == "openai"
        assert wrapper.model == "gpt-4o"
        print(f"Initialized with custom model: {wrapper.model}")
    
    def test_provider_info(self):
        """Test getting provider information"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        wrapper = LLMWrapper("openai")
        info = wrapper.get_provider_info()
        
        # Check required fields
        assert "provider" in info
        assert "model" in info
        assert "base_url" in info
        assert "timeout" in info
        assert "max_retries" in info
        
        assert info["provider"] == "openai"
        assert info["model"] == "gpt-4o-mini"
        assert "api.openai.com" in info["base_url"]
        
        print(f"Provider info: {info}")
    
    @pytest.mark.integration
    def test_simple_chat(self):
        """Test simple chat functionality"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        wrapper = LLMWrapper("openai")
        
        messages = [{"role": "user", "content": "What is 2+2? Respond with just the number."}]
        response = wrapper.chat(messages)
        assert response is not None
        
        # Extract message from response dict
        message = response.get('message', '') if isinstance(response, dict) else response
        assert len(message.strip()) > 0
        print(f"Simple chat response: {message}")
        
        # Check if response contains "4"
        assert "4" in message
    
    @pytest.mark.integration
    def test_chat_with_history(self):
        """Test chat with conversation history"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        wrapper = LLMWrapper("openai")
        
        # Build conversation with proper history
        messages = [
            {"role": "user", "content": "My favorite color is blue. Remember this."},
        ]
        response1 = wrapper.chat(messages)
        assert response1 is not None
        
        message1 = response1.get('message', '') if isinstance(response1, dict) else response1
        print(f"First response: {message1}")
        
        # Add assistant response and follow-up question
        messages.extend([
            {"role": "assistant", "content": message1},
            {"role": "user", "content": "What is my favorite color?"}
        ])
        response2 = wrapper.chat(messages)
        assert response2 is not None
        
        message2 = response2.get('message', '') if isinstance(response2, dict) else response2
        print(f"Second response: {message2}")
        
        # Should remember the color blue
        assert "blue" in message2.lower()
    
    @pytest.mark.integration
    def test_gpt4_model(self):
        """Test using GPT-4 model specifically"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        wrapper = LLMWrapper("openai", "gpt-4o")
        
        messages = [{"role": "user", "content": "Explain quantum computing in one sentence."}]
        response = wrapper.chat(messages)
        assert response is not None
        
        message = response.get('message', '') if isinstance(response, dict) else response
        assert len(message.strip()) > 20  # Should be a substantial response
        print(f"GPT-4 response: {message}")
    
    def test_error_handling(self):
        """Test error handling with invalid API key"""
        # Temporarily set invalid API key
        original_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "sk-invalid_key_for_testing"
        
        try:
            wrapper = LLMWrapper("openai")
            messages = [{"role": "user", "content": "Test message"}]
            with pytest.raises((LLMWrapperError, Exception)):
                wrapper.chat(messages)
        finally:
            # Restore original key
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key
            else:
                os.environ.pop("OPENAI_API_KEY", None)
    
    def test_multiple_models_available(self):
        """Test that OpenAI has multiple model options"""
        models = LLMWrapper.list_models("openai")
        
        # OpenAI should have many models
        assert len(models) > 10
        
        # Check for specific model families
        gpt4_models = [m for m in models if "gpt-4" in m.lower()]
        gpt3_models = [m for m in models if "gpt-3" in m.lower()]
        
        assert len(gpt4_models) > 0
        print(f"GPT-4 models: {gpt4_models}")
        print(f"Total models available: {len(models)}")
    
    @pytest.mark.integration
    def test_long_conversation(self):
        """Test handling longer conversations"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        wrapper = LLMWrapper("openai")
        
        # Build a multi-turn conversation
        messages = []
        
        # First exchange
        messages.append({"role": "user", "content": "I'm planning a trip to Japan."})
        response1 = wrapper.chat(messages)
        message1 = response1.get('message', '') if isinstance(response1, dict) else response1
        messages.append({"role": "assistant", "content": message1})
        
        # Second exchange
        messages.append({"role": "user", "content": "I'm interested in visiting Tokyo and Kyoto."})
        response2 = wrapper.chat(messages)
        message2 = response2.get('message', '') if isinstance(response2, dict) else response2
        messages.append({"role": "assistant", "content": message2})
        
        # Final question referencing context
        messages.append({"role": "user", "content": "Based on what I told you, what should I pack?"})
        response = wrapper.chat(messages)
        assert response is not None
        
        message = response.get('message', '') if isinstance(response, dict) else response
        assert len(message.strip()) > 50  # Should be a detailed response
        print(f"Long conversation response: {message}")
        
        # Should reference Japan or travel context
        assert any(word in message.lower() for word in ["japan", "travel", "trip", "tokyo", "kyoto"])
    
    def test_model_validation_disabled(self):
        """Test that we can use custom models"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        # This should work even if the model isn't in the official list
        wrapper = LLMWrapper("openai", "gpt-4-experimental")
        assert wrapper.model == "gpt-4-experimental"
        print(f"Using experimental model: {wrapper.model}")
    
    @pytest.mark.integration
    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        import threading
        import time
        
        wrapper = LLMWrapper("openai")
        results = []
        
        def make_request(prompt):
            try:
                messages = [{"role": "user", "content": f"Count to {prompt}"}]
                response = wrapper.chat(messages)
                results.append(response)
            except Exception as e:
                results.append(f"Error: {e}")
        
        # Create multiple threads for concurrent requests
        threads = []
        for i in range(3):  # OpenAI can handle more concurrent requests than Gemini
            thread = threading.Thread(target=make_request, args=[i+1])
            threads.append(thread)
            thread.start()
            time.sleep(0.5)  # Shorter delay since OpenAI has higher rate limits
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) == 3
        print(f"Concurrent request results: {len([r for r in results if isinstance(r, dict)])}")
        
        # Most should succeed (OpenAI has generous rate limits)
        successful_results = []
        for r in results:
            if isinstance(r, str) and r.startswith("Error:"):
                continue
            elif isinstance(r, dict) and 'message' in r:
                successful_results.append(r)
        
        assert len(successful_results) >= 2  # At least 2 should succeed
    
    @pytest.mark.integration
    def test_different_model_sizes(self):
        """Test different OpenAI model sizes"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        
        # Test mini model (fast, cheap)
        wrapper_mini = LLMWrapper("openai", "gpt-4o-mini")
        messages = [{"role": "user", "content": "What is AI?"}]
        response_mini = wrapper_mini.chat(messages)
        assert response_mini is not None
        
        message_mini = response_mini.get('message', '') if isinstance(response_mini, dict) else response_mini
        print(f"Mini model response length: {len(message_mini)}")
        
        # Test regular model (more capable)
        time.sleep(1)  # Brief pause between requests
        wrapper_regular = LLMWrapper("openai", "gpt-4o")
        response_regular = wrapper_regular.chat(messages)
        assert response_regular is not None
        
        message_regular = response_regular.get('message', '') if isinstance(response_regular, dict) else response_regular
        print(f"Regular model response length: {len(message_regular)}")
        
        # Both should provide valid responses
        assert len(message_mini.strip()) > 10
        assert len(message_regular.strip()) > 10