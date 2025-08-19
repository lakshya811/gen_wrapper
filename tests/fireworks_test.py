import pytest
import os
from gen_wrapper.llm_wrapper import LLMWrapper, LLMWrapperError
import time

class TestFireworksClient:
    
    def test_list_models(self):
        """Test listing available models"""
        models = LLMWrapper.list_models("fireworks")
        assert isinstance(models, list)
        assert len(models) > 0
        # Check for common Fireworks models
        model_names = [m.lower() for m in models]
        assert any("llama" in m for m in model_names)
        print(f"Available Fireworks models: {models[:5]}...")  # Show first 5
    
    def test_initialization_default_model(self):
        """Test initialization with default model"""
        if not os.getenv("FIREWORKS_API_KEY"):
            pytest.skip("FIREWORKS_API_KEY not set")
        
        wrapper = LLMWrapper("fireworks")
        assert wrapper.provider_name == "fireworks"
        assert wrapper.model == "accounts/fireworks/models/llama-v3p1-8b-instruct"  # Default model from config
        print(f"Initialized with default model: {wrapper.model}")
    
    def test_initialization_custom_model(self):
        """Test initialization with custom model"""
        if not os.getenv("FIREWORKS_API_KEY"):
            pytest.skip("FIREWORKS_API_KEY not set")
        
        wrapper = LLMWrapper("fireworks", "accounts/fireworks/models/llama-v3p1-70b-instruct")
        assert wrapper.provider_name == "fireworks"
        assert wrapper.model == "accounts/fireworks/models/llama-v3p1-70b-instruct"
        print(f"Initialized with custom model: {wrapper.model}")
    
    def test_provider_info(self):
        """Test getting provider information"""
        if not os.getenv("FIREWORKS_API_KEY"):
            pytest.skip("FIREWORKS_API_KEY not set")
        
        wrapper = LLMWrapper("fireworks")
        info = wrapper.get_provider_info()
        
        # Check required fields
        assert "provider" in info
        assert "model" in info
        assert "base_url" in info
        assert "timeout" in info
        assert "max_retries" in info
        
        assert info["provider"] == "fireworks"
        assert info["model"] == "accounts/fireworks/models/llama-v3p1-8b-instruct"
        assert "fireworks.ai" in info["base_url"]
        
        print(f"Provider info: {info}")
    
    @pytest.mark.integration
    def test_simple_chat(self):
        """Test simple chat functionality"""
        if not os.getenv("FIREWORKS_API_KEY"):
            pytest.skip("FIREWORKS_API_KEY not set")
        
        wrapper = LLMWrapper("fireworks")
        
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
        if not os.getenv("FIREWORKS_API_KEY"):
            pytest.skip("FIREWORKS_API_KEY not set")
        
        wrapper = LLMWrapper("fireworks")
        
        # Build conversation with proper history
        messages = [
            {"role": "user", "content": "My favorite programming language is Python. Remember this."},
        ]
        response1 = wrapper.chat(messages)
        assert response1 is not None
        
        message1 = response1.get('message', '') if isinstance(response1, dict) else response1
        print(f"First response: {message1}")
        
        # Add assistant response and follow-up question
        messages.extend([
            {"role": "assistant", "content": message1},
            {"role": "user", "content": "What is my favorite programming language?"}
        ])
        response2 = wrapper.chat(messages)
        assert response2 is not None
        
        message2 = response2.get('message', '') if isinstance(response2, dict) else response2
        print(f"Second response: {message2}")
        
        # Should remember Python
        assert "python" in message2.lower()
    
    @pytest.mark.integration
    def test_deepseek_via_fireworks(self):
        """Test DeepSeek model via Fireworks"""
        if not os.getenv("FIREWORKS_API_KEY"):
            pytest.skip("FIREWORKS_API_KEY not set")
        
        # Check if DeepSeek models are available
        models = LLMWrapper.list_models("fireworks")
        deepseek_models = [m for m in models if "deepseek" in m.lower()]
        
        if not deepseek_models:
            pytest.skip("DeepSeek models not available on Fireworks")
        
        wrapper = LLMWrapper("fireworks", deepseek_models[0])
        
        messages = [{"role": "user", "content": "Explain machine learning in one sentence."}]
        response = wrapper.chat(messages)
        assert response is not None
        
        message = response.get('message', '') if isinstance(response, dict) else response
        assert len(message.strip()) > 20  # Should be a substantial response
        print(f"DeepSeek via Fireworks response: {message}")
    
    def test_error_handling(self):
        """Test error handling with invalid API key"""
        # Temporarily set invalid API key
        original_key = os.environ.get("FIREWORKS_API_KEY")
        os.environ["FIREWORKS_API_KEY"] = "fw_invalid_key_for_testing"
        
        try:
            wrapper = LLMWrapper("fireworks")
            messages = [{"role": "user", "content": "Test message"}]
            with pytest.raises((LLMWrapperError, Exception)):
                wrapper.chat(messages)
        finally:
            # Restore original key
            if original_key:
                os.environ["FIREWORKS_API_KEY"] = original_key
            else:
                os.environ.pop("FIREWORKS_API_KEY", None)
    
    def test_multiple_models_available(self):
        """Test that Fireworks has multiple model options"""
        models = LLMWrapper.list_models("fireworks")
        
        # Fireworks should have several models
        assert len(models) > 5
        
        # Check for specific model families
        llama_models = [m for m in models if "llama" in m.lower()]
        assert len(llama_models) > 0
        print(f"Llama models: {llama_models}")
        print(f"Total models available: {len(models)}")
    
    @pytest.mark.integration
    def test_long_conversation(self):
        """Test handling longer conversations"""
        if not os.getenv("FIREWORKS_API_KEY"):
            pytest.skip("FIREWORKS_API_KEY not set")
        
        wrapper = LLMWrapper("fireworks")
        
        # Build a multi-turn conversation
        messages = []
        
        # First exchange
        messages.append({"role": "user", "content": "I'm learning about AI development."})
        response1 = wrapper.chat(messages)
        message1 = response1.get('message', '') if isinstance(response1, dict) else response1
        messages.append({"role": "assistant", "content": message1})
        
        # Second exchange  
        messages.append({"role": "user", "content": "I'm particularly interested in LLMs and transformers."})
        response2 = wrapper.chat(messages)
        message2 = response2.get('message', '') if isinstance(response2, dict) else response2
        messages.append({"role": "assistant", "content": message2})
        
        # Final question referencing context
        messages.append({"role": "user", "content": "Based on what I told you, what should I study first?"})
        response = wrapper.chat(messages)
        assert response is not None
        
        message = response.get('message', '') if isinstance(response, dict) else response
        assert len(message.strip()) > 50  # Should be a detailed response
        print(f"Long conversation response: {message}")
        
        # Should reference AI, learning, or related context
        assert any(word in message.lower() for word in ["ai", "learn", "study", "transformer", "llm"])
    
    def test_model_validation_disabled(self):
        """Test that we can use custom models"""
        if not os.getenv("FIREWORKS_API_KEY"):
            pytest.skip("FIREWORKS_API_KEY not set")
        
        # This should work even if the model isn't in the official list
        wrapper = LLMWrapper("fireworks", "accounts/fireworks/models/experimental-model")
        assert wrapper.model == "accounts/fireworks/models/experimental-model"
        print(f"Using experimental model: {wrapper.model}")
    
    @pytest.mark.integration
    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        if not os.getenv("FIREWORKS_API_KEY"):
            pytest.skip("FIREWORKS_API_KEY not set")
        
        import threading
        import time
        
        wrapper = LLMWrapper("fireworks")
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
        for i in range(3):  # Fireworks has good rate limits
            thread = threading.Thread(target=make_request, args=[i+1])
            threads.append(thread)
            thread.start()
            time.sleep(1)  # Moderate delay for Fireworks
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) == 3
        print(f"Concurrent request results: {len([r for r in results if isinstance(r, dict)])}")
        
        # Most should succeed
        successful_results = []
        for r in results:
            if isinstance(r, str) and r.startswith("Error:"):
                continue
            elif isinstance(r, dict) and 'message' in r:
                successful_results.append(r)
        
        assert len(successful_results) >= 2  # At least 2 should succeed
    
    @pytest.mark.integration
    def test_different_llama_models(self):
        """Test different Llama model sizes on Fireworks"""
        if not os.getenv("FIREWORKS_API_KEY"):
            pytest.skip("FIREWORKS_API_KEY not set")
        
        # Test smaller model (8B)
        wrapper_8b = LLMWrapper("fireworks", "accounts/fireworks/models/llama-v3p1-8b-instruct")
        messages = [{"role": "user", "content": "What is machine learning?"}]
        response_8b = wrapper_8b.chat(messages)
        assert response_8b is not None
        
        message_8b = response_8b.get('message', '') if isinstance(response_8b, dict) else response_8b
        print(f"8B model response length: {len(message_8b)}")
        
        # Check if 70B model is available
        models = LLMWrapper.list_models("fireworks")
        llama_70b_models = [m for m in models if "70b" in m.lower() and "llama" in m.lower()]
        
        if llama_70b_models:
            try:
                # Test larger model (70B) if available
                time.sleep(5)  # Longer pause to avoid rate limits
                wrapper_70b = LLMWrapper("fireworks", llama_70b_models[0])
                response_70b = wrapper_70b.chat(messages)
                assert response_70b is not None
                
                message_70b = response_70b.get('message', '') if isinstance(response_70b, dict) else response_70b
                print(f"70B model response length: {len(message_70b)}")
                
                # Both should provide valid responses
                assert len(message_70b.strip()) > 10
            except LLMWrapperError as e:
                if "429" in str(e):
                    pytest.skip("Rate limit exceeded for 70B model")
                else:
                    raise
        
        # 8B model should always work
        assert len(message_8b.strip()) > 10
    
    @pytest.mark.integration 
    def test_code_generation(self):
        """Test code generation capabilities"""
        if not os.getenv("FIREWORKS_API_KEY"):
            pytest.skip("FIREWORKS_API_KEY not set")
        
        try:
            time.sleep(3)  # Add delay before test to avoid rate limits
            wrapper = LLMWrapper("fireworks")
            
            messages = [{"role": "user", "content": "Write a simple Python function to add two numbers. Just the code, no explanation."}]
            response = wrapper.chat(messages)
            assert response is not None
            
            message = response.get('message', '') if isinstance(response, dict) else response
            print(f"Code generation response: {message}")
            
            # Should contain Python-like code
            assert any(keyword in message.lower() for keyword in ["def", "return", "python", "+"])
            assert len(message.strip()) > 20
        except LLMWrapperError as e:
            if "429" in str(e):
                pytest.skip("Rate limit exceeded for code generation test")
            else:
                raise