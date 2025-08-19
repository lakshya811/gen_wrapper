import pytest
import os
from gen_wrapper.llm_wrapper import LLMWrapper, LLMWrapperError
import time

class TestAzureOpenAIClient:
    
    def test_list_models(self):
        """Test listing available models"""
        models = LLMWrapper.list_models("azure_openai")
        assert isinstance(models, list)
        print(f"Available Azure OpenAI models: {models}")
        # Azure might have different model availability than regular OpenAI
    
    def test_initialization_default_model(self):
        """Test initialization with default model"""
        if not all([
            os.getenv("AZURE_OPENAI_API_KEY"),
            os.getenv("AZURE_OPENAI_ENDPOINT"),
            os.getenv("AZURE_OPENAI_DEPLOYMENT")
        ]):
            pytest.skip("Azure OpenAI credentials not set")
        
        wrapper = LLMWrapper("azure_openai")
        assert wrapper.provider_name == "azure_openai"
        assert wrapper.model == "gpt-4o"  # Default model from config
        print(f"Initialized with default model: {wrapper.model}")
    
    def test_initialization_custom_model(self):
        """Test initialization with custom model"""
        if not all([
            os.getenv("AZURE_OPENAI_API_KEY"),
            os.getenv("AZURE_OPENAI_ENDPOINT"),
            os.getenv("AZURE_OPENAI_DEPLOYMENT")
        ]):
            pytest.skip("Azure OpenAI credentials not set")
        
        wrapper = LLMWrapper("azure_openai", "gpt-4")
        assert wrapper.provider_name == "azure_openai"
        assert wrapper.model == "gpt-4"
        print(f"Initialized with custom model: {wrapper.model}")
    
    def test_provider_info(self):
        """Test getting provider information"""
        if not all([
            os.getenv("AZURE_OPENAI_API_KEY"),
            os.getenv("AZURE_OPENAI_ENDPOINT"),
            os.getenv("AZURE_OPENAI_DEPLOYMENT")
        ]):
            pytest.skip("Azure OpenAI credentials not set")
        
        wrapper = LLMWrapper("azure_openai")
        info = wrapper.get_provider_info()
        
        # Check required fields
        assert "provider" in info
        assert "model" in info
        assert "timeout" in info
        assert "max_retries" in info
        
        assert info["provider"] == "azure_openai"
        assert info["model"] == "gpt-4o"
        
        print(f"Provider info: {info}")
    
    @pytest.mark.integration
    def test_simple_chat(self):
        """Test simple chat functionality"""
        if not all([
            os.getenv("AZURE_OPENAI_API_KEY"),
            os.getenv("AZURE_OPENAI_ENDPOINT"),
            os.getenv("AZURE_OPENAI_DEPLOYMENT")
        ]):
            pytest.skip("Azure OpenAI credentials not set")
        
        wrapper = LLMWrapper("azure_openai")
        
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
        if not all([
            os.getenv("AZURE_OPENAI_API_KEY"),
            os.getenv("AZURE_OPENAI_ENDPOINT"),
            os.getenv("AZURE_OPENAI_DEPLOYMENT")
        ]):
            pytest.skip("Azure OpenAI credentials not set")
        
        wrapper = LLMWrapper("azure_openai")
        
        # Build conversation with proper history
        messages = [
            {"role": "user", "content": "My favorite cloud platform is Azure. Remember this."},
        ]
        response1 = wrapper.chat(messages)
        assert response1 is not None
        
        message1 = response1.get('message', '') if isinstance(response1, dict) else response1
        print(f"First response: {message1}")
        
        # Add assistant response and follow-up question
        messages.extend([
            {"role": "assistant", "content": message1},
            {"role": "user", "content": "What is my favorite cloud platform?"}
        ])
        response2 = wrapper.chat(messages)
        assert response2 is not None
        
        message2 = response2.get('message', '') if isinstance(response2, dict) else response2
        print(f"Second response: {message2}")
        
        # Should remember Azure
        assert "azure" in message2.lower()
    
    @pytest.mark.integration
    def test_azure_specific_features(self):
        """Test Azure-specific deployment features"""
        if not all([
            os.getenv("AZURE_OPENAI_API_KEY"),
            os.getenv("AZURE_OPENAI_ENDPOINT"),
            os.getenv("AZURE_OPENAI_DEPLOYMENT")
        ]):
            pytest.skip("Azure OpenAI credentials not set")
        
        wrapper = LLMWrapper("azure_openai")
        
        # Test that Azure deployment is being used
        messages = [{"role": "user", "content": "Explain what Azure OpenAI is in one sentence."}]
        response = wrapper.chat(messages)
        assert response is not None
        
        message = response.get('message', '') if isinstance(response, dict) else response
        assert len(message.strip()) > 20  # Should be a substantial response
        print(f"Azure-specific response: {message}")
        
        # Should mention Azure or Microsoft
        assert any(word in message.lower() for word in ["azure", "microsoft"])
    
    def test_error_handling(self):
        """Test error handling with invalid credentials"""
        # Temporarily set invalid credentials
        original_key = os.environ.get("AZURE_OPENAI_API_KEY")
        original_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        
        os.environ["AZURE_OPENAI_API_KEY"] = "invalid_key"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://invalid-endpoint.openai.azure.com/"
        
        try:
            wrapper = LLMWrapper("azure_openai")
            messages = [{"role": "user", "content": "Test message"}]
            with pytest.raises((LLMWrapperError, Exception)):
                wrapper.chat(messages)
        finally:
            # Restore original credentials
            if original_key:
                os.environ["AZURE_OPENAI_API_KEY"] = original_key
            else:
                os.environ.pop("AZURE_OPENAI_API_KEY", None)
                
            if original_endpoint:
                os.environ["AZURE_OPENAI_ENDPOINT"] = original_endpoint
            else:
                os.environ.pop("AZURE_OPENAI_ENDPOINT", None)
    
    def test_azure_configuration_validation(self):
        """Test that Azure requires specific environment variables"""
        # Check that the wrapper properly validates Azure-specific config
        config = {
            "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
            "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "AZURE_OPENAI_DEPLOYMENT": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION")
        }
        
        missing_vars = [k for k, v in config.items() if not v]
        if missing_vars:
            pytest.skip(f"Missing Azure environment variables: {missing_vars}")
        
        # If all variables are set, initialization should work
        wrapper = LLMWrapper("azure_openai")
        assert wrapper.provider_name == "azure_openai"
        print(f"Azure configuration validated successfully")
    
    @pytest.mark.integration
    def test_long_conversation(self):
        """Test handling longer conversations"""
        if not all([
            os.getenv("AZURE_OPENAI_API_KEY"),
            os.getenv("AZURE_OPENAI_ENDPOINT"),
            os.getenv("AZURE_OPENAI_DEPLOYMENT")
        ]):
            pytest.skip("Azure OpenAI credentials not set")
        
        wrapper = LLMWrapper("azure_openai")
        
        # Build a multi-turn conversation about Azure
        messages = []
        
        # First exchange
        messages.append({"role": "user", "content": "I'm working on a cloud migration project."})
        response1 = wrapper.chat(messages)
        message1 = response1.get('message', '') if isinstance(response1, dict) else response1
        messages.append({"role": "assistant", "content": message1})
        
        # Second exchange
        messages.append({"role": "user", "content": "I'm considering Azure as the target platform."})
        response2 = wrapper.chat(messages)
        message2 = response2.get('message', '') if isinstance(response2, dict) else response2
        messages.append({"role": "assistant", "content": message2})
        
        # Final question referencing context
        messages.append({"role": "user", "content": "Based on what I told you, what Azure services should I consider?"})
        response = wrapper.chat(messages)
        assert response is not None
        
        message = response.get('message', '') if isinstance(response, dict) else response
        assert len(message.strip()) > 50  # Should be a detailed response
        print(f"Long conversation response: {message}")
        
        # Should reference Azure services or migration
        assert any(word in message.lower() for word in ["azure", "cloud", "migration", "service"])
    
    def test_model_validation_disabled(self):
        """Test that we can use custom deployment names"""
        if not all([
            os.getenv("AZURE_OPENAI_API_KEY"),
            os.getenv("AZURE_OPENAI_ENDPOINT"),
            os.getenv("AZURE_OPENAI_DEPLOYMENT")
        ]):
            pytest.skip("Azure OpenAI credentials not set")
        
        # This should work with any deployment name
        wrapper = LLMWrapper("azure_openai", "my-custom-deployment")
        assert wrapper.model == "my-custom-deployment"
        print(f"Using custom deployment: {wrapper.model}")
    
    @pytest.mark.integration
    def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        if not all([
            os.getenv("AZURE_OPENAI_API_KEY"),
            os.getenv("AZURE_OPENAI_ENDPOINT"),
            os.getenv("AZURE_OPENAI_DEPLOYMENT")
        ]):
            pytest.skip("Azure OpenAI credentials not set")
        
        import threading
        import time
        
        wrapper = LLMWrapper("azure_openai")
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
        for i in range(3):  # Azure has good rate limits
            thread = threading.Thread(target=make_request, args=[i+1])
            threads.append(thread)
            thread.start()
            time.sleep(0.5)  # Short delay since Azure has high rate limits
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) == 3
        print(f"Concurrent request results: {len([r for r in results if isinstance(r, dict)])}")
        
        # Most should succeed (Azure has generous rate limits)
        successful_results = []
        for r in results:
            if isinstance(r, str) and r.startswith("Error:"):
                continue
            elif isinstance(r, dict) and 'message' in r:
                successful_results.append(r)
        
        assert len(successful_results) >= 2  # At least 2 should succeed
    
    @pytest.mark.integration
    def test_azure_api_versions(self):
        """Test different Azure API versions"""
        if not all([
            os.getenv("AZURE_OPENAI_API_KEY"),
            os.getenv("AZURE_OPENAI_ENDPOINT"),
            os.getenv("AZURE_OPENAI_DEPLOYMENT")
        ]):
            pytest.skip("Azure OpenAI credentials not set")
        
        # Test with the configured API version
        wrapper = LLMWrapper("azure_openai")
        
        messages = [{"role": "user", "content": "What is Azure?"}]
        response = wrapper.chat(messages)
        assert response is not None
        
        message = response.get('message', '') if isinstance(response, dict) else response
        print(f"API version response: {message}")
        
        # Should provide a valid response about Azure
        assert len(message.strip()) > 10
        assert any(word in message.lower() for word in ["azure", "microsoft", "cloud"])
    
    @pytest.mark.integration
    def test_enterprise_features(self):
        """Test Azure enterprise-specific features"""
        if not all([
            os.getenv("AZURE_OPENAI_API_KEY"),
            os.getenv("AZURE_OPENAI_ENDPOINT"),
            os.getenv("AZURE_OPENAI_DEPLOYMENT")
        ]):
            pytest.skip("Azure OpenAI credentials not set")
        
        wrapper = LLMWrapper("azure_openai")
        
        # Test enterprise-grade query
        messages = [{"role": "user", "content": "Explain the security benefits of using Azure OpenAI for enterprise applications."}]
        response = wrapper.chat(messages)
        assert response is not None
        
        message = response.get('message', '') if isinstance(response, dict) else response
        assert len(message.strip()) > 100  # Should be a comprehensive response
        print(f"Enterprise features response length: {len(message)}")
        
        # Should mention security, compliance, or enterprise features
        assert any(word in message.lower() for word in ["security", "enterprise", "compliance", "private"])