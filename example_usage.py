from gen_wrapper.llm_wrapper import LLMWrapper, LLMWrapperError
from dotenv import load_dotenv
load_dotenv()
def example_usage():
    """Demonstrate how to use the LLM wrapper"""
    
    # List available providers
    print("Available providers:", LLMWrapper.list_providers())
    
    try:
        # Initialize wrapper
        wrapper = LLMWrapper("openai", "gpt-4o")
        
        # Show provider info
        print("\nProvider Info:")
        info = wrapper.get_provider_info()
        print(f"Provider: {info['provider']}")
        print(f"Model: {info['model']}")
        print(f"Base URL: {info['base_url']}")
        print(f"LangChain Support: {info['langchain_support']}")
        
        # Simple chat
        response = wrapper.simple_chat("What is the capital of France?")
        print(f"\nSimple chat response: {response}")
        
        # Advanced chat with multiple messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ]
        
        response = wrapper.chat(messages, temperature=0.7, max_tokens=150)
        print(f"\nAdvanced chat response: {response['message']}")
        
    except LLMWrapperError as e:
        print(f"LLM Wrapper Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def test_all_providers():
    """Test all available providers"""
    providers = LLMWrapper.list_providers()
    
    for provider in providers:
        print(f"\nTesting {provider}:")
        try:
            models = LLMWrapper.list_models(provider)
            print(f"  Available models: {', '.join(models)}")
            
            # Try to initialize (will fail if API key missing)
            wrapper = LLMWrapper(provider)
            info = wrapper.get_provider_info()
            print(f"  Default model: {info['model']}")
            print(f"  Status: Ready")
            
        except LLMWrapperError as e:
            print(f"  Status: {e}")

if __name__ == "__main__":
    print("=== Basic Usage Example ===")
    example_usage()
    
    # print("\n=== Testing All Providers ===")
    # test_all_providers()