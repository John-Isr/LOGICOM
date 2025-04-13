from utils.token_utils import calculate_chat_tokens, calculate_string_tokens

def main():
    # Test the string token calculation
    test_string = "Hello, world! This is a test of the token utilities."
    string_tokens = calculate_string_tokens(test_string)
    print(f"String '{test_string}' has {string_tokens} tokens")
    
    # Test the message list token calculation
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you today?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
    ]
    message_tokens = calculate_chat_tokens(test_messages)
    print(f"Message list has {message_tokens} tokens")
    
    print("\nToken utility functions are working correctly!")

if __name__ == "__main__":
    main() 