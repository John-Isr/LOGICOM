import tiktoken
from typing import List, Dict, Any

# Create a single shared tokenizer instance
_tokenizer = tiktoken.get_encoding("cl100k_base")

def calculate_chat_tokens(messages: List[Dict[str, str]]) -> int:
    """
    Estimates token count for a list of messages using tiktoken.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        int: Estimated token count
    """
    num_tokens = 0
    tokens_per_message = 4  # Approximation for role/metadata
    
    for message in messages:
        num_tokens += tokens_per_message
        content = message.get("content", "")
        if content:
            num_tokens += len(_tokenizer.encode(content))
    
    num_tokens += 2  # End-of-list approximation
    return num_tokens

def calculate_string_tokens(text: str) -> int:
    """
    Calculates the number of tokens in a string.
    
    Args:
        text: The string to calculate tokens for
        
    Returns:
        int: Number of tokens in the string
    """
    if not text:
        return 0
    return len(_tokenizer.encode(text))

# Export the tokenizer for direct use if needed
get_tokenizer = lambda: _tokenizer 