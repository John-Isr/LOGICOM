import logging
from typing import List, Dict, Any, Optional
import ollama

# Direct import from project structure
from core.interfaces import LLMInterface

logger = logging.getLogger(__name__)

class OllamaClient(LLMInterface):
    """LLM client implementation for Ollama API using the official library."""
    def __init__(self, 
                 model: str, 
                 base_url: str | None = None, # Optional: Defaults to http://localhost:11434 if None
                 system: Optional[str] = None):
        """Initializes the Ollama client.

        Args:
            model: The name of the Ollama model to use (required).
            base_url: The base URL of the Ollama server (optional).
            system: The system prompt/instruction to use (optional).
        """
        if not model:
            raise ValueError("Ollama client requires a 'model' name.")
        
        self.model = model
        self.system = system # Store system prompt
        
        # Initialize the ollama client targeting the specified host or default
        # The 'host' parameter in ollama.Client corresponds to the base_url
        self.client = ollama.Client(host=base_url) 
        logger.info(f"Initialized OllamaClient for model '{self.model}' at host '{base_url or 'default'}. System prompt {'set' if self.system else 'not set'}.\"")

    def _prepare_messages(self, prompt: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Prepares messages for the Ollama API. 
           Handles system prompt precedence and conversion.
        """
        messages = []
        prompt_system_content = None
        
        # Extract user/assistant messages and find first system message from prompt
        for msg in prompt:
            role = msg.get('role')
            content = msg.get('content')
            if not content:
                continue

            if role == 'system':
                if prompt_system_content is None:
                    prompt_system_content = content
                # Don't add system message to the list here, handled separately
            elif role == 'user':
                messages.append({'role': 'user', 'content': content})
            elif role == 'assistant' or role == 'ai' or role == 'model':
                messages.append({'role': 'assistant', 'content': content}) # Ollama uses 'assistant'
        
        # Determine the effective system prompt
        effective_system_prompt = self.system # Start with the one from init
        if prompt_system_content is not None:
            if effective_system_prompt is not None and prompt_system_content != effective_system_prompt:
                 logger.error(f"CONFLICT: System prompt from init ('{effective_system_prompt[:50]}...') differs from prompt ('{prompt_system_content[:50]}...'). Prioritizing prompt's system message.\"")
            effective_system_prompt = prompt_system_content # Prioritize prompt's system message
        
        # Prepend the effective system prompt if it exists
        if effective_system_prompt:
             # Check if the first message is already a system message (shouldn't happen with current logic, but safe)
             if not messages or messages[0].get('role') != 'system':
                  # Insert system message at the beginning if one exists
                  messages.insert(0, {'role': 'system', 'content': effective_system_prompt})
             elif messages: # If first message IS system (e.g., if logic changes), update it
                  messages[0]['content'] = effective_system_prompt
                  
        if not any(m['role'] == 'user' for m in messages):
             logger.error("Ollama messages prepared without a user role after processing.\"")
             # TODO:Consider how to handle this - raise error or return error string?
             # Returning error for consistency with previous pattern
             return {"error": "Error: No user message found in prompt."}

        return messages

    def generate(self, prompt: List[Dict[str, str]], **kwargs) -> str:
        """Generates a response using the Ollama API via the ollama library."""
        
        # Prepare messages, handle potential error signal from preparation
        messages = self._prepare_messages(prompt)
        if isinstance(messages, dict) and "error" in messages:
             return messages["error"]

        # Extract known options for Ollama, pass others in 'options' dict
        options = {}
        if 'temperature' in kwargs: options['temperature'] = kwargs['temperature']
        if 'max_tokens' in kwargs: options['num_predict'] = kwargs['max_tokens'] # Map max_tokens to num_predict
        # Add any other desired Ollama options from kwargs here
        
        # Prepare parameters for the ollama.chat call
        request_params = {
            'model': self.model,
            'messages': messages,
            'stream': False, # Keep it simple, no streaming for now
            'options': options if options else None,
            'keep_alive': kwargs.get('keep_alive') # Pass keep_alive if provided
        }
        # Filter out None values for cleaner API call
        request_params = {k: v for k, v in request_params.items() if v is not None}

        logger.debug(f"Ollama API Request: {request_params}\"")

        try:
            # Use the client's chat method
            response = self.client.chat(**request_params)
            
            logger.debug(f"Ollama API Response: {response}\"")

            # Extract the content from the response structure
            if response and isinstance(response, dict) and response.get('message'):
                content = response['message'].get('content')
                if content:
                    return content.strip()
                else:
                     logger.warning(f"Ollama response message content is empty. Response: {response}\"")
                     return ""
            else:
                 logger.warning(f"Ollama response missing expected structure or message key. Response: {response}\"")
                 return ""

        except Exception as e:
            # Catch exceptions from the ollama library (e.g., connection errors, API errors)
            logger.error(f"Error during Ollama API call: {e}", exc_info=True)
            # Decide how to handle - returning error string for now
            return f"Error: Ollama API call failed - {e}"

