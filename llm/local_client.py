import os
import json
from typing import List, Dict, Any, Optional
import requests # Dependency: Add 'requests' to requirements.txt
import logging
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type, before_sleep_log, RetryError

# Direct import from project structure
from core.interfaces import LLMInterface

# Configure logging for tenacity
logger = logging.getLogger(__name__)

# Define exceptions to retry on for requests library
RETRYABLE_REQUESTS_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError # Can happen with streaming-like issues
)

# Custom retry condition for HTTP status codes (retry 5xx)
def is_retryable_http_error(exception):
    return (
        isinstance(exception, requests.exceptions.HTTPError) and
        exception.response is not None and 
        exception.response.status_code >= 500 # Retry on server errors
    )

# Combined retry condition
def should_retry_local_exception(exception):
    return isinstance(exception, RETRYABLE_REQUESTS_ERRORS) or is_retryable_http_error(exception)

# Callback function to log before sleeping (retry attempt)
def log_retry_attempt(retry_state):
    wait_time = retry_state.next_action.sleep
    logger.warning(f"Retrying local LLM call (Attempt {retry_state.attempt_number}) after error: {retry_state.outcome.exception()}. Waiting {wait_time:.2f} seconds.")


class BaseLocalClient(LLMInterface):
    """Base class for clients interacting with local LLM APIs."""
    def __init__(self, base_url: str, model_name: str | None = None, api_key: str | None = None, system_instruction: Optional[str] = None):
        if not base_url:
            raise ValueError("base_url must be provided for local LLM clients.")
        self.base_url = base_url.rstrip('/') # Ensure no trailing slash
        self.model_name = model_name
        self.api_key = api_key # May or may not be used by subclass
        self.system_instruction = system_instruction # Store instruction consistently
        self._prepare_session()

    def _prepare_session(self):
        """Prepares the requests session, adding auth if needed."""
        self.session = requests.Session()
        # Add common headers
        self.session.headers.update({"Content-Type": "application/json"})
        # Add API key header if provided (common for OpenAI-compatible APIs)
        if self.api_key:
             self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def generate(self, prompt: List[Dict[str, str]], **kwargs) -> str:
        """Abstract method to be implemented by specific local clients."""
        raise NotImplementedError("Subclasses must implement the generate method.")

    def _prepare_messages_for_api(self, prompt: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Helper to prepend system prompt and filter input messages.
           Handles system instruction prioritization and comparison.
           Can be overridden by subclasses if different format needed.
        """
        # Find system message in the input prompt and separate other messages
        prompt_system_message_content = None
        other_messages = []
        for msg in prompt:
            if msg.get("role") == "system":
                if prompt_system_message_content is None:
                    prompt_system_message_content = msg.get("content")
                # else: No warning if multiple system messages are in the prompt, just use the first.
            else:
                other_messages.append(msg)

        messages_to_send = []
        # Decide which system prompt to use and add it first (based on internal self.system_instruction which came from init's system_instruction)
        if prompt_system_message_content is not None:
            # Prioritize prompt's system message
            if prompt_system_message_content:
                messages_to_send.append({"role": "system", "content": prompt_system_message_content})
            if self.system_instruction: # Check against the internally stored instruction
                # Compare the two system instructions/prompts
                if prompt_system_message_content == self.system_instruction:
                    logger.warning("System instruction provided during init AND a matching system message found in generate() prompt (and they match). Prioritizing the one from generate().")
                else:
                    logger.error("CONFLICT: System instruction provided during init differs from system message in generate() prompt. Prioritizing the one from generate(), but check configuration.")
        elif self.system_instruction:
            # Fallback to init's system instruction
            messages_to_send.append({"role": "system", "content": self.system_instruction})

        # Add the rest of the messages (user/assistant)
        messages_to_send.extend(other_messages)
        
        # Ensure there's at least one non-system message if system prompt was the only thing
        if not any(msg['role'] != 'system' for msg in messages_to_send):
            # Returning error message might be better than raising here, similar to OpenAIClient
            logger.error("Prompt contains only system message(s), cannot make API call.")
            # Raise an error or return a specific signal? Returning error string for now.
            return {"error": "Error: No user/assistant message in prompt."} # Return error signal
            
        return messages_to_send


class OllamaClient(BaseLocalClient):
    """Client for interacting with an Ollama server API."""
    # Note: Ollama typically doesn't use API keys by default

    @retry(
        stop=stop_after_attempt(4), 
        wait=wait_random_exponential(multiplier=3, max=30),
        retry=should_retry_local_exception,
        before_sleep=log_retry_attempt
    )
    def generate(self, prompt: List[Dict[str, str]], **kwargs) -> str:
        """Generates response using Ollama /api/chat endpoint with retry logic."""
        if not self.model_name:
             raise ValueError("Ollama client requires a 'model_name' to be specified.")
             
        api_url = f"{self.base_url}/api/chat"
        
        # Prepare messages using base class helper
        messages_to_send = self._prepare_messages_for_api(prompt)
        if isinstance(messages_to_send, dict) and "error" in messages_to_send:
            return messages_to_send["error"]
            
        # Convert roles for Ollama ('assistant')
        ollama_messages = []
        for msg in messages_to_send:
            role = msg.get('role')
            # Map system role to system for Ollama if needed (check API)
            # Assuming system handled via prompt, map ai/model to assistant
            if role == 'ai' or role == 'model':
                 role = 'assistant'
            elif role == 'system': # Pass system prompt role through if present
                 pass 
            # Keep user role as is
            ollama_messages.append({'role': role, 'content': msg.get('content')})

        options = {}
        if 'temperature' in kwargs: options['temperature'] = kwargs['temperature']
        if 'max_tokens' in kwargs: options['num_predict'] = kwargs['max_tokens'] 
        
        payload = {
            "model": self.model_name,
            "messages": ollama_messages, # Use Ollama formatted messages
            "stream": False, 
            "options": options
        }

        try:
            response = self.session.post(api_url, json=payload)
            response.raise_for_status() 
            response_data = response.json()
            
            if response_data.get('message') and response_data['message'].get('content'):
                return response_data['message']['content'].strip()
            else:
                 logger.warning(f"Ollama response missing expected message content. Response: {response_data}")
                 return ""
                 
        except RetryError as e:
            logger.error(f"Ollama call failed after multiple retries: {e}", exc_info=True)
            raise 
        except requests.exceptions.RequestException as e:
            logger.error(f"Non-retryable error calling Ollama API at {api_url}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during Ollama call: {e}", exc_info=True)
            raise

class GenericAPIClient(BaseLocalClient):
    """Client for local LLMs exposing an OpenAI-compatible /v1/chat/completions endpoint."""

    @retry(
        stop=stop_after_attempt(4), 
        wait=wait_random_exponential(multiplier=3, max=30),
        retry=should_retry_local_exception,
        before_sleep=log_retry_attempt
    )
    def generate(self, prompt: List[Dict[str, str]], **kwargs) -> str:
        """Generates response using OpenAI-compatible /v1/chat/completions endpoint with retry logic."""
        api_url = f"{self.base_url}/v1/chat/completions"
        
        model = kwargs.pop('model', self.model_name)
        if not model:
             logger.warning("No model name specified for generic API call.")

        # Prepare messages using base class helper
        messages_to_send = self._prepare_messages_for_api(prompt)
        if isinstance(messages_to_send, dict) and "error" in messages_to_send:
            return messages_to_send["error"]
            
        payload = {
            "model": model,
            "messages": messages_to_send, # Use prepared messages
            "stream": False,
            **kwargs # Pass other OpenAI-compatible params like temperature, max_tokens, stop
        }

        try:
            response = self.session.post(api_url, json=payload)
            response.raise_for_status() # raise_for_status will be caught by tenacity for 5xx errors
            response_data = response.json()

            # Extract message content (following OpenAI structure)
            if response_data.get('choices'):
                message = response_data['choices'][0].get('message')
                if message and message.get('content'):
                    return message['content'].strip()
            
            logger.warning(f"OpenAI-compatible response missing expected content. Response: {response_data}")
            return "" # Return empty if no content found

        except RetryError as e: # Catch tenacity's RetryError
            logger.error(f"Generic API call failed after multiple retries: {e}", exc_info=True)
            raise # Re-raise the final error
        except requests.exceptions.RequestException as e:
            # Log non-retryable request errors (like 4xx)
            logger.error(f"Non-retryable error calling Generic API at {api_url}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during Generic API call: {e}", exc_info=True)
            raise 