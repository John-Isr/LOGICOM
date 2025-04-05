import os
from typing import List, Dict, Any, Optional
import openai
import logging # Use logging for retry messages

# Direct import from project structure
from core.interfaces import LLMInterface

# Use tenacity for retries
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type, before_sleep_log

# Configure logging for tenacity
logger = logging.getLogger(__name__)

# Define exceptions to retry on for OpenAI
RETRYABLE_OPENAI_ERRORS = (
    openai.APIError, 
    openai.Timeout, 
    openai.APIConnectionError,
    openai.RateLimitError, # Might indicate temporary overload
    openai.APIStatusError, # Retry specifically on 5xx server errors
)

# Custom retry condition for APIStatusError (only retry 5xx)
def is_retryable_status_error(exception):
    return isinstance(exception, openai.APIStatusError) and exception.status_code >= 500

# Combined retry condition
def should_retry_openai_exception(exception):
    return isinstance(exception, RETRYABLE_OPENAI_ERRORS) or is_retryable_status_error(exception)

# Callback function to log before sleeping (retry attempt)
def log_retry_attempt(retry_state):
    wait_time = retry_state.next_action.sleep
    logger.warning(f"Retrying LLM call (Attempt {retry_state.attempt_number}) after error: {retry_state.outcome.exception()}. Waiting {wait_time:.2f} seconds.")

class OpenAIClient(LLMInterface):
    """LLM client implementation for OpenAI API. Manages system prompt internally."""
    def __init__(self, 
                 api_key: str | None = None, 
                 model_name: str = "gpt-3.5-turbo", #TODO: I don't like the fact that this is hardcoded here, consider reading from config/settings.yaml
                 system_instruction: Optional[str] = None):
        # Prioritize passed key, then env var
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided or found in environment variables.")
        self.model_name = model_name
        self.system_instruction = system_instruction
        # Initialize the OpenAI client library. 
        self.client = openai.OpenAI(api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(4), # 1 initial call + 3 retries
        wait=wait_random_exponential(multiplier=5, max=30), # Wait ~5s, then ~10s, ~20s (max 30)
        retry=should_retry_openai_exception, # Use the combined condition function
        before_sleep=log_retry_attempt # Log before waiting
    )
    def generate(self, prompt: List[Dict[str, str]], **kwargs) -> str:
        """Generates a response using the OpenAI API.
        Prioritizes system prompt from 'prompt' argument, falls back to instance's system_instruction.
        Logs a warning or error if both are provided and differ.
        """
        try:
            # Find system message in the input prompt and separate other messages
            prompt_system_message_content = None
            other_messages = []
            for msg in prompt:
                if msg.get("role") == "system":
                    if prompt_system_message_content is None:
                        prompt_system_message_content = msg.get("content")
                    else:
                        # Log if multiple system messages are found in the input prompt itself
                        logger.warning("Multiple system messages found in generate() prompt; using the first one.")
                else:
                    other_messages.append(msg)

            messages_to_send = []
            # Decide which system prompt to use and add it first
            if prompt_system_message_content is not None:
                # Prioritize prompt's system message
                if prompt_system_message_content: # Ensure content is not empty
                    messages_to_send.append({"role": "system", "content": prompt_system_message_content})
                if self.system_instruction:
                    # Compare the two system instructions/prompts
                    if prompt_system_message_content == self.system_instruction:
                         logger.warning("System instruction provided during init and a matching system message found in generate() prompt.")
                    else:
                         logger.error("CONFLICT: System instruction provided during init differs from system message in generate() prompt. Prioritizing the one from generate(), but check configuration.")
            elif self.system_instruction:
                # Fallback to init's system instruction
                messages_to_send.append({"role": "system", "content": self.system_instruction})

            # Add the rest of the messages (user/assistant)
            messages_to_send.extend(other_messages)

            # Ensure there's at least one non-system message
            if not any(msg.get('role') != 'system' for msg in messages_to_send):
                logger.error("Prompt contains only system message(s) or is empty after processing, cannot make OpenAI call.")
                return "Error: No user/assistant message in prompt."

            model = kwargs.pop('model', self.model_name)
            api_params = {
                'model': model,
                'messages': messages_to_send,
                **kwargs  
            }

            response = self.client.chat.completions.create(**api_params)
            
            if response.choices:
                message = response.choices[0].message
                if message.content:
                    return message.content.strip()
            logger.warning("OpenAI response missing expected content.")
            return "" # Return empty string if no content found
        except openai.APIError as e:
            logger.error(f"OpenAI API returned an error after retries (or non-retryable): {e}", exc_info=True)
            raise # Re-raise the final error
        except Exception as e:
            logger.error(f"An unexpected error occurred during OpenAI call: {e}", exc_info=True)
            raise # Re-raise other unexpected errors 