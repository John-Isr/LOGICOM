import os
import time # For sleep in retry logging
from typing import List, Dict, Any
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import logging

# Direct import from project structure
from core.interfaces import LLMInterface
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type, before_sleep_log

logger = logging.getLogger(__name__)

# Exceptions to retry on for Google API Core
RETRYABLE_GOOGLE_ERRORS = (
    google_exceptions.ServiceUnavailable, 
    google_exceptions.DeadlineExceeded,
    google_exceptions.InternalServerError, 
    google_exceptions.TooManyRequests,    
    google_exceptions.ResourceExhausted,
)

# Callback function to log before sleeping (retry attempt)
def log_retry_attempt(retry_state):
    wait_time = retry_state.next_action.sleep
    logger.warning(f"Retrying Gemini call (Attempt {retry_state.attempt_number}) after error: {retry_state.outcome.exception()}. Waiting {wait_time:.2f} seconds.")

class GeminiClient(LLMInterface):
    """LLM client implementation for Google Gemini API.
    Uses the system_instruction parameter for models that support it.
    """
    def __init__(self, 
                 api_key: str | None = None, 
                 model_name: str = "gemini-1.5-flash-8b", #TODO: I don't like the fact that this is hardcoded here, consider reading from config/settings.yaml
                 system_instruction: str | None = None):
        """Initializes the Gemini client, optionally with system instructions."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not provided or found in environment variables.")
        self.model_name = model_name
        self.system_instruction = system_instruction # Store system instruction

        try:
            genai.configure(api_key=self.api_key)
            # Prepare arguments for GenerativeModel, including system_instruction if provided
            model_kwargs = {}
            if self.system_instruction:
                logger.info(f"Initializing Gemini model {self.model_name} with system instruction.")
                model_kwargs['system_instruction'] = self.system_instruction
            else:
                logger.warning(f"Initializing Gemini model {self.model_name} without system instruction.")
            
            self.model = genai.GenerativeModel(self.model_name, **model_kwargs)

        except Exception as e:
            logger.error(f"Failed to configure Google GenAI or create model: {e}", exc_info=True)
            raise

    def _convert_prompt_format(self, prompt: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Converts OpenAI-style prompt list (excluding system) to Gemini format (user, model)."""
        gemini_prompt = []
        prompt_system_message_content = None # Store system message found in prompt

        for message in prompt:
            role = message.get('role')
            content = message.get('content')
            if not content:
                continue

            if role == 'system':
                # System instruction is handled at model initialization, skip here.
                # Log a warning that it was received but won't be directly used in history.
                if content:
                     logger.warning("Gemini received a system message in the prompt. Gemini API only supports a system_instruction at init. This will be ignored in the message history.")
                     if prompt_system_message_content is None: # Store the first one found
                          prompt_system_message_content = content
                continue 
            elif role == 'user':
                gemini_prompt.append({'role': 'user', 'parts': [content]})
            elif role == 'assistant' or role == 'ai' or role == 'model':
                gemini_prompt.append({'role': 'model', 'parts': [content]})
        
        # After the loop, compare the found system message (if any) with the init one
        if prompt_system_message_content is not None and self.system_instruction is not None:
             if prompt_system_message_content == self.system_instruction:
                  logger.warning("System instruction provided during init AND a matching system message found in the prompt. Using the init instruction.")
             else:
                  logger.error("CONFLICT: System instruction provided during init differs from system message found in prompt. Using the init instruction, but check configuration.")

        # Note: Assumes alternating user/model roles in the input after system message removal.
        # TODO: Add validation or merging logic if needed.
        
        return gemini_prompt

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_random_exponential(multiplier=5, max=30),
        retry=retry_if_exception_type(RETRYABLE_GOOGLE_ERRORS),
        before_sleep=log_retry_attempt
    )
    def generate(self, prompt: List[Dict[str, str]], **kwargs) -> str:
        """Generates a response using the Google Gemini API with retry logic."""
        try:
            # Convert prompt
            gemini_prompt = self._convert_prompt_format(prompt)
            
            generation_config = {}
            if 'temperature' in kwargs: generation_config['temperature'] = kwargs['temperature']
            if 'max_tokens' in kwargs: generation_config['max_output_tokens'] = kwargs['max_tokens']
            safety_settings = kwargs.get('safety_settings')
            
            # Call generate_content using the pre-configured self.model
            response = self.model.generate_content(
                gemini_prompt, # This list now only contains user/model turns
                generation_config=genai.types.GenerationConfig(**generation_config) if generation_config else None,
                safety_settings=safety_settings
            )

            # Extract text, handling potential blocks
            if response.parts:
                 return response.text
            elif response.prompt_feedback and response.prompt_feedback.block_reason:
                 logger.warning(f"Gemini prompt blocked due to: {response.prompt_feedback.block_reason}")
                 return f"Error: Prompt blocked ({response.prompt_feedback.block_reason})" # Don't retry prompt blocks
            else:
                 # If parts is empty but no block reason, might be an issue or empty generation
                 logger.warning(f"Gemini response missing expected content or generated empty. Response: {response}")
                 return ""
        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Google API error after retries (or non-retryable): {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Google Gemini call: {e}", exc_info=True)
            raise 