import os
from typing import List, Dict, Any
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import logging

# Direct import from project structure
from core.interfaces import LLMInterface

logger = logging.getLogger(__name__)


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
        self.system_instruction = system_instruction 

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
        prompt_system_message_content = None

        for message in prompt:
            role = message.get('role')
            content = message.get('content')
            if not content:
                continue

            if role == 'system':
                # System instruction is handled at model initialization, skip here.
                # Log a warning that it was received but won't be directly used.
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


    def generate(self, prompt: List[Dict[str, str]], **kwargs) -> str:
        """Generates a response using the Google Gemini API with retry logic."""
        # Convert prompt
        gemini_prompt = self._convert_prompt_format(prompt)
        
        generation_config = {}
        if 'temperature' in kwargs: generation_config['temperature'] = kwargs['temperature']
        if 'max_tokens' in kwargs: generation_config['max_output_tokens'] = kwargs['max_tokens']
        safety_settings = kwargs.get('safety_settings')

        # Log request details at DEBUG level
        logger.debug(f"Gemini API Request: prompt={gemini_prompt}, config={generation_config}, safety={safety_settings}")
        
        # Call generate_content using the pre-configured self.model
        response = self.model.generate_content(
            gemini_prompt,
            generation_config=genai.types.GenerationConfig(**generation_config) if generation_config else None,
            safety_settings=safety_settings
        )
        #  Log the raw response at DEBUG level
        logger.debug(f"Gemini API Response: {response}")

        return response.text.strip()
        