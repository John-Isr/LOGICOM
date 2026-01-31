import os
from typing import List, Dict, Any
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import logging

# Direct import from project structure
from core.interfaces import LLMInterface, INTERNAL_USER_ROLE, INTERNAL_AI_ROLE
from utils.log_main import logger


class GeminiClient(LLMInterface):
    """LLM client implementation for Google Gemini API.
    Uses the system_instruction parameter for models that support it.
    """
    # Role names expected by the Gemini API
    GEMINI_USER_ROLE = "user"
    GEMINI_MODEL_ROLE = "model"

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
                logger.debug(f"Initializing Gemini model {self.model_name} with system instruction.", extra={"msg_type": "system"})
                model_kwargs['system_instruction'] = self.system_instruction
            else:
                logger.debug(f"Initializing Gemini model {self.model_name} without system instruction.", extra={"msg_type": "system"})
            
            self.model = genai.GenerativeModel(self.model_name, **model_kwargs)

        except Exception as e:
            logger.error(f"Failed to configure Google GenAI or create model: {e}", extra={"msg_type": "system"})
            raise

    def _convert_prompt_format(self, prompt: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Converts internal standard prompt list to Gemini format.
           Handles the role mapping from 'assistant' to 'model'.
        """
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
                     logger.warning("Gemini received a system message in the prompt. Gemini API only supports a system_instruction at init. This will be ignored in the message history.", extra={"msg_type": "system"})
                     if prompt_system_message_content is None: # Store the first one found
                          prompt_system_message_content = content
                continue 
            # Map internal standard roles to Gemini roles
            elif role == INTERNAL_USER_ROLE:
                gemini_prompt.append({'role': self.GEMINI_USER_ROLE, 'parts': [content]})
            elif role == INTERNAL_AI_ROLE:
                gemini_prompt.append({'role': self.GEMINI_MODEL_ROLE, 'parts': [content]})
            else:
                # Log unexpected roles but don't necessarily fail, let API handle it?
                logger.warning(f"Gemini client received message with unexpected role '{role}'. Passing through.", extra={"msg_type": "system"})
                gemini_prompt.append({'role': role, 'parts': [content]}) # Pass unknown roles as-is
        
        # After the loop, compare the found system message (if any) with the init one
        if prompt_system_message_content is not None and self.system_instruction is not None:
             if prompt_system_message_content != self.system_instruction:
                  logger.error("CONFLICT: System instruction provided during init differs from system message found in prompt. Using the init instruction.", extra={"msg_type": "system"})

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
        logger.debug("Gemini API Request", extra={"msg_type": "API_request", "model": self.model_name})
        
        try:
            # Call generate_content using the pre-configured self.model
            response = self.model.generate_content(
                gemini_prompt,
                generation_config=genai.types.GenerationConfig(**generation_config) if generation_config else None,
                safety_settings=safety_settings
            )
            #  Log the raw response at DEBUG level
            logger.debug("Gemini API Response", extra={"msg_type": "API_response", "model": self.model_name})

            return response.text.strip()
        
        except google_exceptions.ResourceExhausted as e:
            logger.error(f"Gemini Rate Limit Error (429): {e}", extra={"msg_type": "system", "error_type": "rate_limit"})
            raise
        except google_exceptions.InvalidArgument as e:
            logger.error(f"Gemini Invalid Argument Error: {e}", extra={"msg_type": "system", "error_type": "invalid_argument"})
            raise
        except google_exceptions.PermissionDenied as e:
            logger.error(f"Gemini Permission Denied (Auth Error): {e}", extra={"msg_type": "system", "error_type": "auth_error"})
            raise
        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Gemini API Error: {e}", extra={"msg_type": "system", "error_type": "api_error"})
            raise
        except Exception as e:
            logger.error(f"Unexpected Gemini Error: {e}", extra={"msg_type": "system", "error_type": "unknown"})
            raise
        