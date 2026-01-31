import os
from typing import List, Dict, Any, Optional
import openai
import logging

# Direct import from project structure
from core.interfaces import LLMInterface
from utils.log_main import logger   

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
        # Initialize the OpenAI client library with timeout and retry settings
        self.client = openai.OpenAI(
            api_key=self.api_key,
            timeout=280.0,
            max_retries=1
        )

    def generate(self, prompt: List[Dict[str, str]], **kwargs) -> str:
        """Generates a response using the OpenAI API.
        Prioritizes system prompt from 'prompt' argument, falls back to instance's system_instruction.
        Logs a warning or error if both are provided and differ.
        """
        # Find system message in the input prompt and separate other messages
        prompt_system_message_content = None
        other_messages = []
        for msg in prompt:
            if msg.get("role") == "system":
                if prompt_system_message_content is None:
                    prompt_system_message_content = msg.get("content")
                else:
                    # Log if multiple system messages are found in the input prompt itself
                    logger.error("Multiple system messages found in generate() prompt; using the first one.", extra={"msg_type": "system"})
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
                    logger.warning("System instruction provided during init and a matching system message found in generate() prompt.", extra={"msg_type": "system"})
                else:
                    logger.error("CONFLICT: System instruction provided during init differs from system message in generate() prompt. Prioritizing the one from generate(), but check configuration.", extra={"msg_type": "system"})
        elif self.system_instruction:
            # Fallback to init's system instruction
            messages_to_send.append({"role": "system", "content": self.system_instruction})
            logger.debug(f"Using system instruction from init openai_client.py for {self.model_name}", extra={"msg_type": "system"})

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
        # Log request details at DEBUG level
        logger.debug("OpenAI API Request", extra={"msg_type": "API_request", "model": model})

        try:
            response = self.client.chat.completions.create(**api_params)
            
            # Log the raw response at DEBUG level
            logger.debug("OpenAI API Response", extra={"msg_type": "API_response", "model": model})
            
            return response.choices[0].message.content.strip()
        
        except openai.RateLimitError as e:
            logger.error(f"OpenAI Rate Limit Error (429): {e}", extra={"msg_type": "system", "error_type": "rate_limit"})
            raise
        except openai.APIError as e:
            logger.error(f"OpenAI API Error: {e}", extra={"msg_type": "system", "error_type": "api_error"})
            raise
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI Connection Error: {e}", extra={"msg_type": "system", "error_type": "connection_error"})
            raise
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI Authentication Error: {e}", extra={"msg_type": "system", "error_type": "auth_error"})
            raise
        except Exception as e:
            logger.error(f"Unexpected OpenAI Error: {e}", extra={"msg_type": "system", "error_type": "unknown"})
            raise