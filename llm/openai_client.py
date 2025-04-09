import os
from typing import List, Dict, Any, Optional
import openai
import logging

# Direct import from project structure
from core.interfaces import LLMInterface

logger = logging.getLogger(__name__)

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
                    logger.error("Multiple system messages found in generate() prompt; using the first one.")
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
        # Log request details at DEBUG level
        logger.debug(f"OpenAI API Request: params={api_params}")

        response = self.client.chat.completions.create(**api_params)
        
        # Log the raw response at DEBUG level
        logger.debug(f"OpenAI API Response: {response}")
        
        return response.choices[0].message.content.strip()