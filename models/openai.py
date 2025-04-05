import sys
import tiktoken
from openai import OpenAI
import os

client = OpenAI()
import google.generativeai as genai
from google.generativeai.types import safety_types
from models.base import ModelBackbone
from type import ModelType
from typing import Any, Dict, List

enc = tiktoken.encoding_for_model("gpt-4")


class GeminiModelChatCompletion(ModelBackbone):
    """Gemini model for chat completion using generate_content (stateless)"""
    def __init__(self, model_type: ModelType) -> None:
        super().__init__()
        self.model_type = model_type
        self.token_counter = 0

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        # Separate generation config from other kwargs
        generation_config_params = {
            "temperature": kwargs.pop("temperature", None),
            "top_p": kwargs.pop("top_p", None),
            "top_k": kwargs.pop("top_k", None),
            "max_output_tokens": kwargs.pop("max_output_tokens", None),
            "stop_sequences": kwargs.pop("stop", None) # OpenAI uses 'stop', Gemini uses 'stop_sequences'
        }
        # Filter out None values
        generation_config_params = {k: v for k, v in generation_config_params.items() if v is not None}

        messages = kwargs.pop("messages", []) # Get messages and remove from kwargs

        if not messages:
             raise ValueError("Messages are required for Gemini generate_content.")

        self.token_counter += len(enc.encode(str(messages)))
        
        # Convert OpenAI format messages to Gemini format for the entire history
        gemini_messages: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get('role')
            content = msg.get('content')
            if not role or not content: # Skip if message format is unexpected
                # Consider logging a warning here
                continue 

            if role == "system":
                if not gemini_messages: # If it's the first message
                     gemini_messages.append({"role": "user", "parts": [content]})
            elif role == "user":
                 gemini_messages.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                 gemini_messages.append({"role": "model", "parts": [content]})
    
        if not gemini_messages:
            raise ValueError("Cannot send an empty or invalid message list to Gemini generate_content.")
        

        # Configure safety settings
        safety_settings = {
            genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        }

        # Create GenerationConfig object
        generation_config = genai.types.GenerationConfig(**generation_config_params)

        # Initialize model (stateless for generate_content)
        # System prompt could potentially be passed here if needed and handled differently above
        model = genai.GenerativeModel(
            self.model_type.value, 
            safety_settings=safety_settings 
            # No generation_config here, passed to generate_content directly
            # system_instruction=... # If handling system prompts separately
        )
        
        # Call generate_content with the full history and config
        response = model.generate_content(
            contents=gemini_messages, 
            generation_config=generation_config
        )
        
        # Extract the text response
        # Handle potential errors or empty responses if necessary
        try:
            response_text = response.text
        except ValueError: 
             # Handle cases where the response might be blocked due to safety settings or other issues
             print(f"Warning: Gemini response blocked or invalid. Prompt: {gemini_messages}")
             print(f"Gemini Response object: {response}")
             # Check response.prompt_feedback for block reasons
             if response.prompt_feedback:
                print(f"Prompt Feedback: {response.prompt_feedback}")
             # Check response.candidates for finish reasons
             if response.candidates:
                 print(f"Candidate Finish Reason: {response.candidates[0].finish_reason}")
                 print(f"Candidate Safety Ratings: {response.candidates[0].safety_ratings}")

             response_text = "[Blocked Response]" # Or raise an error, or return None
        except AttributeError:
             # Handle cases where response structure might be different than expected
             print(f"Warning: Unexpected Gemini response structure. Prompt: {gemini_messages}")
             print(f"Gemini Response object: {response}")
             response_text = "[Invalid Response Structure]"

        return response_text # Return just the text content

    @property
    def token_used(self):
        return self.token_counter


class OpenAIModelChatCompletion(ModelBackbone):
    def __init__(self, model_type: ModelType) -> None:
        super().__init__()
        self.model_type = model_type
        self.token_counter = 0

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        if 'messages' in kwargs:
            messages = kwargs['messages']
            self.token_counter += len(enc.encode(str(messages)))

        response = client.chat.completions.create(*args, **kwargs, model=self.model_type.value)
        return response.choices[0].message.content

    @property
    def token_used(self):
        return self.token_counter


class ModelFactory:
    @staticmethod
    def create(model_type: ModelType) -> ModelBackbone:
        default_model_type = ModelType.GPT_3_5_TURBO

        model_class_map = {
            # OpenAI Chat models
            ModelType.GPT_3_5_TURBO: OpenAIModelChatCompletion,
            ModelType.GPT_3_5_TURBO_0125: OpenAIModelChatCompletion,
            ModelType.GPT_4: OpenAIModelChatCompletion,
            ModelType.GPT_4_TURBO: OpenAIModelChatCompletion,
            ModelType.GPT_4_TURBO_0613: OpenAIModelChatCompletion,
            ModelType.GPT_4O: OpenAIModelChatCompletion,
            ModelType.GPT_4O_MINI: OpenAIModelChatCompletion,
            
            # Gemini models
            ModelType.GEMINI_1_5_FLASH_8B: GeminiModelChatCompletion,
            ModelType.GEMINI_1_5_FLASH: GeminiModelChatCompletion,
            ModelType.GEMINI_1_5_PRO: GeminiModelChatCompletion,
            ModelType.GEMINI_2_0_FLASH: GeminiModelChatCompletion,
            ModelType.GEMINI_2_0_FLASH_LITE: GeminiModelChatCompletion,
            ModelType.GEMINI_2_0_PRO: GeminiModelChatCompletion
        }

        model_class = model_class_map.get(model_type, OpenAIModelChatCompletion)
        inst = model_class(model_type or default_model_type)
        return inst
