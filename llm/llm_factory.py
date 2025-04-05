from core.interfaces import LLMInterface
from llm.openai_client import OpenAIClient
from llm.gemini_client import GeminiClient
# Import local clients
from llm.local_client import OllamaClient, GenericAPIClient 
import logging # Added

logger = logging.getLogger(__name__) # Added

class LLMFactory:
    """Factory class to create LLM client instances."""

    @staticmethod
    def create_llm_client(config: dict, system_instruction: str | None = None) -> LLMInterface:
        """
        Creates an LLM client based on the provided configuration dictionary 
        and an optional system instruction.

        Args:
            config: A dictionary containing LLM configuration.
            system_instruction: An optional system instruction to initialize the client with.

        Returns:
            An instance of a class implementing the LLMInterface.

        Raises:
            ValueError: If the provider is unknown or required config is missing.
        """
        provider = config.get('provider')
        if not provider:
            raise ValueError("LLM provider ('provider') must be specified in the configuration.")

        provider = provider.lower()
        api_key = config.get('api_key') # Can be None, clients check env vars too
        model_name = config.get('model_name')
        # Accept potential extra arguments for client constructors
        client_kwargs = config.get('client_kwargs', {}) 

        # Base arguments for all clients (individual clients might ignore some)
        base_client_args = {
            'api_key': api_key,
            'model_name': model_name,
            'system_instruction': system_instruction,
            **client_kwargs
        }
        # Filter out None values, especially for model_name if not provided
        client_args = {k: v for k, v in base_client_args.items() if v is not None}

        if provider == 'openai':
            return OpenAIClient(**client_args)
        
        elif provider == 'gemini':
            return GeminiClient(**client_args)

        elif provider == 'local': 
            api_base_url = config.get('api_base_url')
            if not api_base_url:
                raise ValueError("Local LLM provider requires 'api_base_url' in configuration.")
            
            local_args = {
                'base_url': api_base_url,
                 **client_args
            }
            
            local_type = config.get('local_type', 'ollama').lower() # Default to generic
            
            if local_type == 'ollama':
                local_args.pop('api_key', None) # Ollama doesn't use key in constructor
                if 'model_name' not in local_args:
                     raise ValueError("Ollama client requires 'model_name' in configuration.")
                logger.info(f"Instantiating OllamaClient with args: {local_args}") 
                return OllamaClient(**local_args)
            else:
                 raise ValueError(f"Unsupported local LLM type: '{local_type}'")

        else:
            raise ValueError(f"Unknown or unsupported LLM provider: {provider}")
