import logging

# Direct import from project structure
from core.interfaces import LLMInterface
from llm.openai_client import OpenAIClient
from llm.gemini_client import GeminiClient
from llm.local_client import OllamaClient 

logger = logging.getLogger(__name__)

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
            
            # Prepare args for OllamaClient
            ollama_args = {
                'base_url': api_base_url,
                'model': client_args.get('model_name'),
                'system': client_args.get('system_instruction')
            }

            # Validate required Ollama args
            if not ollama_args['model']:
                 raise ValueError("Ollama client requires 'model_name' (mapped to 'model') in configuration.")

            # Remove None values to avoid passing them to OllamaClient if not set
            ollama_args = {k: v for k, v in ollama_args.items() if v is not None}

            logger.info(f"Instantiating OllamaClient with args: base_url='{api_base_url}', model='{ollama_args.get('model')}', system='{ollama_args.get('system', 'Not Set')}'") 
            return OllamaClient(**ollama_args)

        else:
            raise ValueError(f"Unknown or unsupported LLM provider: {provider}")
