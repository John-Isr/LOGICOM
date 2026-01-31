from utils.log_main import logger
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gc
# Direct import from project structure
from core.interfaces import LLMInterface
from llm.openai_client import OpenAIClient
from llm.gemini_client import GeminiClient
from llm.local_client import OllamaClient, HuggingFaceClient

from utils.log_main import logger

# Cache for loaded Hugging Face models and tokenizers
# Key: tuple(model_name_or_path, quantization_bits) -> Value: tuple(model, tokenizer)
_loaded_hf_models = {}

# Helper function to get quantization config
def _get_hf_quantization_config(quantization_bits: Optional[int]) -> Optional[BitsAndBytesConfig]:
    """Creates a BitsAndBytesConfig if requested."""
    if quantization_bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantization_bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    elif quantization_bits is None:
        return None
    else:
        logger.warning(f"Unsupported quantization_bits value: {quantization_bits}. Disabling quantization.")
        return None

# --- Function to manually clear the HF model cache --- 
def clear_hf_cache():
    """Clears the loaded Hugging Face model cache and attempts to free GPU memory."""
    global _loaded_hf_models
    if not _loaded_hf_models:
        logger.info("HuggingFace model cache is already empty.")
        return

    logger.warning(f"Clearing HuggingFace model cache. Removing {_loaded_hf_models.keys()}...")
    _loaded_hf_models.clear() # Clear the dictionary 

    # Encourage garbage collection
    logger.info("Running garbage collection...")
    gc.collect()

    # Attempt to clear PyTorch CUDA cache
    if torch.cuda.is_available():
        logger.info("Clearing PyTorch CUDA cache...")
        torch.cuda.empty_cache()
        logger.info("CUDA cache cleared.")
    else:
        logger.info("CUDA not available, skipping CUDA cache clear.")
    logger.info("HuggingFace model cache clearing complete.")

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
        api_key = config.get('api_key')
        model_id = config.get('model_name') or config.get('model_name_or_path')

        if not model_id:
            raise ValueError("LLM configuration must include 'model_name' or 'model_name_or_path'.")

        # Accept potential extra arguments for client constructors
        client_kwargs = config.get('client_kwargs', {}) 
        # Create a dictionary of arguments for openai or gemini
        client_args = {
                 'api_key': api_key,
                 'model_name': model_id,
                 'system_instruction': system_instruction,
                 **client_kwargs
             }
        client_args = {k: v for k, v in client_args.items() if v is not None}

        if provider == 'openai':
            logger.info(f"Instantiating OpenAIClient with model: {client_args.get('model_name')}")
            return OpenAIClient(**client_args)
        
        elif provider == 'gemini':
            logger.info(f"Instantiating GeminiClient with model: {client_args.get('model_name')}")
            return GeminiClient(**client_args)

        elif provider == 'local':
            local_type = config.get('local_type')
            local_type = local_type.lower() if local_type else None

            if local_type == 'ollama':
                 logger.error("Attempted to instantiate deprecated OllamaClient via factory.")
                 raise NotImplementedError(
                     "Ollama client is deprecated. Please use 'local_type: huggingface' "
                     "in your models.yaml configuration instead."
                 )

            elif local_type == 'huggingface':
                quantization_bits = config.get('quantization_bits', 4)
                # Create a unique key for caching based on model path and quantization
                cache_key = (model_id, quantization_bits)

                # Check cache first
                if cache_key not in _loaded_hf_models:
                    logger.info(f"HuggingFace model/tokenizer not in cache for key {cache_key}. Loading...")
                    # --- Model Loading Logic ---
                    logger.info(f"Loading tokenizer for {model_id}...")
                    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                    logger.info("Tokenizer loaded.")

                    quant_config = _get_hf_quantization_config(quantization_bits)
                    torch_dtype = torch.float16 if quantization_bits in [4, 8] else "auto"

                    logger.info(f"Loading model {model_id} with quantization: {quantization_bits}-bit, dtype: {torch_dtype} using device_map='auto'...")

                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        quantization_config=quant_config,
                        device_map="auto",
                        torch_dtype=torch_dtype,
                        trust_remote_code=True,
                    )

                    model.eval()
                    logger.info(f"Model {model_id} loaded and set to evaluation mode.")

                    # Store the loaded model and tokenizer in the cache
                    _loaded_hf_models[cache_key] = (model, tokenizer)

                    # --- End Model Loading Logic ---
                else:
                    logger.info(f"Found cached HuggingFace model/tokenizer for key {cache_key}.")

                # Retrieve from cache
                model, tokenizer = _loaded_hf_models[cache_key]

                # Prepare args for HuggingFaceClient (now includes model/tokenizer objects)
                hf_args = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'system_instruction': system_instruction,
                    'generation_defaults': config.get('generation_defaults'),
                    **client_kwargs
                }
                hf_args = {k: v for k, v in hf_args.items() if v is not None}

                logger.debug(f"Instantiating HuggingFaceClient", extra={"msg_type": "system", "model": model_id}) 
                return HuggingFaceClient(**hf_args)

            else:
                raise ValueError(f"Unknown local_type: {local_type}. Must be 'huggingface' or 'ollama'.")

        else:
            raise ValueError(f"Unknown or unsupported LLM provider: {provider}")
