import logging
from typing import List, Dict, Any, Optional
import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel
import accelerate
import jinja2

# Direct import from project structure
from core.interfaces import LLMInterface, INTERNAL_USER_ROLE, INTERNAL_AI_ROLE

logger = logging.getLogger(__name__)

# --- Deprecated OllamaClient ---
class OllamaClient(LLMInterface):
    """(Deprecated) LLM client implementation for Ollama API."""
    def __init__(self, *args, **kwargs):
        logger.error("OllamaClient is deprecated and not supported in this version.")
        raise NotImplementedError(
            "OllamaClient is deprecated. Please configure a 'huggingface' local_type "
            "in models.yaml instead."
        )

    def generate(self, prompt: List[Dict[str, str]], **kwargs) -> str:
         raise NotImplementedError("OllamaClient is deprecated.")


# --- HuggingFace Client ---
class HuggingFaceClient(LLMInterface):
    """LLM client implementation for running models locally via Hugging Face transformers.
       Expects the model and tokenizer to be pre-loaded and passed during initialization.
    """

    model: PreTrainedModel 
    tokenizer: PreTrainedTokenizerBase 

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizerBase,
                 system_instruction: Optional[str] = None,
                 generation_defaults: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initializes the HuggingFace client with pre-loaded model and tokenizer.

        Args:
            model: The pre-loaded Hugging Face model object.
            tokenizer: The pre-loaded Hugging Face tokenizer object.
            system_instruction: Optional default system instruction.
            generation_defaults: Default kwargs for the model.generate() method.
            **kwargs: Consumed (currently unused, but prevents errors if passed).
        """
        # Store pre-loaded objects and configuration
        self.model = model
        self.tokenizer = tokenizer
        self.system_instruction = system_instruction
        self.generation_defaults = generation_defaults if generation_defaults is not None else {}

        # Log basic info
        logger.info(f"HuggingFaceClient initialized for model: {self.model.name_or_path} on device(s): {self.model.device}")

    def _prepend_system_prompt(self, messages: List[Dict[str, str]], system_prompt: str) -> List[Dict[str, str]]:
        """Helper function to prepend the system prompt to the first user message."""
        prepended = False
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                logger.debug(f"Prepending system prompt to first user message for model {self.model.name_or_path}")
                messages[i]["content"] = f"{system_prompt}\n\n{msg['content']}"
                prepended = True
                break
        if not prepended:
            raise ValueError("Cannot apply system prompt: No user message found in the prompt list to prepend to.")
        return messages

    def _prepare_prompt(self, prompt: List[Dict[str, str]]) -> str:
        """
        Prepares the prompt list into a single string using a chat template.
        Handles system prompt precedence and potential template errors for system role.
        """
        # 1. Separate system message from prompt list and other messages
        prompt_system_content = None
        other_messages = [] # Stores user/assistant messages
        for msg in prompt:
            internal_role = msg.get("role")
            content = msg.get("content", "").strip()
            if not content: continue

            if internal_role == "system":
                if prompt_system_content is None:
                    prompt_system_content = content
                    logger.warning("System message found in prompt list as well as in init; using the one from prompt list.")
                else:
                    logger.error("Multiple system messages found in _prepare_prompt input; using the first one.")
            elif internal_role == INTERNAL_USER_ROLE:
                other_messages.append({"role": "user", "content": content})
            elif internal_role == INTERNAL_AI_ROLE:
                other_messages.append({"role": "assistant", "content": content})
            else:
                logger.error(f"HuggingFaceClient encountered unexpected role '{internal_role}'. Skipping message.")
                continue

        # 2. Determine effective system prompt
        effective_system_prompt = None
        if prompt_system_content is not None:
            effective_system_prompt = prompt_system_content
            if self.system_instruction:
                if prompt_system_content == self.system_instruction:
                    logger.warning("System instruction provided during init and a matching system message found in prompt list.")
                else:
                    logger.error("CONFLICT: System instruction provided during init differs from system message found in prompt list. Prioritizing the one from the prompt list.")
        elif self.system_instruction:
            effective_system_prompt = self.system_instruction

        # 3. Construct initial message list including system role if applicable
        initial_messages = []
        if effective_system_prompt:
            initial_messages.append({"role": "system", "content": effective_system_prompt})
        initial_messages.extend(other_messages)

        # 4. Attempt to apply chat template (standard way)
        try:
            formatted_prompt = self.tokenizer.apply_chat_template(
                initial_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            logger.debug("Applied tokenizer chat template directly.")
            return formatted_prompt
        except jinja2.exceptions.TemplateError as e:
            logger.warning(f"Direct apply_chat_template failed: {e}. Checking if it's a system role issue...")
            # Check if error is due to unsupported system role and if we have a system prompt to prepend
            if effective_system_prompt and "system role not supported" in str(e).lower():
                logger.info("Template likely doesn't support system role. Attempting fallback: Prepending system prompt to first user message.")
                try:
                    # Use the helper function to modify the user/assistant messages
                    messages_for_retry = self._prepend_system_prompt(list(other_messages), effective_system_prompt) # Pass a copy
                    # Retry applying template with the modified list (no system role)
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages_for_retry,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    logger.debug("Applied tokenizer chat template using fallback (system prompt prepended to user msg).")
                    return formatted_prompt
                except Exception as fallback_e:
                    logger.error(f"Fallback prompt formatting also failed: {fallback_e}", exc_info=True)
                    # Raise the original template error if fallback fails
                    raise e from fallback_e

    def generate(self, prompt: List[Dict[str, str]], **kwargs) -> str:
        """Generates a response using the loaded Hugging Face model."""
        # prepare and tokenize the prompt
        final_prompt_string = self._prepare_prompt(prompt)
        logger.debug(f"Formatted prompt string: {final_prompt_string}")

        inputs = self.tokenizer(final_prompt_string, return_tensors="pt", add_special_tokens=True)
        inputs = inputs.to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        # Set generation arguments (max new tokens is mandatory)
        gen_kwargs = self.generation_defaults.copy()
        gen_kwargs.update(kwargs)

        if 'max_tokens' in gen_kwargs:
             gen_kwargs['max_new_tokens'] = gen_kwargs.pop('max_tokens')
        if 'max_new_tokens' not in gen_kwargs:
            gen_kwargs['max_new_tokens'] = 512

        # Explicitly set pad_token_id to eos_token_id to suppress warning
        gen_kwargs['pad_token_id'] = self.tokenizer.eos_token_id

        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        # generate the response
        logger.debug(f"Generating response with effective args: {gen_kwargs}")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # decode the response, skipping special tokens and the request itself
        generated_ids = outputs[0, input_length:]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        logger.debug(f"Raw generated text: {output_text}")
        return output_text.strip()

