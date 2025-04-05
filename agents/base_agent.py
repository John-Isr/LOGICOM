from abc import abstractmethod, ABC
from typing import Any, Optional, Dict, List

# Direct import from project structure
from core.interfaces import AgentInterface, LLMInterface, MemoryInterface
from utils.helpers import load_prompt_template
import logging

# Import tiktoken for token estimation
try:
    import tiktoken
except ImportError:
    tiktoken = None
    logging.getLogger(__name__).warning("tiktoken library not found. Token counting will be inaccurate. Run `pip install tiktoken`.")

logger = logging.getLogger(__name__)

# Global tokenizer instance (use default encoding)
_tokenizer = None
if tiktoken:
    try:
        # Try default first, then common alternatives
        _tokenizer = tiktoken.get_encoding("cl100k_base")
        logger.info("Successfully loaded tiktoken encoder 'cl100k_base'.") # Log success
    except Exception as e:
        # Treat failure to load tokenizer as a fatal error
        logger.error(f"CRITICAL: Failed to get tiktoken encoder 'cl100k_base': {e}. Token counting is essential for context management. Halting execution.")
        raise RuntimeError(f"Failed to initialize tiktoken tokenizer: {e}") from e # Re-raise
else:
    # Also treat tiktoken library missing as fatal
    logger.error("CRITICAL: tiktoken library not found. It is required for token counting and context management. Halting execution.")
    raise ImportError("tiktoken library is missing. Please install it via 'pip install tiktoken'")

class BaseAgent(AgentInterface):
    """Base class for all agents, handling common initialization and interaction flow."""

    def __init__(self,
                 llm_client: LLMInterface,
                 memory: MemoryInterface | None, # Allow memory to be None (for Moderator)
                 agent_name: str = "BaseAgent",
                 model_config: Optional[Dict[str, Any]] = None,
                 prompt_wrapper_path: Optional[str] = None):
        """
        Initializes the BaseAgent.

        Args:
            llm_client: An object implementing the LLMInterface.
            memory: An object implementing the MemoryInterface (or None).
            agent_name: A descriptive name for the agent instance.
            model_config: Default configuration for the LLM (e.g., temperature).
            prompt_wrapper_path: Optional path to a prompt wrapper template file.
        """
        self.llm_client = llm_client
        self.memory = memory
        self.agent_name = agent_name
        self.model_config = model_config or {}
        self.prompt_wrapper_path = prompt_wrapper_path # Store path for logging/reference
        self._prompt_wrapper_template: Optional[str] = None # Initialize stored template
        self.token_used: int = 0
        self.prompt_tokens_used: int = 0
        self.completion_tokens_used: int = 0
        self.tokenizer = _tokenizer # Store tokenizer instance on self

        # Load prompt wrapper template content during init
        if self.prompt_wrapper_path:
            try:
                # Use standard file reading
                with open(self.prompt_wrapper_path, 'r', encoding='utf-8') as f:
                     template_content = f.read()
                
                if template_content:
                    self._prompt_wrapper_template = template_content
                    logger.info(f"Successfully loaded prompt wrapper template for {self.agent_name} from {self.prompt_wrapper_path}.")
                else:
                    logger.warning(f"Prompt wrapper file is empty: {self.prompt_wrapper_path}")
            except FileNotFoundError:
                logger.error(f"Prompt wrapper file not found during init: {self.prompt_wrapper_path}. Wrapping will be disabled.")
            except IOError as e: # Catch other potential file reading errors
                logger.error(f"Error reading prompt wrapper template file {self.prompt_wrapper_path}: {e}. Wrapping will be disabled.", exc_info=True)
            except Exception as e:
                logger.error(f"Unexpected error loading prompt wrapper template during init: {e}. Wrapping will be disabled.", exc_info=True)

    @abstractmethod
    def call(self, input_data: Any) -> Any:
        """
        Abstract method for agent-specific logic.
        Subclasses must implement how they process input, interact with LLM,
        update memory, and return output.
        """
        pass

    def reset(self) -> None:
        """
        Resets the agent's memory. Subclasses can override if they have additional state.
        """
        if self.memory:
            self.memory.reset()
        self.token_used = 0
        self.prompt_tokens_used = 0
        self.completion_tokens_used = 0
        logger.info(f"{self.agent_name} memory/state reset.")

    def _generate_response(self, prompt_history: List[Dict[str, str]], **kwargs) -> str:
        """
        Helper method call the LLM and handle basic response/error cases.
        System prompt is now handled by the LLM client itself.
        Uses tiktoken (if available) to estimate prompt and completion tokens.
        """
        # Merge default config with call-specific kwargs
        current_model_config = {**self.model_config, **kwargs}
        
        prompt_tokens = 0
        completion_tokens = 0

        try:
            # Estimate prompt tokens using tiktoken if available
            if self.tokenizer: # Use instance tokenizer
                prompt_tokens = self._estimate_tokens(prompt_history)
            else:
                # Fallback rough estimation (e.g., words)
                prompt_tokens = sum(len(msg.get('content','').split()) for msg in prompt_history)

            response = self.llm_client.generate(prompt_history, **current_model_config)
            
            # Estimate completion tokens
            if self.tokenizer and isinstance(response, str): # Use instance tokenizer
                completion_tokens = len(self.tokenizer.encode(response))
            elif isinstance(response, str):
                completion_tokens = len(response.split()) # Fallback
            
            # Update agent's token counts
            self.prompt_tokens_used += prompt_tokens
            self.completion_tokens_used += completion_tokens
            self.token_used = self.prompt_tokens_used + self.completion_tokens_used 

            # Return stripped processed content and prompt sent
            processed_response = response.strip() if response else ""
            return processed_response, prompt_history
        except Exception as e:
            logger.error(f"Error during LLM generation for {self.agent_name}: {e}", exc_info=True) # Log full error
            # Re-raise the exception
            raise RuntimeError(f"LLM generation failed for {self.agent_name}: {e}") from e

    def _apply_prompt_wrapper(self, prompt_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Applies the pre-loaded prompt wrapper template if available."""
        # Use the pre-loaded template content
        if not self._prompt_wrapper_template or not prompt_history:
            return prompt_history # Return original if no template loaded or no history
            
        try:
            # Assume wrapper template uses {LAST_OPPONENT_MESSAGE}
            wrapper_template = self._prompt_wrapper_template # Use stored template
                 
            # Find the content of the last message (assumed opponent/user)
            last_opponent_message_content = ""
            if prompt_history[-1].get("role") == "user":
                 last_opponent_message_content = prompt_history[-1].get("content", "")
            else:
                 # Find the last user message if the history doesn't end with one
                 logger.error("Last message in history is not a user message. Searching for last user message...")
                 for msg in reversed(prompt_history):
                     if msg.get("role") == "user":
                         last_opponent_message_content = msg.get("content", "")
                         break
                 if not last_opponent_message_content:
                     logger.warning("Could not find last user message in history to apply prompt wrapper. Skipping wrap.")
                     return prompt_history

            # Format the wrapper template with the last opponent message content
            wrapped_content = wrapper_template.replace("{LAST_OPPONENT_MESSAGE}", last_opponent_message_content)

            # Create the new final user message dictionary
            final_user_message = {"role": "user", "content": wrapped_content}

            # Replace the last message in the history
            final_prompt_to_send = prompt_history[:-1] + [final_user_message]
            logger.debug(f"Applied prompt wrapper. Final user message: {wrapped_content[:100]}...")
            return final_prompt_to_send

        # FileNotFoundError already handled in __init__
        except KeyError as e:
            # Log with agent name for context
            logger.error(f"Placeholder {e} not found in wrapper template file {self.prompt_wrapper_path} for {self.agent_name}. Using original prompt history.")
            return prompt_history
        except Exception as e:
            # Log with agent name for context
            logger.error(f"Error formatting prompt wrapper for {self.agent_name} using template from {self.prompt_wrapper_path}: {e}. Using original prompt history.", exc_info=True)
            return prompt_history

    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimates token count for a list of messages using tiktoken.
           Based on ChatSummaryMemory's calculation method.
        """
        if not self.tokenizer: # Use instance tokenizer
            return 0
            
        num_tokens = 0
        tokens_per_message = 4 # Approximation for role/metadata
        for message in messages:
            num_tokens += tokens_per_message
            content = message.get("content")
            if content:
                try:
                    num_tokens += len(self.tokenizer.encode(content)) # Use instance tokenizer
                except Exception as e:
                     logger.error(f"Tiktoken encoding failed for content: {content[:50]}... Error: {e}")
                     num_tokens += len(content) // 4 # Fallback
        num_tokens += 2 # End-of-list approximation
        return num_tokens

    @property
    def last_response(self) -> str:
        """Convenience property to get the last AI message from memory."""
        return self.memory.get_last_ai_message() if self.memory else "" 