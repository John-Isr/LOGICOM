from abc import abstractmethod, ABC
from typing import Any, Optional, Dict, List
import tiktoken
import logging

# Direct import from project structure
from core.interfaces import AgentInterface, LLMInterface, MemoryInterface
from utils.helpers import load_prompt_template
import logging

_tokenizer = tiktoken.get_encoding("cl100k_base")
logger = logging.getLogger(__name__)

class BaseAgent(AgentInterface):
    """Base class for all agents, handling common initialization and interaction flow."""

    def __init__(self,
                 llm_client: LLMInterface,
                 memory: MemoryInterface | None,
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
        self.prompt_wrapper_path = prompt_wrapper_path 
        self._prompt_wrapper_template: Optional[str] = None
        self.token_used: int = 0
        self.prompt_tokens_used: int = 0
        self.completion_tokens_used: int = 0
        self.tokenizer = _tokenizer

        # Load prompt wrapper template content during init
        if self.prompt_wrapper_path:
            with open(self.prompt_wrapper_path, 'r', encoding='utf-8') as f:
                    template_content = f.read()
            self._prompt_wrapper_template = template_content
            if template_content:
                logger.info(f"Successfully loaded prompt wrapper template for {self.agent_name} from {self.prompt_wrapper_path}.")
            else:
                logger.error(f"Prompt wrapper file is empty: {self.prompt_wrapper_path}")

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

    def _generate_response(self, prompt: List[Dict[str, str]], **kwargs) -> str:
        """
        Helper method - calls the LLM and handles basic response/error cases.
        Uses tiktoken to estimate prompt and completion tokens.
        """
        # Merge default config with call-specific kwargs
        current_model_config = {**self.model_config, **kwargs}
        
        prompt_tokens = 0
        completion_tokens = 0

        # Estimate prompt tokens using tiktoken
        prompt_tokens = self._estimate_tokens(prompt)

        response = self.llm_client.generate(prompt, **current_model_config)
        
        # Estimate completion tokens
        # TODO: Consider adding a check to make sure response is a string?
        completion_tokens = len(self.tokenizer.encode(response))
        
        # Update agent's token counts
        self.prompt_tokens_used += prompt_tokens
        self.completion_tokens_used += completion_tokens
        self.token_used = self.prompt_tokens_used + self.completion_tokens_used 

        return response

    def _apply_prompt_wrapper(self, prompt: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Applies the pre-loaded prompt wrapper template if available."""
        # Return original if no template loaded or no prompt
        if not self._prompt_wrapper_template or not prompt:
            return prompt 
            
        # Assume wrapper template uses {LAST_OPPONENT_MESSAGE}
        wrapper_template = self._prompt_wrapper_template 
                
        # Find the content of the last message (assumed opponent/user)
        last_opponent_message_content = ""
        if prompt[-1].get("role") == "user":
                last_opponent_message_content = prompt[-1].get("content", "")
        else:
                raise ValueError("Last message in the prompt isn't a 'user' message. Could not apply the prompt wrapper.")

        # Format the wrapper template with the last opponent message content
        wrapped_content = wrapper_template.replace("{LAST_OPPONENT_MESSAGE}", last_opponent_message_content)

        # Create the new final user message dictionary
        final_user_message = {"role": "user", "content": wrapped_content}

        # Replace the last message in the history
        final_prompt_to_send = prompt[:-1] + [final_user_message]
        logger.debug(f"Applied prompt wrapper. Final user message: {wrapped_content[:100]}...")
        return final_prompt_to_send

    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimates token count for a list of messages using tiktoken."""
        num_tokens = 0
        tokens_per_message = 4 # Approximation for role/metadata
        for message in messages:
            num_tokens += tokens_per_message
            content = message.get("content")
            if content:
                num_tokens += len(self.tokenizer.encode(content))
        num_tokens += 2 # End-of-list approximation
        return num_tokens

    @property
    def last_response(self) -> str:
        """Convenience property to get the last AI message from memory."""
        if self.memory:
            return self.memory.get_last_ai_message()
        else:
            raise AttributeError(f"Cannot get last response from agent '{self.agent_name}': Memory is not configured.") 