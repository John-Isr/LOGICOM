from abc import abstractmethod, ABC
from typing import Any, Optional, Dict, List
import logging

# Direct import from project structure
from core.interfaces import AgentInterface, LLMInterface, MemoryInterface
from utils.token_utils import calculate_chat_tokens, calculate_string_tokens, get_tokenizer

logger = logging.getLogger(__name__)

class BaseAgent(AgentInterface):
    """Base class for all agents, handling common initialization and interaction flow."""

    def __init__(self,
                 llm_client: LLMInterface,
                 memory: MemoryInterface | None,
                 agent_name: str = "BaseAgent",
                 model_config: Optional[Dict[str, Any]] = None,
                 prompt_wrapper: Optional[str] = None):
        """
        Initializes the BaseAgent.

        Args:
            llm_client: An object implementing the LLMInterface.
            memory: An object implementing the MemoryInterface (or None).
            agent_name: A descriptive name for the agent instance.
            model_config: Default configuration for the LLM (e.g., temperature).
            prompt_wrapper: Optional string template for the prompt wrapper.
        """
        self.llm_client = llm_client
        self.memory = memory
        self.agent_name = agent_name
        self.model_config = model_config or {}
        self._prompt_wrapper_template: Optional[str] = prompt_wrapper
        self.token_used: int = 0
        self.prompt_tokens_used: int = 0
        self.completion_tokens_used: int = 0

        if self._prompt_wrapper_template:
            logger.info(f"Prompt wrapper template provided for {self.agent_name}.")
        else:
            logger.info(f"No prompt wrapper template provided for {self.agent_name}.")

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

    def get_memory_tokens(self) -> Dict[str, int]:
        """
        Retrieves token usage from the memory component without modifying agent counts.
        
        Returns:
            Dict with prompt_tokens, completion_tokens, and total_tokens from memory operations.
        """
        if self.memory:
            return self.memory.get_token_usage()
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Because of encapsulation, this getter includes both the agent's usage and the memory's usage.
    def get_token_usage(self) -> Dict[str, int]:
        """
        Returns the agent's current token usage (excluding memory operations).
        
        Returns:
            Dict with prompt_tokens, completion_tokens, and total_tokens used by the agent.
        """
        return {
            "prompt_tokens": self.prompt_tokens_used,
            "completion_tokens": self.completion_tokens_used,
            "total_tokens": self.token_used
        }
        
    def get_total_token_usage(self) -> Dict[str, int]:
        """
        Returns the agent's total token usage including memory operations.
        
        Returns:
            Dict with prompt_tokens, completion_tokens, and total_tokens including memory.
        """
        agent_usage = self.get_token_usage()
        memory_usage = self.get_memory_tokens()
        
        total_prompt = agent_usage["prompt_tokens"] + memory_usage["prompt_tokens"]
        total_completion = agent_usage["completion_tokens"] + memory_usage["completion_tokens"]
        total_tokens = total_prompt + total_completion
        
        return {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion, 
            "total_tokens": total_tokens
        }

    def _generate_response(self, prompt: List[Dict[str, str]], **kwargs) -> str:
        """
        Helper method - calls the LLM and handles basic response/error cases.
        Uses tiktoken to estimate prompt and completion tokens.
        """
        # Merge default config with call-specific kwargs
        current_model_config = {**self.model_config, **kwargs}
        
        # Estimate prompt tokens using shared token utility
        prompt_tokens = calculate_chat_tokens(prompt)

        response = self.llm_client.generate(prompt, **current_model_config)
        
        completion_tokens = calculate_string_tokens(response)
        
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
            
        # Assume wrapper template uses <LAST_OPPONENT_MESSAGE>
        wrapper_template = self._prompt_wrapper_template 
                
        # Find the content of the last message (assumed opponent/user)
        last_opponent_message_content = ""
        if prompt[-1].get("role") == "user":
                last_opponent_message_content = prompt[-1].get("content", "")
        else:
                raise ValueError("Last message in the prompt isn't a 'user' message. Could not apply the prompt wrapper.")

        # Format the wrapper template with the last opponent message content
        wrapped_content = wrapper_template.replace("<LAST_OPPONENT_MESSAGE>", last_opponent_message_content)

        # Create the new final user message dictionary
        final_user_message = {"role": "user", "content": wrapped_content}

        # Replace the last message in the history
        final_prompt_to_send = prompt[:-1] + [final_user_message]
        logger.debug(f"Applied prompt wrapper. Final user message: {wrapped_content[:100]}...")
        return final_prompt_to_send


    @property
    def last_response(self) -> str:
        """Convenience property to get the last AI message from memory."""
        if self.memory:
            return self.memory.get_last_ai_message()
        else:
            raise AttributeError(f"Cannot get last response from agent '{self.agent_name}': Memory is not configured.") 