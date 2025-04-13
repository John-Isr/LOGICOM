from abc import ABC, abstractmethod
from typing import Any, List, Dict
from enum import Enum

# --- Internal Standard Role Names ---
INTERNAL_USER_ROLE = "user"
INTERNAL_AI_ROLE = "assistant"
# ------------------------------------

# --- Standard Helper Type Names ---
HELPER_TYPE_NONE = "No_Helper"
HELPER_TYPE_FALLACY = "Fallacy_Helper"
HELPER_TYPE_LOGICAL = "Logical_Helper"
# ----------------------------------

class LLMInterface(ABC):
    """Interface for Large Language Model interaction."""

    @abstractmethod
    def generate(self, prompt: List[Dict[str, str]], **kwargs) -> str:
        """
        Generates a response from the LLM based on the given prompt.

        Args:
            prompt: A list of message dictionaries (e.g., [{'role': 'user', 'content': '...'}, ...]).
            **kwargs: Additional model-specific parameters (e.g., temperature, max_tokens).

        Returns:
            The generated text response as a string.
        """
        pass

class MemoryInterface(ABC):
    """Interface for conversation memory management."""

    @abstractmethod
    def add_user_message(self, message: str) -> None:
        """Adds a user message to the memory."""
        pass

    @abstractmethod
    def add_ai_message(self, message: str, **kwargs) -> None:
        """Adds an AI (assistant) message to the memory. Kwargs for optional metadata like fallacy type."""
        pass

    @abstractmethod
    def get_history_as_prompt(self) -> List[Dict[str, str]]:
        """
        Formats and returns the current conversation history as a prompt
        suitable for the LLMInterface, potentially handling context length limits.
        """
        pass

    @abstractmethod
    def get_history(self) -> Any:
        """Returns the full, unprocessed conversation history or log."""
        pass

    @abstractmethod
    def get_last_ai_message(self) -> str:
        """Returns the content of the most recent AI message."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the memory, clearing the conversation history."""
        pass


class AgentInterface(ABC):
    """Interface for agents participating in the debate."""

    @abstractmethod
    def call(self, input_data: Any) -> Any:
        """
        Processes input (e.g., opponent's message) and returns the agent's response or action.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the agent's internal state if necessary."""
        pass 