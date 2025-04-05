import re
from typing import Any, Optional, Dict, List, NamedTuple

# Direct imports from project structure
from agents.base_agent import BaseAgent
from core.interfaces import LLMInterface, MemoryInterface

import logging

logger = logging.getLogger(__name__)

class ModeratorAgent(BaseAgent):
    """Agent responsible for performing a *single* specific moderation check."""

    def __init__(self,
                 llm_client: LLMInterface,
                 agent_name: str = "ModeratorCheckAgent", 
                 model_config: Optional[Dict[str, Any]] = None,
                 variables: Optional[Dict] = None):

        # Pass None for memory to BaseAgent, since moderators don't need memory
        super().__init__(llm_client=llm_client, memory=None, agent_name=agent_name, model_config=model_config)

        self.variables = variables or {}

    def reset(self) -> None:
        """Resets internal state (e.g., token count)."""
        # Only need to reset base class state now
        super().reset()

    # Call method takes specific context needed for *this* check 
    # Return value is the *raw result* of the check (e.g., a tag string, a boolean)
    def call(self, context: str | List[Dict[str, str]]) -> str | None:
        """
        Performs the specific moderation check based on the agent's LLM client
        (which holds the system prompt) and the provided context.

        Args:
            context: Relevant conversation history or specific message.

        Returns:
            The raw, stripped response string from the LLM, or None if the
            LLM returns an empty response.
        """
        # Format the user part of the prompt based on the context
        if isinstance(context, list):
             # Simplified history formatting for prompt              
             history_str = "\n".join([f"{msg.get('role')}: {msg.get('content')}" for msg in context])
             user_content = f"Analyze the following recent history:\n{history_str}"
        elif isinstance(context, str):
             user_content = f"Analyze this message: {context}"
        else:
             # Fallback for unexpected context type - Raise Error
             raise TypeError(f"Moderator ({self.agent_name}) received unexpected context type: {type(context)}. Expected str or List[Dict].")

        # Construct the history part (only user message)
        prompt_history = [{"role": "user", "content": user_content}]

        # Call _generate_response. Exceptions will propagate up.
        response_content, prompt_sent = self._generate_response(prompt_history)

        # Return the  response content string and the prompt sent
        return response_content, prompt_sent