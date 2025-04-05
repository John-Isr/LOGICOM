import ast
from typing import Any, Optional, Dict, List
import logging

# Direct imports from project structure
from agents.base_agent import BaseAgent
from core.interfaces import LLMInterface, MemoryInterface

import logging

class DebaterAgent(BaseAgent):
    """Agent responsible for debating against the persuader's points."""

    def __init__(self,
                 llm_client: LLMInterface,
                 memory: MemoryInterface,
                 variables: Optional[Dict] = None,
                 agent_name: str = "DebaterAgent",
                 model_config: Optional[Dict[str, Any]] = None,
                 prompt_wrapper_path: Optional[str] = None):

        # Pass relevant args to BaseAgent, including wrapper path, main LLM client and memory
        super().__init__(llm_client=llm_client, memory=memory, agent_name=agent_name,
                         model_config=model_config, prompt_wrapper_path=prompt_wrapper_path)

        self.variables = variables or {}

    def call(self, opponent_message: str) -> str:
        """Generates a response to the opponent's message."""
        self.memory.add_user_message(opponent_message)
        # Get history from memory
        prompt_history = self.memory.get_prompt()

        # Apply prompt wrapping using the BaseAgent helper method
        # Assumes wrapper uses {LAST_OPPONENT_MESSAGE}
        final_prompt_to_send = self._apply_prompt_wrapper(prompt_history)

        # BaseAgent._generate_response calls the client, which handles system prompt
        response_content, prompt_sent = self._generate_response(final_prompt_to_send)
        
        # Add response to memory with prompt/response metadata
        log_metadata = {
             "prompt_sent": prompt_sent,
             "raw_response": response_content
        }
        self.memory.add_ai_message(response_content, **log_metadata)
        
        return response_content

