from typing import Any, Optional, Dict, List

# Direct imports from project structure
from agents.base_agent import BaseAgent
from core.interfaces import LLMInterface, MemoryInterface


class DebaterAgent(BaseAgent):
    """Agent responsible for debating against the persuader's points."""

    def __init__(self,
                 llm_client: LLMInterface,
                 memory: MemoryInterface,
                 agent_name: str = "DebaterAgent",
                 model_config: Optional[Dict[str, Any]] = None,
                 prompt_wrapper: Optional[str] = None):

        # Pass relevant args to BaseAgent, including wrapper path, main LLM client and memory
        super().__init__(llm_client=llm_client, memory=memory, agent_name=agent_name,
                         model_config=model_config, prompt_wrapper=prompt_wrapper)


    def call(self, opponent_message: str) -> str:
        #TODO: simplify this so it takes the opponent message and wraps it, adds to a memory read, and then generates a response, then adds to memory the original opponent message and the response
        """Generates a response to the opponent's message."""
        self.memory.add_user_message(opponent_message)
        # Get history from memory
        prompt = self.memory.get_history_as_prompt()

        # Apply prompt wrapping using the BaseAgent helper method
        # Assumes wrapper uses {LAST_OPPONENT_MESSAGE}
        final_prompt_to_send = self._apply_prompt_wrapper(prompt)

        response_content = self._generate_response(final_prompt_to_send)
        
        # Add response to memory with prompt/response metadata
        log_metadata = {
             "prompt_sent": final_prompt_to_send,
             "raw_response": response_content
        }
        self.memory.add_ai_message(response_content, **log_metadata)
        
        return response_content

