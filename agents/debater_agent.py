from typing import Any, Optional, Dict, List

# Direct imports from project structure
from agents.base_agent import BaseAgent
from core.interfaces import LLMInterface, MemoryInterface
from utils.log_main import logger  # Import the custom debate logger


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
        logger.debug(f"Debater added opponent message to memory: {opponent_message}", 
                   extra={"msg_type": "memory_operation", "operation": "write", "agent_name": self.agent_name})
        
        # Get history from memory
        prompt = self.memory.get_history_as_prompt()
        logger.debug(f"Debater retrieved message history from memory: {prompt}", 
                    extra={"msg_type": "memory_operation", "operation": "read", "agent_name": self.agent_name})

        # Apply prompt wrapping using the BaseAgent helper method
        # Assumes wrapper uses {LAST_OPPONENT_MESSAGE}
        final_prompt_to_send = self._apply_prompt_wrapper(prompt)
        logger.debug(f"Debater applied prompt wrapper to history: {final_prompt_to_send}", 
                   extra={"msg_type": "prompt_operation", "operation": "apply wrapper", "agent_name": self.agent_name})# TODO:: check_log to see if agent name here is important

        response_content = self._generate_response(final_prompt_to_send)
        logger.debug(f"Debater generated response from LLM: {response_content}", 
                   extra={"msg_type": "llm_operation", "operation": "generate response", "agent_name": self.agent_name})
        
        # Add response to memory with prompt/response metadata
        log_metadata = {
             "prompt_sent": final_prompt_to_send,
             "raw_response": response_content
        }
        self.memory.add_ai_message(response_content, **log_metadata)
        logger.info("Debater added his own response to memory", 
                   extra={"msg_type": "memory_operation", "operation": "write", "agent_name": self.agent_name, "agent_name": self.agent_name})
        logger.info(f"Debater response: {response_content}", 
                   extra={"msg_type": "main debate", "sender": "debater", "receiver": "persuador"})
        return response_content

