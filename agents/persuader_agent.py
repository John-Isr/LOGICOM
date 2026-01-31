import json 
import time
from typing import Any, Optional, Dict, Tuple, List
import logging 

# Direct imports from project structure
from agents.base_agent import BaseAgent
from core.interfaces import LLMInterface, MemoryInterface, INTERNAL_USER_ROLE, INTERNAL_AI_ROLE
from utils.token_utils import calculate_string_tokens, calculate_chat_tokens
from utils.log_main import logger

class PersuaderAgent(BaseAgent):
    """Agent responsible for persuading, potentially using a helper LLM for feedback."""

    def __init__(self,
                 llm_client: LLMInterface,
                 memory: MemoryInterface,
                 initial_prompt: str,
                 agent_name: str = "PersuaderAgent",
                 model_config: Optional[Dict[str, Any]] = None,
                 prompt_wrapper: Optional[str] = None,
                 # Helper-specific components
                 use_helper_feedback: bool = False,
                 helper_llm_client: Optional[LLMInterface] = None,
                 helper_prompt_wrapper: Optional[str] = None,
                 helper_model_config: Optional[Dict[str, Any]] = None):

        # Pass relevant args to BaseAgent, including prompt wrapper, main LLM client and memory
        super().__init__(llm_client=llm_client, memory=memory, agent_name=agent_name,
                         model_config=model_config, prompt_wrapper=prompt_wrapper)

        # Store initial prompt directly
        self.initial_prompt = initial_prompt        
        self.use_helper_feedback = use_helper_feedback
        self.helper_llm_client = helper_llm_client
        self.helper_model_config = helper_model_config or {}
        self._helper_template_content = helper_prompt_wrapper
        
        # Check helper template content if helper is enabled
        if self.use_helper_feedback:
            if not self._helper_template_content:
                # If helper enabled, the wrapper string MUST be provided
                raise ValueError("Helper feedback enabled, but helper_prompt_wrapper string is missing or empty.")
            else:
                 logger.debug("Persuader loaded helper prompt template")

        # Initialize helper token counters
        self.helper_prompt_tokens_used: int = 0
        self.helper_completion_tokens_used: int = 0
        self.helper_token_used: int = 0

        # Final validation: If helper is enabled, ensure client and template are present.
        if self.use_helper_feedback:
            if not self.helper_llm_client:
                 raise ValueError("PersuaderAgent initialized with use_helper_feedback=True but helper_llm_client is None.")
            if not self._helper_template_content:
                 raise ValueError("PersuaderAgent initialized with use_helper_feedback=True but helper_prompt_wrapper string is missing or empty.")

    def reset(self) -> None:
        """Resets agent state including helper token counts."""
        super().reset() # Resets main token counts and memory
        self.helper_prompt_tokens_used = 0
        self.helper_completion_tokens_used = 0
        self.helper_token_used = 0
        logger.debug("Persuader reset helper token counts", 
                   extra={"msg_type": "token_management"})

    def call(self, opponent_message: Optional[str] = None) -> str:
        """
        Generates a response, optionally refining it with helper feedback.

        Args:
            opponent_message: The message from the opponent, or None if this is the first turn.

        Returns:
            The response string to send (potentially refined).
        """
        # --- Handle Initial Turn (Persuader Initiates the debate) --- 
        if opponent_message is None:
            initial_prompt = self.initial_prompt
            self.memory.add_ai_message(initial_prompt)
            logger.info(f"Persuader sending initial prompt to memory: {initial_prompt}", 
                      extra={"msg_type": "memory_operation", "operation": "write", "agent_name": self.agent_name})
            logger.info(f"Persuader opening message: {initial_prompt}", 
                      extra={"msg_type": "main debate", "sender": "persuador", "receiver": "debater"})
            return initial_prompt
        # --- End Initial Turn Handling ---

        # Add opponent message to memory if provided (Standard turn)
        self.memory.add_user_message(opponent_message)
        logger.debug(f"Persuader added opponent message to memory: {opponent_message}", 
                   extra={"msg_type": "memory_operation", "operation": "write", "agent_name": self.agent_name})

        # Get prompt history from memory (includes opponent message)
        prompt_history = self.memory.get_history_as_prompt()
        logger.debug(f"Persuader retrieved message history from memory: {prompt_history}", 
                   extra={"msg_type": "memory_operation", "operation": "read", "agent_name": self.agent_name}) #TODO:: check_log

        # Apply prompt wrapping using the BaseAgent helper method
        final_prompt_to_send = self._apply_prompt_wrapper(prompt_history)
        logger.debug(f"Persuader applied prompt wrapper to history:{final_prompt_to_send}", 
                   extra={"msg_type": "prompt_operation", "agent_name": self.agent_name})#TODO:: check_log

        # Call the main LLM via BaseAgent helper using the wrapped prompt
        response_content = self._generate_response(final_prompt_to_send)
        logger.debug(f"Persuader generated response from LLM: {response_content}", 
                   extra={"msg_type": "llm_operation", "agent_name": self.agent_name})
        
        # Prepare default metadata and final response (in case helper is not used or fails)
        final_response_to_send = response_content
        feedback_tag = None
        
        # Process helper feedback if enabled
        if self.use_helper_feedback:
            # _get_helper_refinement returns (refined_response, feedback_tag_str)
            final_response_to_send, feedback_tag = self._get_helper_refinement(response_content)
            logger.debug(f"Persuader used helper to refine response, feedback tag: {feedback_tag}", 
                       extra={"msg_type": "helper_operation", "agent_name": self.agent_name, "feedback_tag": feedback_tag})

        # Add final AI message to memory with feedback_tag as metadata
        self.memory.add_ai_message(message=final_response_to_send, feedback_tag=feedback_tag)
        logger.debug("Persuador added own response to memory with feedback tag", 
                  extra={"msg_type": "memory_operation", "operation": "write", "agent_name": self.agent_name, "feedback_tag": feedback_tag})
        
        logger.info(f"Persuader response: {final_response_to_send}", 
                       extra={"msg_type": "main debate", "sender": "persuador", "receiver": "debater"})
        # Return only the final response string
        return final_response_to_send

    def _get_helper_refinement(self, persuader_response: str) -> Tuple[str, Optional[str]]: #TODO: Check if more logging needed
        """
        Calls the helper LLM, parses the JSON response, and returns the refined response and feedback tag.
        
        Returns:
            A tuple of (refined_response, feedback_tag). 
            If JSON parsing fails, returns (original_response, None) as fallback.
        """
        if not self.helper_llm_client or not self._helper_template_content: 
            raise RuntimeError("Helper LLM client or prompt template content not properly initialized.")

        history_string = self._format_history_for_helper(self.memory.get_history())
        helper_template_context = {
            "ASSISTANT_RESPONSE": persuader_response,
            "HISTORY": history_string
        }
        
        # Apply formatting using sequential replacement
        formatted_instruction = self._helper_template_content
        for placeholder_key, value in helper_template_context.items():
            placeholder = "<" + placeholder_key + ">" # Construct <HISTORY>, <ASSISTANT_RESPONSE>
            formatted_instruction = formatted_instruction.replace(placeholder, value)
        
        final_user_instruction = formatted_instruction # Use the fully replaced string

        helper_prompt_history = [{"role": "user", "content": final_user_instruction}]

        # Log the helper input for debugging
        logger.debug(f"Helper input - Original response: {persuader_response}", 
                   extra={"msg_type": "helper_operation", "agent_name": self.agent_name, "operation": "input"})
        logger.debug(f"Helper input - Formatted instruction: {final_user_instruction}...", 
                    extra={"msg_type": "helper_operation", "agent_name": self.agent_name, "operation": "input"})

        # Estimate helper prompt tokens
        prompt_tokens = calculate_chat_tokens(helper_prompt_history)
        
        # Call helper LLM
        raw_feedback = self.helper_llm_client.generate(helper_prompt_history, **self.helper_model_config)
        
        # # Log the raw helper output for debugging
        # logger.debug(f"Helper raw output: {raw_feedback}", 
        #            extra={"msg_type": "helper_operation", "agent_name": self.agent_name, "operation": "raw_output"})
        logger.debug("Helper generated response", 
                   extra={"msg_type": "helper_operation", "agent_name": self.agent_name, "operation": "generation"})

        # Estimate helper completion tokens using token_utils
        completion_tokens = calculate_string_tokens(raw_feedback)

        # Update Persuader's helper token counts 
        self.helper_prompt_tokens_used += prompt_tokens
        self.helper_completion_tokens_used += completion_tokens
        self.helper_token_used = self.helper_prompt_tokens_used + self.helper_completion_tokens_used
        logger.debug(f"Updated helper token counts: {self.helper_token_used}", 
                   extra={"msg_type": "token_management"})

        # Attempt to clean and parse the standard JSON format
        cleaned_feedback = raw_feedback.strip()
        if cleaned_feedback.startswith("```json"):
            cleaned_feedback = cleaned_feedback[7:].strip()
            if cleaned_feedback.endswith("```"):
                cleaned_feedback = cleaned_feedback[:-3].strip()
        elif cleaned_feedback.startswith("```"):
            cleaned_feedback = cleaned_feedback[3:].strip()
            if cleaned_feedback.endswith("```"):
                cleaned_feedback = cleaned_feedback[:-3].strip()
        
        try:
            refinement_dict = json.loads(cleaned_feedback)
            
            # Validate required keys based on system prompt instruction
            required_keys = ["response", "feedback_tag"]
            if not isinstance(refinement_dict, dict) or not all(key in refinement_dict for key in required_keys):
                raise ValueError(f"Parsed JSON structure is invalid or missing keys: {required_keys}")

            # Extract needed values
            refined_response = str(refinement_dict["response"])
            feedback_tag_str = str(refinement_dict["feedback_tag"])
            
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: If helper fails to return valid JSON, use original response
            logger.warning(f"Helper LLM returned malformed JSON or invalid structure. Using original response as fallback. Error: {e}. Raw feedback: {raw_feedback[:200]}...", 
                         extra={"msg_type": "helper_operation", "agent_name": self.agent_name, "operation": "fallback"})
            refined_response = persuader_response
            feedback_tag_str = None

        # Log the final parsed result
        logger.debug(f"Helper parsed JSON - Response: {refined_response}", 
                   extra={"msg_type": "helper_operation", "agent_name": self.agent_name, "operation": "parsed_response", "feedback_tag": feedback_tag_str})
        logger.info(f"Feedback tag: {feedback_tag_str}", 
                   extra={"msg_type": "helper_operation", "agent_name": self.agent_name, "operation": "parsed_feedback_tag", "feedback_tag": feedback_tag_str})
        
        # Return both the refined response and feedback tag
        return refined_response, feedback_tag_str

    def _format_history_for_helper(self, history_log: List[Any]) -> str:
        """Helper to convert internal log format to a simple text string for helper prompts."""
        logger.debug("Formatting history")
        history_lines = []
        for entry in history_log:
            if entry.get('type') == 'message' and isinstance(entry.get('data'), dict):
                role = entry['data'].get('role')
                content = entry['data'].get('content')

                if not content:
                     continue
                
                # Map internal roles to simple labels for the text string
                if role == INTERNAL_USER_ROLE:
                    history_lines.append(f"human: {content}")
                elif role == INTERNAL_AI_ROLE:
                     history_lines.append(f"AI: {content}") # Using AI consistently
                else:
                     logger.warning(f"Unexpected role '{role}' in history log entry, skipping for helper text formatting: {entry}")

        # Join the lines into a single string
        return "\n".join(history_lines)
