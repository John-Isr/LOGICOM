import json 
import time
from typing import Any, Optional, Dict, Tuple, List
import logging 

# Direct imports from project structure
from agents.base_agent import BaseAgent
from core.interfaces import LLMInterface, MemoryInterface, INTERNAL_USER_ROLE, INTERNAL_AI_ROLE
from utils.token_utils import calculate_string_tokens

logger = logging.getLogger(__name__)

class PersuaderAgent(BaseAgent):
    """Agent responsible for persuading, potentially using a helper LLM for feedback."""

    def __init__(self,
                 llm_client: LLMInterface,
                 memory: MemoryInterface,
                 initial_prompt: str,
                 agent_name: str = "PersuaderAgent",
                 model_config: Optional[Dict[str, Any]] = None,
                 prompt_wrapper_path: Optional[str] = None,
                 # Helper-specific components
                 use_helper_feedback: bool = False,
                 helper_llm_client: Optional[LLMInterface] = None,
                 helper_prompt_wrapper_path: Optional[str] = None,
                 helper_model_config: Optional[Dict[str, Any]] = None):

        # Pass relevant args to BaseAgent, including wrapper path, main LLM client and memory
        super().__init__(llm_client=llm_client, memory=memory, agent_name=agent_name,
                         model_config=model_config, prompt_wrapper_path=prompt_wrapper_path)

        # Store initial prompt directly
        self.initial_prompt = initial_prompt        
        self.use_helper_feedback = use_helper_feedback
        self.helper_llm_client = helper_llm_client
        self.helper_model_config = helper_model_config or {}
        self._helper_template_content: Optional[str] = None
        
        # Read helper template content during init if helper is enabled
        if self.use_helper_feedback:
            if not helper_prompt_wrapper_path:
                raise ValueError("Helper feedback enabled, but helper_prompt_wrapper_path is missing.")
            try:
                with open(helper_prompt_wrapper_path, 'r', encoding='utf-8') as f:
                    self._helper_template_content = f.read()
                if not self._helper_template_content:
                    logger.error(f"Helper prompt template file is empty: {helper_prompt_wrapper_path}")
                    raise ValueError("Helper prompt template file is empty.")
                logger.info(f"Successfully loaded helper prompt template for {self.agent_name} from {helper_prompt_wrapper_path}")
            except FileNotFoundError:
                logger.error(f"Helper prompt template file not found: {helper_prompt_wrapper_path}")
                raise
            except Exception as e:
                logger.error(f"Error reading helper prompt template file {helper_prompt_wrapper_path}: {e}", exc_info=True)
                raise

        # Initialize helper token counters
        self.helper_prompt_tokens_used: int = 0
        self.helper_completion_tokens_used: int = 0
        self.helper_token_used: int = 0

        # Validate helper setup if enabled
        if self.use_helper_feedback:
            if not all([self.helper_llm_client, self._helper_template_content]):
                raise ValueError("Helper feedback enabled, but helper LLM client or template content is missing/failed to load.")

    def reset(self) -> None:
        """Resets agent state including helper token counts."""
        super().reset() # Resets main token counts and memory
        self.helper_prompt_tokens_used = 0
        self.helper_completion_tokens_used = 0
        self.helper_token_used = 0
        logger.info(f"{self.agent_name} helper token counts reset.")

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
            logger.info(f"{self.agent_name} sending initial prompt.")
            return initial_prompt
        # --- End Initial Turn Handling ---

        # Add opponent message to memory if provided (Standard turn)
        self.memory.add_user_message(opponent_message)

        # Get prompt history from memory (includes opponent message)
        prompt_history = self.memory.get_history_as_prompt()

        # Apply prompt wrapping using the BaseAgent helper method
        final_prompt_to_send = self._apply_prompt_wrapper(prompt_history)

        # Call the main LLM via BaseAgent helper using the wrapped prompt
        response_content = self._generate_response(final_prompt_to_send)
        
        # Prepare default metadata and final response (in case helper is not used or fails)
        final_response_to_send = response_content
        feedback_tag = None
        
        # Store details for logging within memory
        log_metadata = {
             "prompt_sent": final_prompt_to_send,
             "raw_response": response_content 
        }

        # Process helper feedback if enabled
        if self.use_helper_feedback:
            # _get_helper_refinement returns: (refined_response, feedback_tag_str, helper_prompt_history, raw_feedback)
            refined_response, feedback_tag, helper_prompt_sent, helper_raw_response = self._get_helper_refinement(response_content)
            
            # Assign the refined response for sending
            final_response_to_send = refined_response 
                        
            # Add helper details to memory log metadata
            log_metadata["helper_prompt_sent"] = helper_prompt_sent
            log_metadata["helper_raw_response"] = helper_raw_response
            log_metadata["feedback_tag"] = feedback_tag 


        # Add final AI message to memory with collected metadata
        self.memory.add_ai_message(message=final_response_to_send, **log_metadata)

        # Return only the final response string
        return final_response_to_send

    def _get_helper_refinement(self, persuader_response: str) -> Tuple[str, str, List[Dict[str,str]], str]:
        """
        Calls the helper LLM, parses the JSON response, and returns refinement details.
        Expected JSON format: {"response": "<rewritten_text>", "feedback_tag": "<tag_name>"}.
        Raises exceptions if parsing/validation fails.
        """
        if not self.helper_llm_client or not self._helper_template_content: 
            raise RuntimeError("Helper LLM client or prompt template content not properly initialized.")

        history_string = self._format_history_for_helper(self.memory.get_history())
        helper_vars = {
            "ASSISTANT_RESPONSE": persuader_response,
            "HISTORY": history_string
        }
        final_user_instruction = self._helper_template_content.format(**helper_vars)
        helper_prompt_history = [{"role": "user", "content": final_user_instruction}]

        # Estimate helper prompt tokens
        prompt_tokens = self._estimate_tokens(helper_prompt_history)
        
        # Call helper LLM
        raw_feedback = self.helper_llm_client.generate(helper_prompt_history, **self.helper_model_config)
        logger.debug(f"Raw helper response: {raw_feedback}")

        # Estimate helper completion tokens using token_utils
        completion_tokens = calculate_string_tokens(raw_feedback)

        # Update Persuader's helper token counts 
        self.helper_prompt_tokens_used += prompt_tokens
        self.helper_completion_tokens_used += completion_tokens
        self.helper_token_used = self.helper_prompt_tokens_used + self.helper_completion_tokens_used

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
        
        refinement_dict = json.loads(cleaned_feedback)
        
        # Validate required keys based on system prompt instruction
        required_keys = ["response", "feedback_tag"]
        if not isinstance(refinement_dict, dict) or not all(key in refinement_dict for key in required_keys):
            raise ValueError(f"Parsed JSON structure is invalid or missing keys: {required_keys}")

        # Extract needed values
        refined_response = str(refinement_dict["response"])
        feedback_tag_str = str(refinement_dict["feedback_tag"])

        logger.info(f"Successfully parsed helper feedback (Tag: {feedback_tag_str}).")
        
        # Return the necessary information used by 'call'
        # (refined response, feedback tag, prompt, raw response)
        return refined_response, feedback_tag_str, helper_prompt_history, raw_feedback

    def _format_history_for_helper(self, history_log: List[Any]) -> str:
        """Helper to convert internal log format to a simple text string for helper prompts."""
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
