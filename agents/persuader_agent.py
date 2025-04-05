import ast
import json # Import json library
import time # Keep for potential retry logic sleep
from typing import Any, Optional, Dict, Tuple, List

# Direct imports from project structure
from agents.base_agent import BaseAgent
from core.interfaces import LLMInterface, MemoryInterface
# Assume utils handle prompt loading/formatting
import logging # Use logging

logger = logging.getLogger(__name__)

class PersuaderAgent(BaseAgent):
    """Agent responsible for persuading, potentially using a helper LLM for feedback."""

    def __init__(self,
                 llm_client: LLMInterface,
                 memory: MemoryInterface,
                 ai_first_message: Optional[str] = None,
                 variables: Optional[Dict] = None,
                 agent_name: str = "PersuaderAgent",
                 model_config: Optional[Dict[str, Any]] = None,
                 prompt_wrapper_path: Optional[str] = None,
                 # Helper-specific components
                 use_helper_feedback: bool = False,
                 llm_client_helper: Optional[LLMInterface] = None,
                 helper_user_prompt_path: Optional[str] = None,
                 helper_model_config: Optional[Dict[str, Any]] = None,
                 initial_ask_prompt_path: Optional[str] = None):

        # Pass relevant args to BaseAgent, including wrapper path, main LLM client and memory
        super().__init__(llm_client=llm_client, memory=memory, agent_name=agent_name,
                         model_config=model_config, prompt_wrapper_path=prompt_wrapper_path)

        # Reset memory at the start, and add the initial AI message if provided
        self.memory.reset()
        if ai_first_message:
            self.memory.add_ai_message(ai_first_message)

        self.variables = variables or {}
        self.use_helper_feedback = use_helper_feedback
        self.llm_client_helper = llm_client_helper
        self.helper_user_prompt_path = helper_user_prompt_path
        self.helper_model_config = helper_model_config or {}
        self.initial_ask_prompt_path = initial_ask_prompt_path

        # Initialize helper token counters
        self.helper_prompt_tokens_used: int = 0
        self.helper_completion_tokens_used: int = 0
        self.helper_token_used: int = 0

        # Validate helper setup if enabled
        if self.use_helper_feedback:
            if not all([self.llm_client_helper, self.helper_user_prompt_path]):
                raise ValueError("Helper feedback enabled, but helper LLM client or user prompt path is missing.")

    def reset(self) -> None:
        """Resets agent state including helper token counts."""
        super().reset() # Resets main token counts and memory
        self.helper_prompt_tokens_used = 0
        self.helper_completion_tokens_used = 0
        self.helper_token_used = 0
        logger.info(f"{self.agent_name} helper token counts reset.")

    def call(self, opponent_message: Optional[str] = None) -> Tuple[str, str, Optional[Dict], str, str]:
        """
        Generates a response, optionally refining it with helper feedback.
        If helper feedback is enabled and fails (e.g., JSON parsing error after retries),
        an exception will be raised.

        Args:
            opponent_message: The message from the other agent (None if this agent starts).

        Returns:
            A tuple containing:
            - final_response (str): The response to send to the opponent.
            - refinement_type (str): 'Fallacy', 'Logical Argument', or 'No Helper'.
            - refinement_details (Optional[Dict]): Full JSON from helper or None.
            - refinement_technique (str): Specific technique used by helper or empty string.
            - initial_response (str): The response generated before helper feedback.
        """
        # --- Handle Initial Turn (Persuader Asks) --- 
        if opponent_message is None:
            if not self.initial_ask_prompt_path:
                logger.error(f"Persuader ({self.agent_name}) called first but initial_ask_prompt_path is not set.")
                raise ValueError("Persuader cannot start debate without initial_ask_prompt_path configured.")
            try:
                with open(self.initial_ask_prompt_path, 'r', encoding='utf-8') as f:
                    template_content = f.read()
                initial_ask_prompt = template_content.format(**self.variables)
                self.memory.add_ai_message(initial_ask_prompt) # Add to own memory
                logger.info(f"{self.agent_name} sending initial ask prompt.")
                # Return structure: (response, refinement_type, details, technique, initial_response)
                # For this special turn, type is 'Initial Ask', no details, technique, and initial=final.
                return initial_ask_prompt, "Initial Ask", None, "", initial_ask_prompt
            except FileNotFoundError as e:
                 logger.error(f"Initial ask prompt file not found: {self.initial_ask_prompt_path}")
                 raise
            except KeyError as e:
                 logger.error(f"Missing variable {e} for initial ask prompt: {self.initial_ask_prompt_path}")
                 raise
            except Exception as e:
                 logger.error(f"Error loading/formatting initial ask prompt: {e}", exc_info=True)
                 raise
        # --- End Initial Turn Handling ---

        # Add opponent message to memory if provided (Standard turn)
        self.memory.add_user_message(opponent_message)

        # Get prompt history from memory (includes opponent message if added)
        prompt_history = self.memory.get_prompt()

        # Apply prompt wrapping using the BaseAgent helper method
        final_prompt_to_send = self._apply_prompt_wrapper(prompt_history)

        # Call the main LLM via BaseAgent helper using the wrapped prompt
        response_content, prompt_sent = self._generate_response(final_prompt_to_send)
        
        # Store details for logging
        log_metadata = {
             "prompt_sent": prompt_sent,
             "raw_response": response_content 
        }

        # Process helper feedback if enabled
        if self.use_helper_feedback and self.llm_client_helper and not response_content.startswith("ERROR_"):
            try:
                # Call helper refinement
                # Note: _get_helper_refinement returns raw helper response
                refined_response, refinement_type, refinement_details, refinement_technique, helper_prompt_sent, helper_raw_response = self._get_helper_refinement(response_content)
                final_response_to_send = refined_response
                # Add helper details to metadata
                log_metadata["helper_refinement"] = refinement_details
                log_metadata["helper_prompt_sent"] = helper_prompt_sent
                log_metadata["helper_raw_response"] = helper_raw_response
            except Exception as helper_e:
                 # Log helper error and re-raise to stop the turn
                 logger.error(f"Error during helper refinement: {helper_e}", exc_info=True)
                 raise RuntimeError(f"Helper refinement failed: {helper_e}") from helper_e
        else:
            # No helper used or initial generation failed
            final_response_to_send = response_content
            refinement_type = "No Helper" if not response_content.startswith("ERROR_") else "Generation Error"
            refinement_details = None
            refinement_technique = ""

        # Add final AI message to memory with collected metadata
        self.memory.add_ai_message(message=final_response_to_send, **log_metadata)

        # Return structure: (final_response, refinement_type, details, technique, initial_response before helper)
        return final_response_to_send, refinement_type, refinement_details, refinement_technique, response_content # Return original response_content here

    def _get_helper_refinement(self, assistant_response: str) -> Tuple[str, str, Dict, str, List[Dict[str,str]], str]:
        """
        Calls the helper LLM, parses the standardized refinement JSON, and returns details.
        Raises ValueError if parsing fails after retries.
        Expected JSON format: {"refinement_type", "refinement_technique", "response"}.
        """
        if not self.llm_client_helper or not self.helper_user_prompt_path:
            # This check is defensive, should be caught by __init__ validation
            raise RuntimeError("Helper LLM client or prompt template path not properly initialized.")

        history_for_helper = self._format_history_for_helper(self.memory.get_history())
        history_string = json.dumps(history_for_helper)

        helper_vars = {
            **self.variables,
            "ASSISTANT_RESPONSE": assistant_response,
            "HISTORY": history_string
        }

        # Load the wrapper template
        try:
            with open(self.helper_user_prompt_path, 'r', encoding='utf-8') as f:
                user_prompt_template = f.read()
            if not user_prompt_template:
                logger.error(f"Helper prompt template loaded as empty from: {self.helper_user_prompt_path}")
                raise ValueError("Helper prompt template is empty.")

            # Use .format() for substitution
            final_user_instruction = user_prompt_template.format(**helper_vars)

        except FileNotFoundError:
             logger.error(f"Helper prompt template file not found: {self.helper_user_prompt_path}")
             raise
        except Exception as e:
             logger.error(f"Error reading/formatting helper prompt template {self.helper_user_prompt_path}: {e}", exc_info=True)
             raise

        # Construct prompt history for the helper LLM
        helper_prompt_history = [{"role": "user", "content": final_user_instruction}]

        # Updated expected keys
        expected_keys = ["refinement_type", "refinement_technique", "response"]

        # Loop for parsing retries
        max_parse_retries = 2
        for attempt in range(max_parse_retries + 1):
            logger.debug(f"Calling helper LLM for refinement (Attempt {attempt+1})...")
            
            # --- Estimate helper prompt tokens ---
            prompt_tokens = 0
            # Use the instance tokenizer from BaseAgent
            if self.tokenizer: 
                prompt_tokens = self._estimate_tokens(helper_prompt_history)
            else:
                prompt_tokens = sum(len(msg.get('content','').split()) for msg in helper_prompt_history)
            # -------------------------------------
            
            # Call helper LLM
            raw_feedback = self.llm_client_helper.generate(helper_prompt_history, **self.helper_model_config)
            logger.debug(f"Raw helper response: {raw_feedback}")

            # --- Estimate helper completion tokens ---
            completion_tokens = 0
            # Use the instance tokenizer from BaseAgent
            if self.tokenizer and isinstance(raw_feedback, str):
                completion_tokens = len(self.tokenizer.encode(raw_feedback))
            elif isinstance(raw_feedback, str):
                completion_tokens = len(raw_feedback.split())
            # ----------------------------------------

            # --- Update Persuader's helper token counts ---
            self.helper_prompt_tokens_used += prompt_tokens
            self.helper_completion_tokens_used += completion_tokens
            self.helper_token_used = self.helper_prompt_tokens_used + self.helper_completion_tokens_used
            # ----------------------------------------------

            # Attempt to clean and parse the standard JSON format
            cleaned_feedback = raw_feedback.strip()
            # Remove potential markdown code fences
            if cleaned_feedback.startswith("```json"):
                 cleaned_feedback = cleaned_feedback[7:].strip()
                 if cleaned_feedback.endswith("```"):
                      cleaned_feedback = cleaned_feedback[:-3].strip()
            elif cleaned_feedback.startswith("```"):
                 cleaned_feedback = cleaned_feedback[3:].strip()
                 if cleaned_feedback.endswith("```"):
                      cleaned_feedback = cleaned_feedback[:-3].strip()

            try:
                # Rename variable holding the parsed dict
                refinement_dict = json.loads(cleaned_feedback)

                # Use updated expected keys for validation
                if not isinstance(refinement_dict, dict) or not all(key in refinement_dict for key in expected_keys):
                     raise ValueError(f"Parsed JSON is missing required refinement keys: {expected_keys}")

                # Extract values using the new keys
                refined_response = str(refinement_dict.get("response", assistant_response))
                refinement_type_str = str(refinement_dict.get("refinement_type", "Parse Error"))
                refinement_technique_str = str(refinement_dict.get("refinement_technique", "Parse Error"))

                # Validate refinement_type content
                if refinement_type_str not in ["Fallacy", "Logical Argument"]:
                    logger.warning(f"Helper returned unexpected refinement_type: '{refinement_type_str}'")
                    # Decide how to handle - maybe default or raise specific error?
                    # For now, let it pass but log warning.
                    # refinement_type_str = "Unknown Analysis" # Option to overwrite

                logger.info(f"Successfully parsed helper refinement (Type: {refinement_type_str}).")
                # Return values including prompt sent and raw response
                return refined_response, refinement_type_str, refinement_dict, refinement_technique_str, helper_prompt_history, raw_feedback

            except (json.JSONDecodeError, ValueError) as e: # Catch JSON and validation errors
                logger.warning(f'Helper refinement JSON parsing/validation error (Attempt {attempt+1}/{max_parse_retries+1}): {e}')
                logger.warning(f'Raw feedback causing error: {raw_feedback}')
                if attempt == max_parse_retries:
                    logger.error("Max parse/validation retries reached for helper refinement. Raising error.")
                    # Raise the exception to be caught by the orchestrator
                    raise ValueError(f"Failed to parse/validate helper refinement after {max_parse_retries+1} attempts: {raw_feedback}") from e

                # Modify the prompt for the next retry attempt, mentioning the standard keys
                retry_user_content = (
                    f"Your previous response could not be parsed or validated as the required refinement JSON structure. Error: {e}.\n"
                    f"Original problematic response: ```\n{raw_feedback}\n```\n"
                    f"Please ensure your response is ONLY a single, valid JSON object with keys {expected_keys} and string values. Re-generate the response correctly."
                )
                helper_prompt_history = [{"role": "user", "content": retry_user_content}]
                time.sleep(1) # Simple backoff

        # Should not be reached if loop/exception handling is correct
        raise RuntimeError("Exited helper refinement parsing loop unexpectedly after retries.")

    def _format_history_for_helper(self, history_log: List[Any]) -> List[Dict[str, str]]:
        """Helper to convert internal log format to the one needed by helper prompts ([{"human":...}, {"AI":...}])."""
        formatted_history = []
        user_role_name = self.memory.role_map.get('user', 'user')
        ai_role_name = self.memory.role_map.get('ai', 'assistant')

        for entry in history_log:
            if entry.get('type') == 'message' and isinstance(entry.get('data'), dict):
                role = entry['data'].get('role')
                content = entry['data'].get('content')

                if not content:
                     continue

                if role == user_role_name:
                    formatted_history.append({"human": content})
                elif role == ai_role_name:
                     formatted_history.append({"AI": content})
                else:
                     logger.warning(f"Unexpected role '{role}' in history log entry, skipping for helper formatting: {entry}")

        return formatted_history 