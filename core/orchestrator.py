import time
import uuid
import re # For parsing moderator responses
from typing import Dict, Any, Optional, List
import logging
import os
import json 
from colorama import Fore, Style

# Direct imports from project structure
from agents.persuader_agent import PersuaderAgent
from agents.debater_agent import DebaterAgent
from agents.moderator_agent import ModeratorAgent # For type hints
from core.interfaces import MemoryInterface # For type hinting if needed
from utils.helpers import save_debate_log, save_fallacy_data # Import save_fallacy_data
from core.interfaces import INTERNAL_USER_ROLE, INTERNAL_AI_ROLE # Import standard roles

logger = logging.getLogger(__name__)


# TODO: add sides to moderator history, saying who's the debater and who's the persuader, add this in the orchestrator

class DebateOrchestrator:
    """Orchestrates the debate, managing agent turns and moderation checks."""

    # Define constants used for parsing moderator agent outputs
    TERMINATE_TAG = r"<terminate>" # Check for lowercase tag
    KEEP_TALKING_TAG = r"<keep-talking>"
    ON_TOPIC_TAG = r"<on-topic>"
    OFF_TOPIC_TAG = r"<off-topic>"
    SIGNAL_CHECK_POSITIVE_STR = "true"
    SIGNAL_CHECK_NEGATIVE_STR = "false"

    # Define colors (assuming Persuader=BLUE, Debater=GREEN)
    PERSUADER_COLOR = Fore.BLUE
    DEBATER_COLOR = Fore.GREEN
    ROUND_COLOR = Fore.RED
    MODERATOR_COLOR = Fore.YELLOW # For moderator actions
    ERROR_COLOR = Fore.RED + Style.BRIGHT
    RESET_ALL = Style.RESET_ALL # Use this if init(autoreset=True) is not used

    def __init__(self,
                 persuader: PersuaderAgent,
                 debater: DebaterAgent,
                 # Individual moderator agents
                 moderator_terminator: ModeratorAgent,
                 moderator_topic_checker: ModeratorAgent,
                 # Settings
                 max_rounds: int = 12,
                 turn_delay_seconds: int | float = 0,
                 logger_instance: logging.Logger | None = None):
        self.persuader = persuader
        self.debater = debater
        self.moderator_terminator = moderator_terminator
        self.moderator_topic = moderator_topic_checker

        self.max_rounds = max_rounds
        self.turn_delay_seconds = turn_delay_seconds

        self.logger = logger_instance or logging.getLogger(__name__)
        self.log_handlers = {} # Dict to store loggers for different formats

    def _get_recent_history(self, memory: MemoryInterface, count=6) -> List[Dict[str, str]]:
        """Helper to get recent user/assistant turns from memory."""
        # Assuming get_history_as_prompt returns only user/assistant turns now
        full_history = memory.get_history_as_prompt()
        return full_history[-count:]

    def _get_last_opponent_message_content(self, memory: MemoryInterface) -> Optional[str]:
         """Gets the content of the last message assumed to be from the opponent (user role)."""
         prompt_history = memory.get_history_as_prompt()
         if prompt_history and prompt_history[-1].get('role') == INTERNAL_USER_ROLE:
              return prompt_history[-1].get('content')
         # Fallback or error if last message wasn't user? Could indicate issue.
         logger.warning("Could not reliably determine last opponent message for moderator tag check.")
         return None 

    def run_debate(self, topic_id: str, claim: str, log_config: Dict[str, Any], helper_type_name: str) -> Dict[str, Any]:
        """
        Runs a single debate for the given topic.

        Args:
            topic_id: Identifier for the topic being debated.
            claim: The text of the claim.
            log_config: Dictionary with logging parameters ('log_base_path', 'log_formats', etc.).
            helper_type_name: Name identifying this run configuration (for logging).

        Returns:
            A dictionary containing the debate results (result, rounds, finish_reason, chat_id).
        """
        # Reset agents for a new debate
        self.persuader.reset()
        self.debater.reset()
        self.moderator_terminator.reset()
        self.moderator_topic.reset()

        chat_id = str(uuid.uuid4()) # Generate unique ID for this debate instance
        logger.info(f"\n--- Starting Debate --- Topic: {topic_id}, Chat ID: {chat_id} ---")
        logger.info(f"Config: {helper_type_name}")
        logger.info(f"Persuader: {self.persuader.agent_name}, LLM: {self.persuader.llm_client.__class__.__name__}")
        logger.info(f"Debater: {self.debater.agent_name}, LLM: {self.debater.llm_client.__class__.__name__}")
        logger.info(f"Moderator (Terminator): {self.moderator_terminator.agent_name}, LLM: {self.moderator_terminator.llm_client.__class__.__name__}")
        logger.info(f"Moderator (Topic): {self.moderator_topic.agent_name}, LLM: {self.moderator_topic.llm_client.__class__.__name__}")

        keep_talking = True
        round_of_conversation: int = 0
        final_result_status: str = "Max Rounds Reached" # Default status
        finish_reason: str = "Max rounds reached" # Default if loop finishes normally
        logger.info(f"Max rounds limit set to: {self.max_rounds}")

        # Initialize variables for the loop
        current_persuader_response = "" 
        debater_response = "" # Initialize debater response

        # --- Main Debate Loop --- 
        while keep_talking and round_of_conversation < self.max_rounds:
            # Increment round at the start of the loop pair
            round_of_conversation += 1 
            print(f"{self.ROUND_COLOR}\n--- Round {round_of_conversation} ---{self.RESET_ALL}")

            # === Persuader's Turn (Now goes first in the loop) ===
            # Optional delay between full rounds (after first round)
            if self.turn_delay_seconds > 0 and round_of_conversation > 1: 
                 time.sleep(self.turn_delay_seconds)
            try:
                print(f"{self.PERSUADER_COLOR}Persuader responding...{self.RESET_ALL}")
                # If Round 1, opponent_message is None, Persuader generates initial ask.
                # Otherwise, it gets the debater_response from the *end* of the previous iteration.
                opponent_message_for_persuader = debater_response if round_of_conversation > 1 else None
                
                # Expecting only response_string now
                persuader_response = self.persuader.call(opponent_message_for_persuader)
                
                # Metadata is no longer returned, remove extraction logic
                # initial_response = persuader_metadata.get("original_response", "") 
                # feedback_tag = persuader_metadata.get("feedback_tag") 
                
                print(f"{self.PERSUADER_COLOR}Persuader: {persuader_response}{self.RESET_ALL}")

                # Remove immediate logging based on returned metadata
                # The necessary info (original response, tag) is still in the memory log saved at the end.
                # if feedback_tag: 
                #     refinement_type = "Fallacy" if feedback_tag else "Unknown Refinement"
                #     print(f"{self.MODERATOR_COLOR}  (Helper Feedback: {refinement_type} - {feedback_tag}){self.RESET_ALL}")
                #     if refinement_type == "Fallacy":
                #         fallacy_log_path = os.path.join(log_config.get('log_base_path', './logs'), "fallacies.csv")
                #         save_fallacy_data(...)
                
                current_persuader_response = persuader_response # Store for next debater turn
            except Exception as e:
                logger.error(f"Error during Persuader call: {e}", exc_info=True)
                print(f"{self.ERROR_COLOR}Error during Persuader call: {e}{self.RESET_ALL}")
                finish_reason = f"Error in Persuader agent: {e}"
                final_result_status = "Error"
                keep_talking = False
                break # Exit loop on agent error

            # === Debater's Turn ===
            # Optional delay between turns within a round
            if self.turn_delay_seconds > 0: 
                time.sleep(self.turn_delay_seconds)
            try:
                print(f"{self.DEBATER_COLOR}Debater responding...{self.RESET_ALL}")
                # Debater always receives the Persuader's last response (which is initial ask in Round 1)
                message_to_debater = current_persuader_response 
                debater_response = self.debater.call(message_to_debater)
                print(f"{self.DEBATER_COLOR}Debater: {debater_response}{self.RESET_ALL}")
            except Exception as e:
                logger.error(f"Error during Debater call: {e}", exc_info=True)
                print(f"{self.ERROR_COLOR}Error during Debater call: {e}{self.RESET_ALL}")
                finish_reason = f"Error in Debater agent: {e}"
                final_result_status = "Error"
                keep_talking = False
                break # Exit loop on agent error

            # === Moderator's Turn (Multiple Checks) ===
            moderator_logs = [] # Initialize log list for this round
            # Get necessary context (e.g., from debater memory)
            if not self.debater.memory:
                 logger.error("Debater memory not found, cannot perform moderation checks.")
                 finish_reason = "Internal Error: Debater memory missing"
                 final_result_status = "Error"
                 keep_talking = False
                 break
            recent_history = self._get_recent_history(self.debater.memory) # Use helper

            # Initialize checks - will be updated by try blocks or stay None if error
            termination_state: Optional[str] = None
            is_on_topic: Optional[bool] = None

            # --- Termination Check ---
            try:
                print(f"{self.MODERATOR_COLOR}Moderator ({self.moderator_terminator.agent_name}) checking termination...{self.RESET_ALL}")
                termination_result_raw = self.moderator_terminator.call(recent_history) 
                moderator_logs.append({
                    "moderator_name": self.moderator_terminator.agent_name,
                    "raw_response": termination_result_raw 
                })
                # Parse termination result - EXPECTING "CONVINCED", "CONTINUE", "TERMINATE_OTHER" 
                termination_state = self._parse_moderator_termination(termination_result_raw)
            except Exception as e:
                logger.error(f"Error in {self.moderator_terminator.agent_name} or parsing: {e}", exc_info=True)
                finish_reason = f"Moderator Error ({self.moderator_terminator.agent_name}): {e}"
                final_result_status = "Error"
                keep_talking = False
                break 
            
            # --- Topic Check ---
            # If keep_talking is already false, skip subsequent checks
            if keep_talking:
                try:
                    print(f"{self.MODERATOR_COLOR}Moderator ({self.moderator_topic.agent_name}) checking topic...{self.RESET_ALL}")
                    topic_result_raw = self.moderator_topic.call(recent_history)
                    moderator_logs.append({
                        "moderator_name": self.moderator_topic.agent_name,
                        "raw_response": topic_result_raw
                    })
                    is_on_topic = self._parse_moderator_on_topic(topic_result_raw)
                except Exception as e:
                    logger.error(f"Error in {self.moderator_topic.agent_name} or parsing: {e}", exc_info=True)
                    finish_reason = f"Moderator Error ({self.moderator_topic.agent_name}): {e}"
                    final_result_status = "Error"
                    keep_talking = False
                    # No break needed here, will be caught by outer check

            # --- Log Moderator Interactions --- 
            # Log even if errors occurred
            if self.persuader.memory: 
                 self.persuader.memory.log.append({"type": "moderator_check", "data": moderator_logs})
            if self.debater.memory:
                 self.debater.memory.log.append({"type": "moderator_check", "data": moderator_logs})
            
            # If any moderator check failed and set keep_talking to False, exit now
            if not keep_talking: 
                 break
                 
            # --- Process Moderator Decisions --- 
            # This block is only reached if all moderator checks succeeded without exception
            
            # Decision 1: Check Termination State
            if termination_state == "CONVINCED":
                print(f"{self.MODERATOR_COLOR}Moderator Termination Check: CONVINCED.{self.RESET_ALL}")
                finish_reason = "Debater convinced"
                final_result_status = "Persuader Win"
                keep_talking = False
            elif termination_state == "TERMINATE_OTHER":
                print(f"{self.MODERATOR_COLOR}Moderator Termination Check: Early termination signal detected.{self.RESET_ALL}")
                # Use the reason provided by the moderator if possible, otherwise generic
                finish_reason = f"Moderator terminated: {termination_result_raw[:100]}" # Log part of raw response as reason
                final_result_status = "Inconclusive (Terminated)" 
                keep_talking = False
            else: # Assume CONTINUE or unexpected value
                if termination_state != "CONTINUE":
                     logger.warning(f"Unexpected termination state '{termination_state}', assuming CONTINUE.")
                print(f"{self.MODERATOR_COLOR}Moderator Termination Check: Continue debate.{self.RESET_ALL}")

            if not keep_talking: break 

            # Decision 2: Check Topic tag
            if not is_on_topic:
                print(f"{self.MODERATOR_COLOR}Moderator Topic Check: OFF Topic detected! Ending debate.{self.RESET_ALL}")
                finish_reason = "Off-topic detected by moderator"
                final_result_status = "Inconclusive (Off-Topic)"
                keep_talking = False
            else:
                print(f"{self.MODERATOR_COLOR}Moderator Topic Check: ON Topic.{self.RESET_ALL}")
            
            if not keep_talking: break 
            
            # Decision 3: Check Tag Validity (Removed)
            # if not tags_valid:
            #      print(f"{self.MODERATOR_COLOR}Moderator Tag Check: Invalid tags detected! Ending debate.{self.RESET_ALL}")
            #      finish_reason = "Invalid tags detected by moderator"
            #      final_result_status = "Inconclusive (Invalid Tags)"
            #      keep_talking = False
            # else:
            #      print(f"{self.MODERATOR_COLOR}Moderator Tag Check: Tags valid.{self.RESET_ALL}")

            if not keep_talking: break

        # --- End of Debate Loop --- 
        # Determine final status if max rounds reached
        if keep_talking and round_of_conversation >= self.max_rounds:
            finish_reason = "Max rounds reached"
            final_result_status = "Persuader Fail (Max Rounds)"
            logger.info(f"Debate ended: Reached max rounds ({self.max_rounds}).")

        # --- End of Debate ---
        logger.info(f"\n--- Debate Ended --- Round: {round_of_conversation}, Status: {final_result_status}, Reason: {finish_reason} ---")

        # --- Logging ---
        log_base_path = log_config.get('log_base_path', './logs')
        log_formats = log_config.get('log_formats', ['json', 'html'])
        final_history = self.persuader.memory.get_history()

        save_debate_log(
            log_history=final_history,
            log_base_path=log_base_path,
            topic_id=topic_id,
            chat_id=chat_id,
            helper_type=helper_type_name,
            result=final_result_status, # Log the status string
            number_of_rounds=round_of_conversation,
            finish_reason=finish_reason,
            claim=claim,
            save_formats=log_formats
        )

        # --- Token Usage Estimate (Assuming token_used is tracked) ---
        persuader_tokens = self.persuader.token_used if hasattr(self.persuader, 'token_used') else 0
        debater_tokens = self.debater.token_used if hasattr(self.debater, 'token_used') else 0
        mod_term_tokens = self.moderator_terminator.token_used if hasattr(self.moderator_terminator, 'token_used') else 0
        mod_topic_tokens = self.moderator_topic.token_used if hasattr(self.moderator_topic, 'token_used') else 0

        # Accumulate helper tokens if used
        helper_tokens = 0
        if getattr(self.persuader, 'use_helper_feedback', False) and hasattr(self.persuader, 'helper_token_used'): 
            helper_tokens = getattr(self.persuader, 'helper_token_used', 0)

        moderator_tokens = mod_term_tokens + mod_topic_tokens
        
        # Calculate total tokens by summing all components
        total_tokens = persuader_tokens + helper_tokens + debater_tokens + moderator_tokens

        # Log individual components clearly
        if helper_tokens > 0:
             # Log main persuader and helper separately
             logger.info(f"Token Estimates: Persuader(Main)={persuader_tokens}, Persuader(Helper)={helper_tokens}, Debater={debater_tokens}, Moderator(s)={moderator_tokens}, Total={total_tokens}")
        else:
             # Log only main persuader if helper wasn't used
             logger.info(f"Token Estimates: Persuader={persuader_tokens}, Debater={debater_tokens}, Moderator(s)={moderator_tokens}, Total={total_tokens}")

        # Return results
        results = {
            "result": final_result_status, # Return status string
            "rounds": round_of_conversation,
            "finish_reason": finish_reason,
            "chat_id": chat_id,
            "topic_id": topic_id,
            "total_tokens_estimate": total_tokens
        }
        return results

    def reset_orchestrator(self):
        """Resets the orchestrator to use the primary moderators."""
        # self.active_mod_terminator = self.moderator_terminator
        # self.active_mod_topic = self.moderator_topic
        # self.active_mod_tag = self.moderator_tag
        logger.info("Orchestrator reset called (currently no active state change).") 

    # --- Moderator Response Parsing Methods --- 

    def _parse_moderator_termination(self, raw_response: str) -> str:
        """Parses termination signal. Expects 'CONVINCED', 'TERMINATE_OTHER', or 'CONTINUE'."""
        if not raw_response:
             logger.warning("Termination moderator returned empty response. Assuming CONTINUE.")
             return "CONTINUE"
        
        response_clean = raw_response.strip().upper() # Clean and uppercase for direct comparison

        if response_clean == "CONVINCED":
            logger.info("Parser found CONVINCED signal.")
            return "CONVINCED"
        
        if response_clean == "TERMINATE_OTHER":
            logger.info("Parser found TERMINATE_OTHER signal.")
            return "TERMINATE_OTHER"

        # Default if no exact match found
        if response_clean != "CONTINUE": # Log if it wasn't exactly CONTINUE either
             logger.warning(f"Termination moderator returned unexpected response '{raw_response}'. Defaulting to CONTINUE.")
        
        return "CONTINUE"

    def _parse_moderator_on_topic(self, raw_response: str) -> bool:
        """Parses the raw response from the topic checker moderator.
        Expects a JSON object: {"on_topic": true/false}.
        Raises ValueError if parsing fails.
        """
        if not raw_response:
             logger.error("Topic moderator returned empty response.")
             raise ValueError("Topic moderator returned empty response.")

        try:
            # Clean potential markdown fences
            cleaned_response = str(raw_response).strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:].strip()
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3].strip()
            elif cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:].strip()
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3].strip()
                    
            topic_json = json.loads(cleaned_response)
            if isinstance(topic_json, dict) and isinstance(topic_json.get('on_topic'), bool):
                is_on_topic = topic_json['on_topic']
                logger.info(f"Parsed topic check: on_topic={is_on_topic}")
                return is_on_topic
            else:
                logger.error(f"Topic moderator returned invalid JSON format or missing/wrong type for 'on_topic': {raw_response}")
                raise ValueError(f"Invalid JSON format from topic moderator: {raw_response}")
        except json.JSONDecodeError as e:
            logger.error(f"Topic moderator failed to parse JSON: {raw_response}. Error: {e}")
            raise ValueError(f"Failed to parse JSON from topic moderator: {raw_response}") from e
        except Exception as e:
             logger.error(f"Unexpected error parsing moderator topic check: {e}", exc_info=True)
             raise ValueError(f"Unexpected error parsing topic moderator: {raw_response}") from e

    # --- Removed _parse_moderator_tag_check --- 
    # def _parse_moderator_tag_check(self, raw_response: str) -> bool:
    #     ...

    def reset_orchestrator(self):
        """Resets the orchestrator to use the primary moderators."""
        # self.active_mod_terminator = self.moderator_terminator
        # self.active_mod_topic = self.moderator_topic
        # self.active_mod_tag = self.moderator_tag
        logger.info("Orchestrator reset called (currently no active state change).") 