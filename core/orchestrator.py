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
                 moderator_tag_checker: ModeratorAgent,
                 # Settings
                 max_rounds: int = 12,
                 turn_delay_seconds: int | float = 0,
                 logger_instance: logging.Logger | None = None):
        self.persuader = persuader
        self.debater = debater
        self.moderator_terminator = moderator_terminator
        self.moderator_topic = moderator_topic_checker
        self.moderator_tag = moderator_tag_checker

        self.max_rounds = max_rounds
        self.turn_delay_seconds = turn_delay_seconds

        self.logger = logger_instance or logging.getLogger(__name__)
        self.log_handlers = {} # Dict to store loggers for different formats

    def _get_recent_history(self, memory: MemoryInterface, count=6) -> List[Dict[str, str]]:
        """Helper to get recent user/assistant turns from memory."""
        # Assuming get_prompt returns only user/assistant turns now
        full_history = memory.get_prompt()
        return full_history[-count:]

    def _get_last_opponent_message_content(self, memory: MemoryInterface) -> Optional[str]:
         """Gets the content of the last message assumed to be from the opponent (user role)."""
         prompt_history = memory.get_prompt()
         if prompt_history and prompt_history[-1].get('role') == memory.role_map.get('user', 'user'):
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
        self.moderator_tag.reset()

        chat_id = str(uuid.uuid4()) # Generate unique ID for this debate instance
        logger.info(f"\n--- Starting Debate --- Topic: {topic_id}, Chat ID: {chat_id} ---")
        logger.info(f"Config: {helper_type_name}")
        logger.info(f"Persuader: {self.persuader.agent_name}, LLM: {self.persuader.llm_client.__class__.__name__}")
        logger.info(f"Debater: {self.debater.agent_name}, LLM: {self.debater.llm_client.__class__.__name__}")
        logger.info(f"Moderator (Terminator): {self.moderator_terminator.agent_name}, LLM: {self.moderator_terminator.llm_client.__class__.__name__}")
        logger.info(f"Moderator (Topic): {self.moderator_topic.agent_name}, LLM: {self.moderator_topic.llm_client.__class__.__name__}")
        logger.info(f"Moderator (Tag): {self.moderator_tag.agent_name}, LLM: {self.moderator_tag.llm_client.__class__.__name__}")

        keep_talking = True
        round_of_conversation: int = 0
        final_result: bool = False
        finish_reason: str = "Max rounds reached" # Default if loop finishes normally
        not_convinced_counter: int = 0 # Counter managed by orchestrator
        max_rounds = log_config.get("max_rounds", 5) # Get from config or default
        logger.info(f"Max rounds not convinced limit set to: {max_rounds}")

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
                persuader_response, refinement_type, refinement_details, refinement_technique, initial_response = self.persuader.call(opponent_message_for_persuader)
                print(f"{self.PERSUADER_COLOR}Persuader: {persuader_response}{self.RESET_ALL}")

                # Log helper refinement details (only if not Initial Ask)
                if refinement_type != "Initial Ask" and refinement_type in ["Fallacy", "Logical Argument"] and refinement_technique:
                    print(f"{self.MODERATOR_COLOR}  (Helper Refinement: {refinement_type} - {refinement_technique}){self.RESET_ALL}")
                    # Log specifically if it was a fallacy
                    if refinement_type == "Fallacy":
                        fallacy_log_path = os.path.join(log_config.get('log_base_path', './logs'), "fallacies.csv")
                        save_fallacy_data(
                            csv_path=fallacy_log_path,
                            data_to_append={
                                "Topic_ID": topic_id,
                                "Chat_ID": chat_id,
                                "Argument": initial_response, # Log initial before refinement
                                "Counter_Argument": opponent_message_for_persuader, # Log the message persuader received
                                "Fallacy": refinement_technique,
                                "Fallacious_Argument": refinement_technique
                            }
                        )
                current_persuader_response = persuader_response # Store for next debater turn
            except Exception as e:
                logger.error(f"Error during Persuader call: {e}", exc_info=True)
                print(f"{self.ERROR_COLOR}Error during Persuader call: {e}{self.RESET_ALL}")
                finish_reason = f"Error in Persuader agent: {e}"
                final_result = False
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
                final_result = False
                keep_talking = False
                break # Exit loop on agent error

            # === Moderator's Turn (Multiple Checks) ===
            # Moderator checks run after *both* agents have spoken in a round
            if self.turn_delay_seconds > 0: 
                time.sleep(self.turn_delay_seconds)
            
            # Get necessary context from DEBATER's memory (contains Persuader msg + Debater response)
            debater_memory = self.debater.memory 
            if not debater_memory:
                 logger.error("CRITICAL: Debater memory object is None. Cannot proceed with debate or moderation.")
                 raise ValueError("Debater agent lacks a memory object during debate execution.")
            recent_history = self._get_recent_history(debater_memory, count=6)
            
            # --- Perform Checks --- 
            # Store moderator interaction details
            moderator_logs = []
            
            terminate_result_raw = None
            topic_result_raw = None
            tag_result_raw = None
            mod_error = False

            try:
                 print(f"{self.MODERATOR_COLOR}Moderator ({self.moderator_terminator.agent_name}) checking termination...{self.RESET_ALL}")
                 # Capture response and prompt
                 mod_response, mod_prompt = self.moderator_terminator.call(recent_history)
                 terminate_result_raw = mod_response
                 moderator_logs.append({
                     "moderator_name": self.moderator_terminator.agent_name,
                     "prompt_sent": mod_prompt,
                     "raw_response": mod_response
                 })
            except Exception as e:
                 logger.error(f"Error in {self.moderator_terminator.agent_name}: {e}")
                 mod_error = True 
                 moderator_logs.append({"moderator_name": self.moderator_terminator.agent_name, "error": str(e)})

            
            try:
                print(f"{self.MODERATOR_COLOR}Moderator ({self.moderator_topic.agent_name}) checking topic...{self.RESET_ALL}")
                # Capture response and prompt
                mod_response, mod_prompt = self.moderator_topic.call(recent_history)
                topic_result_raw = mod_response
                moderator_logs.append({
                    "moderator_name": self.moderator_topic.agent_name,
                    "prompt_sent": mod_prompt,
                    "raw_response": mod_response
                })
            except Exception as e:
                logger.error(f"Error in {self.moderator_topic.agent_name}: {e}")
                mod_error = True
                moderator_logs.append({"moderator_name": self.moderator_topic.agent_name, "error": str(e)})

            
            try:
                print(f"{self.MODERATOR_COLOR}Moderator ({self.moderator_tag.agent_name}) checking tags...{self.RESET_ALL}")
                # Capture response and prompt
                mod_response, mod_prompt = self.moderator_tag.call(recent_history)
                tag_result_raw = mod_response
                moderator_logs.append({
                    "moderator_name": self.moderator_tag.agent_name,
                    "prompt_sent": mod_prompt,
                    "raw_response": mod_response
                })
            except Exception as e:
                logger.error(f"Error in {self.moderator_tag.agent_name}: {e}")
                mod_error = True
                moderator_logs.append({"moderator_name": self.moderator_tag.agent_name, "error": str(e)})

            # --- Log Moderator Interactions --- 
            if self.persuader.memory: # Ensure persuader has memory
                 self.persuader.memory.log.append({
                     "type": "moderator_check", 
                     "data": moderator_logs
                 }) 
            # ---------------------------------
            
            # --- Decision Logic  --- 
            if mod_error:
                 logger.error("Stopping debate due to moderator check error(s).")
                 finish_reason = "Moderator check failed"
                 final_result = False # Debate ended due to error
                 keep_talking = False
                 break # Exit the main debate loop

            # If no moderator error, proceed with checks:
            # 1. Check tag for convinced signal (TRUE/FALSE expected)
            # Ensure case-insensitive comparison and stripping whitespace
            is_convinced = str(tag_result_raw).strip().lower() == self.SIGNAL_CHECK_POSITIVE_STR
            if is_convinced:
                logger.info("Moderator detected convinced signal.")
                finish_reason = "Debater signal detected (convinced)."
                final_result = True
                keep_talking = False
                break # Terminate immediately if convinced
            else:
                not_convinced_counter += 1 # Increment counter if not convinced

            # 2. Check counter
            if not_convinced_counter >= max_rounds:
                logger.info(f"Max rounds not convinced ({max_rounds}) reached.")
                finish_reason = f"Debater not convinced after {max_rounds} checks."
                final_result = False
                keep_talking = False
                break

            # 3. Check terminator signal (<TERMINATE> / <KEEP-TALKING> expected)
            # Search for the tag within the response, case-insensitive
            if re.search(self.TERMINATE_TAG, str(terminate_result_raw), re.IGNORECASE):
                logger.info("Moderator detected termination signal.")
                finish_reason = "Termination condition met (e.g., greeting)."
                final_result = False
                keep_talking = False
                break

            # 4. Check topic (<ON-TOPIC> / <OFF-TOPIC> expected)
            # --- Updated Topic Check Logic --- 
            is_on_topic = True # Assume on topic unless explicitly false
            if topic_result_raw:
                try:
                    # Clean potential markdown fences
                    cleaned_topic_raw = str(topic_result_raw).strip()
                    if cleaned_topic_raw.startswith("```json"):
                        cleaned_topic_raw = cleaned_topic_raw[7:].strip()
                        if cleaned_topic_raw.endswith("```"):
                            cleaned_topic_raw = cleaned_topic_raw[:-3].strip()
                    elif cleaned_topic_raw.startswith("```"):
                        cleaned_topic_raw = cleaned_topic_raw[3:].strip()
                        if cleaned_topic_raw.endswith("```"):
                            cleaned_topic_raw = cleaned_topic_raw[:-3].strip()
                            
                    topic_json = json.loads(cleaned_topic_raw)
                    if isinstance(topic_json, dict) and isinstance(topic_json.get('on_topic'), bool):
                        is_on_topic = topic_json['on_topic']
                    else:
                        logger.warning(f"Moderator topic check returned invalid JSON format or missing/wrong type for 'on_topic': {topic_result_raw}")
                except json.JSONDecodeError:
                    logger.warning(f"Moderator topic check failed to parse JSON: {topic_result_raw}")
                except Exception as e:
                     logger.error(f"Unexpected error parsing moderator topic check: {e}", exc_info=True)
            # --- End Updated Logic --- 
                
            if not is_on_topic: # Check the parsed boolean value
                logger.info(f"Moderator detected off-topic conversation (Result: {topic_result_raw}).")
                finish_reason = "Conversation went off-topic."
                final_result = False
                keep_talking = False
                break
                
            # If no termination condition met, continue implicitly
            logger.info("Moderator checks passed, continuing debate.")

        # --- End of Debate ---
        logger.info(f"\n--- Debate Ended --- Round: {round_of_conversation}, Reason: {finish_reason} ---")

        # --- Logging ---
        log_base_path = log_config.get('log_base_path', './logs')
        log_formats = log_config.get('log_formats', ['json', 'html'])
        # Get final history, ensuring metadata (like helper_analysis) is included
        final_history = self.persuader.memory.get_history()

        save_debate_log(
            log_history=final_history,
            log_base_path=log_base_path,
            topic_id=topic_id,
            chat_id=chat_id,
            helper_type=helper_type_name,
            result=final_result,
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
        mod_tag_tokens = self.moderator_tag.token_used if hasattr(self.moderator_tag, 'token_used') else 0

        # Accumulate helper tokens if used
        helper_tokens = 0
        if self.persuader.use_helper_feedback and hasattr(self.persuader, 'helper_token_used'): # Check for helper_token_used on persuader
            helper_tokens = self.persuader.helper_token_used # Get value from persuader

        moderator_tokens = mod_term_tokens + mod_topic_tokens + mod_tag_tokens
        # Adjust total to exclude helper tokens from main persuader count if helper used, as they are now separate
        if helper_tokens > 0:
             persuader_main_tokens = persuader_tokens - helper_tokens # Estimate main persuader tokens
             total_tokens = persuader_main_tokens + debater_tokens + moderator_tokens + helper_tokens
             logger.info(f"Token Estimates: Persuader(Main)={persuader_main_tokens}, Persuader(Helper)={helper_tokens}, Debater={debater_tokens}, Moderator(s)={moderator_tokens}, Total={total_tokens}")
        else:
             total_tokens = persuader_tokens + debater_tokens + moderator_tokens + helper_tokens
             logger.info(f"Token Estimates: Persuader={persuader_tokens}, Debater={debater_tokens}, Moderator(s)={moderator_tokens}, Total={total_tokens}")

        # Return results
        results = {
            "result": final_result,
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