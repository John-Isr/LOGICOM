import time
import uuid
import re # For parsing moderator responses
from typing import Dict, Any, Optional, List, Tuple
import logging
import os
import json 
from colorama import Fore, Style

# Direct imports from project structure
from agents.persuader_agent import PersuaderAgent
from agents.debater_agent import DebaterAgent
from agents.moderator_agent import ModeratorAgent # For type hints
from core.interfaces import MemoryInterface # For type hinting if needed
from utils.log_debate import save_debate_log
from core.interfaces import INTERNAL_USER_ROLE, INTERNAL_AI_ROLE # Import standard roles

logger = logging.getLogger(__name__)


# TODO: add sides to moderator history, saying who's the debater and who's the persuader, add this in the orchestrator

class DebateOrchestrator:
    """Orchestrates the debate, managing agent turns and moderation checks."""

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

    def _get_recent_history(self, memory: MemoryInterface, count=None) -> List[Dict[str, str]]:
        """Helper to get conversation history from memory.
        If count is specified, returns only that many recent messages.
        If count is None, returns full history.
        """
        full_history = memory.get_history_as_prompt()
        if count is not None:
            return full_history[-count:]
        return full_history

    def _get_last_opponent_message_content(self, memory: MemoryInterface) -> Optional[str]:
         """Gets the content of the last message assumed to be from the opponent (user role)."""
         prompt_history = memory.get_history_as_prompt()
         if prompt_history and prompt_history[-1].get('role') == INTERNAL_USER_ROLE:
              return prompt_history[-1].get('content')
         # Fallback or error if last message wasn't user? Could indicate issue.
         logger.warning("Could not reliably determine last opponent message for moderator tag check.")
         return None 
    
# The main loop of the debate
    def run_debate(self, topic_id: str, claim: str, log_config: Dict[str, Any], helper_type_name: str) -> Dict[str, Any]:
        """
        Runs a single debate for the given topic.

        Args:
            topic_id
            claim
            log_config: Dictionary with logging parameters ('log_base_path', 'log_formats', etc.).
            helper_type_name: Name identifying (for logging).
        """
        # Initialize debate
        chat_id = self._initialize_debate(topic_id, helper_type_name)
        
        # Initialize state
        keep_talking = True
        round_number = 0
        final_result_status = None  # Will be set based on actual outcome
        finish_reason = None  # Will be set based on actual outcome
        current_persuader_response = ""
        debater_response = ""
        has_debater_responded = False

        # --- Main Debate Loop --- 
        while keep_talking and round_number < self.max_rounds:
            round_number += 1
            print(f"{self.ROUND_COLOR}\n--- Round {round_number} ---{self.RESET_ALL}")

            # Run Persuader's turn
            success, current_persuader_response = self._run_persuader_turn(round_number, debater_response)
            if not success: #terminate if error
                keep_talking = False
                final_result_status = "Error"
                finish_reason = f"Error in Persuader agent"
                break

            # Run Debater's turn
            success, debater_response = self._run_debater_turn(current_persuader_response)
            if not success:
                keep_talking = False
                final_result_status = "Error"
                finish_reason = f"Error in Debater agent"
                break
            has_debater_responded = True

            # Run Moderation checks only after debater has responded at least once
            if has_debater_responded:
                keep_talking, final_result_status, finish_reason = self._run_moderation_checks(
                    persuader_memory=self.persuader.memory,
                    debater_memory=self.debater.memory
                )

        # Handle max rounds reached
        if keep_talking and round_number >= self.max_rounds:
            final_result_status = "Persuader Fail (Max Rounds)"
            finish_reason = "Max rounds reached"
            logger.info(f"Debate ended: Reached max rounds ({self.max_rounds}).")

        # Return results
        return self._finalize_debate(
            topic_id=topic_id,
            chat_id=chat_id,
            claim=claim,
            round_number=round_number,
            final_result_status=final_result_status,
            finish_reason=finish_reason,
            log_config=log_config,
            helper_type_name=helper_type_name
        )

    def _initialize_debate(self, topic_id: str, helper_type_name: str) -> str:
        """Initialize a new debate session."""
        # Reset all agents
        self.persuader.reset()
        self.debater.reset()
        self.moderator_terminator.reset()
        self.moderator_topic.reset()

        # Generate unique ID
        chat_id = str(uuid.uuid4())
        
        # Log initial setup
        logger.info(f"\n--- Starting Debate --- Topic: {topic_id}, Chat ID: {chat_id} ---")
        logger.info(f"Config: {helper_type_name}")
        logger.info(f"Persuader: {self.persuader.agent_name}, LLM: {self.persuader.llm_client.__class__.__name__}")
        logger.info(f"Debater: {self.debater.agent_name}, LLM: {self.debater.llm_client.__class__.__name__}")
        logger.info(f"Moderator (Terminator): {self.moderator_terminator.agent_name}")
        logger.info(f"Moderator (Topic): {self.moderator_topic.agent_name}")
        logger.info(f"Max rounds limit set to: {self.max_rounds}")

        return chat_id

    def _run_persuader_turn(self, round_number: int, previous_debater_response: str) -> Tuple[bool, str]:
        """Run the persuader's turn in the debate."""
        try:
            # Add delay between rounds if configured
            if self.turn_delay_seconds > 0 and round_number > 1:
                time.sleep(self.turn_delay_seconds)

            print(f"{self.PERSUADER_COLOR}Persuader responding...{self.RESET_ALL}")
            
            # First round has no opponent message
            opponent_message = previous_debater_response if round_number > 1 else None
            persuader_response = self.persuader.call(opponent_message)
            
            print(f"{self.PERSUADER_COLOR}Persuader: {persuader_response}{self.RESET_ALL}")
            return True, persuader_response

        except Exception as e:
            logger.error(f"Error during Persuader call: {e}", exc_info=True)
            print(f"{self.ERROR_COLOR}Error during Persuader call: {e}{self.RESET_ALL}")
            return False, ""

    def _run_debater_turn(self, persuader_message: str) -> Tuple[bool, str]:
        """Run the debater's turn in the debate."""
        try:
            # Add delay between turns if configured
            if self.turn_delay_seconds > 0:
                time.sleep(self.turn_delay_seconds)

            print(f"{self.DEBATER_COLOR}Debater responding...{self.RESET_ALL}")
            debater_response = self.debater.call(persuader_message)
            print(f"{self.DEBATER_COLOR}Debater: {debater_response}{self.RESET_ALL}")
            
            return True, debater_response

        except Exception as e:
            logger.error(f"Error during Debater call: {e}", exc_info=True)
            print(f"{self.ERROR_COLOR}Error during Debater call: {e}{self.RESET_ALL}")
            return False, ""

    def _run_moderation_checks(self, persuader_memory: MemoryInterface, debater_memory: MemoryInterface) -> Tuple[bool, str, str]:
        """Run all moderation checks and return updated debate state."""
        try:
            if not debater_memory:
                raise ValueError("Debater memory not found")

            # Get recent history for both checks (limit to last few messages)
            recent_history = self._get_recent_history(debater_memory, count=6)  # Last few messages sufficient for both checks
            moderator_logs = []

            # Run termination check
            keep_talking, result_status, finish_reason = self._run_termination_check(recent_history, moderator_logs)
            if not keep_talking:
                return keep_talking, result_status, finish_reason

            # Run topic check with same recent history
            is_on_topic = self._run_topic_check(recent_history, moderator_logs)
            if not is_on_topic:
                return False, "Inconclusive (Off-Topic)", "Off-topic detected by moderator"

            # Log moderation results
            self._log_moderation_results(persuader_memory, debater_memory, moderator_logs)
            
            return True, "", ""

        except Exception as e:
            logger.error(f"Error during moderation: {e}", exc_info=True)
            return False, "Error", f"Moderation Error: {str(e)}"

    def _run_termination_check(self, history: List[Dict[str, str]], moderator_logs: List[Dict[str, Any]]) -> Tuple[bool, str, str]:
        """Run the termination check moderation and handle the result."""
        print(f"{self.MODERATOR_COLOR}Moderator checking termination...{self.RESET_ALL}")
        
        termination_result = self.moderator_terminator.call(history)
        moderator_logs.append({
            "moderator_name": self.moderator_terminator.agent_name,
            "raw_response": termination_result
        })
        
        # Parse the termination result
        termination_state = self._parse_moderator_termination(termination_result)
        
        # Handle the termination state
        if termination_state == "TERMINATE":
            print(f"{self.MODERATOR_COLOR}Moderator Termination Check: TERMINATE.{self.RESET_ALL}")
            return False, "Inconclusive (Terminated)", "Early termination by moderator"
        else:  # KEEP-TALKING
            print(f"{self.MODERATOR_COLOR}Moderator Termination Check: Continue debate.{self.RESET_ALL}")
            return True, "", ""

    def _run_topic_check(self, history: List[Dict[str, str]], moderator_logs: List[Dict[str, Any]]) -> bool:
        """Run the topic check moderation."""
        print(f"{self.MODERATOR_COLOR}Moderator checking topic...{self.RESET_ALL}")
        
        topic_result = self.moderator_topic.call(history)
        moderator_logs.append({
            "moderator_name": self.moderator_topic.agent_name,
            "raw_response": topic_result
        })
        
        return self._parse_moderator_on_topic(topic_result)

    def _log_moderation_results(self, persuader_memory: MemoryInterface, debater_memory: MemoryInterface, moderator_logs: List[Dict[str, Any]]):
        """Log the results of moderation checks."""
        if persuader_memory:
            persuader_memory.log.append({"type": "moderator_check", "data": moderator_logs})
        if debater_memory:
            debater_memory.log.append({"type": "moderator_check", "data": moderator_logs})

    def _finalize_debate(self, topic_id: str, chat_id: str, claim: str, round_number: int, 
                        final_result_status: str, finish_reason: str, log_config: Dict[str, Any], 
                        helper_type_name: str) -> Dict[str, Any]:
        """Finalize the debate by saving logs and preparing results."""
        logger.info(f"\n--- Debate Ended --- Round: {round_number}, Status: {final_result_status}, Reason: {finish_reason} ---")

        # Save debate logs
        log_base_path = log_config.get('log_base_path', './logs')
        log_formats = log_config.get('log_formats', ['json', 'html'])
        final_history = self.persuader.memory.get_history()

        save_debate_log(
            log_history=final_history,
            log_base_path=log_base_path,
            topic_id=topic_id,
            chat_id=chat_id,
            helper_type=helper_type_name,
            result=final_result_status,
            number_of_rounds=round_number,
            finish_reason=finish_reason,
            claim=claim,
            save_formats=log_formats
        )

        # Calculate and log token usage
        token_usage = self._calculate_token_usage()
        self._log_token_usage(token_usage)

        return {
            "result": final_result_status,
            "rounds": round_number,
            "finish_reason": finish_reason,
            "chat_id": chat_id,
            "topic_id": topic_id,
            "total_tokens_estimate": sum(token_usage.values())
        }

    def _calculate_token_usage(self) -> Dict[str, int]:
        """Calculate token usage for all components, including memory operations."""
        # Get token counts directly from each agent
        persuader_tokens = self.persuader.get_total_token_usage()["total_tokens"]
        debater_tokens = self.debater.get_total_token_usage()["total_tokens"]
        term_mod_tokens = self.moderator_terminator.get_total_token_usage()["total_tokens"]
        topic_mod_tokens = self.moderator_topic.get_total_token_usage()["total_tokens"]
        
        # Helper tokens are tracked separately
        helper_tokens = self.persuader.helper_token_used if self.persuader.use_helper_feedback else 0
        
        return {
            'persuader': persuader_tokens,
            'debater': debater_tokens,
            'moderator': term_mod_tokens + topic_mod_tokens,
            'helper': helper_tokens
        }

    def _log_token_usage(self, usage: Dict[str, int]):
        """Log token usage statistics."""
        if usage['helper'] > 0:
            logger.info(
                f"Token Estimates: Persuader(Main)={usage['persuader']}, "
                f"Persuader(Helper)={usage['helper']}, "
                f"Debater={usage['debater']}, "
                f"Moderator(s)={usage['moderator']}, "
                f"Total={sum(usage.values())}"
            )
        else:
            logger.info(
                f"Token Estimates: Persuader={usage['persuader']}, "
                f"Debater={usage['debater']}, "
                f"Moderator(s)={usage['moderator']}, "
                f"Total={sum(usage.values())}"
            )

    def reset_orchestrator(self):
        """Resets the orchestrator to use the primary moderators."""
        # self.active_mod_terminator = self.moderator_terminator
        # self.active_mod_topic = self.moderator_topic
        # self.active_mod_tag = self.moderator_tag
        logger.info("Orchestrator reset called (currently no active state change).") 

    # --- Moderator Response Parsing Methods --- 

    def _parse_moderator_termination(self, raw_response: str) -> str:
        """Parses termination signal. Expects either <TERMINATE> or <KEEP-TALKING> tag."""
        if not raw_response:
            logger.warning("Termination moderator returned empty response. Assuming KEEP-TALKING.")
            return "KEEP-TALKING"
        
        raw_text = raw_response.strip().upper()
        
        if '<TERMINATE>' in raw_text:
            logger.info("Parser found TERMINATE signal.")
            return "TERMINATE"
        
        if '<KEEP-TALKING>' in raw_text:
            logger.info("Parser found KEEP-TALKING signal.")
            return "KEEP-TALKING"
            
        # Default if no clear signal found
        logger.warning(f"Termination moderator returned unexpected response '{raw_response}'. Defaulting to KEEP-TALKING.")
        return "KEEP-TALKING"

    def _parse_moderator_on_topic(self, raw_response: str) -> bool:
        """Parses the raw response from the topic checker moderator.
        Expects either <ON-TOPIC> or <OFF-TOPIC> tag.
        """
        raw_text = raw_response.strip().upper()
        
        if '<ON-TOPIC>' in raw_text:
            return True
        if '<OFF-TOPIC>' in raw_text:
            return False
            
        # Default to on-topic if no clear signal found
        logger.warning(f"Topic check response format unclear: {raw_response}. Defaulting to on-topic.")
        return True

    # --- Removed _parse_moderator_tag_check --- 
    # def _parse_moderator_tag_check(self, raw_response: str) -> bool: