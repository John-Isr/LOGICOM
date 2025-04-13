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
                 max_rounds: int,
                 turn_delay_seconds: float):
        self.persuader = persuader
        self.debater = debater
        self.moderator_terminator = moderator_terminator
        self.moderator_topic = moderator_topic_checker
        self.turn_delay_seconds = turn_delay_seconds
        self.max_rounds = max_rounds

        
        self.log_handlers = {} # Dict to store loggers for different formats

    
# The main loop of the debate
    def run_debate(self, topic_id: str, claim: str, log_config: Dict[str, Any], helper_type: str) -> Dict[str, Any]:
        """
        Runs a single debate for the given topic.

        Args:
            topic_id: Identifier for the topic being debated.
            claim: The text of the claim.
            log_config: Dictionary with logging parameters ('log_base_path', 'log_formats', etc.).
            helper_type: Name identifying (for logging).
        """
        # Initialize debate
        chat_id = self._initialize_debate(topic_id, helper_type)
        
        # Initialize state
        keep_talking = True
        round_number = 0
        final_result_status = None  # Will be set based on actual outcome
        finish_reason = None  # Will be set based on actual outcome
        current_persuader_response = ""
        debater_response = ""

        # --- Main Debate Loop --- 
        while keep_talking and round_number < self.max_rounds:
            round_number += 1
            print(f"{self.ROUND_COLOR}\n--- Round {round_number} ---{self.RESET_ALL}")

            # Run Persuader's turn
            current_persuader_response = self._run_persuader_turn(debater_response)


            # Run Debater's turn
            debater_response = self._run_debater_turn(current_persuader_response)
            

            keep_talking, final_result_status, finish_reason = self._run_moderation_checks(
                persuader_memory=self.persuader.memory, #TODO: memory shouldnt be accessed from orchastrator
                debater_memory=self.debater.memory
            )

        # Handle max rounds reached
        if round_number >= self.max_rounds:
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
            helper_type=helper_type
        )

    def _initialize_debate(self, topic_id: str, helper_type: str) -> str:
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
        logger.info(f"Config: {helper_type}")
        logger.info(f"Persuader: {self.persuader.agent_name}, LLM: {self.persuader.llm_client.__class__.__name__}")
        logger.info(f"Debater: {self.debater.agent_name}, LLM: {self.debater.llm_client.__class__.__name__}")
        logger.info(f"Moderator (Terminator): {self.moderator_terminator.agent_name}, LLM: {self.moderator_terminator.llm_client.__class__.__name__}")
        logger.info(f"Moderator (Topic): {self.moderator_topic.agent_name}, LLM: {self.moderator_topic.llm_client.__class__.__name__}")
        logger.info(f"Max rounds limit set to: {self.max_rounds}")

        return chat_id

    def _run_persuader_turn(self, previous_debater_response: str) -> str:
        """Run the persuader's turn in the debate."""
        # --- Add Turn Delay ---
        if self.turn_delay_seconds > 0:
            logger.info(f"Waiting for {self.turn_delay_seconds:.2f} seconds before debater's turn.")
            time.sleep(self.turn_delay_seconds)
        # ----------------------

        # First round has no opponent message
        opponent_message = previous_debater_response if previous_debater_response else None
        persuader_response = self.persuader.call(opponent_message)
            
        print(f"{self.PERSUADER_COLOR}Persuader: {persuader_response}{self.RESET_ALL}")
        return persuader_response


    def _run_debater_turn(self, persuader_message: str) -> str:
        """Run the debater's turn in the debate."""
        # --- Add Turn Delay ---
        if self.turn_delay_seconds > 0:
            logger.info(f"Waiting for {self.turn_delay_seconds:.2f} seconds before debater's turn.")
            time.sleep(self.turn_delay_seconds)
        # ----------------------

        debater_response = self.debater.call(persuader_message)
        print(f"{self.DEBATER_COLOR}Debater: {debater_response}{self.RESET_ALL}")
            
        return debater_response



    def _run_moderation_checks(self, persuader_memory: MemoryInterface, debater_memory: MemoryInterface) -> Tuple[bool, str, str]:
        """Run all moderation checks and return updated debate state."""
            
        # Get recent history for both checks (limit to last few messages)
        recent_history = self._get_recent_history(debater_memory, count=6)  # Last few messages sufficient for both checks
        moderator_logs = []

        # Run termination check
        if not self._run_termination_check(recent_history, moderator_logs):
            return False, "Inconclusive (Terminated)", "Early termination by moderator"

        # Run topic check with same recent history
        if not self._run_topic_check(recent_history, moderator_logs):
            return False, "Inconclusive (Off-Topic)", "Off-topic detected by moderator"

        # append results to memories
        self._append_moderation_results_to_memories(persuader_memory, debater_memory, moderator_logs)
            
        return True, "", ""


    def _run_termination_check(self, history: List[Dict[str, str]], moderator_logs: List[Dict[str, Any]]) -> bool:
        """Run the termination check moderation and handle the result."""
        print(f"{self.MODERATOR_COLOR}Moderator checking termination...{self.RESET_ALL}")
        
        termination_result = self.moderator_terminator.call(history)
        moderator_logs.append({
            "moderator_name": self.moderator_terminator.agent_name,
            "raw_response": termination_result
        })
        raw_text = termination_result.strip().upper()
        if '<TERMINATE>' in raw_text:
            logger.info("Parser found TERMINATE signal.")
            print(f"{self.MODERATOR_COLOR}Moderator Termination Check: TERMINATE.{self.RESET_ALL}")
            return False
        
        elif '<KEEP-TALKING>' in raw_text:
            logger.info("Parser found KEEP-TALKING signal.")
            print(f"{self.MODERATOR_COLOR}Moderator Termination Check: Continue debate.{self.RESET_ALL}")
            return True
        
        else: #TODO: Decide if this should be a warning or an error
            logger.warning(f"Termination moderator returned unexpected response '{termination_result}'. Defaulting to KEEP-TALKING.")
            return True


    def _run_topic_check(self, history: List[Dict[str, str]], moderator_logs: List[Dict[str, Any]]) -> bool:
        """Run the topic check moderation."""
        print(f"{self.MODERATOR_COLOR}Moderator checking topic...{self.RESET_ALL}")
        
        topic_result = self.moderator_topic.call(history)
        moderator_logs.append({
            "moderator_name": self.moderator_topic.agent_name,
            "raw_response": topic_result
        })
        raw_text = topic_result.strip().upper()
        
        if '<ON-TOPIC>' in raw_text:
            return True
        elif '<OFF-TOPIC>' in raw_text:
            return False
        else: #TODO: Decide if this should be a warning or an error
            logger.warning(f"Topic check response format unclear: {topic_result}. Defaulting to on-topic.")
            return True
    #TODO: make the memory incapsuled in agents
    def _append_moderation_results_to_memories(self, persuader_memory: MemoryInterface, debater_memory: MemoryInterface, moderator_logs: List[Dict[str, Any]]):
        """Log the results of moderation checks."""
        if persuader_memory:
            persuader_memory.log.append({"type": "moderator_check", "data": moderator_logs})
        if debater_memory:
            debater_memory.log.append({"type": "moderator_check", "data": moderator_logs})

    def _finalize_debate(self, topic_id: str, chat_id: str, claim: str, round_number: int, 
                        final_result_status: str, finish_reason: str, log_config: Dict[str, Any], 
                        helper_type: str) -> Dict[str, Any]:
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
            helper_type=helper_type,
            result=final_result_status,
            number_of_rounds=round_number,
            finish_reason=finish_reason,
            claim=claim,
            save_formats=log_formats
        )

        # Calculate and log token usage
        token_usage = self._calculate_token_usage()
       

        return {
            "result": final_result_status,
            "rounds": round_number,
            "finish_reason": finish_reason,
            "chat_id": chat_id,
            "topic_id": topic_id,
            "total_tokens_estimate": token_usage
        }
    #TODO, make the agents independent of the orchestrator
    def _calculate_token_usage(self) -> int:
        """Calculate token usage for all components, including memory operations."""
        # Get token counts directly from each agent
        persuader_tokens = self.persuader.get_total_token_usage()["total_tokens"]
        debater_tokens = self.debater.get_total_token_usage()["total_tokens"]
        term_mod_tokens = self.moderator_terminator.get_total_token_usage()["total_tokens"]
        topic_mod_tokens = self.moderator_topic.get_total_token_usage()["total_tokens"]
        
        # Helper tokens are tracked separately
        helper_tokens = self.persuader.helper_token_used if self.persuader.use_helper_feedback else 0
        
        total_tokens = persuader_tokens + debater_tokens + term_mod_tokens + topic_mod_tokens + helper_tokens
        
        # Log the token counts
        logger.info(
            f"Token Estimates: Persuader={persuader_tokens}, "
            f"Debater={debater_tokens}, "
            f"Moderator={term_mod_tokens + topic_mod_tokens}, "
            f"Helper={helper_tokens}, "
            f"Total={total_tokens}"
        )
        
        return total_tokens
        
    def _get_recent_history(self, memory: MemoryInterface, count=None) -> List[Dict[str, str]]:
        """Helper to get conversation history from memory.
        If count is specified, returns only that many recent messages.
        If count is None, returns full history.
        """
        full_history = memory.get_history_as_prompt()
        if count is not None:
            return full_history[-count:]
        return full_history


    # def reset_orchestrator(self):



    # --- Moderator Response Parsing Methods --- 


    # --- Removed _parse_moderator_tag_check --- 
    # def _parse_moderator_tag_check(self, raw_response: str) -> bool: