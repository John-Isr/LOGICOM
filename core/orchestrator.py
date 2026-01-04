import time
import uuid
import re # For parsing moderator responses
from typing import Dict, Any, Optional, List, Tuple


from colorama import Fore, Style

# Direct imports from project structure
from agents.persuader_agent import PersuaderAgent
from agents.debater_agent import DebaterAgent
from agents.moderator_agent import ModeratorAgent # For type hints
from core.interfaces import MemoryInterface # For type hinting if needed
from utils.log_main import logger



# TODO: add sides to moderator history, saying who's the debater and who's the persuader, add this in the orchestrator

class DebateOrchestrator:
    """Orchestrates the debate, managing agent turns and moderation checks."""

    def __init__(self,
                 # Agents
                 persuader: PersuaderAgent,
                 debater: DebaterAgent,
                 moderator_terminator: ModeratorAgent,
                 moderator_topic_checker: ModeratorAgent,
                 moderator_conviction: ModeratorAgent,
                 moderator_argument_quality: ModeratorAgent,
                 moderator_debate_quality: ModeratorAgent,
                 # Settings
                 max_rounds: int,
                 turn_delay_seconds: float):
        self.persuader = persuader
        self.debater = debater
        self.moderator_terminator = moderator_terminator
        self.moderator_topic = moderator_topic_checker
        self.moderator_conviction = moderator_conviction
        self.moderator_argument_quality = moderator_argument_quality
        self.moderator_debate_quality = moderator_debate_quality
        self.turn_delay_seconds = turn_delay_seconds
        self.max_rounds = max_rounds

        self.log_handlers = {} # Dict to store loggers for different formats

    
# The main loop of the debate
    def run_debate(self, topic_id: str, claim: str, log_config: Dict[str, Any], helper_type: str, chat_id: str) -> Dict[str, Any]:
        """
        Runs a single debate for the given topic.

        Args:
            topic_id: Identifier for the topic being debated.
            claim: The text of the claim.
            log_config: Dictionary with logging parameters ('log_base_path', 'log_formats', etc.).
            helper_type: The type of helper used (for logging).
            chat_id: Unique identifier for this chat (required).
        """
        # Initialize debate
        self._initialize_debate(topic_id, helper_type, chat_id)
        
        # Initialize state
        keep_talking = True
        round_number = 0
        final_result_status = None
        finish_reason = None
        current_persuader_response = ""
        debater_response = ""

        # --- Main Debate Loop --- 
        while keep_talking and round_number < self.max_rounds:
            round_number += 1
            logger.debug(f"\n--- Round number is {round_number} ---", extra={"msg_type": "main debate", "round": round_number})

            # Run Persuader's turn
            current_persuader_response = self._run_persuader_turn(debater_response)


            # Run Debater's turn
            debater_response = self._run_debater_turn(current_persuader_response)
            
            # Check argument quality
            argument_quality_rate = self._run_argument_quality_check(persuader_memory=self.persuader.memory, debater_memory=self.debater.memory)
            
            # Update the argument quality rate for the persuader's most recent message - TODO:: decide if save it differently
            if hasattr(self.persuader.memory, 'argument_quality_rates') and len(self.persuader.memory.argument_quality_rates) > 0:
                self.persuader.memory.argument_quality_rates[-1] = argument_quality_rate
            
            # Check if debater was convinced after their response and get conviction rate
            is_convinced, conviction_rate = self._run_conviction_check(debater_response=debater_response, debater_memory=self.debater.memory)
            
            # Update the conviction rate for the debater's most recent message - TODO:: decide if save it differently
            if hasattr(self.debater.memory, 'conviction_rates') and len(self.debater.memory.conviction_rates) > 0:
                self.debater.memory.conviction_rates[-1] = conviction_rate


            if is_convinced:
                final_result_status = "Convinced"
                finish_reason = "Debater convinced"
                break



            keep_talking, finish_reason = self._run_moderation_checks(
                persuader_memory=self.persuader.memory, #TODO: memory shouldnt be accessed from orchastrator
                debater_memory=self.debater.memory
            )
            
            # Set result status based on status tag if debate should end
            if not keep_talking:
                    final_result_status = "Inconclusive"

        # Handle max rounds reached
        # Only set "Not convinced" if status wasn't already set to "Convinced"
        # If status is "Inconclusive" after max rounds, it's OK to change to "Not convinced"
        if round_number >= self.max_rounds and final_result_status != "Convinced":
            final_result_status = "Not convinced"
            finish_reason = "Max rounds reached"
            logger.debug(f"Debate ended: Reached max rounds ({self.max_rounds})." , extra={"msg_type": "main debate"})

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

    def _initialize_debate(self, topic_id: str, helper_type: str, chat_id: str = None) -> str:
        """Initialize a new debate session."""
        # Reset all agents
        self.persuader.reset()
        self.debater.reset()
        self.moderator_terminator.reset()
        self.moderator_topic.reset()
        self.moderator_conviction.reset()
        self.moderator_argument_quality.reset()
        self.moderator_debate_quality.reset()
        
        # Log initial setup 
        logger.debug(f"Starting Debate, Topic: {topic_id}, Chat ID: {chat_id}", 
                   extra={"msg_type": "system", "topic_id": topic_id, "chat_id": chat_id})
        logger.debug(f"Config: {helper_type}", extra={"msg_type": "system"})
        logger.debug(f"Persuader: {self.persuader.agent_name}, LLM: {self.persuader.llm_client.__class__.__name__}", 
                   extra={"msg_type": "system"})
        logger.debug(f"Debater: {self.debater.agent_name}, LLM: {self.debater.llm_client.__class__.__name__}", 
                   extra={"msg_type": "system"})
        logger.debug(f"Moderator (Terminator): {self.moderator_terminator.agent_name}", extra={"msg_type": "system"})
        logger.debug(f"Moderator (Topic): {self.moderator_topic.agent_name}", extra={"msg_type": "system"})
        logger.debug(f"Moderator (Conviction): {self.moderator_conviction.agent_name}", extra={"msg_type": "system"})
        logger.debug(f"Moderator (Argument Quality): {self.moderator_argument_quality.agent_name}", extra={"msg_type": "system"})
        logger.debug(f"Moderator (Debate Quality): {self.moderator_debate_quality.agent_name}", extra={"msg_type": "system"})
        logger.debug(f"Max rounds limit set to: {self.max_rounds}", extra={"msg_type": "system"})

        return

    def _run_persuader_turn(self, previous_debater_response: str) -> str:
        """Run the persuader's turn in the debate."""
        # --- Add Turn Delay ---
        if self.turn_delay_seconds > 0:
            logger.info(f"Waiting for {self.turn_delay_seconds:.2f} seconds before persuader's turn.")
            time.sleep(self.turn_delay_seconds)
        # ----------------------

        # First round has no opponent message
        opponent_message = previous_debater_response if previous_debater_response else None
        persuader_response = self.persuader.call(opponent_message)
            
       # logger.debug(f"Persuader: {persuader_response}", extra={"msg_type": "main debate", "speaker": "persuader"})
        return persuader_response


    def _run_debater_turn(self, persuader_message: str) -> str:
        """Run the debater's turn in the debate."""
        # --- Add Turn Delay ---
        if self.turn_delay_seconds > 0:
            logger.info(f"Waiting for {self.turn_delay_seconds:.2f} seconds before debater's turn.")
            time.sleep(self.turn_delay_seconds)
        # ----------------------

        debater_response = self.debater.call(persuader_message)
        #logger.debug(f"Debater: {debater_response}", extra={"msg_type": "debate_verbose", "speaker": "debater"})
            
        return debater_response



    def _run_moderation_checks(self, persuader_memory: MemoryInterface, debater_memory: MemoryInterface) -> Tuple[bool, str]:
        """Run all moderation checks and return updated debate state.
        
        Returns:
            Tuple of (should_continue, status_tag)
            status_tag is "TERMINATE", "OFF-TOPIC", or "" (empty if checks pass)
        """
            
        # Get recent history for both checks (limit to last few messages)
        recent_history = self._get_recent_history(debater_memory, count=6)  # Last few messages sufficient for both checks
        moderator_logs = []

        # Run termination check
        should_continue, status_tag = self._run_termination_check(recent_history, moderator_logs)
        if not should_continue:
            return False, status_tag

        # Run topic check with same recent history
        is_on_topic, status_tag = self._run_topic_check(recent_history, moderator_logs)
        if not is_on_topic:
            return False, status_tag

        # append results to memories
        self._append_moderation_results_to_memories(persuader_memory, debater_memory, moderator_logs)
            
        return True, ""


    def _run_termination_check(self, history: List[Dict[str, str]], moderator_logs: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Run the termination check moderation and handle the result.
        
        Returns:
            Tuple of (should_continue, status_tag)
            status_tag is either "TERMINATE" or "KEEP-TALKING"
        """
       # logger.debug(f"Moderator checking termination...", extra={"msg_type": "main debate", "speaker": "moderator"})
        
        termination_result = self.moderator_terminator.call(history)
        moderator_logs.append({
            "moderator_name": self.moderator_terminator.agent_name,
            "raw_response": termination_result
        })
        raw_text = termination_result.strip().upper()
        if 'TERMINATE' in raw_text:
            logger.debug("Parser found TERMINATE signal." , extra={"msg_type": "main debate", "sender": "moderator"})
            return False, "TERMINATE"
        
        elif 'KEEP-TALKING' in raw_text:
            logger.debug("Parser found KEEP-TALKING signal." , extra={"msg_type": "main debate", "sender": "moderator"})
            return True, "KEEP-TALKING"
                
        else:
            # Default to TERMINATE on unparseable responses (safer - prevents resource waste)
            logger.error(f"Termination moderator returned unexpected response '{termination_result}'. Defaulting to TERMINATE for safety." , 
                        extra={"msg_type": "main debate", "sender": "moderator"})
            return False, "TERMINATE"


    def _run_topic_check(self, history: List[Dict[str, str]], moderator_logs: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Run the topic check moderation.
        
        Returns:
            Tuple of (is_on_topic, status_tag)
            status_tag is either "ON-TOPIC" or "OFF-TOPIC"
        """
       # logger.debug(f"Moderator checking topic...", extra={"msg_type": "main debate", "speaker": "moderator"})
        
        topic_result = self.moderator_topic.call(history)
        moderator_logs.append({
            "moderator_name": self.moderator_topic.agent_name,
            "raw_response": topic_result
        })
        raw_text = topic_result.strip().upper()
        
        if 'ON-TOPIC' in raw_text:
            return True, "ON-TOPIC"
        elif 'OFF-TOPIC' in raw_text:
            return False, "OFF-TOPIC"
        else: #TODO: Decide if this should be a warning or an error
            logger.warning(f"Topic check response format unclear: {topic_result}. Defaulting to on-topic." , extra={"msg_type": "main debate", "sender": "moderator"})
            return True, "ON-TOPIC"

    def _run_conviction_check(self, debater_response: str, debater_memory: MemoryInterface) -> Tuple[bool, Optional[int]]:
        """Run the conviction check moderation and return whether debate should end with conviction.
        
        Returns:
            Tuple of (is_convinced, conviction_rate)
            conviction_rate is an integer 1-10, or None if not parseable
        """
        # Get recent history for conviction check - focus on the last few exchanges
        recent_history = self._get_recent_history(debater_memory, count=4)  # Last 2 exchanges
        moderator_logs = []
        
        conviction_result = self.moderator_conviction.call(recent_history)
        moderator_logs.append({
            "moderator_name": self.moderator_conviction.agent_name,
            "raw_response": conviction_result
        })
        
        raw_text = conviction_result.strip().upper()
        
        # Log the raw moderator response for debugging
        logger.debug(f"Conviction moderator raw response: '{conviction_result}'",
                   extra={"msg_type": "main debate", "sender": "moderator"})
        
        # Parse conviction rate using helper
        conviction_rate = self._parse_rate(conviction_result, "conviction rate")
        
        if 'CONVINCED' in raw_text and 'NOT-CONVINCED' not in raw_text:
            logger.debug(f"Parser found CONVINCED signal with rate {conviction_rate}.", 
                       extra={"msg_type": "main debate", "sender": "moderator"})
            return True, conviction_rate
        
        elif 'NOT-CONVINCED' in raw_text:
            logger.debug(f"Parser found NOT-CONVINCED signal with rate {conviction_rate}.", 
                       extra={"msg_type": "main debate", "sender": "moderator"})
            return False, conviction_rate
        
        else:
            # Default to NOT-CONVINCED on unparseable responses (conservative - requires explicit conviction)
            logger.error(f"Conviction moderator returned unexpected response '{conviction_result}'. Defaulting to NOT-CONVINCED.", 
                       extra={"msg_type": "main debate", "sender": "moderator"})
            return False, conviction_rate

    def _run_argument_quality_check(self, persuader_memory: MemoryInterface, debater_memory: MemoryInterface) -> Optional[int]:
        """Run the argument quality check moderation and return the quality rating.
        
        Returns:
            argument_quality_rate: An integer 1-10 representing argument quality, or None if not parseable
        """
        # Get recent history for argument quality check - focus on the last few exchanges
        recent_history = self._get_recent_history(debater_memory, count=6)  # Last few exchanges
        moderator_logs = []
        
        argument_quality_result = self.moderator_argument_quality.call(recent_history)
        moderator_logs.append({
            "moderator_name": self.moderator_argument_quality.agent_name,
            "raw_response": argument_quality_result
        })
        
        raw_text = argument_quality_result.strip().upper()
        
        # Log the raw moderator response for debugging
        logger.debug(f"Argument quality moderator raw response: '{argument_quality_result}'",
                   extra={"msg_type": "main debate", "sender": "moderator"})
        
        # Parse argument quality rate using helper
        argument_quality_rate = self._parse_rate(argument_quality_result, "argument quality rate")
        if argument_quality_rate is not None:
            logger.debug(f"Parser found argument quality rate {argument_quality_rate}.",
                       extra={"msg_type": "main debate", "sender": "moderator"})
        
        # Append moderation results to memories
        self._append_moderation_results_to_memories(persuader_memory, debater_memory, moderator_logs)
        
        return argument_quality_rate

    def _run_debate_quality_check(self, topic_id: str, claim: str, chat_id: str, helper_type: str, log_config: Dict[str, Any]) -> Tuple[Optional[int], str]:
        """Run the debate quality moderator to assess the overall debate quality.
        
        Parses the debate transcript from the debate_main.log JSONL file to get the complete, un-summarized debate.
        
        Returns:
            Tuple of (debate_quality_rating, debate_quality_review)
            debate_quality_rating: An integer 1-10 representing overall debate quality, or None if not parseable
            debate_quality_review: A string containing the professional review
        """
        import json
        import os
        
        # Construct the log file path - debate_main.log is saved in debates/topic_id/helper_type/chat_id
        debates_base_dir = log_config.get('debates_base_dir', 'debates')
        log_directory = os.path.join(debates_base_dir, topic_id, helper_type, chat_id)
        log_file_path = os.path.join(log_directory, "debate_main.log")
        
        # Parse debate transcript from JSONL log file
        debate_transcript = []
        try:
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            log_entry = json.loads(line)
                            # Only process main debate messages
                            if log_entry.get('msg_type') == 'main debate':
                                sender = log_entry.get('sender', '')
                                message = log_entry.get('message', '')
                                
                                # Filter out moderator messages and system messages
                                if sender and sender.lower() not in ['moderator', 'orchestrator'] and message:
                                    # Map sender names to display names
                                    if 'persuader' in sender.lower() or sender.lower() == 'persuador':
                                        debate_transcript.append({"role": "PERSUADER", "content": message})
                                    elif 'debater' in sender.lower():
                                        debate_transcript.append({"role": "DEBATER", "content": message})
                        except json.JSONDecodeError:
                            # Skip invalid JSON lines
                            continue
            else:
                # Log at debug level instead of warning since fallback works fine
                logger.debug(f"Log file not found at {log_file_path}, falling back to memory", 
                             extra={"msg_type": "main debate", "sender": "moderator"})
                # Fallback to memory if log file doesn't exist yet
                persuader_history = self.persuader.memory.get_history_as_prompt()
                debater_history = self.debater.memory.get_history_as_prompt()
                
                if persuader_history:
                    debate_transcript.append({"role": "PERSUADER", "content": persuader_history[0].get('content', '')})
                
                min_len = min(len(persuader_history) - 1, len(debater_history))
                for i in range(min_len):
                    if i < len(debater_history):
                        debate_transcript.append({"role": "DEBATER", "content": debater_history[i].get('content', '')})
                    if i + 1 < len(persuader_history):
                        debate_transcript.append({"role": "PERSUADER", "content": persuader_history[i + 1].get('content', '')})
        except Exception as e:
            logger.error(f"Error parsing debate from log file {log_file_path}: {e}", 
                        extra={"msg_type": "main debate", "sender": "moderator"})
            # Fallback to empty transcript
            debate_transcript = []
        
        # Format transcript as text for the moderator
        if debate_transcript:
            transcript_text = "\n\n".join([f"{msg['role']}:\n{msg['content']}" for msg in debate_transcript])
        else:
            transcript_text = "No debate transcript available."
        
        # Format the full context for the moderator (includes the debate transcript)
        full_context = f"Here is the complete debate transcript:\n\n{transcript_text}"
        
        # Call the debate quality moderator
        debate_quality_result = self.moderator_debate_quality.call(full_context)
        
        # Log the raw moderator response
        logger.debug(f"Debate quality moderator raw response: '{debate_quality_result}'",
                   extra={"msg_type": "main debate", "sender": "moderator"})
        
        # Parse rating and review from response
        debate_quality_rating = None
        debate_quality_review = ""
        
        try:
            import re
            raw_text = debate_quality_result.strip()
            
            # Extract rating using helper (handles "Rating:" or "RATE:" patterns)
            debate_quality_rating = self._parse_rate(debate_quality_result, "debate quality rating")
            # Also try "RATING:" pattern specifically for debate quality
            if debate_quality_rating is None:
                rate_match = re.search(r'RATING:\s*(\d+)', raw_text, re.IGNORECASE)
                if rate_match:
                    rate = int(rate_match.group(1))
                    if 1 <= rate <= 10:
                        debate_quality_rating = rate
                    else:
                        logger.warning(f"Debate quality rating {rate} out of range 1-10, using None",
                                     extra={"msg_type": "main debate", "sender": "moderator"})
            
            # Extract review (everything after "Review:" or "REVIEW:")
            review_match = re.search(r'REVIEW:\s*(.+?)(?:\n\n|\Z)', raw_text, re.IGNORECASE | re.DOTALL)
            if review_match:
                debate_quality_review = review_match.group(1).strip()
            else:
                # If no explicit Review: tag, try to extract everything after the rating
                if rate_match:
                    review_start = rate_match.end()
                    debate_quality_review = raw_text[review_start:].strip()
                    # Remove any remaining "Review:" prefix if present
                    debate_quality_review = re.sub(r'^REVIEW:\s*', '', debate_quality_review, flags=re.IGNORECASE).strip()
                else:
                    debate_quality_review = raw_text
            
            if not debate_quality_review:
                debate_quality_review = "No review provided."
                
        except Exception as e:
            logger.warning(f"Could not parse debate quality rating/review from '{debate_quality_result}': {e}", 
                         extra={"msg_type": "main debate", "sender": "moderator"})
            debate_quality_review = debate_quality_result if debate_quality_result else "Error parsing review."
        
        return debate_quality_rating, debate_quality_review
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
        # Calculate token usage
        token_usage = self._calculate_token_usage()
        
        # Get feedback tags from persuader's memory
        feedback_tags = self.persuader.memory.get_feedback_tags()
        
        # Get conviction rates from debater's memory
        conviction_rates = self.debater.memory.get_conviction_rates()
        
        # Get argument quality rates from persuader's memory
        argument_quality_rates = self.persuader.memory.get_argument_quality_rates()
        
        # Run debate quality moderator to get overall debate rating and review
        # Only run if there is more than one round to the debate
        if round_number > 1:
            debate_quality_rating, debate_quality_review = self._run_debate_quality_check(topic_id, claim, chat_id, helper_type, log_config)
        else:
            debate_quality_rating = -1
            debate_quality_review = "Debate quality check skipped: only one round completed"
        
        # Log debate end with all metadata needed for HTML/XLSX generation, including token usage, feedback tags, conviction rates, argument quality rates, and debate quality
        logger.info(f"Debate ended with result: {final_result_status} !!!!", 
                   extra={"msg_type": "main debate", "sender": "orchestrator", "topic_id": topic_id, 
                          "chat_id": chat_id, "helper_type": helper_type, "result": final_result_status, 
                          "rounds": round_number, "finish_reason": finish_reason, "claim": claim,
                          "token_usage": token_usage, "feedback_tags": feedback_tags, "conviction_rates": conviction_rates,
                          "argument_quality_rates": argument_quality_rates, "debate_quality_rating": debate_quality_rating,
                          "debate_quality_review": debate_quality_review})
        
        # Log the debate quality review as a separate main debate message
        logger.info(f"Debate Quality Assessment - Rating: {debate_quality_rating}/10\nReview: {debate_quality_review}",
                   extra={"msg_type": "main debate", "sender": "moderator", "topic_id": topic_id, 
                          "chat_id": chat_id})

        # Return just the essential summary information
        return {
            "status": "Success",
            "result": final_result_status,
            "rounds": round_number,
            "finish_reason": finish_reason, 
            "total_tokens_estimate": token_usage,
            "feedback_tags": feedback_tags,
            "conviction_rates": conviction_rates,
            "argument_quality_rates": argument_quality_rates,
            "debate_quality_rating": debate_quality_rating,
            "debate_quality_review": debate_quality_review
        }
    #TODO, make the agents independent of the orchestrator
    def _calculate_token_usage(self) -> int:
        """Calculate token usage for all components, including memory operations."""
        # Get token counts directly from each agent
        persuader_tokens = self.persuader.get_total_token_usage()["total_tokens"]
        debater_tokens = self.debater.get_total_token_usage()["total_tokens"]
        term_mod_tokens = self.moderator_terminator.get_total_token_usage()["total_tokens"]
        topic_mod_tokens = self.moderator_topic.get_total_token_usage()["total_tokens"]
        conviction_mod_tokens = self.moderator_conviction.get_total_token_usage()["total_tokens"]
        arg_quality_mod_tokens = self.moderator_argument_quality.get_total_token_usage()["total_tokens"]
        debate_quality_mod_tokens = self.moderator_debate_quality.get_total_token_usage()["total_tokens"]
        
        # Helper tokens are tracked separately
        helper_tokens = self.persuader.helper_token_used if self.persuader.use_helper_feedback else 0
        
        total_tokens = persuader_tokens + debater_tokens + term_mod_tokens + topic_mod_tokens + conviction_mod_tokens + arg_quality_mod_tokens + debate_quality_mod_tokens + helper_tokens
        
        # Log the token counts
        logger.info(
            f"Token Estimates: Persuader={persuader_tokens}, "
            f"Debater={debater_tokens}, "
            f"Moderator={term_mod_tokens + topic_mod_tokens + conviction_mod_tokens + arg_quality_mod_tokens + debate_quality_mod_tokens}, "
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


    def _parse_rate(self, text: str, rate_name: str = "rate") -> Optional[int]:
        """Parse a rate (1-10) from moderator response text.
        
        Args:
            text: The text to parse
            rate_name: Name for logging (e.g., "conviction rate", "argument quality rate")
            
        Returns:
            Integer 1-10 if found and valid, None otherwise
        """
        raw_text = text.strip().upper()
        try:
            # Try multiple patterns in order of specificity
            rate_match = re.search(r'RATE:\s*(\d+)', raw_text)
            if not rate_match:
                rate_match = re.search(r'-\s*(\d+)', raw_text)
            if not rate_match:
                rate_match = re.search(r'\b(\d{1,2})\b', raw_text)
            
            if rate_match:
                rate = int(rate_match.group(1))
                if 1 <= rate <= 10:
                    return rate
                else:
                    logger.warning(f"{rate_name.capitalize()} {rate} out of range 1-10, using None",
                                 extra={"msg_type": "main debate", "sender": "moderator"})
        except Exception as e:
            logger.warning(f"Could not parse {rate_name} from '{text}': {e}", 
                         extra={"msg_type": "main debate", "sender": "moderator"})
        return None
