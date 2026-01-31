from typing import List, Dict, Any, Optional
from copy import deepcopy
import logging

from core.interfaces import MemoryInterface, LLMInterface, INTERNAL_USER_ROLE, INTERNAL_AI_ROLE
from utils.token_utils import calculate_chat_tokens, calculate_string_tokens
from utils.log_main import logger


class ChatSummaryMemory(MemoryInterface):
    """Stores chat history, using LLM summarization to manage context length based on token counts."""
    
    def __init__(self,
                 summarizer_llm: LLMInterface,
                 summarization_trigger_tokens: int, 
                 target_prompt_tokens: int,
                 keep_messages_after_summary: int):

        self.summarizer_llm = summarizer_llm
        self.summarization_trigger_tokens = summarization_trigger_tokens
        self.target_prompt_tokens = target_prompt_tokens 
        self.keep_messages_after_summary = keep_messages_after_summary
            
        self.messages: List[Dict[str, str]] = []
        self.log: List[Any] = []
        
        # Token tracking for memory operations
        self.prompt_tokens_used: int = 0
        self.completion_tokens_used: int = 0
        self.total_tokens_used: int = 0
        
        # Cache for current messages token count (invalidated when messages change)
        self._cached_token_count: Optional[int] = None
        
        # Persistent feedback tags storage (survives summarization)
        self.feedback_tags: List[Optional[str]] = []
        
        # Persistent conviction rates storage (survives summarization)
        self.conviction_rates: List[Optional[int]] = []
        
        # Persistent argument quality rates storage (survives summarization)
        self.argument_quality_rates: List[Optional[int]] = []

    def add_user_message(self, message: str) -> None:
        """Adds a user message using the internal standard role."""
        logger.debug("Adding user message to memory", extra={"msg_type": "memory_operation", "role": INTERNAL_USER_ROLE}) #TODO:: check_log
        entry = {"role": INTERNAL_USER_ROLE, "content": message}
        self.messages.append(entry)
        self.log.append({"type": "message", "data": deepcopy(entry)})
        self._cached_token_count = None  # Invalidate cache
        self._check_context_length()

    def add_ai_message(self, message: str, **kwargs) -> None:
        """Adds an AI message using the internal standard role."""
        logger.debug("Adding AI message to memory", extra={"msg_type": "memory_operation", "role": INTERNAL_AI_ROLE}) #TODO:: check_log
        entry = {"role": INTERNAL_AI_ROLE, "content": message}
        self.messages.append(entry)
        self.log.append({"type": "message", "data": deepcopy(entry), "metadata": kwargs})
        
        # Store feedback_tag in persistent field (survives summarization)
        feedback_tag = kwargs.get('feedback_tag')
        self.feedback_tags.append(feedback_tag)
        
        # Store conviction_rate in persistent field (survives summarization)
        conviction_rate = kwargs.get('conviction_rate')
        self.conviction_rates.append(conviction_rate)
        
        # Store argument_quality_rate in persistent field (survives summarization)
        argument_quality_rate = kwargs.get('argument_quality_rate')
        self.argument_quality_rates.append(argument_quality_rate)
        
        self._cached_token_count = None  # Invalidate cache
        self._check_context_length()

    def get_history_as_prompt(self) -> List[Dict[str, str]]:
        """Returns the current history using internal standard roles."""
        return deepcopy(self.messages)

    def get_history(self) -> List[Any]:
        """Returns the detailed, *un-summarized* conversation log."""
        return deepcopy(self.log)

    def get_last_ai_message(self) -> str:
        """Returns the content of the most recent AI message using the internal standard role."""
        for message in reversed(self.messages):
            if message.get('role') == INTERNAL_AI_ROLE:
                return message.get('content', "")
        return ""

    def reset(self) -> None:
        """Resets the memory, clearing messages, log, feedback tags, conviction rates, and argument quality rates."""
        self.messages = []
        self.log = []
        self.feedback_tags = []
        self.conviction_rates = []
        self.argument_quality_rates = []
        self.prompt_tokens_used = 0
        self.completion_tokens_used = 0
        self.total_tokens_used = 0
        self._cached_token_count = None

    def get_token_usage(self) -> Dict[str, int]:
        """Returns the token usage by memory operations (primarily summarization)."""
        return {
            "prompt_tokens": self.prompt_tokens_used,
            "completion_tokens": self.completion_tokens_used,
            "total_tokens": self.total_tokens_used
        }

    def get_feedback_tags(self) -> List[Optional[str]]:
        """Returns the list of feedback tags collected during the conversation."""
        return self.feedback_tags.copy()

    def get_conviction_rates(self) -> List[Optional[int]]:
        """Returns the list of conviction rates collected during the conversation."""
        return self.conviction_rates.copy()

    def get_argument_quality_rates(self) -> List[Optional[int]]:
        """Returns the list of argument quality rates collected during the conversation."""
        return self.argument_quality_rates.copy()

    def update_last_argument_quality_rate(self, rate: Optional[int]) -> None:
        """Updates the argument quality rate for the most recent AI message."""
        if len(self.argument_quality_rates) > 0:
            self.argument_quality_rates[-1] = rate

    def update_last_conviction_rate(self, rate: Optional[int]) -> None:
        """Updates the conviction rate for the most recent AI message."""
        if len(self.conviction_rates) > 0:
            self.conviction_rates[-1] = rate

    def _check_context_length(self) -> None:
        """Checks token count and triggers summarization if trigger threshold is exceeded."""
        # Allow disabling summarization via trigger_tokens <= 0
        if self.summarization_trigger_tokens <= 0:
            logger.debug("Summarization trigger token limit is 0 or less. Skipping context length check.", extra={"msg_type": "memory_operation"})
            return
        
        # Use cached token count if available, otherwise calculate and cache
        if self._cached_token_count is None:
            self._cached_token_count = calculate_chat_tokens(self.messages)
        current_tokens = self._cached_token_count
        logger.debug(f"Current prompt token count estimate: {current_tokens}", extra={"msg_type": "memory_operation"})

        # Trigger based on summarization_trigger_tokens
        if current_tokens > self.summarization_trigger_tokens:
            logger.warning(
                f"Token count ({current_tokens}) exceeds trigger threshold ({self.summarization_trigger_tokens}). "
                f"Attempting summarization (Target prompt size: {self.target_prompt_tokens})...",
                extra={"msg_type": "memory_operation"}
            )
            self._summarize()
        
    def _summarize(self) -> None:
        """Summarizes the chat history using the configured LLM and prompt.
           Raises RuntimeError if summarization fails.
        """
        num_to_keep = self.keep_messages_after_summary
        # Need at least: first message + messages to summarize + last messages to keep
        if len(self.messages) <= num_to_keep + 1:
            logger.info("Not enough messages to summarize significantly.", extra={"msg_type": "memory_operation"})
            return

        first_message = self.messages[0]
        messages_to_summarize = self.messages[1:-num_to_keep]
        messages_kept = self.messages[-num_to_keep:]
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages_to_summarize])
        
        summary = None
        # Use hardcoded prompt via f-string for conciseness
        user_content = f"Summarize this conversation history:\n{history_text}"
        summarizer_prompt = [{"role": "user", "content": user_content}]
        
        # Calculate prompt tokens
        prompt_tokens = calculate_chat_tokens(summarizer_prompt)
        
        logger.info(f"Calling summarizer LLM to summarize messages.", extra={"msg_type": "memory_operation"})
        summary = self.summarizer_llm.generate(summarizer_prompt)
        # Calculate completion tokens
        completion_tokens = calculate_string_tokens(summary) if summary else 0
        
        # Update token tracking
        self.prompt_tokens_used += prompt_tokens
        self.completion_tokens_used += completion_tokens
        self.total_tokens_used += prompt_tokens + completion_tokens

        # Check if the summary is valid (i.e., not None or empty)
        if summary: # Simplified check
            # Process successful summary
            # Prepend summary using the standard AI role for compatibility
            summary_content = f"Summary of prior conversation: {summary}"
            summary_message = {"role": INTERNAL_AI_ROLE, "content": summary_content}
            self.messages = [first_message, summary_message] + messages_kept
            # Also add the summarization action to the detailed log
            self.log.append({
                "type": "summarization", 
                "data": {"summary": summary, "messages_summarized": len(messages_to_summarize)}, 
                "context_injection": summary_content
            })
            # Recalculate and cache token count after summarization
            self._cached_token_count = calculate_chat_tokens(self.messages)
            logger.info(f"History summarized. New token count estimate: {self._cached_token_count}", extra={"msg_type": "memory_operation"})
        else:
            # Handle case where generate() succeeded but returned None or empty string

            logger.error("Summarization failed: LLM returned empty summary.", extra={"msg_type": "memory_operation"})
            raise RuntimeError("Summarization failed: LLM returned empty summary.") 

