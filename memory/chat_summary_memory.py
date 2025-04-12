from typing import List, Dict, Any, Optional
from copy import deepcopy
import logging

from core.interfaces import MemoryInterface, LLMInterface, INTERNAL_USER_ROLE, INTERNAL_AI_ROLE
from utils.token_utils import calculate_chat_tokens, calculate_string_tokens

logger = logging.getLogger(__name__)


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

    def add_user_message(self, message: str) -> None:
        """Adds a user message using the internal standard role."""
        entry = {"role": INTERNAL_USER_ROLE, "content": message}
        self.messages.append(entry)
        self.log.append({"type": "message", "data": deepcopy(entry)})
        self._check_context_length()

    def add_ai_message(self, message: str, **kwargs) -> None:
        """Adds an AI message using the internal standard role."""
        entry = {"role": INTERNAL_AI_ROLE, "content": message}
        self.messages.append(entry)
        self.log.append({"type": "message", "data": deepcopy(entry), "metadata": kwargs})
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
        """Resets the memory, clearing messages and log."""
        self.messages = []
        self.log = []
        self.prompt_tokens_used = 0
        self.completion_tokens_used = 0
        self.total_tokens_used = 0

    def get_token_usage(self) -> Dict[str, int]:
        """Returns the token usage by memory operations (primarily summarization)."""
        return {
            "prompt_tokens": self.prompt_tokens_used,
            "completion_tokens": self.completion_tokens_used,
            "total_tokens": self.total_tokens_used
        }

    def _check_context_length(self) -> None:
        """Checks token count and triggers summarization if trigger threshold is exceeded."""
        # Allow disabling summarization via trigger_tokens <= 0
        if self.summarization_trigger_tokens <= 0:
            logger.debug("Summarization trigger token limit is 0 or less. Skipping context length check.")
            return
            
        current_tokens = calculate_chat_tokens(self.messages)
        logger.debug(f"Current prompt token count estimate: {current_tokens}")

        # Trigger based on summarization_trigger_tokens
        if current_tokens > self.summarization_trigger_tokens:
            logger.warning(
                f"Token count ({current_tokens}) exceeds trigger threshold ({self.summarization_trigger_tokens}). "
                f"Attempting summarization (Target prompt size: {self.target_prompt_tokens})..."
            )
            self._summarize()
        
    def _summarize(self) -> None:
        """Summarizes the chat history using the configured LLM and prompt.
           Raises RuntimeError if summarization fails.
        """
        num_to_keep = self.keep_messages_after_summary
        if len(self.messages) <= num_to_keep:
            logger.info("Not enough messages to summarize significantly.")
            return

        messages_to_summarize = self.messages[:-num_to_keep]
        messages_kept = self.messages[-num_to_keep:]
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages_to_summarize])
        
        summary = None
        # Use hardcoded prompt via f-string for conciseness
        user_content = f"Summarize this conversation history:\n{history_text}"
        summarizer_prompt = [{"role": "user", "content": user_content}]
        
        # Calculate prompt tokens
        prompt_tokens = calculate_chat_tokens(summarizer_prompt)
        
        logger.info(f"Calling summarizer LLM ({self.summarizer_llm.__class__.__name__}) to summarize {len(messages_to_summarize)} messages...")
        summary = self.summarizer_llm.generate(summarizer_prompt)
        logger.info("Summarization attempt complete.")

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
            self.messages = [summary_message] + messages_kept
            # Also add the summarization action to the detailed log
            self.log.append({
                "type": "summarization", 
                "data": {"summary": summary, "messages_summarized": len(messages_to_summarize)}, 
                "context_injection": summary_content
            })
            new_token_count = calculate_chat_tokens(self.messages)
            logger.info(f"History summarized. New token count estimate: {new_token_count}")
        else:
            # Handle case where generate() succeeded but returned None or empty string
            error_msg = f"Summarization failed: LLM returned empty summary."
            logger.error(error_msg)
            raise RuntimeError(error_msg) 

