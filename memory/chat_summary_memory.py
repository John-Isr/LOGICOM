from typing import List, Dict, Any, Optional
from copy import deepcopy
import logging
import tiktoken 

from core.interfaces import MemoryInterface, LLMInterface

logger = logging.getLogger(__name__)

# Default settings if not provided via config
DEFAULT_SUMMARIZATION_TRIGGER_TOKENS = 6000
DEFAULT_TARGET_PROMPT_TOKENS = 4000
DEFAULT_KEEP_MESSAGES_AFTER_SUMMARY = 4
DEFAULT_SUMMARIZER_ENCODING = "cl100k_base" 

class ChatSummaryMemory(MemoryInterface):
    """Stores chat history, using LLM summarization to manage context length based on token counts."""
    
    def __init__(self,
                 summarizer_llm: Optional[LLMInterface] = None,
                 summarization_trigger_tokens: int = DEFAULT_SUMMARIZATION_TRIGGER_TOKENS,
                 target_prompt_tokens: int = DEFAULT_TARGET_PROMPT_TOKENS,
                 keep_messages_after_summary: int = DEFAULT_KEEP_MESSAGES_AFTER_SUMMARY,
                 role_map: Optional[Dict[str, str]] = None,
                 encoding_name: str = DEFAULT_SUMMARIZER_ENCODING):

        self.summarizer_llm = summarizer_llm
        # Store new config values
        self.summarization_trigger_tokens = summarization_trigger_tokens
        self.target_prompt_tokens = target_prompt_tokens # Goal after summarization
        self.keep_messages_after_summary = keep_messages_after_summary
        self.role_map = role_map or {'user': 'user', 'ai': 'assistant'}
        self.encoding_name = encoding_name
        try:
            self.tokenizer = tiktoken.get_encoding(self.encoding_name)
        except ValueError:
            logger.warning(f"Encoding '{self.encoding_name}' not found. Using default cl100k_base.")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            
        # Check if summarization is possible
        self.summarization_enabled = bool(self.summarizer_llm)
        if not self.summarization_enabled:
             logger.warning("Summarizer LLM not provided. Summarization disabled.")

        # Stores the potentially summarized conversation history for generating prompts
        self.messages: List[Dict[str, str]] = []
        # Stores the full, detailed, un-summarized log
        self.log: List[Any] = []

    def add_user_message(self, message: str) -> None:
        """Adds a user message, logs it, and checks context length."""
        user_role = self.role_map.get('user', 'user')
        entry = {"role": user_role, "content": message}
        self.messages.append(entry)
        self.log.append({"type": "message", "data": deepcopy(entry)})
        self._check_context_length()

    def add_ai_message(self, message: str, **kwargs) -> None:
        """Adds an AI message, logs it, and checks context length."""
        ai_role = self.role_map.get('ai', 'assistant')
        entry = {"role": ai_role, "content": message}
        self.messages.append(entry)
        self.log.append({"type": "message", "data": deepcopy(entry), "metadata": kwargs})
        self._check_context_length()

    def get_prompt(self) -> List[Dict[str, str]]:
        """Returns the current (potentially summarized) user/assistant conversation history."""
        return deepcopy(self.messages)

    def get_history(self) -> List[Any]:
        """Returns the detailed, *un-summarized* conversation log."""
        return deepcopy(self.log)

    def get_last_ai_message(self) -> str:
        """Returns the content of the most recent AI message from the prompt history."""
        ai_role = self.role_map.get('ai', 'assistant')
        for message in reversed(self.messages):
            if message.get('role') == ai_role:
                return message.get('content', "")
        return ""

    def reset(self) -> None:
        """Resets the memory, clearing messages and log."""
        self.messages = []
        self.log = []

    def _calculate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimates token count for a list of messages using tiktoken."""
        num_tokens = 0
        # OpenAI counts tokens per message based on role and content
        # Using a simplified approach: count tokens for content only
        # Add a small buffer per message as an approximation for role/metadata tokens
        tokens_per_message = 4 # Approximation
        for message in messages:
            num_tokens += tokens_per_message
            content = message.get("content")
            if content:
                try:
                    num_tokens += len(self.tokenizer.encode(content))
                except Exception as e:
                     logger.error(f"Tiktoken encoding failed for content: {content[:50]}... Error: {e}")
                     # Fallback: estimate based on characters/words
                     num_tokens += len(content) // 4 
        num_tokens += 2 # Add tokens for end-of-list approximation
        return num_tokens

    def _check_context_length(self) -> None:
        """Checks token count and triggers summarization if trigger threshold is exceeded."""
        if not self.summarization_enabled:
            return # Cannot summarize if not configured
            
        current_tokens = self._calculate_tokens(self.messages)
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
        if not self.summarization_enabled:
            # This case should ideally not be reached if _check_context_length checks first
            logger.error("Summarization called but not enabled/configured.") 
            # Raise error as we cannot manage context if summarization was expected
            raise RuntimeError("Summarization expected but not configured.") 

        num_to_keep = self.keep_messages_after_summary
        if len(self.messages) <= num_to_keep:
            logger.info("Not enough messages to summarize significantly.")
            return

        messages_to_summarize = self.messages[:-num_to_keep]
        messages_kept = self.messages[-num_to_keep:]
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages_to_summarize])
        
        summary = None
        try:
            # Use hardcoded prompt via f-string for conciseness
            user_content = f"Summarize this conversation history:\n{history_text}"
            summarizer_prompt = [{"role": "user", "content": user_content}]
            
            logger.info(f"Calling summarizer LLM ({self.summarizer_llm.__class__.__name__}) to summarize {len(messages_to_summarize)} messages...")
            summary = self.summarizer_llm.generate(summarizer_prompt)
            logger.info("Summarization attempt complete.")

            if summary and not summary.startswith("Error:"): 
                summary_message = {"role": "system", "content": f"Summary of prior conversation: {summary}"}
                self.messages = [summary_message] + messages_kept
                self.log.append({"type": "summarization", "data": {"summary": summary, "messages_summarized": len(messages_to_summarize)}})
                new_token_count = self._calculate_tokens(self.messages)
                logger.info(f"History summarized. New token count estimate: {new_token_count}")
            else:
                 # Summarization failed (empty or error response)
                 error_msg = f"Summarization failed or returned empty/error: {summary}"
                 logger.error(error_msg)
                 # Stop execution as context cannot be managed reliably
                 raise RuntimeError(error_msg) 

        except Exception as e:
            # Summarization failed (exception during call)
            error_msg = f"Error during summarization process: {e}"
            logger.error(error_msg, exc_info=True)
            # Stop execution as context cannot be managed reliably
            raise RuntimeError(error_msg) from e 