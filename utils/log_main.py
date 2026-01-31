import logging
import json
import datetime as st
from typing import override
import queue
import logging.config
import os

# Create a queue for log records
log_queue = queue.Queue(-1)  # No limit on size

logger = logging.getLogger("debate_logger")

class MyJSONFormatter(logging.Formatter):
    def __init__(self, fmt_keys: dict):
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}
        super().__init__()
    
    @override
    def format(self, record: logging.LogRecord):
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)
    
    @override
    def _prepare_log_dict(self, record: logging.LogRecord):
        always_include = {
            "message": record.getMessage(),
            "timestamp": st.datetime.fromtimestamp(record.created, tz=st.timezone.utc).isoformat(), # TODO: change to local time
        }
        
        message = {}
        for key, val in self.fmt_keys.items():
            if val in always_include:
                message[key] = always_include.pop(val)
            else:
                message[key] = getattr(record, val)
        
        # Add specific extra attributes we care about
        for attr in ["msg_type", "speaker", "receiver", "sender", "round", "topic", 
                    "topic_id", "chat_id", "helper_type", "result", "finish_reason", 
                    "rounds", "claim", "token_usage", "feedback_tags", "feedback_tag", 
                    "conviction_rates", "conviction_rate"]:
            if hasattr(record, attr) and attr not in message:
                message[attr] = getattr(record, attr)
        
        message.update(always_include)
        return message

# Filter classes for directing logs to the right handlers
class MainDebateFilter(logging.Filter):
    """Filter that only allows log records related to the main debate"""
    def filter(self, record):
        return getattr(record, "msg_type", None) == "main debate"

class PersuadorHelperFilter(logging.Filter):
    """Filter that only allows log records related to persuador-helper communication"""
    def filter(self, record):
        return getattr(record, "msg_type", None) == "helper_operation"

class SystemMessageFilter(logging.Filter):
    """Filter that only allows log records with msg_type = 'system'"""
    def filter(self, record):
        return getattr(record, "msg_type", None) == "system"

def setup_logging(log_directory: str = "logs"):
    """Initialize logging configuration with proper queue setup
    
    Args:
        log_directory: Directory where log files should be written. Defaults to 'logs'.
    """
    
    # Ensure log directory exists
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
        
    # CRITICAL: Remove all existing handlers from the logger to avoid conflicts
    # This allows each debate instance to have its own set of handlers
    for handler in logger.handlers[:]:  # Use slice to copy list since we're modifying it
        handler.close()
        logger.removeHandler(handler)
    
    # Load the logging configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'log.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Update file paths in handlers to use the specified log directory
        for handler_name, handler_config in config.get('handlers', {}).items():
            if 'filename' in handler_config:
                # Extract just the filename from the original path
                original_filename = os.path.basename(handler_config['filename'])
                # Set new path using the specified log directory
                handler_config['filename'] = os.path.join(log_directory, original_filename)
            
        # Configure the queue handler to use our queue
        if 'handlers' in config and 'queue_handler' in config['handlers']:
            config['handlers']['queue_handler']['queue'] = log_queue
            
        # Set incremental to False to prevent adding duplicate handlers
        # This replaces the configuration instead of adding to it
        config['incremental'] = False
        
        logging.config.dictConfig(config)
        
        # Start a QueueListener to handle records from the queue if needed
        # This is commented out as you might want to customize how you process the queue
        # from logging.handlers import QueueListener
        # listener = QueueListener(log_queue, <handlers>)
        # listener.start()
    else:
        logging.basicConfig(level=logging.INFO)
        logger.warning(f"Logging config not found at {config_path}. Using basic configuration.")

    # --- Suppress noisy logs from underlying libraries ---
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("google.api_core").setLevel(logging.WARNING)
    # --------------------------------



