import os
import json
import html
import shutil
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging 

# Direct import from project structure
from core.interfaces import INTERNAL_USER_ROLE, INTERNAL_AI_ROLE

logger = logging.getLogger(__name__) # Added

# --- Prompt Handling --- 

def replace_variables(text: str, variables: Dict[str, Any]) -> str:
    """Replaces placeholder variables in a string with their values."""
    if not variables:
        return text
    for variable, value in variables.items():
        # Ensure placeholders are distinct (e.g., <VARIABLE>)
        placeholder = f"<{variable.strip('<> ')}>" 
        text = text.replace(placeholder, str(value))
    return text

def load_prompt_template(file_path: str, variables: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[str]]:
    """
    Loads a prompt template from a file, extracts system and assistant sections,
    and replaces variables.
    Assumes specific markers: ==== SYSTEM ==== and ==== ASSISTANT ====
    
    Args:
        file_path: Path to the prompt template file.
        variables: Dictionary of variables to replace in the prompts.

    Returns:
        A tuple (system_prompt, assistant_prompt). Assistant prompt can be None.
        Returns (full_content, None) if markers are not found.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        logger.error(f"Prompt file not found at {file_path}") # Use logger
        raise # Re-raise the FileNotFoundError
    except Exception as e:
        logger.error(f"Error reading prompt file {file_path}: {e}", exc_info=True) # Use logger
        return f"Error reading prompt file: {e}", None

    system_start_marker = "==== SYSTEM ===="
    assistant_start_marker = "==== ASSISTANT ====" # Using ASSISTANT marker as in original

    system_start_idx = content.find(system_start_marker)
    assistant_start_idx = content.find(assistant_start_marker)

    if system_start_idx != -1 and assistant_start_idx != -1:
        system_start = system_start_idx + len(system_start_marker)
        system_end = assistant_start_idx
        system_text = content[system_start:system_end].strip()

        assistant_start = assistant_start_idx + len(assistant_start_marker)
        assistant_text = content[assistant_start:].strip()
        
        # Replace variables if provided
        system_text = replace_variables(system_text, variables or {})
        assistant_text = replace_variables(assistant_text, variables or {})
        
        return system_text, assistant_text
    elif system_start_idx != -1:
        # Only system prompt found
        system_start = system_start_idx + len(system_start_marker)
        system_text = content[system_start:].strip()
        system_text = replace_variables(system_text, variables or {})
        return system_text, None
    else:
        # No markers found, treat entire file as system prompt (or maybe user? TBD)
        logger.warning(f"Markers not found in {file_path}. Treating full content as system prompt.") # Use logger
        full_content = replace_variables(content, variables or {})
        return full_content, None

# --- Variable Extraction --- 

def extract_claim_data_for_prompt(claim_data: pd.Series, column_mapping: Dict[str, str]) -> Dict[str, str]:
    """
    Extracts variables for prompts from input data using configurable column names.
    Currently extracts: TOPIC, CLAIM, REASON.
    
    Args:
        claim_data: The data record (e.g., a row from a pandas DataFrame).
        column_mapping: Dictionary mapping standard prompt variable names 
                        (e.g., "TOPIC", "CLAIM", "ORIGINAL_TEXT", "REASON", 
                         "WARRANT_ONE", "WARRANT_TWO") to the actual column 
                         names in the claim_data Series.

    Returns:
        Dictionary of variables for the prompt template.
    """
    base_vars = {}
    # Map standard variables to columns using the provided mapping
    # Only extract variables currently used in prompts
    vars_to_extract = ["TOPIC", "CLAIM", "REASON"] 
    for var_name in vars_to_extract:
        column_name = column_mapping.get(var_name)
        if column_name:
            base_vars[var_name] = str(claim_data.get(column_name, ''))
        else:
            logger.warning(f"No column mapping provided for standard variable '{var_name}'. It will be empty.")
            base_vars[var_name] = ''


    return base_vars

# --- Logging --- 

def create_directory(directory_path: str, overwrite: bool = True) -> None:
    """Creates a directory, optionally removing it first if it exists."""
    if overwrite and os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
            logger.info(f"Removed existing directory: '{directory_path}'") # Use logger
        except OSError as e:
            logger.error(f"Error removing directory '{directory_path}': {e}") # Use logger
            # Decide if we should proceed or raise error
            return # Exit if removal failed
    elif not overwrite and os.path.exists(directory_path):
         logger.info(f"Directory '{directory_path}' already exists and overwrite is False. Skipping creation.") # Use logger
         return

    try:
        os.makedirs(directory_path, exist_ok=True) # exist_ok=True handles race conditions
        logger.info(f"Ensured directory exists: '{directory_path}'") # Use logger
    except OSError as e:
        logger.error(f"Error creating directory '{directory_path}': {e}")
        raise # Re-raise error if directory creation fails

def _save_log_json(log_history: List[Any], file_path: str, metadata: Dict[str, Any]) -> None:
    """Saves the conversation log history and metadata as a JSON file."""
    data_to_save = {
        **metadata, 
        "log": log_history # Assumes log_history is directly serializable
    }
    try:
        with open(file_path, "w", encoding='utf-8') as json_file:
            json.dump(data_to_save, json_file, indent=4)
        logger.info(f"Saved JSON log to: {file_path}") # Use logger
    except Exception as e:
        logger.error(f"Error saving JSON log to {file_path}: {e}", exc_info=True) # Use logger

def _save_log_html(log_history: List[Any], file_path: str, metadata: Dict[str, Any]) -> None:
    """Saves the conversation log history as an HTML file."""
    colors = ['blue', 'green']  # Original: Persuader=blue, Debater=green?
    round_number_color = 'red'
    
    html_content = '<html><head><meta charset="UTF-8"><style>'
    html_content += 'body { font-family: Arial, sans-serif; }'
    html_content += '.log-container { width: 80%; margin: 0 auto; word-wrap: break-word; white-space: pre-wrap; }'
    html_content += f'.round-number {{ color: {round_number_color}; font-weight: bold; margin-top: 1em; }}'
    html_content += f'.log-entry {{ font-weight: normal; margin-left: 1em; margin-bottom: 0.5em;}}'
    html_content += f'.role-label {{ font-weight: bold; margin-right: 0.5em;}}'
    html_content += '</style></head><body>'
    html_content += '<div class="log-container">'
    html_content += '<h1>Debate Log</h1>'
    html_content += f'<p><strong>Topic ID:</strong> {html.escape(str(metadata.get("Topic_ID", "N/A")))}</p>'
    html_content += f'<p><strong>Chat ID:</strong> {html.escape(str(metadata.get("Chat_ID", "N/A")))}</p>'
    html_content += f'<p><strong>Helper Type:</strong> {html.escape(str(metadata.get("Helper_Type", "N/A")))}</p>'
    html_content += f'<hr>'

    round_num = 0
    # Flag to track if the round number has been printed for the current Persuader turn
    round_printed_for_persuader = True # Start true so Debater doesn't trigger initially

    # Assuming Persuader is 'assistant' role and Debater is 'user' role for display
    # This is an assumption based on Persuader often starting.
    # A more robust solution would require explicit roles passed in metadata.
    role_display_map = {
        'assistant': 'Persuader', 
        'user': 'Debater'
    }
    # Colors assigned based on assumed roles
    role_color_map = {
         'Persuader': 'blue', # Persuader = blue
         'Debater': 'green' # Debater = green
    }

    for i, entry in enumerate(log_history):
        if entry.get('type') == 'message':
            data = entry.get('data', {})
            role = data.get('role') # Get the actual role (e.g., 'user', 'assistant')
            content = data.get('content')
            metadata_msg = entry.get('metadata') # e.g., fallacy info

            if role and content:
                 # Determine display role and color based on actual role
                 display_role = role_display_map.get(role, role.capitalize()) # Fallback to capitalized role
                 color = role_color_map.get(display_role, 'black') # Fallback to black

                 # Increment round number and print heading BEFORE the Persuader's message
                 if display_role == 'Persuader' and not round_printed_for_persuader:
                      round_num += 1
                      html_content += f'<div class="round-number">Round {round_num}</div>'
                      round_printed_for_persuader = True # Mark as printed for this turn
                 elif display_role == 'Debater':
                      round_printed_for_persuader = False # Reset flag after Debater speaks
                
                 html_content += f'<div class="log-entry" style="color:{color};">'
                 html_content += f'<span class="role-label">{html.escape(display_role)}:</span>'
                 html_content += f'<span>{html.escape(content)}</span>'
                 if metadata_msg:
                     html_content += f'<br><small><i>Metadata: {html.escape(str(metadata_msg))}</i></small>'
                 html_content += '</div>'
                 # Removed turn_in_round counter

    html_content += '<hr>'                 
    html_content += f'<p><strong>Result:</strong> {html.escape(str(metadata.get("Result", "N/A")))}</p>'
    html_content += f'<p><strong>Stop Reason:</strong> {html.escape(str(metadata.get("Stop_Reason", "N/A")))}</p>'
    html_content += f'<p><strong>Number of Rounds:</strong> {html.escape(str(metadata.get("Number_of_Rounds", "N/A")))}</p>'
    html_content += '</div></body></html>'

    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(html_content)
        logger.info(f"Saved HTML log to: {file_path}") # Use logger
    except Exception as e:
        logger.error(f"Error saving HTML log to {file_path}: {e}", exc_info=True) # Use logger

def _save_log_txt(log_history: List[Any], file_path: str, metadata: Dict[str, Any]) -> None:
    """Saves the conversation log history as a plain text file (JSON dump)."""
    # Original just dumped the list of messages, let's dump the whole log with metadata
    data_to_save = {
        **metadata,
        "log": log_history 
    }
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data_to_save, file, indent=4)
            # Alternatively, write a more human-readable text format:
            # file.write(f"Topic ID: {metadata.get('Topic_ID', 'N/A')}\n")
            # ... other metadata ...
            # for entry in log_history:
            #     if entry.get('type') == 'message': ... write role: content ...
        logger.info(f"Saved TXT log (JSON dump) to: {file_path}") # Use logger
    except Exception as e:
        logger.error(f"Error saving TXT log to {file_path}: {e}", exc_info=True) # Use logger

def _save_log_xlsx(xlsx_path: str, data_to_append: Dict[str, Any]) -> None:
    """
    Appends or updates a record in an Excel file summary.

    Args:
        xlsx_path: Path to the Excel file.
        data_to_append: Dictionary containing data for one row. Keys should match
                          potential columns. Must include 'Topic_ID' and the 
                          specific 'helper_type' fields being updated.
    """
    topic_id = data_to_append.get("Topic_ID")
    helper_type = data_to_append.get("Helper_Type")
    result = data_to_append.get("Result")
    num_rounds = data_to_append.get("Number_of_Rounds")
    chat_id = data_to_append.get("Chat_ID")
    claim = data_to_append.get("Claim", "") # Get claim if available

    if not all([topic_id, helper_type]):
        logger.error("Missing Topic_ID or Helper_Type for XLSX save.") # Use logger
        return

    # Define expected columns based on original structure
    # This structure might need updating based on refactored goals
    all_columns = ["Topic_ID", "Claim", 
                   "No_Helper", "Vanilla_Helper", "Fallacy_Helper", # Result columns per helper type
                   "No_Helper_Round", "Vanilla_Helper_Round", "Fallacy_Helper_Round", # Round columns
                   "Chat_ID_No_Helper", "Chat_ID_Vanilla_Helper", "Chat_ID_Fallacy_Helper"] # Chat ID columns
    
    result_col = f'{helper_type}'
    round_col = f'{helper_type}_Round'
    chat_id_col = f'Chat_ID_{helper_type}'

    if not all([c in all_columns for c in [result_col, round_col, chat_id_col]]):
         logger.error(f"Helper type '{helper_type}' does not map to valid XLSX columns.") # Use logger
         return

    try:
        if os.path.exists(xlsx_path):
            df = pd.read_excel(xlsx_path)
            # Ensure all expected columns exist, add if missing
            for col in all_columns:
                 if col not in df.columns:
                      df[col] = None # Add missing column
            df = df[all_columns] # Reorder columns
        else:
            df = pd.DataFrame(columns=all_columns)

        # Check if topic_id exists
        if topic_id in df["Topic_ID"].values:
            # Update existing row
            row_index = df.index[df["Topic_ID"] == topic_id].tolist()[0]
            df.loc[row_index, result_col] = result
            df.loc[row_index, round_col] = num_rounds
            df.loc[row_index, chat_id_col] = str(chat_id)
            if claim and pd.isna(df.loc[row_index, "Claim"]):
                 df.loc[row_index, "Claim"] = claim # Add claim if missing
        else:
            # Add new row
            new_row = {"Topic_ID": topic_id, "Claim": claim}
            new_row[result_col] = result
            new_row[round_col] = num_rounds
            new_row[chat_id_col] = str(chat_id)
            # Use concat instead of append
            df = pd.concat([df, pd.DataFrame([new_row], columns=df.columns)], ignore_index=True)

        # Save the DataFrame
        df.to_excel(xlsx_path, index=False)
        logger.info(f"Updated XLSX summary at: {xlsx_path}") # Use logger

    except Exception as e:
        logger.error(f"Error saving XLSX log to {xlsx_path}: {e}", exc_info=True) # Use logger

def save_debate_log(log_history: List[Any], 
                      log_base_path: str, 
                      topic_id: str, 
                      chat_id: str, 
                      helper_type: str, 
                      result: bool, 
                      number_of_rounds: int, 
                      finish_reason: str, 
                      claim: Optional[str] = None, 
                      save_formats: List[str] = ['json', 'html', 'txt', 'xlsx']) -> None:
    """
    Saves the debate log in specified formats to a structured directory.

    Args:
        log_history: The conversation log from MemoryInterface.get_history().
        log_base_path: The root directory to save logs (e.g., 'Reworked/logs').
        topic_id: Identifier for the debate topic.
        chat_id: Unique identifier for this specific chat instance.
        helper_type: Identifier for the configuration/helper used (e.g., 'No_Helper').
        result: Final outcome of the debate (e.g., True if convinced).
        number_of_rounds: Total rounds in the debate.
        finish_reason: Reason why the debate ended.
        claim: The text of the claim being debated (optional, for summary).
        save_formats: List of formats to save ('json', 'html', 'txt', 'xlsx').
    """
    
    log_directory = os.path.join(log_base_path, topic_id, helper_type)
    create_directory(log_directory, overwrite=False) # Don't overwrite subdirs if they exist
    
    base_filename = str(chat_id)
    xlsx_summary_path = os.path.join(log_base_path, "all_debates_summary.xlsx")

    # Common metadata for individual logs and summary
    metadata = {
        "Topic_ID": topic_id,
        "Chat_ID": chat_id,
        "Helper_Type": helper_type,
        "Result": result,
        "Number_of_Rounds": number_of_rounds,
        "Stop_Reason": finish_reason,
        "Claim": claim
    }

    if 'json' in save_formats:
        json_path = os.path.join(log_directory, f"{base_filename}.json")
        _save_log_json(log_history, json_path, metadata)
        
    if 'html' in save_formats:
        html_path = os.path.join(log_directory, f"{base_filename}.html")
        _save_log_html(log_history, html_path, metadata)
        
    if 'txt' in save_formats: # Saves JSON dump as .txt
        txt_path = os.path.join(log_directory, f"{base_filename}.txt")
        _save_log_txt(log_history, txt_path, metadata)
        
    if 'xlsx' in save_formats:
        _save_log_xlsx(xlsx_summary_path, metadata)

    # --- Process Log History for Fallacies (Moved from Orchestrator) --- 
    # Check if fallacy logging is desired (e.g., based on save_formats or a specific flag)
    # Assuming we always process if the function is called, and save_fallacy_data handles file creation.
    # Alternatively, could check if a specific 'fallacy_csv' format is in save_formats.
    
    fallacy_log_path = os.path.join(log_base_path, "fallacies.csv") # Define path
    logger.info(f"Processing log history for fallacies (logging to {fallacy_log_path})...")
    
    last_opponent_message = None 
    processed_fallacy_count = 0
    for i, entry in enumerate(log_history):
        if entry.get("type") == "message":
            entry_data = entry.get("data", {})
            entry_role = entry_data.get("role")
            
            if entry_role == INTERNAL_USER_ROLE: # Uses INTERNAL_USER_ROLE from core.interfaces
                last_opponent_message = entry_data.get("content")
            elif entry_role == INTERNAL_AI_ROLE: # Uses INTERNAL_AI_ROLE from core.interfaces
                entry_metadata = entry.get("metadata", {})
                feedback_tag = entry_metadata.get("feedback_tag") 
                
                if feedback_tag: 
                    original_response = entry_metadata.get("raw_response")
                    if original_response:
                            save_fallacy_data(
                                csv_path=fallacy_log_path,
                                data_to_append={
                                    "Topic_ID": topic_id,
                                    "Chat_ID": chat_id,
                                    "Argument": original_response,
                                    "Counter_Argument": last_opponent_message,
                                    "Fallacy": feedback_tag,
                                }
                            )
                            processed_fallacy_count += 1
                    else:
                            logger.warning(f"Found feedback_tag '{feedback_tag}' in log entry {i} but missing raw_response in metadata.")
    logger.info(f"Finished processing fallacies. Found and logged {processed_fallacy_count} instances.")
    # ------------------------------------------------------------------

# --- Other Utilities --- 

def save_fallacy_data(csv_path: str, data_to_append: Dict[str, Any]) -> None:
    """
    Appends fallacy analysis data to a CSV file.

    Args:
        csv_path: Path to the CSV file.
        data_to_append: Dictionary containing fallacy data for one instance.
                        Expected keys: Topic_ID, Chat_ID, Argument, 
                        Counter_Argument, Fallacy, Fallacious_Argument.
    """
    expected_columns=["Topic_ID", "Chat_ID", "Argument", "Counter_Argument", "Fallacy", 'Fallacious_Argument']
    
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
             # Ensure all expected columns exist
            for col in expected_columns:
                 if col not in df.columns:
                      df[col] = None
        else:
            df = pd.DataFrame(columns=expected_columns)
        
        # Ensure data_to_append only has expected columns before concatenating
        new_row_data = {k: [data_to_append.get(k)] for k in expected_columns}
        new_row = pd.DataFrame(new_row_data)

        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved fallacy data to: {csv_path}") # Use logger

    except Exception as e:
        logger.error(f"Error saving fallacy data to {csv_path}: {e}", exc_info=True) # Use logger 