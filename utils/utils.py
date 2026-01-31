import os
import shutil
import json
import pandas as pd
from openpyxl import load_workbook
import openpyxl
from filelock import FileLock
from utils.log_main import logger

def create_debate_directory(topic_id, chat_id, helper_type, debates_base_dir="debates"):
    """
    Creates directory structure for debate logs with given topic_id, chat_id, and helper_type.
    
    Args:
        topic_id: Identifier for the debate topic
        chat_id: Unique identifier for this specific chat instance
        helper_type: Type of helper used (no_helper, vanilla, fallacy)
        debates_base_dir: Base directory for debates (default: "debates")
    
    Returns:
        str: Path to the created directory
    """
    # Define base debates directory
    debates_dir = debates_base_dir
    
    # Create general debates folder if it doesn't exist
    if not os.path.exists(debates_dir):
        os.makedirs(debates_dir)
    
    # Create topic directory if it doesn't exist
    topic_dir = os.path.join(debates_dir, str(topic_id)) # Creates debates/topic_id
    if not os.path.exists(topic_dir):
        os.makedirs(topic_dir)
    
    # Create helper-type subdirectory if it doesn't exist
    helper_dir = os.path.join(topic_dir, helper_type)
    if not os.path.exists(helper_dir):
        os.makedirs(helper_dir)
    
    # Saving current debate:
    chat_dir = os.path.join(topic_dir, helper_type, str(chat_id))
    if not os.path.exists(chat_dir):
        os.makedirs(chat_dir)
    
    return chat_dir

# save_debate_logs function removed - logs are now written directly to the correct location

def save_debate_in_excel(topic_id, claim_data, helper_type, chat_id, result, rounds, finish_reason="", 
                         conviction_rates=None, feedback_tags=None, argument_quality_rates=None,
                         debate_quality_rating=None, debate_quality_review=None, excel_file_path=None):
    """
    Save debate results to a central Excel file.
    Creates the file if it doesn't exist, otherwise appends to it.
    
    Args:
        topic_id: Identifier for the debate topic
        claim_data: Dictionary with claim information including the claim text
        helper_type: Type of helper used (no_helper, vanilla_helper, fallacy_helper)
        chat_id: Unique identifier for this specific chat instance
        result: Integer result status (1=convinced, 0=not convinced, 2=inconclusive, -1=error)
        rounds: Number of rounds the debate lasted
        finish_reason: Reason why the debate ended (e.g., "Max rounds reached", "TERMINATE")
        conviction_rates: List of conviction rates per round (optional)
        feedback_tags: List of feedback tags per round (optional)
        argument_quality_rates: List of argument quality rates per round (optional)
        debate_quality_rating: Overall debate quality rating 1-10 (optional)
        debate_quality_review: Professional review of debate quality (optional)
        excel_file_path: Path to Excel file (default: "all_debates_summary.xlsx" in current directory)
        
    Returns:
        bool: True if successful, False otherwise
    """
    excel_file = excel_file_path if excel_file_path else "all_debates_summary.xlsx"
    lock_file = excel_file + ".lock"
    
    # Use file lock to ensure only one process writes at a time
    lock = FileLock(lock_file, timeout=30)
    
    try:
        with lock:
            # Get the claim text from the claim data
            claim = claim_data.get('claim', 'Unknown claim')
            
            # Store full vectors as JSON strings
            import json
            conviction_rates = conviction_rates or []
            feedback_tags = feedback_tags or []
            argument_quality_rates = argument_quality_rates or []
            conviction_rates_json = json.dumps(conviction_rates)
            feedback_tags_json = json.dumps(feedback_tags)
            argument_quality_rates_json = json.dumps(argument_quality_rates)
            
            # Prepare the new row data for Summary sheet
            new_row = {
                'topic_id': topic_id,
                'claim': claim,
                'helper_type': helper_type,
                'result': result,
                'rounds': rounds,
                'finish_reason': finish_reason,
                'conviction_rates_vector': conviction_rates_json,
                'feedback_tags_vector': feedback_tags_json,
                'argument_quality_rates_vector': argument_quality_rates_json,
                'debate_quality_rating': debate_quality_rating,
                'debate_quality_review': debate_quality_review or '',
                'chat_id': chat_id
            }
            
            # Check if the file already exists
            if os.path.exists(excel_file):
                # Load existing file - try to read Summary sheet
                try:
                    df_summary = pd.read_excel(excel_file, sheet_name='Summary')
                    # Append new row
                    df_summary = pd.concat([df_summary, pd.DataFrame([new_row])], ignore_index=True)
                except Exception as e:
                    logger.debug(f"Could not read Summary sheet, creating new: {e}", extra={"msg_type": "system"})
                    # If Summary sheet doesn't exist, create it
                    df_summary = pd.DataFrame([new_row])
            else:
                # Create new dataframe with headers
                df_summary = pd.DataFrame([new_row])
            
            # Save dataframe to Excel
            with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a' if os.path.exists(excel_file) else 'w') as writer:
                # Remove existing sheet if it exists
                if os.path.exists(excel_file) and 'Summary' in writer.book.sheetnames:
                    del writer.book['Summary']
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
                
            logger.info(f"Successfully updated debate summary in {excel_file}", 
                       extra={"msg_type": "system"})
            return True
        
    except Exception as e:
        logger.error(f"Error saving debate to Excel: {e}", 
                   extra={"msg_type": "system"})
        return False



