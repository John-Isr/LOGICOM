import sys
import argparse
import pandas as pd
import uuid
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple, Union, List
import json
import logging.config
from datetime import datetime
import os

# --- Direct Imports ---
from utils.log_main import logger as debate_logger, setup_logging 
from utils.set_api_keys import set_environment_variables_from_file, API_KEYS_PATH
from config.loader import load_app_config
from core.orchestrator import DebateOrchestrator
from core.debate_setup import DebateInstanceSetup
from utils.utils import create_debate_directory, save_debate_in_excel

# Use colorama for terminal colors
from colorama import init
init(autoreset=True)

# Use our custom debate logger instead of creating a new one
logger = debate_logger

# --- Helper Functions ---

def _setup_api_keys():
    """Sets API keys from the API_keys file."""
    set_environment_variables_from_file(API_KEYS_PATH)

# --- Helper to format prompts for a specific claim ---
def format_prompts_for_claim(debate_settings: Dict[str, Any], 
                               claim_data: pd.Series, 
                               loaded_prompts: Dict[str, str]) -> Tuple[Dict[str, str], str, str]:
    """Formats all loaded prompts using data from the current claim row.

    Raises:
        KeyError: If required keys are missing in debate_settings (mapping, column names)
                  or in claim_data (data columns), or if placeholders missing in format string.
    Returns:
        Tuple containing (formatted_prompts_dict, topic_id_str, claim_text_str)
    """
    logger.debug("Formatting prompts for current claim...", extra={"msg_type": "system"})
    
    # 1. Get config values needed for data extraction
    mapping = debate_settings['column_mapping']
    # Get column names from the mapping dictionary
    topic_id_col_name = mapping['TOPIC_ID']
    claim_col_name = mapping['CLAIM']
    topic_col_name = mapping['TOPIC']
    reason_col_name = mapping['REASON']

    # 2. Extract required data using these column names
    topic_id = str(claim_data[topic_id_col_name])
    claim_text = str(claim_data[claim_col_name])
    topic_text = str(claim_data[topic_col_name])
    reason_text = str(claim_data[reason_col_name])

    # 3. Build context dictionary with keys matching placeholders
    str_context = {
        "CLAIM": claim_text,
        "TOPIC": topic_text,
        "REASON": reason_text
    }

    # Debugging: Log the keys available right before formatting
    logger.debug(f"Context keys available for formatting: {list(str_context.keys())}", extra={"msg_type": "system"})

    # 4. Format prompts by sequential replacement
    formatted_prompts: Dict[str, str] = {}
    for prompt_name, template_content in loaded_prompts.items():
        formatted_string = template_content
        for placeholder_key, value in str_context.items():
            placeholder = "<" + placeholder_key + ">" # Construct placeholder like <CLAIM>
            if placeholder in formatted_string:
                formatted_string = formatted_string.replace(placeholder, value)
        
        formatted_prompts[prompt_name] = formatted_string
        # Log if any replacements happened
        if formatted_string != template_content:
             logger.debug(f"Formatted prompt for prompt: {prompt_name}", extra={"msg_type": "system"})
        else:
             logger.debug(f"No initial placeholders found for prompt: {prompt_name}", extra={"msg_type": "system"})

    logger.debug("Prompts formatted successfully.", extra={"msg_type": "system"})
    return formatted_prompts, topic_id, claim_text

# --- Argument Parsing --- 
def define_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI Debates with Reworked Architecture")
    parser.add_argument("--run_name", required=True,
                        help="Name for this run (will be combined with timestamp for results folder)")
    parser.add_argument("--helper_type", default="Default_No_Helper", 
                        help="Name of the helper type configuration in settings.yaml to use.")
    parser.add_argument("--claim_index", type=int, default=None, 
                        help="Index of the specific claim in the claims_file to run (0-based). Runs all if not specified.")
    parser.add_argument("--settings_path", default="./config/settings.yaml", 
                        help="Path to the main settings configuration file.")
    parser.add_argument("--models_path", default="./config/models.yaml", 
                        help="Path to the LLM models configuration file.")
    parser.add_argument("--max_rounds", type=int, default=None,
                        help="Override the maximum number of debate rounds (default is from settings.yaml)")
    parser.add_argument("--results_dir", default=None,
                        help="Path to results directory (default: creates results/<timestamp>_<run_name>/)")
    args = parser.parse_args()
    return args

# --- New Helper Function to Run a Single Debate --- 
def _run_single_debate(index: int, 
                         claim_data: pd.Series, 
                         debate_settings: Dict, 
                         agent_config: Dict, 
                         prompt_templates: Dict, 
                         helper_type: str,
                         debates_base_dir: str = "debates",
                         excel_file_path: str = None) -> Dict:
    """Sets up and runs a single debate instance, handling errors."""
    topic_id = "N/A"
    run_result = {}
    try:
        # Generate chat ID early
        chat_id = str(uuid.uuid4())
        if not chat_id:
            logger.error(f"Failed to generate a valid chat_id", extra={"msg_type": "system"})
            raise ValueError("Failed to generate a valid chat_id")

        # Format prompts for this claim (includes extracting topic_id, claim_text)
        formatted_prompts, topic_id, claim_text = format_prompts_for_claim(debate_settings, claim_data, prompt_templates)

        # Create directory structure for logs - do this early before anything can fail
        chat_dir = create_debate_directory(topic_id, chat_id, helper_type, debates_base_dir)
        
        # Setup logging to write directly to the debate directory
        # MUST be done before any logger calls
        setup_logging(log_directory=chat_dir)
        
        logger.info(f"Created debate chat directory: {chat_dir}", extra={"msg_type": "system"})
        logger.info(f"Preparing Claim Index: {index}, Topic ID: {topic_id}", extra={"msg_type": "system"})

        # Instantiate setup class for this claim
        setup = DebateInstanceSetup(
            agents_configuration=agent_config, 
            debate_settings=debate_settings,
            formatted_prompts=formatted_prompts
        )

        # Instantiate orchestrator 
        orchestrator = DebateOrchestrator(
            persuader=setup.persuader, 
            debater=setup.debater,
            moderator_terminator=setup.moderator_terminator,
            moderator_topic_checker=setup.moderator_topic_checker,
            moderator_conviction=setup.moderator_conviction,
            moderator_argument_quality=setup.moderator_argument_quality,
            moderator_debate_quality=setup.moderator_debate_quality,
            max_rounds=int(debate_settings['max_rounds']),
            turn_delay_seconds=float(debate_settings['turn_delay_seconds'])
        )
        
        # Run debate
        # Add debates_base_dir to log_config so orchestrator knows where to find log files
        log_config_with_debates_dir = {**debate_settings, 'debates_base_dir': debates_base_dir}
        run_result_data = orchestrator.run_debate(
            topic_id=topic_id, 
            claim=claim_text, 
            log_config=log_config_with_debates_dir,
            helper_type=helper_type,
            chat_id=chat_id
        )
        
        # --- Post-Debate Processing ---
        
        # Convert result to integer for Excel
        # 1 = convinced, 0 = not convinced, 2 = inconclusive, -1 = error
        result_status = run_result_data.get('result', 'Unknown')
        finish_reason = run_result_data.get('finish_reason', '')
        
        if result_status == "Convinced":
            result_code = 1
        elif result_status == "Not convinced":
            result_code = 0
        elif result_status.startswith("Inconclusive"):
            result_code = 2  # Inconclusive (TERMINATE, OFF-TOPIC, etc.)
        else:
            result_code = 2  # Default to inconclusive for unknown statuses
        
        # Extract conviction rates, feedback tags, argument quality rates, and debate quality
        conviction_rates = run_result_data.get('conviction_rates', [])
        feedback_tags = run_result_data.get('feedback_tags', [])
        argument_quality_rates = run_result_data.get('argument_quality_rates', [])
        debate_quality_rating = run_result_data.get('debate_quality_rating')
        debate_quality_review = run_result_data.get('debate_quality_review', '')
        
        # Save debate summary to Excel (with round details)
        rounds = run_result_data.get('rounds', 0)
        excel_success = save_debate_in_excel(
            topic_id,
            claim_data,
            helper_type,
            chat_id,
            result_code,
            rounds,
            finish_reason,
            conviction_rates,
            feedback_tags,
            argument_quality_rates,
            debate_quality_rating,
            debate_quality_review,
            excel_file_path=excel_file_path
        )
        if excel_success:
            logger.info(f"Successfully saved debate summary to Excel", extra={"msg_type": "system"})
        else:
            logger.warning(f"Failed to save debate summary to Excel", extra={"msg_type": "system"})
        
        # Combine orchestrator results with status/IDs
        run_result = {
             "topic_id": topic_id,
             "claim_index": index,
             "status": 'Success',
             **run_result_data # Merge results from orchestrator
        }

    except Exception as e:
        # Handle setup or runtime errors for this specific debate
        current_topic_id = topic_id if topic_id != "N/A" else f"Index_{index}"
        logger.error(f"!!!!! Error running debate for Topic ID {current_topic_id}: {e} !!!!!", 
                     extra={"msg_type": "system"})
        
        # Save error to Excel with result code -1 (error)
        try:
            # Generate a chat_id if we don't have one yet (error before chat_id generation)
            error_chat_id = chat_id if 'chat_id' in locals() else str(uuid.uuid4())
            error_finish_reason = f"ERROR: {str(e)}"
            excel_success = save_debate_in_excel(
                current_topic_id,
                claim_data,
                helper_type,
                error_chat_id,
                result=-1,  # Error
                rounds=0,
                finish_reason=error_finish_reason,
                conviction_rates=[],
                feedback_tags=[],
                excel_file_path=excel_file_path
            )
            if excel_success:
                logger.info(f"Successfully saved error to Excel with result code -1", extra={"msg_type": "system"})
            else:
                logger.warning(f"Failed to save error to Excel", extra={"msg_type": "system"})
        except Exception as excel_error:
            logger.error(f"Failed to save error to Excel: {excel_error}", extra={"msg_type": "system"})
        
        run_result = {
            "topic_id": current_topic_id,
            "claim_index": index,
            "status": "ERROR",
            "error_message": str(e)
        }
    return run_result

# # --- New Helper Function to Summarize Results --- 
# def _summarize_results(results_summary: List[Dict]):
#     """Logs the summary of successful and failed debate runs."""
#     logger.info("\n===== Debate Run Summary ====", extra={"msg_type": "system"})
#     successful_runs = [r for r in results_summary if r.get('status') == 'Success']
#     failed_runs = [r for r in results_summary if r.get('status') not in ['Success', None]] # Count ERROR and CONFIG_ERROR etc.
#     logger.info(f"Total Debates Attempted: {len(results_summary)}", extra={"msg_type": "system"})
#     logger.info(f"Successful: {len(successful_runs)}", extra={"msg_type": "system"})
#     logger.info(f"Failed: {len(failed_runs)}", extra={"msg_type": "system"})
#     if failed_runs:
#          logger.warning("\nFailed Runs:")
#          for fail in failed_runs:
#               logger.warning(f"  Index: {fail.get('claim_index','N/A')}, Topic: {fail.get('topic_id','N/A')}, Status: {fail.get('status', 'UNKNOWN')}, Error: {fail.get('error_message','Unknown')}", extra={"msg_type": "system"})

# --- Main Execution Logic --- 
def main():
    # Print statement to verify the script is running
    print("Starting application...")
    
    # Note: Logging will be configured per debate instance to avoid conflicts
    print("Application starting...")

    args = define_arguments()
    
    _setup_api_keys()

    # Determine results directory
    if args.results_dir:
        # Use provided results directory (from multiple_runs.py)
        results_dir = args.results_dir
    else:
        # Create new results directory for manual run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_base = "results"
        if not os.path.exists(results_base):
            os.makedirs(results_base)
        results_dir = os.path.join(results_base, f"{timestamp}_{args.run_name}")
        os.makedirs(results_dir, exist_ok=True)
        print(f"Created results directory: {results_dir}")
    
    # Set debates and Excel paths based on results directory
    debates_dir = os.path.join(results_dir, "debates")
    excel_file_path = os.path.join(results_dir, "all_debates_summary.xlsx")

    try:
        # Load configuration directly using the loader
        
        debate_settings, agent_config, prompt_templates = load_app_config(
            settings_path=args.settings_path,
            models_path=args.models_path,
            run_config_name=args.helper_type
        )
        print("Configuration loaded successfully.")
        print(f"Loading configuration for run: '{args.helper_type}'...")

        # Load claims data
        claims_file_path = debate_settings['claims_file_path']
        print(f"Loading claim data from: {claims_file_path}")
        claims_df = pd.read_csv(claims_file_path)
        num_claims = len(claims_df)
        print(f"Loaded {num_claims} claims.")

        # Determine claims to run
        if args.claim_index is not None:
            print(f"Running only for specified claim index: {args.claim_index}")
            claim_indices_to_run = [args.claim_index]
        else:
            print(f"Running for all {num_claims} claims.")
            claim_indices_to_run = list(range(num_claims))

        # Override max_rounds if provided via command line
        if args.max_rounds is not None:
            print(f"Overriding max_rounds from {debate_settings['max_rounds']} to {args.max_rounds}")
            debate_settings['max_rounds'] = args.max_rounds

        # Get helper type from the resolved config
        helper_type = agent_config['helper_type']

        # --- Run Debates Loop --- 
        # results_summary = []
        for index in tqdm(claim_indices_to_run, total=len(claim_indices_to_run), desc="Running Debates"):
            claim_data = claims_df.iloc[index]
            run_result = _run_single_debate(
                index=index,
                claim_data=claim_data,
                debate_settings=debate_settings,
                agent_config=agent_config,
                prompt_templates=prompt_templates,
                helper_type=helper_type,
                debates_base_dir=debates_dir,
                excel_file_path=excel_file_path
            )
            # results_summary.append(run_result)

        # --- Print Summary --- 
        # _summarize_results(results_summary)

    except Exception as e:
        print(f"\nAn unexpected error occurred in main execution: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 