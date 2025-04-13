import sys
import argparse
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple, Union, List
import logging

# --- Direct Imports ---
from utils.set_api_keys import set_environment_variables_from_file, API_KEYS_PATH
from config.loader import load_app_config
from core.orchestrator import DebateOrchestrator
from core.debate_setup import DebateInstanceSetup

# Use colorama for terminal colors
from colorama import init
init(autoreset=True)

# Define logger at module level
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def _setup_logging():
    """Configures application-wide logging."""
    logging.basicConfig(
        level=logging.DEBUG, # change from DEBUG to WARNING for regular runs
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Suppress noisy logs from underlying libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("google.api_core").setLevel(logging.WARNING)
    logger.info("Logging configured.")

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
    logger.debug("Formatting prompts for current claim...")
    
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
    logger.debug(f"Context keys available for formatting: {list(str_context.keys())}")

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
             logger.debug(f"Formatted prompt for prompt: {prompt_name}")
        else:
             logger.debug(f"No initial placeholders found for prompt: {prompt_name}")

    logger.debug("Prompts formatted successfully.")
    return formatted_prompts, topic_id, claim_text

# --- Argument Parsing --- 
def define_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI Debates with Reworked Architecture")
    parser.add_argument("--config_run_name", default="Default_NoHelper", 
                        help="Name of the agent configuration section in settings.yaml to use.")
    parser.add_argument("--claim_index", type=int, default=None, 
                        help="Index of the specific claim in the claims_file to run (0-based). Runs all if not specified.")
    parser.add_argument("--settings_path", default="./config/settings.yaml", 
                        help="Path to the main settings configuration file.")
    parser.add_argument("--models_path", default="./config/models.yaml", 
                        help="Path to the LLM models configuration file.")
    args = parser.parse_args()
    return args

# --- New Helper Function to Run a Single Debate --- 
def _run_single_debate(index: int, 
                         claim_data: pd.Series, 
                         debate_settings: Dict, 
                         agent_config: Dict, 
                         prompt_templates: Dict, 
                         helper_type: str) -> Dict:
    """Sets up and runs a single debate instance, handling errors."""
    topic_id = "N/A"
    run_result = {}
    try:
        # Format prompts for this claim (includes extracting topic_id, claim_text)
        formatted_prompts, topic_id, claim_text = format_prompts_for_claim(debate_settings, claim_data, prompt_templates)

        # Log start using extracted topic_id
        logger.info(f"\n===== Preparing Claim Index: {index}, Topic ID: {topic_id} ====")

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
            max_rounds=int(debate_settings['max_rounds']),
            turn_delay_seconds=float(debate_settings['turn_delay_seconds'])
        )
        
        # Run debate
        run_result_data = orchestrator.run_debate(
            topic_id=topic_id, 
            claim=claim_text, 
            log_config=debate_settings,
            helper_type=helper_type
        )
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
        logger.error(f"!!!!! Error running debate for Topic ID {current_topic_id}: {e} !!!!!", exc_info=True)
        run_result = {
            "topic_id": current_topic_id,
            "claim_index": index,
            "status": "ERROR",
            "error_message": str(e)
        }
    return run_result

# --- New Helper Function to Summarize Results --- 
def _summarize_results(results_summary: List[Dict]):
    """Logs the summary of successful and failed debate runs."""
    logger.info("\n===== Debate Run Summary ====")
    successful_runs = [r for r in results_summary if r.get('status') == 'Success']
    failed_runs = [r for r in results_summary if r.get('status') not in ['Success', None]] # Count ERROR and CONFIG_ERROR etc.
    logger.info(f"Total Debates Attempted: {len(results_summary)}")
    logger.info(f"Successful: {len(successful_runs)}")
    logger.info(f"Failed: {len(failed_runs)}")
    if failed_runs:
         logger.warning("\nFailed Runs:")
         for fail in failed_runs:
              logger.warning(f"  Index: {fail.get('claim_index','N/A')}, Topic: {fail.get('topic_id','N/A')}, Status: {fail.get('status', 'UNKNOWN')}, Error: {fail.get('error_message','Unknown')}")

# --- Main Execution Logic --- 
def main():
    # --- Central Logging Configuration ---
    _setup_logging()
    logger.info("Application starting...")
    # -----------------------------------

    args = define_arguments()
    
    _setup_api_keys()

    try:
        # Load configuration directly using the loader
        logger.info(f"Loading configuration for run: '{args.config_run_name}'...")
        debate_settings, agent_config, prompt_templates = load_app_config(
            settings_path=args.settings_path,
            models_path=args.models_path,
            run_config_name=args.config_run_name
        )
        logger.info("Configuration loaded successfully.")

        # Load claims data
        claims_file_path = debate_settings['claims_file_path']
        logger.info(f"Loading claim data from: {claims_file_path}")
        claims_df = pd.read_csv(claims_file_path)
        num_claims = len(claims_df)
        logger.info(f"Loaded {num_claims} claims.")

        # Determine claims to run
        if args.claim_index is not None:
            logger.info(f"Running only for specified claim index: {args.claim_index}")
            claim_indices_to_run = [args.claim_index]
        else:
            logger.info(f"Running for all {num_claims} claims.")
            claim_indices_to_run = list(range(num_claims))

        # Get helper type from the resolved config
        helper_type = agent_config['helper_type']

        # --- Run Debates Loop --- 
        results_summary = []
        for index in tqdm(claim_indices_to_run, total=len(claim_indices_to_run), desc="Running Debates"):
            claim_data = claims_df.iloc[index]
            run_result = _run_single_debate(
                index=index,
                claim_data=claim_data,
                debate_settings=debate_settings,
                agent_config=agent_config,
                prompt_templates=prompt_templates,
                helper_type=helper_type
            )
            results_summary.append(run_result)

        # --- Print Summary --- 
        _summarize_results(results_summary)

    except Exception as e:
        logger.critical(f"\nAn unexpected critical error occurred during setup or execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main() 