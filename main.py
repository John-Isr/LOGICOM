import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
import time
import uuid
from typing import Dict, Any, Optional, Tuple
import logging

# --- Direct Imports (assuming script run from project root) ---
from config.loader import load_app_config
from llm.llm_factory import LLMFactory
from memory.chat_summary_memory import ChatSummaryMemory
from agents.persuader_agent import PersuaderAgent
from agents.debater_agent import DebaterAgent
from agents.moderator_agent import ModeratorAgent
from utils.helpers import extract_claim_data_for_prompt, save_debate_log
# Import the function and path from set_api_keys
from utils.set_api_keys import set_environment_variables_from_file, API_KEYS_PATH

# Import AgentType from interfaces
from core.interfaces import AgentType                            
# Import Orchestrator
from core.orchestrator import DebateOrchestrator

# Use colorama for terminal colors
from colorama import init, Fore, Style
init(autoreset=True)

# Define logger at module level
logger = logging.getLogger(__name__)

# --- Argument Parsing --- 
def define_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI Debates with Reworked Architecture")

    parser.add_argument("--config_run_name", default="Default_NoHelper", 
                        help="Name of the agent configuration section in settings.yaml to use.")
    parser.add_argument("--claim_index", type=int, default=None, 
                        help="Index of the specific claim in the dataset to run (0-based). Runs all if not specified.")
    parser.add_argument("--settings_path", default="./config/settings.yaml", 
                        help="Path to the main settings configuration file.")
    parser.add_argument("--models_path", default="./config/models.yaml", 
                        help="Path to the LLM models configuration file.")
    # API keys can be set via environment variables (OPENAI_API_KEY, GOOGLE_API_KEY) 
    # or potentially in models.yaml (less secure)
    # parser.add_argument("--openai_api_key", help="OpenAI API Key (overrides env var)")
    # parser.add_argument("--google_api_key", help="Google API Key (overrides env var)")

    args = parser.parse_args()
    return args

# --- Agent/Dependency Instantiation --- 
def setup_debate_environment(config: Dict[str, Any], run_name: str, claim_data: pd.Series) -> Dict[str, Any]:
    """Loads prompts, creates LLM clients and agents for a specific debate run configuration and claim."""
    
    logger.info(f"Setting up environment for run: '{run_name}'")
    agent_configs = config['settings']['agent_configurations'].get(run_name)
    if not agent_configs:
        raise ValueError(f"Configuration '{run_name}' not found in settings.")
    
    # Get debate settings, including new memory settings
    debate_settings = config.get('settings', {}).get('debate_settings', {}) 
    memory_config = {
        'memory_type': debate_settings.get('memory_type', 'summarize'), # Default to summarize to match settings.yaml
        'summarization_trigger_tokens': debate_settings.get('summarization_trigger_tokens', 6000),
        'target_prompt_tokens': debate_settings.get('target_prompt_tokens', 4000),
        'keep_messages_after_summary': debate_settings.get('keep_messages_after_summary', 4),
        'summarizer_llm_config_ref': debate_settings.get('summarizer_llm_config_ref')
    }

    llm_clients: Dict[str, Any] = {}
    resolved_llm_providers = config.get('resolved_llm_providers', {})

    # --- LLM Client Creation Helper ---
    def get_llm_client(ref_name: str, system_instruction: str | None = None):
        """Gets or creates an LLM client, passing system instruction for client initialization."""
        cache_key = f"{ref_name}_{system_instruction}" if system_instruction else ref_name
        
        if cache_key not in llm_clients:
            llm_model_config = resolved_llm_providers.get(ref_name)
            if not llm_model_config:
                raise ValueError(f"LLM configuration reference '{ref_name}' not found in models.yaml")
            logger.info(f"Creating LLM client for: {ref_name} (Provider: {llm_model_config.get('provider')}) {'with system instruction.' if system_instruction else 'without system instruction.'}")
            llm_clients[cache_key] = LLMFactory.create_llm_client(llm_model_config, system_instruction=system_instruction)
        return llm_clients[cache_key]

    # --- Load Prompts and Prepare Variables ---
    # Simplified function to just read file content
    def read_instruction_file(file_path: str, variables: Dict) -> str:
        if not file_path: return "" # Handle case where path might be optional (though validation should catch required)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Perform variable substitution if needed
            return content.format(**variables) # Use format for substitution
        except FileNotFoundError:
            logger.error(f"Instruction file not found: {file_path}")
            raise # Re-raise error to halt setup
        except KeyError as e:
            logger.error(f"Missing variable {e} needed for instruction file: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading or formatting instruction file {file_path}: {e}", exc_info=True)
            raise

    def load_moderator_prompts(mod_config: Dict, default_vars: Dict) -> Dict[str, str]:
        prompts = {}
        for key in ['terminator', 'tag_checker', 'topic_checker']:
            path_key = f'prompt_{key}_path'
            path = mod_config.get(path_key)
            if not path: raise ValueError(f"{path_key} missing in moderator config")
            # Directly read moderator instruction file
            prompts[f'system_prompt_{key}'] = read_instruction_file(path, default_vars)
        return prompts

    # --- Instantiate Summarizer LLM (if needed) ---
    summarizer_llm_client = None
    if memory_config['memory_type'] == 'summarize':
         summarizer_ref = memory_config.get('summarizer_llm_config_ref')
         if not summarizer_ref:
              raise ValueError("memory_type is 'summarize' but summarizer_llm_config_ref is missing in debate_settings.")
         # Summarizer LLM typically doesn't need a dynamic system prompt during creation
         summarizer_llm_client = get_llm_client(summarizer_ref, system_instruction=None)

    # --- Create Memory Factory Function ---
    def create_memory_instance():
         if memory_config['memory_type'] == 'summarize':
              logger.info("Creating ChatSummaryMemory with summarization enabled.")
              return ChatSummaryMemory(
                   summarizer_llm=summarizer_llm_client,
                   summarization_trigger_tokens=memory_config.get('summarization_trigger_tokens'), # Use get() for safety
                   target_prompt_tokens=memory_config.get('target_prompt_tokens'),
                   keep_messages_after_summary=memory_config.get('keep_messages_after_summary')
              )
         else:
              # Fallback or default to truncation (current ChatSummaryMemory handles this internally now)
              # We might create a separate TruncationMemory class later for clarity,
              # but for now, ChatSummaryMemory without summarizer acts like truncation.
              logger.info("Creating ChatSummaryMemory with truncation (summarization disabled).")
              return ChatSummaryMemory(summarizer_llm=None) # Pass None to disable summarization

    # --- Instantiate Agents --- 
    agents = {}
    
    # --- Define Column Mapping (get from debate_settings or use defaults) ---
    column_mapping = debate_settings.get('column_mapping', {})
    # Ensure default mappings if not provided in config
    default_mapping = {
        "TOPIC": "title",
        "CLAIM": "claim",
        "ORIGINAL_TEXT": "original_text",
        "REASON": "reason",
        "WARRANT_ONE": "warrant_one",
        "WARRANT_TWO": "warrant_two"
    }
    # Merge defaults with provided mapping (provided mapping takes precedence)
    final_column_mapping = {**default_mapping, **column_mapping} 
    logger.info(f"Using column mapping for prompt variables: {final_column_mapping}")

    default_vars_persuader = extract_claim_data_for_prompt(claim_data, AgentType.PERSUADER_AGENT, final_column_mapping)
    default_vars_debater = extract_claim_data_for_prompt(claim_data, AgentType.DEBATER_AGENT, final_column_mapping)

    # Persuader
    p_config = agent_configs.get('persuader')
    if p_config:
        # Load system instruction directly
        p_sys_instruction_path = p_config.get('system_instruction_path')
        if not p_sys_instruction_path: raise ValueError("Persuader system_instruction_path missing.")
        p_sys_instruction = read_instruction_file(p_sys_instruction_path, default_vars_persuader)
        # Removed ai_first_message extraction
        # p_sys_instruction, p_ai_first_msg = load_agent_prompt(p_config, default_vars_persuader)
        p_ai_first_msg = None # Feature removed for simplicity
        
        llm_ref = p_config.get('llm_config_ref')
        p_llm_client = get_llm_client(llm_ref, system_instruction=p_sys_instruction)
        resolved_persuader_config = p_config.get('_resolved_llm_config', {}) 
        p_model_cfg = resolved_persuader_config.get('default_config', {})
        p_model_cfg.update(p_config.get('model_config_override', {}))
        # Use memory factory
        p_memory = create_memory_instance()
        # Get wrapper path
        p_wrapper_path = p_config.get('prompt_wrapper_path')

        # Helper setup
        p_use_helper = p_config.get('use_helper_feedback', False)
        p_helper_llm_client = None
        p_helper_sys_prompt_text = None 
        p_helper_user_prompt_path = None 
        p_helper_model_cfg = {}
        if p_use_helper:
             helper_ref = p_config.get('llm_config_ref_helper')
             if not helper_ref: raise ValueError("helper llm_config_ref_helper missing.")
             helper_sys_prompt_path = p_config.get('helper_system_prompt_path')
             if not helper_sys_prompt_path: raise ValueError("helper_system_prompt_path missing.")
             # Renamed back to user prompt path
             p_helper_user_prompt_path = p_config.get('helper_user_prompt_path') 
             if not p_helper_user_prompt_path: raise ValueError("helper_user_prompt_path missing.")
             
             # Load helper system prompt text directly
             p_helper_sys_prompt_text = read_instruction_file(helper_sys_prompt_path, default_vars_persuader)
             # p_helper_sys_prompt_text, _ = load_prompt_template(helper_sys_prompt_path, default_vars_persuader)
             
             # Create helper client, passing its system prompt text
             p_helper_llm_client = get_llm_client(helper_ref, system_instruction=p_helper_sys_prompt_text)
             resolved_helper_config = p_config.get('_resolved_llm_config_helper', {}) 
             p_helper_model_cfg = resolved_helper_config.get('default_config', {})
             p_helper_model_cfg.update(p_config.get('helper_model_config_override', {}))

        # Get initial ask path
        p_initial_ask_path = p_config.get('initial_ask_prompt_path')
        if not p_initial_ask_path: raise ValueError("Persuader initial_ask_prompt_path missing.")

        agents['persuader'] = PersuaderAgent(
            llm_client=p_llm_client,
            memory=p_memory,
            ai_first_message=p_ai_first_msg,
            variables=default_vars_persuader,
            model_config=p_model_cfg,
            prompt_wrapper_path=p_wrapper_path,
            use_helper_feedback=p_use_helper,
            llm_client_helper=p_helper_llm_client,
            helper_user_prompt_path=p_helper_user_prompt_path, 
            helper_model_config=p_helper_model_cfg,
            initial_ask_prompt_path=p_initial_ask_path
        )
    else: raise ValueError("Persuader configuration missing.")

    # Debater
    d_config = agent_configs.get('debater')
    if d_config:
        # Load system instruction directly
        d_sys_instruction_path = d_config.get('system_instruction_path')
        if not d_sys_instruction_path: raise ValueError("Debater system_instruction_path missing.")
        # Pass the final_column_mapping to read_instruction_file if needed, though typically prompts use the extracted vars.
        # Make sure default_vars_debater is used here.
        d_sys_instruction = read_instruction_file(d_sys_instruction_path, default_vars_debater)
        # d_sys_instruction, _ = load_agent_prompt(d_config, default_vars_debater)
        
        llm_ref_debater = d_config.get('llm_config_ref')
        d_llm_client = get_llm_client(llm_ref_debater, system_instruction=d_sys_instruction)
        resolved_debater_config = d_config.get('_resolved_llm_config', {}) 
        d_model_cfg = resolved_debater_config.get('default_config', {})
        d_model_cfg.update(d_config.get('model_config_override', {}))
        # Use memory factory
        d_memory = create_memory_instance()
        # Get wrapper path
        d_wrapper_path = d_config.get('prompt_wrapper_path')

        agents['debater'] = DebaterAgent(
            llm_client=d_llm_client,
            memory=d_memory,
            variables=default_vars_debater,
            model_config=d_model_cfg,
            prompt_wrapper_path=d_wrapper_path
        )
    else: raise ValueError("Debater configuration missing.")

    # --- Moderator Setup --- 
    m_config = agent_configs.get('moderator_primary')
    if not m_config: raise ValueError("Moderator (Primary) configuration missing.")

    # Moderators likely use the debater's perspective/variables for context
    mod_prompts = load_moderator_prompts(m_config, default_vars_debater)

    def create_moderator(prompt_key, config, prompts):
        llm_ref = config.get('llm_config_ref')
        if not llm_ref: raise ValueError("llm_config_ref missing in moderator config")
        # System prompt text is already loaded into prompts dict
        sys_prompt_text = prompts.get(f'system_prompt_{prompt_key}')
        if not sys_prompt_text: raise ValueError(f"Missing system_prompt_{prompt_key}...")
        
        llm_client = get_llm_client(llm_ref, system_instruction=sys_prompt_text)
        # Resolve config similar to other agents (assuming loader resolves it)
        resolved_mod_config = config.get('_resolved_llm_config', {})
        model_cfg = resolved_mod_config.get('default_config', {})
        model_cfg.update(config.get('model_config_override', {}))
        agent_name_suffix = prompt_key.replace("_", " ").title().replace(" ", "")
        return ModeratorAgent(
            llm_client=llm_client,
            variables=default_vars_debater, # Pass debater vars to moderator
            model_config=model_cfg,
            agent_name=f"Moderator{agent_name_suffix}"
        )

    agents['moderator_terminator'] = create_moderator('terminator', m_config, mod_prompts)
    agents['moderator_topic_checker'] = create_moderator('topic_checker', m_config, mod_prompts)
    agents['moderator_tag_checker'] = create_moderator('tag_checker', m_config, mod_prompts)

    return agents

# --- Main Execution Logic --- (Using Orchestrator) ---

def main():
    # --- Central Logging Configuration ---
    logging.basicConfig(
        level=logging.INFO, # Set default logging level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # --- Suppress noisy httpx logs --- 
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # --------------------------------
    logger.info("Application starting...")
    # -----------------------------------

    # --- Set API Keys from file --- 
    # Call the function from set_api_keys.py at the start
    # This ensures environment variables are set within this process
    logger.info(f"Attempting to set environment variables from {API_KEYS_PATH}...")
    set_environment_variables_from_file(API_KEYS_PATH)
    # ------------------------------

    args = define_arguments()
    
    try:
        # Load configurations
        config = load_app_config(args.settings_path, args.models_path)
        debate_settings = config['settings']['debate_settings']
        run_config_name = args.config_run_name
        agent_configs = config['settings']['agent_configurations'].get(run_config_name)
        if not agent_configs:
            logger.error(f"Run configuration '{run_config_name}' not found in settings.")
            sys.exit(1)
        helper_type_name = agent_configs.get('helper_type_name', run_config_name)

        # Load dataset
        data_path = debate_settings['data_path']
        logger.info(f"Loading data from: {data_path}")
        
        # Check if the configured path exists directly
        if not os.path.exists(data_path):
             # If not, try resolving it relative to the project root (assuming main.py is in the root)
             project_root = os.path.dirname(os.path.abspath(__file__))
             alt_path_from_root = os.path.join(project_root, data_path.lstrip('./')) 
             logger.info(f"Configured data path '{data_path}' not found. Trying relative to project root: '{alt_path_from_root}'")
             if os.path.exists(alt_path_from_root):
                  data_path = alt_path_from_root
                  logger.info(f"Using alternative data path: {data_path}")
             else:
                  # Use f-string for clarity
                  logger.error(f"Data file not found at specified path '{debate_settings['data_path']}' or relative to project root '{alt_path_from_root}'")
                  sys.exit(1)
        
        # Proceed with loading now data_path is confirmed
        data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(data)} claims.")

        # Determine claims to run
        claim_indices_to_run = []
        if args.claim_index is not None:
            if 0 <= args.claim_index < len(data):
                claim_indices_to_run = [args.claim_index]
            else:
                logger.error(f"Error: Invalid claim_index {args.claim_index}. Must be between 0 and {len(data)-1}.")
                sys.exit(1)
        else:
            claim_indices_to_run = list(range(len(data)))
            logger.info(f"Running for all {len(claim_indices_to_run)} claims.")

        # --- Iterate through claims and run debates ---
        results_summary = []
        topic_id_col_name = debate_settings.get('topic_id_column', 'id') # Read column name from config, default to 'id'
        claim_col_name = debate_settings.get('claim_column', 'claim') # Also get claim column name configurably
        logger.info(f"Using column '{topic_id_col_name}' for Topic ID (fallback to index).")
        logger.info(f"Using column '{claim_col_name}' for Claim Text.")

        for index in tqdm(claim_indices_to_run, desc="Running Debates"):
            claim_data = data.iloc[index]
            topic_id = str(claim_data.get(topic_id_col_name, index)) # Use the configured column name or fallback to index
            claim_text = str(claim_data.get(claim_col_name, '')) # Revert to simple .get() with empty string fallback

            # Optional check if you still want to skip empty claims after simple get
            if not claim_text:
                 logger.warning(f"Warning: Empty claim text found for Topic ID {topic_id} (Index {index}) using column '{claim_col_name}'. Skipping.")
                 continue

            logger.info(f"\n===== Preparing Claim Index: {index}, Topic ID: {topic_id} ====")
            
            try:
                # Setup agents and environment for this specific claim
                agents = setup_debate_environment(config, run_config_name, claim_data)
                
                # Instantiate the Orchestrator
                orchestrator = DebateOrchestrator(
                    persuader=agents['persuader'],
                    debater=agents['debater'],
                    moderator_terminator=agents['moderator_terminator'],
                    moderator_topic_checker=agents['moderator_topic_checker'],
                    moderator_tag_checker=agents['moderator_tag_checker'],
                    max_rounds=debate_settings.get('max_rounds', 12),
                    turn_delay_seconds=float(debate_settings.get('turn_delay_seconds', 0.0)),
                )

                # Run the debate using the orchestrator
                result = orchestrator.run_debate(
                    topic_id=topic_id,
                    claim=claim_text,
                    log_config=debate_settings, # Pass relevant log settings
                    helper_type_name=helper_type_name
                )
                results_summary.append(result)
                
                # Optional delay between debates
                # time.sleep(1)

            except Exception as e:
                logger.error(f"!!!!! Error running debate for Topic ID {topic_id} (Index {index}): {e} !!!!!", exc_info=True)
                results_summary.append({
                    "topic_id": topic_id,
                    "claim_index": index,
                    "status": "ERROR",
                    "error_message": str(e)
                })

        # --- Print Summary --- 
        logger.info("\n===== Debate Run Summary ====")
        successful_runs = [r for r in results_summary if r.get('status') != 'ERROR']
        failed_runs = [r for r in results_summary if r.get('status') == 'ERROR']
        logger.info(f"Total Debates Run: {len(results_summary)}")
        logger.info(f"Successful: {len(successful_runs)}")
        logger.info(f"Failed: {len(failed_runs)}")
        # Could print more details from successful_runs (avg rounds, results breakdown, etc.)
        if failed_runs:
             logger.warning("\nFailed Runs:")
             for fail in failed_runs:
                  logger.warning(f"  Index: {fail['claim_index']}, Topic: {fail['topic_id']}, Error: {fail['error_message']}")

    except Exception as e:
        logger.critical(f"\nAn unexpected error occurred in main execution: {e}", exc_info=True)
        # import traceback # Already logged via exc_info=True
        # traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 