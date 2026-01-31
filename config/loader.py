import yaml
import os
from typing import Dict, Any, Tuple
import logging
from utils.log_main import logger

# Define default paths relative to the loader file's location
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SETTINGS_PATH = os.path.join(CONFIG_DIR, 'settings.yaml')
DEFAULT_MODELS_PATH = os.path.join(CONFIG_DIR, 'models.yaml')

def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if config is None:
                logger.warning(f"Config file {file_path} is empty.", extra={"msg_type": "system"})
                return {}
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {file_path}", extra={"msg_type": "system"})
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}", extra={"msg_type": "system"})
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred loading config {file_path}: {e}", extra={"msg_type": "system"})
        raise

def _load_prompt(file_path: str) -> str:
    """Loads content from a text file, assuming path is relative to main project directory.
       Raises FileNotFoundError or OSError on failure."""
    logger.debug(f"Attempting to load prompt from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.debug(f"Successfully loaded prompt from: {file_path}")
        if not content:
             logger.error(f"Prompt file loaded but is empty: {file_path}")
             # Raise error if file is empty
             raise ValueError(f"Prompt file loaded but is empty: {file_path}")
        return content
    except FileNotFoundError:
         logger.error(f"Prompt file not found: {file_path}")
         raise 
    except Exception as e:
         logger.error(f"Error reading prompt file {file_path}: {e}", exc_info=True)
         raise

def _resolve_models_for_config(target_agent_config: Dict[str, Any], 
                             all_models_dict: Dict[str, Any]) -> None:
    """Finds model_name refs in the target config and injects resolved model dicts.

    Modifies target_agent_config in place.
    Raises KeyError if a referenced model_name is not found in all_models_dict.
    """
    for agent_name, agent_details in target_agent_config.items():
        # Skip non-dict items like helper_type
        if not isinstance(agent_details, dict):
             continue

        # Resolve main model
        model_name_value = agent_details.get('model_name')
        if model_name_value:
             if model_name_value in all_models_dict:
                 agent_details['_resolved_llm_config'] = all_models_dict[model_name_value]
             else:
                 logger.error(f"Model name '{model_name_value}' referenced by agent '{agent_name}' not found in models file!")
                 raise KeyError(f"Model name '{model_name_value}' not found in models file.")
        
        # Resolve helper model
        helper_model_name_value = agent_details.get('helper_model_name')
        if helper_model_name_value:
            if helper_model_name_value in all_models_dict:
                 agent_details['_resolved_llm_config_helper'] = all_models_dict[helper_model_name_value]
            else:
                logger.error(f"Helper model name '{helper_model_name_value}' referenced by agent '{agent_name}' not found in models file!")
                raise KeyError(f"Helper model name '{helper_model_name_value}' not found in models file.")

def load_app_config(settings_path: str = DEFAULT_SETTINGS_PATH, 
                      models_path: str = DEFAULT_MODELS_PATH,
                      run_config_name: str = "Default_No_Helper") -> Tuple[Dict, Dict, Dict]:
    """
    Loads debate settings, all prompt templates, and the specific agent
    configuration for the run, resolving LLM model details.

    Args:
        settings_path: Path to the main settings YAML file.
        models_path: Path to the LLM models YAML file.
        run_config_name: The name of the specific agent configuration to load and resolve.

    Returns:
        A tuple containing: (debate_settings_dict, resolved_agent_config_dict, prompt_templates_dict).
    """
    logger.debug(f"Loading settings from: {settings_path}", extra={"msg_type": "system"})
    logger.debug(f"Loading models from: {models_path}", extra={"msg_type": "system"})
    logger.debug(f"Target run configuration: '{run_config_name}'", extra={"msg_type": "system"})
    
    settings_config = load_yaml_config(settings_path)
    models_config = load_yaml_config(models_path)

    # 1. Extract global debate settings
    debate_settings = settings_config['debate_settings']
    
    # 2. Load prompt templates using paths from debate_settings
    prompt_paths = debate_settings['prompt_paths']
    prompt_templates: Dict[str, str] = {}
    logger.info("Loading prompt templates specified in settings...")
    for key, path in prompt_paths.items():
        content = _load_prompt(path)
        prompt_templates[key] = content
        logger.debug(f"Loaded prompt '{key}' from {path}")
    logger.info("Prompt templates loaded successfully by config loader.")

    # 3. Extract the dictionary of all available models
    resolved_llm_providers = models_config['llm_models']
    if not resolved_llm_providers:
        logger.warning("No 'llm_models' defined in the models file.")

    # 4. Extract the specific agent configuration for this run
    agent_configs = settings_config['agent_configurations']
    target_agent_config = agent_configs[run_config_name]
                                        
    # 5. Resolve and inject LLM configs into the target configuration using the helper
    _resolve_models_for_config(target_agent_config, resolved_llm_providers)

    # 6. Return the results as a tuple
    logger.debug("Loaded agent configurations", extra={"msg_type": "system"})
    return debate_settings, target_agent_config, prompt_templates

