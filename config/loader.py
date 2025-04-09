import yaml
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

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
                logger.warning(f"Config file {file_path} is empty.")
                return {}
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {file_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred loading config {file_path}: {e}", exc_info=True)
        raise

def load_app_config(settings_path: str = DEFAULT_SETTINGS_PATH, 
                      models_path: str = DEFAULT_MODELS_PATH) -> Dict[str, Any]:
    """
    Loads the main application settings and LLM model configurations.
    Merges model provider details into the main settings under a 'resolved_llm_providers' key.

    Args:
        settings_path: Path to the main settings YAML file.
        models_path: Path to the LLM models YAML file.

    Returns:
        A dictionary containing the combined configuration.
    """
    logger.info(f"Loading settings from: {settings_path}")
    logger.info(f"Loading models from: {models_path}")
    
    settings_config = load_yaml_config(settings_path)
    models_config = load_yaml_config(models_path)

    # Store the raw loaded configs
    combined_config = {
        "settings": settings_config,
        "models": models_config
    }
    
    # Resolve LLM references in agent configurations for easier access
    # This adds the actual model provider dictionary to the agent config
    resolved_llm_providers = models_config.get('llm_models', {})
    combined_config['resolved_llm_providers'] = resolved_llm_providers

    agent_configs = settings_config.get('agent_configurations', {})
    for run_name, run_config in agent_configs.items():
        for agent_name, agent_details in run_config.items():
            if isinstance(agent_details, dict):
                llm_ref = agent_details.get('llm_config_ref')
                if llm_ref and llm_ref in resolved_llm_providers:
                    # Inject the resolved LLM provider config
                    agent_details['_resolved_llm_config'] = resolved_llm_providers[llm_ref]
                
                # Resolve helper LLM ref if present
                llm_ref_helper = agent_details.get('llm_config_ref_helper')
                if llm_ref_helper and llm_ref_helper in resolved_llm_providers:
                     agent_details['_resolved_llm_config_helper'] = resolved_llm_providers[llm_ref_helper]

    return combined_config

