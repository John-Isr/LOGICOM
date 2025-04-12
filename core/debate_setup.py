import logging
from typing import Dict, Any, Optional
import pandas as pd
import os 

# --- Import necessary classes ---
from core.interfaces import LLMInterface, MemoryInterface, AgentInterface
from memory.chat_summary_memory import ChatSummaryMemory
from agents.persuader_agent import PersuaderAgent
from agents.debater_agent import DebaterAgent
from agents.moderator_agent import ModeratorAgent
from llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)

# Helper functions
def _extract_claim_data_for_prompt(claim_data: pd.Series, column_mapping: Dict[str, str]) -> Dict[str, str]:
    """
    Extracts variables for prompts from input data using configurable column names.
    Currently extracts: TOPIC, CLAIM, REASON.
    
    Args:
        claim_data: The data record (e.g., a row from a pandas DataFrame).
        column_mapping: Dictionary mapping standard prompt variable names 
                        (e.g., "TOPIC", "CLAIM", "ORIGINAL_TEXT", "REASON") 
                        to the actual column names in the claim_data Series.

    Returns:
        Dictionary of variables for the prompt template.
    """
    base_vars = {}
    # Map standard variables to columns using the provided mapping
    vars_to_extract = ["TOPIC", "CLAIM", "REASON"] 
    for var_name in vars_to_extract:
        column_name = column_mapping.get(var_name)
        if column_name:
            base_vars[var_name] = str(claim_data.get(column_name, ''))
        else:
            logger.warning(f"No column mapping provided for standard variable '{var_name}'. It will be empty.")
            base_vars[var_name] = ''

    return base_vars

class DebateInstanceSetup:
    """Handles claim-specific setup: formatting prompts, creating clients & agents."""
    def __init__(self, 
                 agents_configuration: Dict[str, Any],
                 debate_settings: Dict[str, Any],
                 initial_prompt_template: str, 
                 claim_data: pd.Series):
        
        logger.info(f"Setting up instance for Claim ID: {claim_data.get(debate_settings.get('topic_id_column', 'id'), 'N/A')}") # TODO: Don't use get, if this isn't there, an error should be raised
        self.agents_configuration = agents_configuration
        self.debate_settings = debate_settings
        self.initial_prompt_template = initial_prompt_template
        self.claim_data = claim_data

        # --- Setup Steps --- 
        # Step 1: Extract Debate Details 
        column_mapping = self.debate_settings.get('column_mapping', {}) # TODO: Don't use get, if this isn't there, an error should be raised
        self.debate_details = _extract_claim_data_for_prompt(self.claim_data, column_mapping)
        logger.debug("Extracted debate details.")
        
        # Step 2: Format Prompts
        self._format_prompts() 
        
        # Step 3: Create LLM Clients
        self._create_llm_clients() 
        
        # Step 4: Create Memories
        self._create_memories() 
        
        # Step 5: Create Agents
        self._create_agents() 

        logger.info("Debate instance setup complete.")
        
    # --- Private Helper Methods --- 
    def _load_and_populate_prompt(self, file_path: str, debate_context: Dict) -> str:
         """Reads a prompt template file and populates placeholders."""
         #TODO: make it so main passes prompt templates as strings, then this object doesn't need to access files at all
         if not file_path: return ""
         if not os.path.exists(file_path):
              logger.error(f"Prompt file path does not exist: {file_path}")
              raise FileNotFoundError(f"Prompt file path does not exist: {file_path}")
         try:
             with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
             # Ensure values are strings for formatting
             str_context = {k: str(v) for k, v in debate_context.items()}
             return content.format(**str_context) 
         except KeyError as e: logger.error(f"Missing var {e} for: {file_path}"); raise
         except Exception as e: logger.error(f"Read/format error {file_path}: {e}"); raise

    def _format_prompts(self):
        """Formats all necessary prompts using claim details."""
        p_config = self.agents_configuration.get('persuader')
        d_config = self.agents_configuration.get('debater')
        m_config = self.agents_configuration.get('moderator')
        if not all([p_config, d_config, m_config]): raise ValueError("Missing agent config block.")

        # Initial Prompt
        try: self.initial_prompt_content = self.initial_prompt_template.format(**self.debate_details)
        except KeyError as e: logger.error(f"Missing variable {e} for initial prompt."); raise

        # Agent System Prompts
        p_sys_path = p_config['system_instruction_path']; self.p_sys_instruction = self._load_and_populate_prompt(p_sys_path, self.debate_details)
        d_sys_path = d_config['system_instruction_path']; self.d_sys_instruction = self._load_and_populate_prompt(d_sys_path, self.debate_details)
        
        # Moderator System Prompts 
        self.mod_sys_instructions = {}
        for mod_key, path_key in [('terminator', 'prompt_terminator_path'), ('topic_checker', 'prompt_topic_checker_path')]:
            path = m_config[path_key]; self.mod_sys_instructions[mod_key] = self._load_and_populate_prompt(path, self.debate_details)
            
        # Helper System Prompt (if needed)
        self.helper_sys_instruction = None
        if p_config.get('use_helper_feedback', False):
            helper_sys_path = p_config['helper_system_prompt_path']
            self.helper_sys_instruction = self._load_and_populate_prompt(helper_sys_path, self.debate_details)
            
        # Summarizer System Prompt
        summ_sys_path = self.debate_settings.get('summarizer_system_prompt_path')
        if not summ_sys_path: raise ValueError("summarizer_system_prompt_path missing in debate_settings.")
        # Format with empty dict assuming no claim-specific vars needed
        self.summarizer_system_instruction = self._load_and_populate_prompt(summ_sys_path, {}) #TODO: if no claim-specific vars needed, move this logic to main
        
        logger.debug("Formatted all prompts.")

    def _create_llm(self, provider_config: Dict[str, Any], system_instruction: Optional[str] = None) -> LLMInterface:
        """Helper to create LLM clients using the factory and resolved config."""
        if not provider_config: 
             raise ValueError(f"Resolved LLM provider config is missing or empty.")
        # LLMFactory expects the resolved provider config dictionary
        return LLMFactory.create_llm_client(provider_config, system_instruction=system_instruction)

    def _create_llm_clients(self):
        """Creates all necessary LLM client instances for this debate,
           including dedicated clients for memory summarization."""
        p_config = self.agents_configuration['persuader']
        d_config = self.agents_configuration['debater']
        m_config = self.agents_configuration['moderator']

        # Get pre-resolved agent/helper configs (injected by loader)
        p_provider_config = p_config.get('_resolved_llm_config')
        d_provider_config = d_config.get('_resolved_llm_config')
        m_provider_config = m_config.get('_resolved_llm_config')
        h_provider_config = None
        if p_config.get('use_helper_feedback', False):
            h_provider_config = p_config.get('_resolved_llm_config_helper')
        
        # --- Check if configs were found --- 
        if not p_provider_config: raise ValueError("Resolved LLM config missing for persuader.")
        if not d_provider_config: raise ValueError("Resolved LLM config missing for debater.")
        if not m_provider_config: raise ValueError("Resolved LLM config missing for moderator.")
        if p_config.get('use_helper_feedback', False) and not h_provider_config: 
            raise ValueError("LLM config missing for helper.")
        # --- End Checks --- 

        # --- Create Main Clients (with Agent System Prompts) --- 
        self.p_llm_client = self._create_llm(p_provider_config, self.p_sys_instruction)
        self.d_llm_client = self._create_llm(d_provider_config, self.d_sys_instruction)
        self.mod_term_client = self._create_llm(m_provider_config, self.mod_sys_instructions['terminator'])
        self.mod_topic_client = self._create_llm(m_provider_config, self.mod_sys_instructions['topic_checker'])
        self.p_helper_llm_client = None
        if h_provider_config: 
             self.p_helper_llm_client = self._create_llm(h_provider_config, self.helper_sys_instruction)
        
        # --- Create Dedicated Summarizer Clients (with Summarizer System Prompt) --- 
        self.p_summarizer_client = self._create_llm(p_provider_config, system_instruction=self.summarizer_system_instruction)
        self.d_summarizer_client = self._create_llm(d_provider_config, system_instruction=self.summarizer_system_instruction)
             
        logger.debug("Created all LLM clients (including dedicated summarizers).")

    def _create_memories(self):
        """Creates separate ChatSummaryMemory instances for Persuader and Debater,
           using dedicated summarizer clients based on agent LLM config."""
        try:
            trigger = int(self.debate_settings['summarization_trigger_tokens'])
            target = int(self.debate_settings['target_prompt_tokens'])
            keep = int(self.debate_settings['keep_messages_after_summary'])
        except KeyError as e: raise ValueError(f"Missing memory setting: {e}")
        except ValueError as e: raise ValueError(f"Invalid memory setting value: {e}")
        
        # Check if summarizer clients exist (created in _create_llm_clients)
        if not hasattr(self, 'p_summarizer_client') or not self.p_summarizer_client:
             raise RuntimeError("Persuader summarizer LLM client not created before memory.")
        if not hasattr(self, 'd_summarizer_client') or not self.d_summarizer_client:
             raise RuntimeError("Debater summarizer LLM client not created before memory.")
             
        self.persuader_memory = ChatSummaryMemory(
            summarizer_llm=self.p_summarizer_client,
            summarization_trigger_tokens=trigger, 
            target_prompt_tokens=target,
            keep_messages_after_summary=keep
        )
        logger.info("Created Persuader ChatSummaryMemory.")
        
        self.debater_memory = ChatSummaryMemory(
            summarizer_llm=self.d_summarizer_client,
            summarization_trigger_tokens=trigger, 
            target_prompt_tokens=target,
            keep_messages_after_summary=keep
        )
        logger.info("Created Debater ChatSummaryMemory.")

    def _create_agents(self):
        #TODO: further encapsulare this, it's a mess of mega long lines right now
        p_config=self.agents_configuration['persuader']; d_config=self.agents_configuration['debater']; m_config=self.agents_configuration['moderator']
        # ... (Get model configs) ...
        p_provider_config = p_config.get('_resolved_llm_config', {})
        d_provider_config = d_config.get('_resolved_llm_config', {})
        m_provider_config = m_config.get('_resolved_llm_config', {})
        h_provider_config = {}
        if p_config.get('use_helper_feedback', False): h_provider_config = p_config.get('_resolved_llm_config_helper', {})
        p_model_cfg = {**p_provider_config.get('default_config', {}), **p_config.get('model_config_override', {})}; d_model_cfg = {**d_provider_config.get('default_config', {}), **d_config.get('model_config_override', {})}; mod_model_cfg = {**m_provider_config.get('default_config', {}), **m_config.get('model_config_override', {})}; p_helper_model_cfg = {}
        if p_config.get('use_helper_feedback', False): p_helper_model_cfg = {**h_provider_config.get('default_config', {}), **p_config.get('helper_model_config_override', {})}

        self.persuader = PersuaderAgent(llm_client=self.p_llm_client, memory=self.persuader_memory, initial_prompt=self.initial_prompt_content, model_config=p_model_cfg, prompt_wrapper_path=p_config.get('prompt_wrapper_path'), use_helper_feedback=p_config.get('use_helper_feedback', False), helper_llm_client=self.p_helper_llm_client, helper_prompt_wrapper_path=p_config.get('helper_prompt_wrapper_path'), helper_model_config=p_helper_model_cfg)
        self.debater = DebaterAgent(llm_client=self.d_llm_client, memory=self.debater_memory, variables=self.debate_details, model_config=d_model_cfg, prompt_wrapper_path=d_config.get('prompt_wrapper_path'))
        self.moderator_terminator = ModeratorAgent(llm_client=self.mod_term_client, agent_name="ModeratorTerminator", model_config=mod_model_cfg, variables=self.debate_details)
        self.moderator_topic_checker = ModeratorAgent(llm_client=self.mod_topic_client, agent_name="ModeratorTopicChecker", model_config=mod_model_cfg, variables=self.debate_details)
        logger.info("Created all agents.") 