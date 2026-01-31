import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import os 

# Import our custom logger
from utils.log_main import logger as debate_logger

# --- Import necessary classes ---
from core.interfaces import HELPER_TYPE_NONE, HELPER_TYPE_FALLACY, HELPER_TYPE_LOGICAL
from memory.chat_summary_memory import ChatSummaryMemory
from agents.persuader_agent import PersuaderAgent
from agents.debater_agent import DebaterAgent
from agents.moderator_agent import ModeratorAgent
from llm.llm_factory import LLMFactory
_create_llm_client_func = LLMFactory.create_llm_client
# Use our custom debate_logger instead of creating a new one
logger = debate_logger


class DebateInstanceSetup:
    """Handles claim-specific setup: formatting prompts, creating clients & agents."""
    def __init__(self, 
                 agents_configuration: Dict[str, Any],
                 debate_settings: Dict[str, Any],
                 formatted_prompts: Dict[str, str]):
        
        logger.debug("Setting up new debate instance...")
        self.agents_configuration = agents_configuration
        self.debate_settings = debate_settings
        self.prompts = formatted_prompts

        # --- Setup Steps --- 
        
        # Step 2: Create LLM Clients
        self._create_llm_clients() 
        
        # Step 3: Create Memories
        self._create_memories() 
        
        # Step 4: Create Agents
        self._create_agents() 

        logger.debug("Debate instance setup complete.")
        
    # --- Private Helper Methods --- 
    def _create_llm_clients(self):
        """Creates all necessary LLM client instances for this debate,
           including dedicated clients for memory summarization,
           using formatted system instructions from self.prompts."""
        p_config = self.agents_configuration['persuader']
        d_config = self.agents_configuration['debater']
        m_config = self.agents_configuration['moderator']

        # Determine if helper is configured based on helper_type
        helper_type = self.agents_configuration['helper_type']
        use_helper = helper_type != HELPER_TYPE_NONE

        # Get resolved agent/helper LLM configurations
        p_provider_config = p_config['_resolved_llm_config']
        d_provider_config = d_config['_resolved_llm_config']
        m_provider_config = m_config['_resolved_llm_config']
        # Get helper config only if helper is actually used according to helper_type
        h_provider_config = p_config['_resolved_llm_config_helper'] if use_helper else None 
        
        # --- Check if configs were found --- 
        if not p_provider_config: raise ValueError("Resolved LLM config missing for persuader.")
        if not d_provider_config: raise ValueError("Resolved LLM config missing for debater.")
        if not m_provider_config: raise ValueError("Resolved LLM config missing for moderator.")
        # Check helper config only if use_helper is True
        if use_helper and not h_provider_config: 
            raise ValueError("LLM config missing for helper.")
        # --- End Checks --- 

        # --- Create Main Clients --- 
        self.p_llm_client = _create_llm_client_func(p_provider_config, self.prompts['persuader_system'])
        self.d_llm_client = _create_llm_client_func(d_provider_config, self.prompts['debater_system'])
        
        # Create separate moderator clients, ensuring prompts exist
        mod_term_instr = self.prompts['moderator_terminator']
        if mod_term_instr is None: raise ValueError("Formatted prompt 'moderator_terminator' not found.")
        self.mod_term_client = _create_llm_client_func(m_provider_config, mod_term_instr)

        mod_topic_instr = self.prompts['moderator_topic']
        if not mod_topic_instr: raise ValueError("Formatted prompt for 'moderator_topic' not found.")
        self.mod_topic_client = _create_llm_client_func(m_provider_config, mod_topic_instr)
        
        mod_conv_instr = self.prompts['moderator_conviction']
        if not mod_conv_instr: raise ValueError("Formatted prompt for 'moderator_conviction' not found.")
        self.mod_conv_client = _create_llm_client_func(m_provider_config, mod_conv_instr)

        mod_arg_quality_instr = self.prompts.get('moderator_argument_quality')
        if not mod_arg_quality_instr: 
            raise ValueError(f"Formatted prompt for 'moderator_argument_quality' not found. Available prompts: {list(self.prompts.keys())}")
        mod_arg_quality_config = self.agents_configuration['moderator_argument_quality']
        if not mod_arg_quality_config: raise ValueError("Configuration for 'moderator_argument_quality' not found.")
        mod_arg_quality_provider_config = mod_arg_quality_config['_resolved_llm_config']
        if not mod_arg_quality_provider_config: raise ValueError("Resolved LLM config missing for moderator_argument_quality.")
        self.mod_arg_quality_client = _create_llm_client_func(mod_arg_quality_provider_config, mod_arg_quality_instr)

        mod_debate_quality_instr = self.prompts.get('moderator_debate_quality')
        if not mod_debate_quality_instr: 
            raise ValueError(f"Formatted prompt for 'moderator_debate_quality' not found. Available prompts: {list(self.prompts.keys())}")
        self.mod_debate_quality_client = _create_llm_client_func(mod_arg_quality_provider_config, mod_debate_quality_instr)

        self.p_helper_llm_client = None
        # Create helper client only if use_helper is True and config was resolved
        if h_provider_config: 
             # Get model identifier for logging (different models use different keys)
             model_id = h_provider_config.get('model_name_or_path', h_provider_config.get('model_name', 'Unknown'))
             logger.info(f"Attempting to create helper LLM client using config: {model_id}...")
             # Retrieve the correct system prompt for the determined helper type
             helper_system_instr, _ = self._determine_helper_prompts(helper_type)
             if use_helper and helper_system_instr is None:
                  raise ValueError(f"Formatted helper system prompt not found for helper type '{helper_type}'. Check settings 'prompt_paths'.")

             self.p_helper_llm_client = _create_llm_client_func(h_provider_config, helper_system_instr)
             if self.p_helper_llm_client:
                  logger.info("Helper LLM client created successfully.")
        
        # --- Create Dedicated Summarizer Clients (with Summarizer System Prompt) --- 
        summarizer_instr = self.prompts['memory_summarizer']
        if not summarizer_instr: raise ValueError("Formatted prompt for 'memory_summarizer' not found.")
        # Assuming summarizer uses same base LLM config as agent, just different system prompt
        self.p_summarizer_client = _create_llm_client_func(p_provider_config, system_instruction=summarizer_instr)
        self.d_summarizer_client = _create_llm_client_func(d_provider_config, system_instruction=summarizer_instr)
             
        logger.debug("Created all LLM clients (including dedicated summarizers).")

    def _determine_helper_prompts(self, helper_type: str) -> Tuple[Optional[str], Optional[str]]:
        """Determines the system and wrapper prompt keys based on helper_type."""
        if helper_type == HELPER_TYPE_NONE:
            # No helper prompts needed
            return None, None
        elif helper_type == HELPER_TYPE_FALLACY:
            system_key = "helper_fallacy_system"
            wrapper_key = "helper_fallacy_wrapper"
        elif helper_type == HELPER_TYPE_LOGICAL:
            system_key = "helper_logical_system"
            wrapper_key = "helper_logical_wrapper"
        else:
            # Handle unknown helper types
            raise ValueError(f"Unknown helper_type '{helper_type}' specified. No helper prompts will be loaded.")
        
        # Retrieve the actual prompt content using the determined keys
        system_prompt = self.prompts[system_key] 
        wrapper_prompt = self.prompts[wrapper_key]

        return system_prompt, wrapper_prompt

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
        logger.debug("Created Persuader ChatSummaryMemory.")
        
        self.debater_memory = ChatSummaryMemory(
            summarizer_llm=self.d_summarizer_client,
            summarization_trigger_tokens=trigger, 
            target_prompt_tokens=target,
            keep_messages_after_summary=keep
        )
        logger.debug("Created Debater ChatSummaryMemory.")

    def _create_agents(self):
        #TODO: further encapsulare this, it's a mess of mega long lines right now
        p_config=self.agents_configuration['persuader']; d_config=self.agents_configuration['debater']; m_config=self.agents_configuration['moderator']
        helper_type = self.agents_configuration['helper_type']
        if helper_type is None: raise ValueError("Helper type not found in agents_configuration.")
        use_helper = helper_type != HELPER_TYPE_NONE

        # Get resolved LLM configs again to extract default generation params
        p_provider_config = p_config['_resolved_llm_config']
        d_provider_config = d_config['_resolved_llm_config']
        m_provider_config = m_config['_resolved_llm_config']
        h_provider_config = p_config['_resolved_llm_config_helper'] if use_helper else {}
        mod_arg_quality_config = self.agents_configuration['moderator_argument_quality']
        if not mod_arg_quality_config: raise ValueError("Configuration for 'moderator_argument_quality' not found.")
        mod_arg_quality_provider_config = mod_arg_quality_config['_resolved_llm_config']
        if not mod_arg_quality_provider_config: raise ValueError("Resolved LLM config missing for moderator_argument_quality.")

        # Combine default generation params from model config with agent-specific overrides
        p_model_cfg = {**p_provider_config.get('default_config', {}), **p_config.get('model_config_override', {})}; d_model_cfg = {**d_provider_config.get('default_config', {}), **d_config.get('model_config_override', {})}; mod_model_cfg = {**m_provider_config.get('default_config', {}), **m_config.get('model_config_override', {})}; mod_arg_quality_model_cfg = {**mod_arg_quality_provider_config.get('default_config', {}), **mod_arg_quality_config.get('model_config_override', {})}; p_helper_model_cfg = {}
        if use_helper: p_helper_model_cfg = {**h_provider_config.get('default_config', {}), **p_config.get('helper_model_config_override', {})}

        # Retrieve required prompt content from self.prompts using direct access
        persuader_initial_prompt = self.prompts['persuader_initial']
        persuader_wrapper = self.prompts['persuader_wrapper']
        debater_wrapper = self.prompts['debater_wrapper']

        # Determine correct helper prompts based on type
        helper_system_prompt, helper_wrapper = self._determine_helper_prompts(helper_type)

        # Check helper prompts only if helper is actually used
        if use_helper:
            # Check resolved provider config was found earlier
            if not h_provider_config: raise ValueError(f"Helper type '{helper_type}' requires a helper_model_name to be set in settings.yaml")
            # Check helper LLM client was successfully created
            if self.p_helper_llm_client is None:
                 raise ValueError(f"Helper LLM client failed to initialize for helper type '{helper_type}'. Check model config '{p_config.get('helper_model_name')}'.")

        # Instantiate agents, passing prompt content instead of paths
        self.persuader = PersuaderAgent(llm_client=self.p_llm_client, memory=self.persuader_memory, initial_prompt=persuader_initial_prompt, model_config=p_model_cfg, prompt_wrapper=persuader_wrapper, use_helper_feedback=use_helper, helper_llm_client=self.p_helper_llm_client, helper_prompt_wrapper=helper_wrapper, helper_model_config=p_helper_model_cfg)
        self.debater = DebaterAgent(llm_client=self.d_llm_client, memory=self.debater_memory, model_config=d_model_cfg, prompt_wrapper=debater_wrapper)
        self.moderator_terminator = ModeratorAgent(llm_client=self.mod_term_client, agent_name="ModeratorTerminator", model_config=mod_model_cfg)
        self.moderator_topic_checker = ModeratorAgent(llm_client=self.mod_topic_client, agent_name="ModeratorTopicChecker", model_config=mod_model_cfg)
        self.moderator_conviction = ModeratorAgent(llm_client=self.mod_conv_client, agent_name="ModeratorConviction", model_config=mod_model_cfg)
        self.moderator_argument_quality = ModeratorAgent(llm_client=self.mod_arg_quality_client, agent_name="ModeratorArgumentQuality", model_config=mod_arg_quality_model_cfg)
        self.moderator_debate_quality = ModeratorAgent(llm_client=self.mod_debate_quality_client, agent_name="ModeratorDebateQuality", model_config=mod_arg_quality_model_cfg)
        logger.info("Created all agents.") 