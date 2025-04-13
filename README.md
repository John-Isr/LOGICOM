# LOGICOM - Rewritten Architecture

This directory contains a refactored version of the LOGICOM project, designed for improved modularity, maintainability, and extensibility, particularly regarding the use of different Large Language Models (LLMs).

## Architecture Overview

This project simulates and analyzes debates between AI agents on specific topics/claims using a structured, modular architecture.

**Core Architecture Principles:**

1.  **Orchestration Pattern:** The system is centered around a `DebateOrchestrator` (`core/orchestrator.py`), which manages the turn-by-turn flow of the debate, including agent interactions and moderation checks.
2.  **Agent-Based System:** The debate involves distinct AI agents, each with specific roles:
    *   `PersuaderAgent`: Aims to convince the `DebaterAgent` of the claim. It can optionally use a "helper" LLM to refine its arguments or identify fallacies in the opponent's reasoning.
    *   `DebaterAgent`: Responds to the `PersuaderAgent`, typically arguing against the claim or seeking clarification.
    *   `ModeratorAgent`: Multiple instances monitor the debate based on specific criteria (termination conditions, topic relevance, signal detection like `[CONVINCED]`) using dedicated prompts.
3.  **Modular Components & Interfaces:** Key functionalities are separated into modules with clearly defined interfaces (`core/interfaces.py`):
    *   `AgentInterface`: Abstract base for all agent types.
    *   `LLMInterface`: Abstract base for interacting with different Large Language Models.
    *   `MemoryInterface`: Abstract base for managing conversation history for each agent (e.g., `ChatSummaryMemory` handles context limits via summarization or truncation).
4.  **Configuration Driven:** System behavior is heavily controlled by YAML configuration files (`config/settings.yaml`, `config/models.yaml`). This allows easy modification of:
    *   Debate parameters (max rounds, delays).
    *   Agent configurations (LLM choice, prompts, helper usage).
    *   LLM provider details (model names, API keys/endpoints).
    *   Data paths and logging settings.
5.  **LLM Abstraction:** An `LLMFactory` (`llm/llm_factory.py`) creates specific LLM client instances (e.g., `OpenAIClient`, `GeminiClient`, `LocalClient`) based on the configuration, abstracting the underlying API details.
6.  **Prompt Management:** Agent prompts (system instructions, wrappers, helper prompts) are loaded from external files (typically in `prompts/`) and dynamically formatted with claim-specific data.
7.  **Data Handling:** Debates are initiated based on claims loaded from a dataset (e.g., CSV), specified in the configuration.
8.  **Entry Point & Setup (`main.py`):** The main script handles command-line arguments, loads configurations (`config/loader.py`), sets up API keys (`utils/set_api_keys.py`), prepares the debate environment *for each claim* by instantiating `core.debate_setup.DebateInstanceSetup` (which creates agents, LLMs, memory), iterates through claims, and invokes the orchestrator.
9.  **Logging & Output:** Detailed debate logs (transcripts, results, metadata, token usage) are saved to structured directories within the path defined by `log_base_path` in `config/settings.yaml` (default: `logs/`). Logs are saved per claim and configuration in the format `<log_base_path>/<topic_id>/<helper_type>/<chat_id>.[format]`. A summary Excel file (`all_debates_summary.xlsx`) and fallacy CSV (`fallacies.csv`) are saved directly in the `log_base_path`. Logging is handled by `utils/helpers.save_debate_log`.

**Overall Flow:**

`main.py` -> Load Config -> Load Data -> For each Claim -> Instantiate `DebateInstanceSetup` (Create LLMs, Memory, Agents based on config) -> Instantiate `DebateOrchestrator` -> `Orchestrator.run_debate()` -> [Debate Loop: Persuader Turn -> Debater Turn -> Moderator Checks -> Evaluate Termination Conditions] -> `save_debate_log` (Save Logs & Results, Process Fallacies) -> Summarize Run.

This architecture promotes modularity and configurability, making it easier to experiment with different LLMs, prompts, agent strategies, and debate parameters.

## Directory Structure

```
LOGICOM/ # Project Root
├── main.py                 # Main entry point: parses args, loads config, sets up, runs debates
├── core/
│   ├── interfaces.py       # Defines core ABCs (LLMInterface, AgentInterface, MemoryInterface)
│   ├── orchestrator.py     # High-level debate loop logic, manages agent turns and moderation
│   └── debate_setup.py     # Handles claim-specific setup (prompts, clients, agents, memory)
├── llm/
│   ├── __init__.py
│   ├── llm_factory.py      # Factory to create LLM clients based on config
│   ├── openai_client.py    # Implementation for OpenAI
│   ├── gemini_client.py    # Implementation for Google Gemini
│   └── local_client.py     # Implementations for local LLMs (Ollama, Generic OpenAI-compatible)
├── agents/
│   ├── __init__.py
│   ├── base_agent.py       # Base class for all agents
│   ├── persuader_agent.py  # Persuader agent logic
│   ├── debater_agent.py    # Debater agent logic
│   └── moderator_agent.py  # Moderator agent logic (performs single check)
├── memory/
│   ├── __init__.py
│   └── chat_summary_memory.py # Memory implementation (stores history, formats prompts)
├── prompts/                # Contains prompt template files used by agents
│   └── ...
├── config/
│   ├── __init__.py
│   ├── settings.yaml       # Main config: debate settings, agent setups, LLM references
│   ├── models.yaml         # LLM provider configs: API keys (optional), endpoints, model names
│   └── loader.py           # Logic to load and parse YAML configuration files
├── utils/
│   ├── __init__.py
│   ├── set_api_keys.py     # Script to set API keys as environment variables
│   └── log_debate.py       # Logging, file handling
│   └── token_utils.py      # Token calculation utilities for LLMs
├── claims/                 # Default data directory containing claim datasets (e.g., CSV)
│   └── ...
├── logs/                   # Default output directory for debate logs, summaries, and fallacy reports
│   ├── <topic_id>/         # Subdirectory for each claim/topic
│   │   └── <helper_type>/  # Subdirectory for each agent config run
│   │       └── <chat_id>.[json|html|txt] # Individual debate logs
│   ├── all_debates_summary.xlsx # Summary of all runs
│   └── fallacies.csv       # Log of detected fallacies (if applicable)
├── debates/                # (legacy / alternative log directory - TODO:check usage)
│   └── ...
├── API_keys                # (Gitignored) File to store API keys locally
├── API_keys.template       # Template for the API keys file
├── requirements.txt        # Python package dependencies
├── .gitignore              # Specifies intentionally untracked files for Git
├── LICENSE                 # Project license information
└── README.md               # This file
```

## Setup

1.  **Install Dependencies:** From within the `Reworked` directory:
    ```bash
    pip install -r requirements.txt
    ```
2.  **API Keys:** You need to provide API keys for OpenAI and/or Google Gemini. You can do this in one of the following ways (the application checks in this order):
    *   **Recommended: Use the `set_api_keys.py` script:**
        1. Copy `API_keys.template` to `API_keys` in the project root directory.
        2. Edit `API_keys` with your actual keys, uncommenting the relevant lines.
        3. Run the script *from within the project root directory* to set environment variables for the current session:
           ```bash
           python utils/set_api_keys.py
           ```
    *   **Manual Environment Variables:** Set the variables directly in your terminal session *before* running `main.py`:
        ```bash
        export OPENAI_API_KEY="your_openai_key"
        export GOOGLE_API_KEY="your_google_api_key"
        ```
    *   **(Less Secure) Edit `config/models.yaml`:** Add your keys directly into the `models.yaml` file under the respective provider configurations.

3.  **Data:** Ensure the dataset specified in `config/settings.yaml` (`debate_settings.data_path`) is accessible. The default configuration points to `./claims/all-claim-not-claim.csv` relative to the project root. `main.py` will attempt to resolve this path relative to its own location if the direct path isn't found.
4.  **Prompts:** Ensure the prompt files referenced in `config/settings.yaml` exist within the `prompts/` directory or paths specified.

## Running

Execute the main script from the project root directory:

```bash
python main.py [OPTIONS]
```

**Options:**

*   `--config_run_name <NAME>`: Specifies which agent configuration section from `settings.yaml` to use (default: `Default_NoHelper`).
*   `--claim_index <INDEX>`: Run only for a specific claim index (0-based) in the dataset. If omitted, runs for all claims.
*   `--settings_path <PATH>`: Path to the settings YAML file (default: `./config/settings.yaml`).
*   `--models_path <PATH>`: Path to the models YAML file (default: `./config/models.yaml`).

**Example:** Run the default 'Default_NoHelper' configuration for claim index 5:

```bash
python main.py --config_run_name Default_NoHelper --claim_index 5 
```

**Example:** Run a hypothetical 'LocalRun_Llama3' configuration for all claims:

```bash
python main.py --config_run_name LocalRun_Llama3
```

## Configuration

*   **`config/models.yaml`**: Define different LLM providers (OpenAI, Gemini, local) and their connection details (API keys/endpoints, model names, default parameters).
*   **`config/settings.yaml`**: 
    *   `debate_settings`: Configure data paths (using `claims/` by default), logging options (`log_base_path` defaults to `logs/`, log formats), max rounds, memory settings, column mappings.
    *   `agent_configurations`: Define different named setups (e.g., `Default_NoHelper`, `Default_FallacyHelper`). Each setup specifies which LLM config (`llm_config_ref`), prompt template paths, and specific parameters to use for the Persuader, Debater, and Moderator(s).

## Local LLMs

To use a local LLM:

1.  Ensure your local LLM server (e.g., Ollama, llama-cpp-python with API) is running.
2.  Define a configuration for it in `config/models.yaml` under `llm_models`.
    *   Set `provider: local`.
    *   Set `local_type: ollama` or `local_type: generic` (for OpenAI-compatible APIs).
    *   Specify the `api_base_url` and `model_name`.
3.  Create a run configuration in `config/settings.yaml` under `agent_configurations` that references your local LLM config using `llm_config_ref`.
4.  Run `main.py` using the `--config_run_name` option pointing to your local run configuration. 