# LOGICOM - Rewritten Architecture

This directory contains a refactored version of the LOGICOM project, designed for improved modularity, maintainability, and extensibility, particularly regarding the use of different Large Language Models (LLMs).

## Architecture Overview

The core principles behind this refactoring are:

1.  **Interface-Based Design:** Core components (LLMs, Memory, Agents) interact through defined interfaces (`core/interfaces.py`), decoupling implementation details.
2.  **Dependency Injection:** Components like LLM clients and memory objects are created based on configuration and injected into agents, rather than being created internally.
3.  **Configuration-Driven:** Debate settings, LLM choices, prompt paths, and agent parameters are managed via YAML configuration files (`config/`).
4.  **Clear Directory Structure:** Code is organized by functionality (`llm`, `agents`, `memory`, `core`, `utils`, `config`).

## Directory Structure

```
Reworked/
├── main.py                 # Main entry point: parses args, loads config, sets up, runs debates
├── core/
│   ├── interfaces.py       # Defines core ABCs (LLMInterface, AgentInterface, MemoryInterface, AgentType)
│   └── orchestrator.py     # High-level debate loop logic, manages agent turns and moderation
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
│   └── helpers.py          # Utility functions (prompt loading, logging, etc.)
├── data/                   # Data directory referenced in config
│   └── ...
├── logs/                   # Default output directory for debate logs
│   └── ...
├── API_keys                # (Gitignored) File to store API keys locally
├── API_keys.template       # Template for the API keys file
├── requirements.txt        # Python package dependencies
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

3.  **Data:** Ensure the dataset specified in `config/settings.yaml` (`debate_settings.data_path`) is accessible. The default configuration points to `./data/claims/all-claim-not-claim.csv` relative to the project root.
4.  **Prompts:** Ensure the prompt files referenced in `config/settings.yaml` exist within the `prompts/` directory.

## Running

Execute the main script from the project root directory:

```bash
python main.py [OPTIONS]
```

**Options:**

*   `--config_run_name <NAME>`: Specifies which agent configuration section from `settings.yaml` to use (default: `OriginalDefault`).
*   `--claim_index <INDEX>`: Run only for a specific claim index (0-based) in the dataset. If omitted, runs for all claims.
*   `--settings_path <PATH>`: Path to the settings YAML file (default: `./config/settings.yaml`).
*   `--models_path <PATH>`: Path to the models YAML file (default: `./config/models.yaml`).

**Example:** Run the default 'OriginalDefault' configuration for claim index 5:

```bash
python main.py --config_run_name OriginalDefault --claim_index 5 
```

**Example:** Run a hypothetical 'LocalRun_Llama3' configuration for all claims:

```bash
python main.py --config_run_name LocalRun_Llama3
```

## Configuration

*   **`config/models.yaml`**: Define different LLM providers (OpenAI, Gemini, local) and their connection details (API keys/endpoints, model names, default parameters).
*   **`config/settings.yaml`**: 
    *   `debate_settings`: Configure data paths, logging options, max rounds.
    *   `agent_configurations`: Define different named setups (e.g., `OriginalDefault`, `LocalRun_Llama3`). Each setup specifies which LLM config (`llm_config_ref`), prompt template, and specific parameters to use for the Persuader, Debater, and Moderator(s).

## Local LLMs

To use a local LLM:

1.  Ensure your local LLM server (e.g., Ollama, llama-cpp-python with API) is running.
2.  Define a configuration for it in `config/models.yaml` under `llm_models`.
    *   Set `provider: local`.
    *   Set `local_type: ollama` or `local_type: generic` (for OpenAI-compatible APIs).
    *   Specify the `api_base_url` and `model_name`.
3.  Create a run configuration in `config/settings.yaml` under `agent_configurations` that references your local LLM config using `llm_config_ref`.
4.  Run `main.py` using the `--config_run_name` option pointing to your local run configuration. 