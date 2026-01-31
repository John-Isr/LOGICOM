# LOGICOM - Rewritten Architecture

A modular system for simulating and analyzing debates between AI agents on specific topics/claims using different Large Language Models (LLMs).

## Architecture Overview

The system uses a **DebateOrchestrator** to manage turn-by-turn debates between:
- **PersuaderAgent**: Convinces the debater of a claim (optionally uses helper LLM)
- **DebaterAgent**: Responds to the persuader
- **ModeratorAgent**: Multiple instances monitor termination, topic relevance, and conviction signals

**Key Features:**
- **Modular Design**: Interfaces for agents, LLMs, and memory (`core/interfaces.py`)
- **LLM Abstraction**: Factory pattern supports OpenAI, Gemini, and local LLMs
- **Configuration Driven**: YAML files control debate parameters, agent setups, and LLM choices
- **Structured Output**: All runs create timestamped `results/<timestamp>_<run_name>/` directories

**Flow:** `main.py` → Load Config → For each Claim → Create Agents/LLMs → Run Debate → Save Results

## Directory Structure

```
LOGICOM/
├── main.py                 # Main entry point
├── core/                   # Orchestrator, interfaces, debate setup
├── llm/                    # LLM clients (OpenAI, Gemini, local)
├── agents/                 # Persuader, Debater, Moderator agents
├── memory/                 # Conversation history management
├── prompts/                # Prompt templates
├── config/                 # settings.yaml, models.yaml
├── utils/                  # Utilities (logging, analysis, multiple_runs)
├── claims/                 # Claim datasets (CSV)
└── results/                # Output: results/<timestamp>_<run_name>/
    ├── debates/            # Debate logs
    ├── all_debates_summary.xlsx
    └── analysis/           # Analysis results (batch runs)
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API keys** (choose one method):
   - **Recommended:** Copy `API_keys.template` to `API_keys`, edit with your keys, then run:
     ```bash
     python utils/set_api_keys.py
     ```
   - **Manual:** Set environment variables:
     ```bash
     export OPENAI_API_KEY="your_key"
     export GOOGLE_API_KEY="your_key"
     ```

3. **Configure:** Ensure `config/settings.yaml` points to your claims dataset and prompts exist in `prompts/`.

## Running

### Single Debates (`main.py`)

```bash
python main.py --run_name <NAME> [OPTIONS]
```

**Required:**
- `--run_name <NAME>`: Name for this run (creates `results/<timestamp>_<run_name>/`)

**Options:**
- `--helper_type <TYPE>`: Helper configuration (default: `Default_No_Helper`)
- `--claim_index <INDEX>`: Run specific claim (0-based), or omit for all claims
- `--max_rounds <N>`: Override max rounds from settings
- `--settings_path <PATH>`: Custom settings file (default: `./config/settings.yaml`)
- `--models_path <PATH>`: Custom models file (default: `./config/models.yaml`)

**Examples:**
```bash
# Single claim
python main.py --run_name test --helper_type Default_No_Helper --claim_index 5

Run all claims with a specific helper type:
```bash
python main.py --helper_type Default_Fallacy_Helper
```

### Batch Runs (`multiple_runs.py`)

```bash
python utils/multiple_runs.py --run_name <NAME> [OPTIONS]
```

**Required:**
- `--run_name <NAME>`: Name for batch run

**Options:**
- `--helper_types <TYPE1> <TYPE2> ...`: Helper types to run (default: all)
- `--claim_indexes <INDEX1> <INDEX2> ...`: Specific claims or ranges (e.g., `0-199`)
- `--max_workers <N>`: Parallel workers (default: 4, use 1 for sequential)
- `--list_helpers`: List available helper types

**Examples:**
```bash
# All helper types, all claims
python utils/multiple_runs.py --run_name batch_experiment

# Specific helper types and claims
python utils/multiple_runs.py --run_name test --helper_types Default_No_Helper Default_Fallacy_Helper --claim_indexes 0 1 2
```

## Output

All runs save to `results/<timestamp>_<run_name>/`:
- **Debate logs:** `debates/<topic_id>/<helper_type>/<chat_id>/` (JSONL files)
- **Summary:** `all_debates_summary.xlsx` (Excel with all debate results)
- **Analysis:** `analysis/` directory (batch runs only, auto-generated)

## Configuration

- **`config/models.yaml`**: Define LLM providers (OpenAI, Gemini, local) with API keys/endpoints and model names
- **`config/settings.yaml`**: 
  - `debate_settings`: Data paths, max rounds, memory settings, column mappings
  - `agent_configurations`: Named setups (e.g., `Default_No_Helper`) specifying LLM models for each agent

## Local LLMs


1.  Ensure your local LLM server (e.g., Ollama, llama-cpp-python with API) is running.
2.  Define a configuration for it in `config/models.yaml` under `llm_models`.
    *   Set `provider: local`.
    *   Set `local_type: huggingface` (for Hugging Face models with quantization).
    *   Specify the `model_name_or_path` and `quantization_bits` (e.g., 4 for 4-bit quantization).
3.  Create a run configuration in `config/settings.yaml` under `agent_configurations` that references your local LLM model name.
4.  Run `main.py` or `multiple_runs.py` using the `--helper_type` option pointing to your local run configuration. 
