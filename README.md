# LLM Fallacy Susceptibility Analysis

This project extends the [LOGICOM](https://arxiv.org/abs/2308.09853) research on LLM susceptibility to logical fallacies, with a focus on:

1. Reproducing the original findings using the LOGICOM framework
2. Updating the framework to leverage modern reasoning-focused LLMs 
3. Replacing cloud-based models with a local open-source model
4. Possibly investigating how attention weight perturbations affect fallacy susceptibility

## Project Phases

### Phase 1: LOGICOM Reproduction
Reproducing the original paper which implements a sophisticated multi-agent debate framework. This involves:
- Understanding and running the complex agent architecture (persuader, debater, moderator, and helper agents)
- Setting up the experimental environment with different debate scenarios
- Configuring the necessary API keys and model connections
- Validating results against the published findings
- Optionally documenting challenges and solutions in the reproduction process

### Phase 2: Modern LLM Integration
Updating the framework to use more advanced cloud-based models released since the original research:
- Integrating newer reasoning-focused LLMs (e.g., GPT-4o, Claude 3, etc.)
- Adapting the agent architecture to leverage improved reasoning capabilities
- Comparing fallacy susceptibility across model generations
- Exploring whether newer models exhibit improved resistance to fallacious reasoning

### Phase 3: Local Model Integration
Replacing the cloud-based GPT models with locally deployed open-weights models:
- Setting up local model infrastructure
- Modifying the model interface to work with open-source LLMs
- Performance comparison between API-based and local models
- Optimizing for efficiency and resource usage

### Phase 4: Attention Weight Experimentation (Future Work)
Investigating whether specific perturbations to LLM attention weights affect fallacy susceptibility:
- Implementing mechanisms to modify attention weights
- Testing specific fallacy types (e.g., false dichotomy)
- Measuring changes in model opinions and susceptibility
- Analyzing the relationship between attention patterns and reasoning ability

## Original LOGICOM Background

This work builds upon the [LOGICOM paper](https://arxiv.org/abs/2308.09853) which investigates the rational thinking capability of Large Language Models (LLMs) in multi-round argumentative debates by presenting a diagnostic benchmark to assess LLM robustness against logical fallacies.

<figure>
  <img src="https://github.com/Amir-pyh/LOGICOM/blob/main/figs/LOGICOM.png" alt="LOGICOM demonstration" style="width:100%">
  <figcaption>LOGICOM: A demonstration of three scenarios evaluating LLMs' reasoning skills and vulnerability to logical fallacies.</figcaption>
</figure>

### Key Findings from Original Research

The original research addressed two main questions:

**RQ1**: Can large language models (with fixed weights) change their opinions through reasoning when faced with new arguments?

<figure>
  <img src="https://github.com/Amir-pyh/LOGICOM/blob/main/figs/Q1.png" alt="RQ1 Results" style="width:50%">
  <figcaption>Percentage of instances in which the debater agent changes its stance from disagreement to agreement.</figcaption>
</figure>

**RQ2**: Are large language models susceptible to fallacious reasoning?

<figure>
  <img src="https://github.com/Amir-pyh/LOGICOM/blob/main/figs/Q2-1.png" alt="RQ2 Results 1" style="width:70%">
  <figcaption>The average, taken from three repetitions, in which the persuader agent successfully convinced the debater agent for each scenario.</figcaption>
</figure>

<figure>
  <img src="https://github.com/Amir-pyh/LOGICOM/blob/main/figs/Q2-2-GPT-3_5.png" alt="RQ2 Results 2" style="width:70%">
</figure>

<figure>
  <img src="https://github.com/Amir-pyh/LOGICOM/blob/main/figs/Q2-2-GPT-4.png" alt="RQ2 Results 3" style="width:70%">
  <figcaption>Analyzing the susceptibility of GPT models to fallacious arguments. In the consistent agreement instances ("Three Success"), it shows a higher level of success rate for fallacious persuader compared to the logical persuaders for both GPT-3.5 and GPT-4 debater agents. Furthermore, the number of instances in the bar chart groups for "One Success" and "Two Success" can be seen as indications of level of inconsistency in debater agent's reasoning which is higher in GPT-3.5 compared to GPT-4.</figcaption>
</figure>

## Running the Code

### Basic Usage (Phase 1)
```bash
python main.py --api_key_openai <insert your OpenAI API key> --api_key_palm <insert your PaLM API key> --helper_prompt_instruction <No_Helper|Fallacy_Helper|Vanilla_Helper>
```

### Local Model Usage (Phase 2)
todo: *Documentation for running with local open-source models will be added after implementation*

## Repository Structure

```
LOGICOM/
├── agents/                     # Agent implementations for the debate system
│   ├── base.py                 # Abstract base class for all agents
│   ├── debaterAgent.py         # Implementation of the debater agent
│   ├── persuaderAgent.py       # Implementation of the persuader agent
│   └── modertorAgent.py        # Implementation of the moderator agent
│
├── claims/                     # Dataset of claims used for debates
│   └── all-claim-not-claim.csv # CSV file with claims for debate topics
│
├── config/                     # Configuration classes for LLM models
│   ├── gptconfig.py            # Configuration for GPT models
│   └── palmconfig.py           # Configuration for PaLM models
│
├── debates/                    # Debate logs generated during experiments
│   └── all-debates.zip         # Archive of debate logs
│
├── figs/                       # Figures and visualizations used in the paper
│
├── logical-fallacies-dataset/  # Logical fallacies dataset created from debates
│
├── memory/                     # Conversation history and summarization
│   ├── base.py                 # Base memory classes for conversation tracking
│   └── chatsummary.py          # Implementation of chat summarization
│
├── models/                     # LLM backend implementations
│   ├── openai.py               # Implementation for OpenAI models
│   └── base.py                 # Base model classes
│
├── prompts/                    # Prompt templates for different agents
│   ├── debater/                # Prompts for the debater agent
│   ├── persuader/              # Prompts for the persuader agent
│   ├── moderator/              # Prompts for the moderator agent
│   ├── helper/                 # Prompts for the helper agent
│   └── summary/                # Prompts for conversation summarization
│
├── main.py                     # Main script that orchestrates the debate experiments
├── utils.py                    # Utility functions for the project
├── type.py                     # Type definitions and enumerations
└── requirements.txt            # Required Python packages
```

## Logical Fallacy Dataset
The original research proposed a dataset containing over 5k pairs of logical/fallacious arguments. Each pair is extracted from debates generated by LLMs on 100 controversial subjects during the experiment. The CSV file for this dataset is located in the [logical-fallacies-dataset](https://github.com/Amir-pyh/LOGICOM/tree/main/logical-fallacies-dataset) folder.

## Citation
When building upon this work, please cite the original LOGICOM paper:

```bibtex
@misc{payandeh2023susceptible,
      title={How susceptible are LLMs to Logical Fallacies?},
      author={Amirreza Payandeh and Dan Pluth and Jordan Hosier and Xuesu Xiao and Vijay K. Gurbani},
      year={2023},
      eprint={2308.09853},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
