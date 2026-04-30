# RCPP-Agent

## Overview


<p align="justify">
RCPP-Agent is a LLM-based multi-agent architecture for fine-grained roadside charging pile planning. This architecture integrates six specialized agents to operationalize an expert-inspired planning workflow, from multimodal environmental perception and suitability assessment to multi-scenario evaluation and phased decision-making.
</p>

## Prerequisites

- Python **3.9+**
- SpatiaLite
- Neo4j Database
- Fine-tuned LLM and MLLM
- OpenAI API key

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/NNUSCENELAB/RCPP-Agent.git
cd RCPP-Agent
```

Install the required packages (Python 3.9+):

```bash
pip install -r requirements.txt
```

## Configuration

- Create **`configs/.env`** for secrets and overrides.
- **`configs/config.py`** defines global multi-agent settings.
- **`configs/data_config.py`** defines workspace paths, Neo4j env names, and SpatiaLite defaults.

## Usage

Import study-area data into the SpatiaLite database (before `rcpp_agent.py`):

```bash
python tools/data_flow_tool/import_spatialite_tool.py --area <district_slug>
```

Run the main agent:

```bash
cd src
python rcpp_agent.py
```


## Project Structure

```
RCPP-Agent/
├── configs/
│   ├── config.py               # Multi-agent configurations
│   └── data_config.py          # Data configurations
├── rcpp_core/
│   ├── prompts.py              # Prompt templates for agents
│   ├── scene_semantic_inference.py   # Multimodal inference glue
│   ├── build_scene_semantic_memory.py  # Neo4j load, merge, save helpers
│   └── neo4j_scene_semantic_store.py   # LangGraph store on Neo4j
├── src/
│   ├── rcpp_agent.py           # Main application
│   ├── task_orchestration_agent.py   # Top orchestrator
│   ├── environment_perception_agent.py   # scene-semantic memory construction
│   ├── suitability_assessment_agent.py   # RCP suitability scoring
│   ├── multi_scenario_evaluation_agent.py   # Scenario weights via LLM-AHP
│   ├── phased_decision_making_agent.py   # Phased strategy and candidate scores
│   └── review_agent.py         # Plan review and readjust signal
├── tools/
│   ├── base.py / registry.py   # Tool base class and registration
│   ├── llm_ahp_tool.py         # LLM-AHP subgraph
│   ├── rcpp_paths.py           # Workspace path helpers
│   ├── data_flow_tool/         # SpatiaLite ingest
│   ├── memory_construction_tool/   # Perception pipeline tools
│   ├── multi_scenario_evaluation_tool/   # MCDA calculators
│   └── review_tool/            # Review metrics tools
```

## Changelog

### 2026-04-22 v1.0

1. Created the RCPP-Agent.

