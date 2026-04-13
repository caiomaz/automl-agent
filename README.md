# AutoML-Agent

AutoML-Agent is the official implementation of **AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML** (ICML 2025).

[Paper](https://arxiv.org/abs/2410.02958) | [Poster](static/pdfs/poster.pdf) | [Project Website](https://deepauto-ai.github.io/automl-agent/)

## 1. What This Project Does

AutoML-Agent turns a task description into a locally executed machine learning workflow. Instead of stopping at model search, it coordinates multiple LLM-driven agents that:

- parse the user request into structured requirements,
- retrieve datasets and implementation knowledge,
- propose and compare end-to-end plans,
- generate executable Python for the selected solution,
- run that code locally, retry on failures, and save artifacts for inspection.

The current workflow is **CLI-first**, with optional programmatic usage through `AgentManager`.

## 2. Why It Is Different From Classical AutoML

- It is **multi-agent**, with separate planning, data, modeling, and execution roles.
- It is **full-pipeline**, covering retrieval, preprocessing, training, evaluation, explainability, and deployment.
- It is **execution-grounded**, because generated code is run locally and corrected when it fails.
- It is **environment-aware**, because the CLI can inject installed packages and hardware information into the generation loop.
- It is **workspace-aware**, because all generated artifacts are written under `agent_workspace/` with stable path conventions.

## 3. Quick Start

### 3.1 Prerequisites

- Python 3.11+
- An OpenRouter API key for the default workflow
- Optional provider credentials depending on the dataset or retrieval path you want to use

### 3.2 Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install local heavy dependencies only when you need local training stacks such as PyTorch, vision libraries, or vLLM:

```bash
pip install -r requirements-local.txt
```

### 3.3 Configure The Environment

```bash
cp .env.example .env
```

At minimum, set:

```bash
OPENROUTER_API_KEY=your-key-here
```

You can keep the default backbone or change it through `.env`:

```bash
LLM_BACKBONE=or-glm-5
LLM_PROMPT_AGENT=or-gpt-5-nano
```

### 3.4 Run The Interactive CLI

```bash
python -m cli
```

The CLI walks through six stages:

1. backbone and Prompt Agent model selection,
2. task type selection,
3. dataset path or URL selection,
4. task description,
5. optional constraints,
6. advanced options such as RAP and revision count.

### 3.5 Run Non-Interactive Mode

```bash
python -m cli run \
  --task tabular_regression \
  --prompt "Predict crab age from both categorical and numerical features in the uploaded Crab Age Dataset" \
  --llm or-glm-5 \
  --n-plans 1 \
  --n-revise 1
```

### 3.6 Inspect Available LLM Aliases

```bash
python -m cli list-models
```

## 4. Supported Task Types

| Task Type | Typical Modality | Primary Metric |
| --- | --- | --- |
| `image_classification` | image | accuracy |
| `text_classification` | text | accuracy |
| `tabular_classification` | tabular | F1 |
| `tabular_regression` | tabular | RMSLE |
| `tabular_clustering` | tabular | RI |
| `node_classification` | graph | accuracy |
| `ts_forecasting` | time series | RMSLE |

## 5. Workspace Layout

All runs use the canonical workspace defined in `utils.workspace`:

```text
agent_workspace/
├── datasets/
├── exp/
└── trained_models/
```

- `datasets/` stores uploaded or downloaded input data.
- `exp/` stores generated scripts and experiment outputs.
- `trained_models/` stores serialized pipelines and model artifacts.

## 6. Documentation Map

The full human-facing documentation lives in `docs/` and follows a numbered structure:

- [Documentation Index](docs/00_INDEX.md)
- [Project Overview](docs/01_PROJECT_OVERVIEW.md)
- [Architecture And Agents](docs/02_ARCHITECTURE_AND_AGENTS.md)
- [Setup And Environment](docs/03_SETUP_AND_ENVIRONMENT.md)
- [Quickstart Tutorial](docs/04_QUICKSTART_TUTORIAL.md)
- [CLI Reference](docs/05_CLI_REFERENCE.md)
- [Workspace And Datasets](docs/06_WORKSPACE_AND_DATASETS.md)
- [Execution Pipeline And Artifacts](docs/07_EXECUTION_PIPELINE_AND_ARTIFACTS.md)
- [Task Types And Metrics](docs/08_TASK_TYPES_AND_METRICS.md)
- [LLM Configuration](docs/09_LLM_CONFIGURATION.md)
- [Examples And Applicability](docs/10_EXAMPLES_AND_APPLICABILITY.md)
- [Troubleshooting](docs/11_TROUBLESHOOTING.md)
- [Development And Testing](docs/12_DEVELOPMENT_AND_TESTING.md)
- [ADR Index](docs/90_ADR_INDEX.md)

## 7. Programmatic Usage

The CLI is the default entry point, but the orchestration layer can also be used directly:

```python
from agent_manager import AgentManager

manager = AgentManager(
    task="tabular_regression",
    llm="or-glm-5",
    interactive=False,
    data_path="agent_workspace/datasets/CrabAgePrediction.csv",
)

manager.initiate_chat(
    "Predict crab age from both categorical and numerical features in the uploaded Crab Age Dataset"
)
```

## 8. Citation

```bibtex
@inproceedings{AutoML_Agent,
  title={Auto{ML}-Agent: A Multi-Agent {LLM} Framework for Full-Pipeline Auto{ML}},
  author={Trirat, Patara and Jeong, Wonyong and Hwang, Sung Ju},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=p1UBWkOvZm}
}
```

## 9. License

This project is licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license. Commercial use is prohibited.
