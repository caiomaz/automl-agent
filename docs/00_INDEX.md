# 00. Documentation Index

## 1. Purpose

This documentation suite is the canonical human-facing reference for AutoML-Agent. It is organized as a numbered manual so readers can move from orientation to execution, then into extension and governance.

Use this index as the default entry point before reading deeper chapters.

## 2. Recommended Reading Paths

### 2.1 New Users

Read in this order:

1. [01. Project Overview](01_PROJECT_OVERVIEW.md)
2. [03. Setup And Environment](03_SETUP_AND_ENVIRONMENT.md)
3. [04. Quickstart Tutorial](04_QUICKSTART_TUTORIAL.md)
4. [05. CLI Reference](05_CLI_REFERENCE.md)
5. [11. Troubleshooting](11_TROUBLESHOOTING.md)

### 2.2 Evaluators And Researchers

Read in this order:

1. [01. Project Overview](01_PROJECT_OVERVIEW.md)
2. [02. Architecture And Agents](02_ARCHITECTURE_AND_AGENTS.md)
3. [07. Execution Pipeline And Artifacts](07_EXECUTION_PIPELINE_AND_ARTIFACTS.md)
4. [08. Task Types And Metrics](08_TASK_TYPES_AND_METRICS.md)
5. [10. Examples And Applicability](10_EXAMPLES_AND_APPLICABILITY.md)
6. [90. ADR Index](90_ADR_INDEX.md)

### 2.3 Contributors And Maintainers

Read in this order:

1. [02. Architecture And Agents](02_ARCHITECTURE_AND_AGENTS.md)
2. [06. Workspace And Datasets](06_WORKSPACE_AND_DATASETS.md)
3. [07. Execution Pipeline And Artifacts](07_EXECUTION_PIPELINE_AND_ARTIFACTS.md)
4. [09. LLM Configuration](09_LLM_CONFIGURATION.md)
5. [12. Development And Testing](12_DEVELOPMENT_AND_TESTING.md)
6. [90. ADR Index](90_ADR_INDEX.md)

## 3. Document Map

| Document | Focus |
| --- | --- |
| [01. Project Overview](01_PROJECT_OVERVIEW.md) | Product scope, positioning, workflows, and non-goals |
| [02. Architecture And Agents](02_ARCHITECTURE_AND_AGENTS.md) | Agent responsibilities, data contracts, orchestration, and state flow |
| [03. Setup And Environment](03_SETUP_AND_ENVIRONMENT.md) | Installation, `.env`, provider keys, and optional local Prompt Agent setup |
| [04. Quickstart Tutorial](04_QUICKSTART_TUTORIAL.md) | End-to-end beginner tutorial based on a real successful CLI run |
| [05. CLI Reference](05_CLI_REFERENCE.md) | Interactive wizard, non-interactive mode, arguments, and examples |
| [06. Workspace And Datasets](06_WORKSPACE_AND_DATASETS.md) | Workspace layout, dataset flows, supported formats, and caching behavior |
| [07. Execution Pipeline And Artifacts](07_EXECUTION_PIPELINE_AND_ARTIFACTS.md) | Planning, verification, retries, generated scripts, and saved outputs |
| [08. Task Types And Metrics](08_TASK_TYPES_AND_METRICS.md) | Supported task families, primary metrics, prompt guidance, and constraints |
| [09. LLM Configuration](09_LLM_CONFIGURATION.md) | Registry semantics, aliases, raw OpenRouter slugs, overrides, and vLLM |
| [10. Examples And Applicability](10_EXAMPLES_AND_APPLICABILITY.md) | Usage scenarios, applicability boundaries, example prompts, and references |
| [11. Troubleshooting](11_TROUBLESHOOTING.md) | Common failure modes, root causes, and recovery actions |
| [12. Development And Testing](12_DEVELOPMENT_AND_TESTING.md) | Repository structure, extension points, testing markers, and contribution rules |
| [90. ADR Index](90_ADR_INDEX.md) | Architectural decision record overview and governance trail |

## 4. Core Terminology

### 4.1 AutoML-Agent

The overall multi-agent system implemented in this repository.

### 4.2 Agent Manager

The orchestration layer that parses requirements, generates plans, coordinates specialized agents, and drives verification and revision.

### 4.3 Prompt Agent

The agent responsible for converting a human task description into structured JSON requirements.

### 4.4 Data Agent

The agent responsible for dataset retrieval, data handling guidance, and data-oriented planning.

### 4.5 Model Agent

The agent responsible for model-family selection, tuning strategy, and model-oriented planning.

### 4.6 Operation Agent

The agent that writes executable Python, runs it locally, captures failures, and retries with improved instructions.

### 4.7 Task Type

The primary machine learning problem class selected in the CLI, such as `tabular_regression` or `image_classification`.

### 4.8 Workspace

The canonical `agent_workspace/` directory tree containing datasets, experiment outputs, and trained models.

### 4.9 RAP

Short for **Retrieval-Augmented Planning**. When enabled, the planning stage may retrieve external examples and knowledge to improve plan quality.

### 4.10 Plan Decomposition

The project’s practice of decomposing or re-understanding plans before execution so later stages receive clearer instructions.

### 4.11 Verification Loop

The pre-execution and post-execution checks used to decide whether to accept, revise, or retry a solution.

## 5. Reference Assets

- [Project README](../README.md)
- [Paper PDF](automl-agent-paper-2410.02958v2.pdf)
- [Poster PDF](../static/pdfs/poster.pdf)
- [Example plans directory](../example_plans/)

## 6. Maintenance Rule

If a code change alters CLI behavior, workspace paths, task types, artifact names, or agent responsibilities, at least one chapter in this manual must be updated in the same change set.