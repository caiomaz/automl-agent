# 01. Project Overview

## 1. Scope

AutoML-Agent is a multi-agent LLM system for building end-to-end machine learning workflows from natural-language requirements. The repository is designed to run locally while still using external LLM providers for reasoning, planning, and code generation.

The current product surface is centered on:

1. a six-step interactive CLI,
2. a non-interactive CLI subcommand for scripted runs,
3. a programmatic `AgentManager` API,
4. a canonical local workspace for datasets, experiment outputs, and trained models.

## 2. Problem The Project Solves

Classical AutoML frameworks usually assume that the dataset is already available, the problem formulation is already fixed, and the output is a model score or a selected estimator. AutoML-Agent extends that workflow by trying to automate the steps around the model as well.

It is designed for users who want help with:

- turning a free-form request into structured ML requirements,
- locating or downloading datasets when none are provided,
- assembling an end-to-end experimental plan,
- generating locally executable code,
- saving deployable artifacts rather than only reporting leaderboard results.

## 3. Core Product Characteristics

### 3.1 Multi-Agent Specialization

The repository separates planning, data work, modeling work, and operational code execution into distinct agents. This is a core architectural decision, not a presentation detail.

### 3.2 Full-Pipeline Orientation

The intended output of a successful run is not just a trained estimator. A successful run can also include:

- downloaded or resolved datasets,
- preprocessing logic,
- hyperparameter search,
- evaluation reports,
- explainability artifacts,
- serialized pipelines,
- a small deployment surface such as Gradio.

### 3.3 Execution-Grounded Generation

Generated code is executed locally. Failures are surfaced back into the orchestration loop and can trigger plan revision or instruction revision.

### 3.4 Environment Awareness

The CLI can collect system information and inject it into the generation process so the Operation Agent favors libraries already installed on the machine.

### 3.5 Workspace Isolation

All file I/O is expected to stay inside the canonical workspace under `agent_workspace/`. This reduces path ambiguity and keeps generated artifacts discoverable.

## 4. Supported User Workflows

| Workflow | Typical Entry Point | When To Use It |
| --- | --- | --- |
| Interactive CLI | `python -m cli` | Human-guided runs, exploration, demos, first-time usage |
| Non-interactive CLI | `python -m cli run ...` | Repeatable scripted runs, automation, shell workflows |
| Programmatic orchestration | `AgentManager(...)` | Notebook usage, integrations, advanced experiments |

## 5. Typical Run Outcomes

When a run succeeds, users usually get some combination of the following:

1. a generated Python script under `agent_workspace/exp/`,
2. structured metrics and analysis artifacts,
3. a serialized model or pipeline under `agent_workspace/trained_models/`,
4. optional local deployment output such as a Gradio app,
5. logs that show the planning, execution, and validation stages.

## 6. Project Boundaries

AutoML-Agent is not currently designed to be:

1. a fully deterministic build system,
2. a replacement for careful dataset governance,
3. a guarantee that every generated script will succeed on the first attempt,
4. a fully offline system when external LLM providers are used.

The repository should be read as a practical multi-agent AutoML framework with strong experimental capabilities, not as a guaranteed one-shot production compiler.

## 7. Differentiators Against Traditional AutoML

### 7.1 Traditional AutoML

Traditional AutoML usually starts from a prepared table or tensor dataset and focuses on estimator selection, feature processing, and hyperparameter search.

### 7.2 AutoML-Agent

AutoML-Agent adds orchestration around the model search itself:

- requirement parsing,
- dataset retrieval options,
- planning with external knowledge,
- code generation as a first-class output,
- execution-time retries,
- artifact management for downstream deployment.

## 8. Relationship To The Rest Of This Manual

- Read [02. Architecture And Agents](02_ARCHITECTURE_AND_AGENTS.md) next if you need to understand how the system is composed.
- Read [03. Setup And Environment](03_SETUP_AND_ENVIRONMENT.md) next if you want to install and run the project.
- Read [04. Quickstart Tutorial](04_QUICKSTART_TUTORIAL.md) next if you want the fastest path to a successful run.