# AutoML-Agent Copilot Instructions

## 1. Project Identity

AutoML-Agent is a **CLI-first, multi-agent AutoML framework** that turns natural-language requirements into locally executed machine learning workflows.

The public workflow is centered on:

1. `python -m cli`
2. `python -m cli list-models`
3. `python -m cli run ...`

Do not document or reintroduce `python -m cli --run` as the primary interface. The current non-interactive interface is the `run` subcommand.

## 2. Fast Navigation Map

Use this map before making changes.

| Area | Files To Read First |
| --- | --- |
| CLI behavior | `cli.py`, `__main__.py`, `README.md`, `docs/04_QUICKSTART_TUTORIAL.md`, `docs/05_CLI_REFERENCE.md` |
| Agent orchestration | `agent_manager/__init__.py`, `docs/02_ARCHITECTURE_AND_AGENTS.md`, `docs/07_EXECUTION_PIPELINE_AND_ARTIFACTS.md` |
| Prompt parsing | `prompt_agent/__init__.py`, `prompt_agent/schema.json`, `docs/02_ARCHITECTURE_AND_AGENTS.md` |
| Dataset retrieval and downloads | `data_agent/retriever.py`, `utils/workspace.py`, `docs/06_WORKSPACE_AND_DATASETS.md`, `docs/11_TROUBLESHOOTING.md` |
| Model retrieval | `model_agent/retriever.py`, `docs/02_ARCHITECTURE_AND_AGENTS.md` |
| Code generation and retries | `operation_agent/__init__.py`, `operation_agent/execution.py`, `docs/07_EXECUTION_PIPELINE_AND_ARTIFACTS.md` |
| LLM registry and providers | `configs.py`, `.env.example`, `docs/03_SETUP_AND_ENVIRONMENT.md`, `docs/09_LLM_CONFIGURATION.md` |
| Task types and metrics | `configs.py`, `prompt_pool/`, `docs/08_TASK_TYPES_AND_METRICS.md` |
| Testing | `pytest.ini`, `tests/`, `docs/12_DEVELOPMENT_AND_TESTING.md` |
| Decision history | `docs/90_ADR_INDEX.md`, `docs/adr/` |

## 3. Non-Negotiable Invariants

### 3.1 Workspace Contract

All handwritten code and generated code must respect the canonical workspace in `utils.workspace`:

1. `agent_workspace/datasets/`
2. `agent_workspace/exp/`
3. `agent_workspace/trained_models/`

Do not hardcode replacement paths like `/app/`, `/data/`, `/tmp/project`, or ad hoc absolute directories.

### 3.2 Agent Separation

Keep the distinction between:

1. Prompt Agent,
2. Agent Manager,
3. Data Agent,
4. Model Agent,
5. Operation Agent.

Do not merge responsibilities casually. If you intentionally change role boundaries, update the architecture docs and ADRs.

### 3.3 LLM Role Split

The backbone LLM and the Prompt Agent LLM are separate concepts.

- `LLM_BACKBONE` drives planning, verification, and code generation.
- `LLM_PROMPT_AGENT` drives requirement parsing.
- `OPENROUTER_MODEL` is a workflow-wide override path.

Preserve alias-vs-raw-slug behavior in `configs.py`.

### 3.4 Generated Artifacts Are Not Source Of Truth

Outputs under `agent_workspace/` and `bkp/` are runtime artifacts. Do not make new features depend on committed generated outputs. Keep source-of-truth logic in code, tests, docs, and prompt templates.

## 4. Coding Patterns To Preserve

### 4.1 Retriever Design

Retriever functions should:

1. accept structured inputs or `**kwargs` that come from parsed requirements,
2. degrade gracefully when optional provider libraries are unavailable,
3. prefer cache reuse when local data already exists,
4. return structured metadata that downstream agents can use.

### 4.2 Operation Agent Behavior

Preserve the current execution-grounded model:

1. synthesize code from instructions,
2. inject workspace constraints and system info,
3. run code locally,
4. capture stderr and stdout,
5. retry on failure.

Do not bypass the retry loop without a strong reason.

### 4.3 Task Template Coupling

Task behavior is split across:

1. `configs.py` metric mapping,
2. CLI task selection,
3. `prompt_pool/` task templates,
4. documentation in `docs/08_TASK_TYPES_AND_METRICS.md`.

If one changes, audit the others.

## 5. Testing Rules

### 5.1 Unit Tests

Unit tests must not depend on live external services.

### 5.2 Integration Tests

Integration tests may use live services, but they must skip cleanly when required keys are missing.

### 5.3 Existing Patterns

Preserve these existing conventions:

1. `pytest.ini` markers `unit` and `integration`,
2. Kaggle import mocking in `tests/conftest.py`,
3. workspace helper coverage in `tests/test_workspace.py`.

## 6. Documentation Sync Rules

When you change any of the following, update the paired documentation in the same change set.

### 6.1 CLI Or User Workflow Changes

Update:

1. `README.md`
2. `docs/04_QUICKSTART_TUTORIAL.md`
3. `docs/05_CLI_REFERENCE.md`
4. `docs/11_TROUBLESHOOTING.md` if failure modes change

### 6.2 Workspace Or Artifact Changes

Update:

1. `docs/06_WORKSPACE_AND_DATASETS.md`
2. `docs/07_EXECUTION_PIPELINE_AND_ARTIFACTS.md`
3. `.gitignore` if artifact types change

### 6.3 LLM Registry Or Environment Changes

Update:

1. `.env.example`
2. `README.md`
3. `docs/03_SETUP_AND_ENVIRONMENT.md`
4. `docs/09_LLM_CONFIGURATION.md`

### 6.4 Task Type Or Metric Changes

Update:

1. `configs.py`
2. `prompt_pool/`
3. `README.md`
4. `docs/08_TASK_TYPES_AND_METRICS.md`

### 6.5 Agent Or Architecture Changes

Update:

1. `docs/02_ARCHITECTURE_AND_AGENTS.md`
2. `docs/07_EXECUTION_PIPELINE_AND_ARTIFACTS.md`
3. the relevant ADR in `docs/adr/` or add a new ADR

## 7. Preferred Terminology

Use these terms consistently in code comments, docs, and generated repository guidance:

1. Agent Manager
2. Prompt Agent
3. Data Agent
4. Model Agent
5. Operation Agent
6. Task Type
7. Workspace
8. Experiment Outputs
9. Trained Models
10. Retrieval-Augmented Planning (RAP)
11. Plan Decomposition
12. Verification Loop

Prefer `Task Type` in human-facing docs when possible. Use `downstream_task` only when referring to the specific JSON field or internal code contract.

## 8. Documentation Style Rules For This Repo

1. Human-facing docs in `docs/` are in English.
2. `docs/` uses numbered filenames.
3. `docs/00_INDEX.md` is the entry point.
4. ADRs live under `docs/adr/` and are indexed from `docs/90_ADR_INDEX.md`.
5. Keep README concise and route deeper details into `docs/`.

## 9. Artifact Hygiene

If a change introduces new generated output types, update `.gitignore` so those outputs remain ephemeral unless the repository deliberately wants to track a curated sample.

Do not silently normalize generated artifacts into handwritten source files.

## 10. Default Decision Rule

If you are unsure where to document or preserve a behavior:

1. treat the CLI as the public interface,
2. treat `agent_workspace/` as the runtime sandbox,
3. treat `docs/` as the human manual,
4. treat this file as the fast navigation and invariants sheet for agents.