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

### 4.4 Development Method Is TDD And Spec-Driven

Treat **TDD** and **Spec-Driven Development** as the default engineering process for this repository.

For non-trivial work, follow this order:

1. define the expected behavior and acceptance criteria from the prompt, docs, and code contracts,
2. identify the affected modules, invariants, and public surfaces,
3. write or update tests first when practical,
4. implement the minimum code required to satisfy the spec,
5. refactor only after behavior is covered,
6. update the paired documentation in the same change set.

Do not start with exploratory refactors that change behavior before the expected behavior is pinned down.

If the task is ambiguous, first make the implicit spec explicit in code comments, tests, or docs rather than improvising behavior.

### 4.5 SOLID Is Mandatory

Apply SOLID rigorously for all new code and refactors.

1. **SRP**: each module, class, and function should have one clear reason to change.
2. **OCP**: prefer extending via new modules, adapters, policies, or configuration points over editing unrelated logic branches.
3. **LSP**: replacement implementations must honor the same contracts, especially for retrievers, tracing helpers, schedulers, and runtime backends.
4. **ISP**: avoid broad interfaces when narrower task-specific interfaces or helper contracts are sufficient.
5. **DIP**: depend on explicit contracts and injected collaborators rather than hardwiring infrastructure and providers deep into orchestration logic.

When a change would violate SOLID to save time, prefer a small explicit abstraction over a shortcut that will harden into debt.

### 4.6 Avoid Overengineering

Keep solutions clean, minimal, and efficient.

1. implement the smallest design that satisfies the current spec and the known roadmap,
2. do not add speculative abstractions without a concrete use in the current task or documented roadmap,
3. do not introduce new frameworks, background services, or storage layers without explicit justification,
4. prefer simple data contracts and plain Python objects when they are sufficient,
5. add complexity only when required by safety, observability, scalability, or an already-approved future phase in `TASKS.md`.

### 4.7 Code Style, Naming, And Consistency

Keep a coherent coding standard across the repository.

1. preserve existing naming style and file layout unless the task explicitly changes the architecture,
2. use descriptive names that reflect domain intent,
3. avoid synonyms for the same concept across modules,
4. keep public APIs and data contracts stable unless the task explicitly requires a breaking change,
5. prefer small functions with explicit inputs and outputs over long procedures with hidden state,
6. keep side effects near the boundaries of the system,
7. centralize cross-cutting concerns such as tracing, workspace rules, provider setup, and runtime policies.

For new names, prefer terminology already standardized in this file and in the docs.

### 4.8 Base Stack Is Inviolable Unless The User Explicitly Changes It

Treat the approved base stack as fixed unless the user gives an explicit instruction to change it.

For the current and planned roadmap, the default stack is:

1. Python for the agent core,
2. pytest for automated testing,
3. the existing CLI-first interface as the primary surface,
4. the canonical `agent_workspace/` layout,
5. OpenAI-compatible providers through the current LLM registry model,
6. FastAPI for the future minimal backend when that phase is executed,
7. Next.js/React with `shadcn/ui` for the future minimal frontend when that phase is executed,
8. Docker Compose and Traefik for the future container/deploy phases when those phases are executed,
9. LangSmith as the intended long-term external observability layer when the modernization phase is executed.

Do not replace the base stack with alternative frameworks, infrastructure, or patterns without express user approval.

### 4.9 Documentation Consultation Is Required For External Integrations

When changing library integrations, provider behavior, deployment configuration, or framework-specific code, consult the official documentation first whenever possible.

Preferred order:

1. official library or framework docs,
2. official provider docs,
3. repository docs and ADRs,
4. current source code.

When tools are available, prefer fetching documentation from the official URLs or the current library docs source before making changes to integrations such as:

1. LangChain,
2. LangSmith,
3. FastAPI,
4. Next.js,
5. Docker Compose,
6. Traefik,
7. provider SDKs.

Do not rely on stale memory for integration details when the task depends on recent APIs or behavior.

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

### 5.4 Required Test Levels For Meaningful Changes

For significant changes, design a test strategy that covers the relevant layers instead of stopping at a single happy-path test.

Use the following levels when they apply:

1. **unit tests** for pure logic and small contracts,
2. **integration tests** for provider, runtime, and cross-module interactions,
3. **regression tests** for previously observed failures and behavior that must not break,
4. **system tests** for end-to-end workflow slices across agents,
5. **functional tests** for user-facing flows such as CLI or future API behavior,
6. **basic performance tests** for clearly bounded concerns such as scheduling overhead, excessive token growth, or slow path regressions,
7. **smoke tests** for fast confidence that the main runtime surfaces still boot and execute core paths.

Not every task requires all seven levels, but substantial architectural work should explicitly evaluate which of these levels are needed.

### 5.5 TDD Expectations For Test Writing

When the change is more than trivial:

1. write the failing test first when practical,
2. name the test after the required behavior,
3. keep one behavioral reason per test,
4. add regression coverage for every bug fixed,
5. avoid tests that merely restate implementation details without protecting behavior.

### 5.6 Test Quality Rules

All new tests should be:

1. deterministic,
2. isolated,
3. explicit about setup,
4. explicit about assertions,
5. fast by default unless intentionally integration-level,
6. written to reveal contract breakage, not just line execution.

Avoid weak tests that pass regardless of the real outcome.

### 5.7 Safety Net Requirement For Large Refactors

For changes involving architecture, tracing, workspace rules, runtime execution, provider setup, or repository restructuring:

1. create baseline tests before the refactor,
2. keep regression tests for old behavior that must remain valid,
3. add tests for the new behavior introduced,
4. verify that existing public workflows still pass,
5. do not declare the refactor complete without meaningful automated coverage.

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

### 6.6 Testing And Quality Workflow Changes

Update when relevant:

1. `docs/12_DEVELOPMENT_AND_TESTING.md`
2. `README.md` if the public developer workflow changed,
3. `pytest.ini` when markers or execution modes change.

### 6.7 Platform, Packaging, Or Deployment Changes

Update when relevant:

1. `README.md`
2. `docs/03_SETUP_AND_ENVIRONMENT.md`
3. `docs/11_TROUBLESHOOTING.md`
4. `DEPLOY.md` when it exists,
5. any relevant ADR for infrastructure or runtime architecture.

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

### 8.1 Documentation Standard

Documentation should be:

1. explicit about assumptions,
2. explicit about prerequisites,
3. explicit about supported versus unsupported behavior,
4. updated in the same change set as behavior changes,
5. consistent in terminology with code and tests,
6. written as operational guidance, not marketing copy.

For major features, document at least:

1. what changed,
2. why it exists,
3. how to use it,
4. what it produces,
5. how to test it,
6. how it fails and how to recover.

## 9. Artifact Hygiene

If a change introduces new generated output types, update `.gitignore` so those outputs remain ephemeral unless the repository deliberately wants to track a curated sample.

Do not silently normalize generated artifacts into handwritten source files.

## 10. Default Decision Rule

If you are unsure where to document or preserve a behavior:

1. treat the CLI as the public interface,
2. treat `agent_workspace/` as the runtime sandbox,
3. treat `docs/` as the human manual,
4. treat this file as the fast navigation and invariants sheet for agents.

## 11. Commit And Change Management Rules

### 11.1 Semantic Commit Standard

Use a consistent semantic commit style from now on.

Preferred pattern:

```text
<type>(<scope>): <summary>
```

Allowed types:

1. `feat`
2. `fix`
3. `refactor`
4. `test`
5. `docs`
6. `build`
7. `ci`
8. `perf`
9. `chore`

Recommended scopes for this repo:

1. `cli`
2. `manager`
3. `prompt-agent`
4. `data-agent`
5. `model-agent`
6. `operation-agent`
7. `workspace`
8. `tracing`
9. `tests`
10. `docs`
11. `infra`
12. `backend`
13. `frontend`

Examples:

1. `feat(tracing): add run and handoff event ledger`
2. `fix(cli): preserve graceful cancel as cancelled status`
3. `refactor(workspace): isolate per-run experiment outputs`
4. `test(manager): cover scheduler fallback to serial queue`

If a change is large, prefer multiple coherent commits over one mixed commit.

### 11.2 Change Scope Discipline

Each change set should be internally coherent.

1. do not mix unrelated refactors with behavior changes,
2. do not hide architectural changes inside formatting-only or cleanup commits,
3. keep tests and docs close to the code they validate,
4. preserve backward compatibility unless the task explicitly authorizes a breaking change.

## 12. Future Roadmap Alignment

When implementing tasks from `TASKS.md`, prioritize:

1. preserving the CLI-first contract until a new surface is explicitly made official,
2. keeping agent responsibilities separate,
3. making tracing, tests, and docs land with the code,
4. preventing accidental stack drift,
5. avoiding architectural shortcuts that would block the planned `agent/backend/frontend` split,
6. keeping runtime behavior observable and auditable.

If a proposed implementation conflicts with `TASKS.md`, this file, or the existing docs, resolve the conflict explicitly instead of silently choosing one interpretation.