# 12. Development And Testing

## 1. Repository Layout

The repository is organized around a small set of high-value modules:

| Path | Purpose |
| --- | --- |
| `cli.py` | interactive and non-interactive CLI entry point |
| `agent_manager/` | orchestration, planning, verification, and retries |
| `prompt_agent/` | requirement parsing and schema-driven prompt interpretation |
| `data_agent/` | data retrieval and data-oriented reasoning |
| `model_agent/` | model retrieval and model-oriented reasoning |
| `operation_agent/` | code generation, subprocess execution, and runtime retry loop |
| `prompt_pool/` | task-specific code templates and prompt scaffolding |
| `utils/` | workspace helpers, embeddings, tracing, system info |
| `tests/` | unit and integration tests |

## 2. Development Invariants

### 2.1 Respect The Workspace Contract

All project code and generated code must treat `agent_workspace/` as the canonical local workspace.

### 2.2 Preserve Agent Separation

Do not collapse Prompt Agent, Data Agent, Model Agent, and Operation Agent concerns unless you are deliberately making an architectural change.

### 2.3 Keep The CLI As The Default User Surface

The CLI is the main onboarding path. Changes that affect runtime behavior should be reflected there and documented accordingly.

## 3. Adding New Capabilities

### 3.1 Add A New Task Type

When adding a task type, update at least:

1. `configs.py` task metrics,
2. `prompt_pool/`,
3. CLI task selection,
4. documentation in [08. Task Types And Metrics](08_TASK_TYPES_AND_METRICS.md),
5. README if the public surface changed.

### 3.2 Add A New Retriever

When adding a data or model source:

1. preserve graceful failure behavior,
2. keep return shapes compatible with existing retrieval flows,
3. avoid hard dependency assumptions when a provider library is missing.

### 3.3 Add A New LLM Alias Or Provider

When extending the LLM registry:

1. update `configs.py`,
2. verify the CLI model listing still renders clearly,
3. update `.env.example` or [09. LLM Configuration](09_LLM_CONFIGURATION.md) if the public setup changes.

## 4. Testing Strategy

The repository uses pytest.

### 4.1 Test Markers

`pytest.ini` defines:

1. `unit` for fast tests without external dependencies,
2. `integration` for tests that may touch external APIs.

### 4.2 Running Tests

Run the full suite:

```bash
pytest
```

Run only unit tests:

```bash
pytest -m unit
```

Run only integration tests:

```bash
pytest -m integration
```

### 4.3 External Dependency Discipline

Unit tests should not rely on live provider access. Integration tests should skip cleanly when required environment variables are absent.

## 5. Existing Test Conventions

Important patterns already present in the repository:

1. Kaggle imports are mocked in `tests/conftest.py`,
2. workspace helpers are covered by dedicated tests,
3. the repository distinguishes testable pure helpers from provider-backed integrations.

## 6. Documentation Sync Rules

Update documentation in the same change set whenever you change:

1. CLI commands or prompts,
2. workspace path rules,
3. task types or metrics,
4. LLM configuration behavior,
5. artifact naming or save locations,
6. agent responsibilities or state flow.

## 7. Artifact Policy For Contributors

Generated datasets, experiment outputs, backup runs, and trained models are runtime artifacts. Treat them as ephemeral unless there is an explicit reason to version a curated example.

## 8. Review Checklist For Changes

Before finishing a change, ask:

1. did I preserve the workspace contract,
2. did I update docs if public behavior changed,
3. did I avoid introducing provider assumptions into unit-tested code,
4. did I keep the task-type, metric, and prompt-template story aligned,
5. did I avoid turning generated artifacts into source-of-truth files.

## 9. Reading Continuation

- Read [02. Architecture And Agents](02_ARCHITECTURE_AND_AGENTS.md) before making structural changes.
- Read [07. Execution Pipeline And Artifacts](07_EXECUTION_PIPELINE_AND_ARTIFACTS.md) before changing retries or verification loops.
- Read [90. ADR Index](90_ADR_INDEX.md) before challenging a project-level decision.