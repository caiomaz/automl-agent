# 07. Execution Pipeline And Artifacts

## 1. End-To-End Flow

AutoML-Agent executes work in layered stages rather than going directly from prompt to code.

The practical runtime flow is:

1. parse the user request,
2. validate and summarize the requirements,
3. generate one or more candidate plans,
4. derive data and model solutions from those plans,
5. verify whether a candidate is worth implementing,
6. synthesize a single implementation instruction,
7. generate and run Python code locally,
8. verify the implementation result,
9. revise if necessary.

## 2. Phase 1: Requirement Parsing

The Prompt Agent converts a free-form request into structured JSON. The Agent Manager then checks:

1. whether the request is relevant to machine learning or AI,
2. whether the resulting requirement object is sufficient to proceed.

This is where the selected task type matters most. The task type from the CLI constrains how the downstream request is interpreted.

## 3. Phase 2: Planning

The Agent Manager generates one or more plans. If RAP is enabled, the planning stage can retrieve supporting knowledge before writing the plans.

Each plan is intended to cover the full ML lifecycle when appropriate, including:

1. data handling,
2. preprocessing,
3. model choice,
4. tuning,
5. evaluation,
6. deployment-oriented outputs.

## 4. Phase 3: Specialist Execution Across Plans

For each plan, the repository runs:

1. a Data Agent execution path,
2. a Model Agent execution path.

These plan-level executions are parallelized across the number of plans selected by the user.

The result is a structured candidate solution made from both the data and model perspectives.

## 5. Phase 4: Pre-Execution Verification

Before code is written, the Agent Manager can ask the backbone LLM whether the proposed solution appears to satisfy the user requirements.

This step helps avoid unnecessary implementation attempts when the plan is clearly not aligned.

If no candidate passes this stage and revision budget remains, the Agent Manager revises the plans and tries again.

## 6. Phase 5: Instruction Synthesis

Once a candidate solution is accepted for implementation, the Agent Manager produces a single detailed instruction set for the Operation Agent.

That instruction typically includes:

1. selected model family,
2. data split expectations,
3. preprocessing strategy,
4. tuning targets,
5. evaluation expectations,
6. artifact save locations.

## 7. Phase 6: Operation Agent Execution

The Operation Agent performs the code-first steps of the workflow:

1. merges the selected instruction with task-specific prompt templates,
2. injects environment constraints and installed-package context,
3. writes a Python file under `agent_workspace/exp/`,
4. runs that file locally as a subprocess,
5. captures stdout and stderr,
6. retries when execution fails.

## 8. Phase 7: Post-Execution Verification

After the script finishes, the Agent Manager evaluates the generated code and the observed results against the original requirements.

Possible outcomes:

1. success and end of run,
2. instruction revision followed by another implementation attempt,
3. failure after the revision budget is exhausted.

## 9. Typical Artifact Locations

| Location | Typical Content |
| --- | --- |
| `agent_workspace/datasets/` | uploaded datasets, URL downloads, cached retrievals |
| `agent_workspace/exp/` | generated scripts and experiment outputs |
| `agent_workspace/trained_models/` | serialized pipelines, checkpoints, final artifacts |

## 10. Common Artifact Types

Not every run produces the same files, but successful runs commonly create:

1. a generated Python script,
2. metrics files such as JSON or CSV,
3. artifact-specific outputs such as residuals, SHAP values, confusion matrices, or forecast logs,
4. a serialized `joblib`, `pkl`, or task-specific model file,
5. optional demo or deployment entry points.

## 11. Naming Conventions

Generated script names usually include some combination of:

1. task type,
2. timestamp or unique identifier,
3. backbone model identifier,
4. plan or feature flags such as RAP or verification.

The exact filename is runtime-specific, but the directory location remains stable under `agent_workspace/exp/`.

## 12. Real-World Example Of Retry Behavior

The reference Crab Age run showed two important behaviors:

1. the generated script initially failed because the dataset was not present locally,
2. a later retry fixed a `joblib` serialization call and completed successfully.

This is not an edge case. It is part of the intended execution-grounded design.

## 13. What To Inspect After A Run

Recommended inspection order:

1. open the generated script in `agent_workspace/exp/`,
2. inspect saved metrics and CSV outputs,
3. inspect the serialized model under `agent_workspace/trained_models/`,
4. verify whether the reported deployment surface actually launched.

## 14. Reading Continuation

- Read [08. Task Types And Metrics](08_TASK_TYPES_AND_METRICS.md) to understand how planning targets differ by task.
- Read [11. Troubleshooting](11_TROUBLESHOOTING.md) for failure recovery.
- Read [12. Development And Testing](12_DEVELOPMENT_AND_TESTING.md) if you plan to extend the pipeline.