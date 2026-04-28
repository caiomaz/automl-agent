# 11. Troubleshooting

## 1. Startup Failures

### 1.1 `OPENROUTER_API_KEY not set in .env`

**Meaning**: The default workflow cannot start because the required provider key is missing.

**Fix**:

1. copy `.env.example` to `.env`,
2. set `OPENROUTER_API_KEY`,
3. rerun `python -m cli`.

### 1.2 `Invalid choice` During CLI Selection

**Meaning**: The interactive wizard did not recognize the model or task input.

**Fix**:

1. enter a listed number,
2. enter a registered alias,
3. enter a raw OpenRouter slug containing `/`.

## 2. Dataset Failures

### 2.1 `FileNotFoundError: No CSV files found in .../agent_workspace/datasets`

**Meaning**: The generated script expected tabular data locally, but the dataset directory was empty.

**Fix**:

1. provide a local dataset path in the CLI,
2. provide a direct URL,
3. allow the retry loop to attempt automatic retrieval if the task is recoverable.

### 2.2 Kaggle Retrieval Does Not Work

**Meaning**: Kaggle credentials or connectivity are not configured correctly.

**Fix**:

1. set `KAGGLE_API_TOKEN` in `.env`, or
2. create `~/.kaggle/kaggle.json` and secure it with `chmod 600`,
3. retry the run.

### 2.3 Wrong Dataset Is Retrieved Automatically

**Meaning**: The task description was too vague, or the retrieval stage matched a different public dataset.

**Fix**:

1. use a more explicit prompt,
2. provide a dataset URL,
3. provide a local path to remove ambiguity.

## 3. Generated Code Failures

### 3.1 First Script Attempt Fails

**Meaning**: This is normal in some runs. The Operation Agent is designed to capture failures and retry.

**Fix**:

1. wait for the retry path to finish,
2. inspect the next generated script if the failure persists,
3. reduce task ambiguity or set constraints on the next run.

### 3.2 `AttributeError: module 'joblib' has no attribute 'dumps'`

**Meaning**: The generated code used an invalid serialization call.

**Fix**:

1. allow the Operation Agent retry path to correct the implementation,
2. if you are editing manually, replace the invalid call with a supported serialization strategy.

### 3.3 Missing Dependency Errors

**Meaning**: The generated script referenced a package not available in the current environment.

**Fix**:

1. keep system-info collection enabled,
2. install the required package if it is genuinely needed,
3. rerun with a lighter or more constrained setup.

## 4. Performance And Runtime Issues

### 4.1 Runs Take Too Long

**Fix**:

1. set `n_plans=1`,
2. reduce revision count,
3. disable RAP,
4. set a training-time limit,
5. use lighter task constraints such as `LightGBM` for tabular work.

### 4.2 The Generated Model Is Too Heavy For The Machine

**Fix**:

1. keep system-info injection enabled,
2. choose lighter backbones for planning,
3. specify a preferred algorithm that matches the machine,
4. add an inference budget.

## 5. Gradio Issues

### 5.1 Gradio Does Not Open Automatically

**Meaning**: The generated script may still be running correctly, but you need to open the local URL manually.

**Fix**:

1. open `http://localhost:7860` in the browser,
2. confirm that the port is free,
3. check terminal output for launch errors.

### 5.2 Port Conflict On 7860

**Fix**:

1. stop the conflicting process,
2. rerun the workflow,
3. adjust the generated script manually if you need a different port.

## 6. Prompt And Task Mismatch

### 6.1 Wrong Task Type Chosen

**Meaning**: The selected task type and the natural-language prompt do not describe the same problem.

**Fix**:

1. rerun the CLI,
2. select the correct task type,
3. make the prompt consistent with that selection.

## 7. When To Escalate To Manual Inspection

Inspect the generated script directly when:

1. retries are exhausted,
2. the wrong dataset is used repeatedly,
3. provider outputs are clearly drifting from the machine constraints,
4. you want to keep a partial success but refine the final code yourself.

## 8. Cancellation And Scheduler

### 8.1 The run does not stop on Ctrl+C

The first Ctrl+C (or `SIGTERM`) flips a graceful-cancel flag: the current step finishes, the manifest is written with `status="cancelled"`, and the post-run summary still prints. If something is wedged inside a worker:

1. press Ctrl+C a **second** time to escalate to a hard `KeyboardInterrupt`.

### 8.2 `scheduler_fallback_serial` event in `events.jsonl`

`BranchScheduler` raised `SchedulerFallback` from a worker (typical causes: rate limit, provider error). It re-runs the remaining branches one at a time. The run does not fail because of this event by itself; check the surrounding `agent_finished` / `llm_call_completed` records to see what triggered the fallback.

### 8.3 Forcing serial execution

Use `--scheduler-mode serial` or `--max-concurrency 1` on `python -m cli run` to avoid the thread pool entirely (helpful when debugging or when a provider rate-limits aggressively).

## 9. HITL And Constraints

### 9.1 The CLI silently auto-approves a checkpoint

The default `--hitl-level off` skips every checkpoint and applies the documented default. Use `--hitl-level standard` to enforce only safety-critical ones (destructive cleanup, deploy) or `--hitl-level strict` to enforce every known checkpoint. Each checkpoint always emits `hitl_requested` + `hitl_resolved` events with matching `hitl_id`, even when skipped, so the audit trail is preserved.

### 9.2 Invalid value for `--split-policy` / `--token-economy` / `--hitl-level`

These flags use `choices=` in argparse, so misspelled values fail at parse time before the run starts. The structured normalizer (`utils.constraints.normalize_constraints`) also raises `ValueError` if the same misconfiguration arrives via the Prompt Agent payload — failing fast is preferred to running with a bogus policy.

### 9.3 `analyses/constraints.json` missing

Persistence runs only when at least one valid constraint was supplied. An empty constraint dict is a no-op. Check the `constraints_recorded` event in `events.jsonl` — if it's absent, no constraints were attached to the run.

## 10. Reading Continuation

- Read [05. CLI Reference](05_CLI_REFERENCE.md) if the problem starts in the wizard.
- Read [06. Workspace And Datasets](06_WORKSPACE_AND_DATASETS.md) for data-ingress issues.
- Read [07. Execution Pipeline And Artifacts](07_EXECUTION_PIPELINE_AND_ARTIFACTS.md) to understand why retries happen.