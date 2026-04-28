# 05. CLI Reference

## 1. Overview

The CLI is the default user interface for AutoML-Agent.

It supports three entry points:

1. `python -m cli` for the interactive wizard,
2. `python -m cli list-models` to show registered LLM aliases,
3. `python -m cli run ...` for non-interactive execution.

## 2. Interactive Mode

Start it with:

```bash
python -m cli
```

The interactive workflow is organized into six numbered steps.

### 2.1 Step 1: LLMs

The CLI asks for:

1. a backbone LLM,
2. a Prompt Agent LLM.

You can enter either:

- a registered alias such as `or-glm-5`, or
- a raw OpenRouter slug such as `z-ai/glm-5.1`.

### 2.2 Step 2: Task Type

Available task types are:

1. `image_classification`
2. `text_classification`
3. `tabular_classification`
4. `tabular_regression`
5. `tabular_clustering`
6. `node_classification`
7. `ts_forecasting`

### 2.3 Step 3: Dataset

You can choose one of three dataset entry modes:

1. local path or previously discovered dataset,
2. remote URL,
3. blank input, which lets the agents attempt retrieval automatically.

If you choose a directory, the CLI expands it into the underlying file list before passing it into the Agent Manager.

### 2.4 Step 4: Task Description

This is the most important human input. Keep it concise but explicit about:

1. the predictive goal,
2. the data modality,
3. any domain-specific context that matters.

### 2.5 Step 5: Constraints

Optional constraint fields include:

- preferred model or algorithm,
- performance metric and target value,
- maximum training time,
- maximum inference time per sample,
- system information collection.

These constraints are injected into the planning and implementation instructions.

### 2.6 Step 6: Advanced Options

Advanced options include:

1. number of plans,
2. maximum revisions,
3. RAP toggle.

## 3. `list-models` Command

Use:

```bash
python -m cli list-models
```

This prints:

1. all registered aliases,
2. the underlying model slug,
3. a provider hint,
4. the current defaults from `.env`.

## 4. `run` Subcommand

Use:

```bash
python -m cli run [options]
```

This is a real subcommand, not a `--run` flag.

### 4.1 Required Arguments

| Argument | Meaning |
| --- | --- |
| `--task` | Task type |
| `--prompt` | User task description |

### 4.2 Optional Arguments

| Argument | Meaning |
| --- | --- |
| `--llm` | Backbone alias or raw slug |
| `--data` | Dataset path |
| `--n-plans` | Number of plans to generate |
| `--n-revise` | Maximum revision count |
| `--no-rap` | Disable Retrieval-Augmented Planning |
| `--model` | Preferred model or algorithm |
| `--perf-metric` | Performance metric to optimize or target |
| `--perf-value` | Numeric performance target |
| `--max-train-time` | Training time budget |
| `--max-inference-time` | Per-sample inference budget |
| `--system-info` | Explicitly enable system info collection |
| `--no-system-info` | Disable system info collection |
| `--deploy` | Build and launch the Gradio web app at the end (default: on) |
| `--no-deploy` | Skip Gradio deployment — generate modeling pipeline only |
| `--cleanup-mode` | Workspace cleanup policy applied before the run starts: `preserve` (default), `archive` (move prior `runs/<id>` subtrees under `agent_workspace/archive/<timestamp>/`), or `purge` (delete them; the dataset cache is always preserved) |
| `--scheduler-mode` | How plan branches are scheduled: `parallel` (default, thread pool) or `serial` (one branch at a time) |
| `--max-concurrency` | Maximum concurrent branches when running in parallel mode (default: `--n-plans`). Setting `1` forces serial execution. |
| `--seed` | Reproducibility seed propagated to the structured constraints (`constraints.seed` in `run_manifest.json`) |
| `--framework` | Preferred ML framework hint (`sklearn`, `xgboost`, `lightgbm`, `pytorch`, ...) |
| `--split-policy` | Train/val/test split policy: `holdout`, `k-fold`, `stratified-k-fold`, `group-split`, `time-split` |
| `--token-economy` | How aggressively to compact context for token savings: `off` (default), `moderate`, `aggressive` |
| `--hitl-level` | Human-in-the-loop policy: `off` (default, skip every checkpoint), `standard` (enforce only safety-critical checkpoints — destructive cleanup, deploy), `strict` (enforce every known checkpoint) |
| `--critic-policy` | Critic Agent policy: `off` (skip review), `warn` (default, log findings only), `request_hitl` (escalate findings to a human checkpoint), `block` (fail the run on error-severity findings). Reports persisted under `exp/runs/<id>/analyses/critic/` |

After every `run`, the CLI prints a one-screen post-run summary with the `run_id`, final status, recorded event count, and pointers to `cost_summary.json`, `terminal.log`, and the `exp/runs/<id>/` artifact directory.

### 4.3 Listing past runs

```bash
python -m cli list-runs
```

Reads every `run_manifest.json` under `agent_workspace/exp/runs/` and prints `run_id`, `status`, `task_type`, and `started_at` ordered from most recent to oldest. Useful for finding the artifacts of an older run.

### 4.4 Graceful cancellation

The `run` and interactive commands install handlers for `SIGINT` (Ctrl+C) and `SIGTERM`:

1. **First signal** — graceful cancel: the run finishes the current step, the manifest is finalized as `cancelled`, and the post-run summary still prints.
2. **Second signal** — escalation: a `KeyboardInterrupt` is raised immediately as an escape hatch when the graceful path is stuck.

### 4.3 Minimal Example

```bash
python -m cli run \
  --task tabular_regression \
  --prompt "Predict crab age from both categorical and numerical features in the uploaded Crab Age Dataset"
```

### 4.4 Constrained Example

```bash
python -m cli run \
  --task tabular_regression \
  --prompt "Predict crop price from soil composition, environmental factors, and crop management features in the Crop Price Prediction dataset" \
  --llm or-glm-5 \
  --model LightGBM \
  --perf-metric RMSLE \
  --perf-value 0.15 \
  --max-train-time "30 minutes" \
  --n-plans 1 \
  --n-revise 1
```

## 5. Interactive Prompts To Remember

The CLI defaults to collecting system information. This is recommended on most machines because it reduces the chance of the Operation Agent generating code that depends on packages you do not have installed.

If you are working on a smaller local machine, a good safe starting point is:

1. `LightGBM` as the preferred algorithm for tabular work,
2. no explicit performance target on the first run,
3. a modest training budget such as `30 minutes`,
4. `n_plans=1` and `n_revise=1` while validating the workflow.

## 6. Output Behavior

The CLI prints:

1. a request summary from the Prompt Agent,
2. the generated plan or plans,
3. plan-verification status,
4. streamed subprocess output from the generated Python script.

The command itself does not hide retries. If the Operation Agent regenerates code, you will see another execution attempt in the same terminal session.

## 7. Related Reading

- Read [04. Quickstart Tutorial](04_QUICKSTART_TUTORIAL.md) for a fully worked first run.
- Read [09. LLM Configuration](09_LLM_CONFIGURATION.md) for model alias and override details.
- Read [11. Troubleshooting](11_TROUBLESHOOTING.md) for common CLI failures.