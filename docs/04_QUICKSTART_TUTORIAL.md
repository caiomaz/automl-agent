# 04. Quickstart Tutorial

## 1. Goal

This tutorial walks through a successful first run using the interactive CLI. It is based on a real `tabular_regression` session that predicted crab age and produced:

1. a generated experiment script,
2. evaluation metrics,
3. SHAP-based explainability outputs,
4. a serialized deployment pipeline,
5. a local Gradio demo.

## 2. Before You Start

Make sure you have:

1. installed `requirements.txt`,
2. copied `.env.example` to `.env`,
3. set `OPENROUTER_API_KEY`.

Then activate your environment and run:

```bash
python -m cli
```

## 3. Recommended Answers For The First Demo Run

### 3.1 Step 1 Of 6: LLMs

For a first end-to-end run, these values are known to work well:

- Backbone LLM: `z-ai/glm-5.1`
- Prompt Agent LLM: `qwen/qwen3.5-flash-02-23`

You can also choose aliases from the numbered list, but using raw slugs is fully supported.

### 3.2 Step 2 Of 6: Task Type

Choose:

```text
tabular_regression
```

### 3.3 Step 3 Of 6: Dataset

For this walkthrough, leave both prompts blank:

1. local path: press Enter,
2. remote URL: press Enter.

This exercises the repository’s automatic dataset retrieval behavior.

### 3.4 Step 4 Of 6: Task Description

Use the prompt below exactly:

```text
Predict crab age from both categorical and numerical features in the uploaded Crab Age Dataset
```

### 3.5 Step 5 Of 6: Constraints

For the simplest first run:

- Preferred model or algorithm: leave blank
- Performance target: `n`
- Max training time: leave blank
- Max inference time: leave blank
- Include system info: `y`

Keeping system information enabled helps the Operation Agent choose code that matches the current machine.

### 3.6 Step 6 Of 6: Advanced Options

To shorten the first run while still demonstrating the full pipeline:

- Number of plans: `1`
- Max revisions: `1`
- RAP: `n`

These settings mirror a real successful run and keep the tutorial focused.

## 4. What Happens Next

After confirmation, you should see the following sequence:

1. the Prompt Agent summarizes your request,
2. the Agent Manager prints one plan,
3. the Data Agent and Model Agent process that plan,
4. the Agent Manager verifies the candidate solution,
5. the Operation Agent writes and executes a script under `agent_workspace/exp/`.

## 5. Expected Runtime Behavior

### 5.1 Dataset Fallback

In the reference run, the first generated script initially failed because no CSV existed in `agent_workspace/datasets/`. The retry path then downloaded the Crab Age dataset from Kaggle automatically and continued.

This is expected behavior. A first failure does not mean the whole run failed.

### 5.2 Operation Agent Retry

The same reference run also hit a runtime error caused by `joblib.dumps`. The Operation Agent retried, corrected the generated code, and completed successfully.

This demonstrates an important property of the system: generated code is executed and can be revised without restarting the full workflow manually.

## 6. Example Successful Outcome

The reference run produced the following final evaluation snapshot:

```text
Best CV MAE : 1.4585
MAE=1.5049  RMSE=2.1543  R²=0.5800
Model size=1.6663 MB  Inf latency=0.0996 ms/sample
```

It also reported the top SHAP features:

1. `shell_weight`
2. `shucked_weight_ratio`
3. `height`
4. `viscera_weight_ratio`
5. `volume`

Your exact values may differ because generation, search, and model behavior can vary.

## 7. Artifacts You Should Expect

After a successful run, inspect:

```text
agent_workspace/
├── datasets/
├── exp/
└── trained_models/
```

Typical outputs include:

1. a generated Python script in `agent_workspace/exp/`,
2. metrics JSON and CSV outputs,
3. residual and SHAP files,
4. a serialized pipeline in `agent_workspace/trained_models/`.

## 8. Inspecting The Outputs

Useful commands after the run:

```bash
find agent_workspace/exp -maxdepth 2 -type f | sort
find agent_workspace/trained_models -maxdepth 3 -type f | sort
```

If the run launched Gradio successfully, open:

```text
http://localhost:7860
```

## 9. Recommended Next Experiments

After the first run, try one variation at a time:

1. provide a local dataset path instead of auto-retrieval,
2. set `LightGBM` or `XGBoost` as a preferred model,
3. enable RAP,
4. increase `n_plans` to explore more candidate plans,
5. move to non-interactive mode with `python -m cli run ...`.

## 10. Where To Go Next

- Read [05. CLI Reference](05_CLI_REFERENCE.md) for all command options.
- Read [07. Execution Pipeline And Artifacts](07_EXECUTION_PIPELINE_AND_ARTIFACTS.md) to understand the retry and verification loops.
- Read [11. Troubleshooting](11_TROUBLESHOOTING.md) if your first run does not complete.