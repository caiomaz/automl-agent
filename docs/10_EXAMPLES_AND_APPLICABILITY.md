# 10. Examples And Applicability

## 1. Good Fits For AutoML-Agent

AutoML-Agent is a strong fit when you want:

1. a guided AutoML workflow from prompt to local artifacts,
2. flexible dataset entry modes,
3. a local script you can inspect after generation,
4. explainability or deployment artifacts rather than just a score,
5. a research-oriented multi-agent workflow for experimentation.

## 2. Weak Fits For AutoML-Agent

It is a weaker fit when you need:

1. completely deterministic outputs,
2. strictly offline execution with no provider-backed LLMs,
3. guaranteed one-shot code generation,
4. a fully locked production pipeline with no human review of generated code.

## 3. Scenario Catalog

### 3.1 Local Dataset, Interactive Flow

Best for:

1. first-time usage,
2. demonstrations,
3. quick validation on known data.

### 3.2 Remote URL Download

Best for:

1. sharable public datasets,
2. reproducible links,
3. simple data ingress without manual copying.

### 3.3 Automatic Retrieval

Best for:

1. exploratory tasks,
2. public benchmark discovery,
3. rapid experimentation when exact dataset path is not known.

### 3.4 Non-Interactive CLI

Best for:

1. shell automation,
2. repeated experiments,
3. scripted comparison runs.

### 3.5 Programmatic `AgentManager`

Best for:

1. notebooks,
2. experimental integrations,
3. advanced orchestration use.

## 4. Example Prompts By Task

### 4.1 Tabular Regression

```text
Predict crop price from soil composition, environmental factors, and crop management features in the Crop Price Prediction dataset
```

### 4.2 Tabular Classification

```text
Classify banana quality as Good or Bad based on size, weight, sweetness, softness, harvest time, ripeness, and acidity
```

### 4.3 Text Classification

```text
Classify e-commerce product descriptions into Electronics, Household, Books, and Clothing categories
```

### 4.4 Image Classification

```text
Classify butterfly species from the uploaded dataset of butterfly wing images
```

### 4.5 Node Classification

```text
Predict paper category for each node in the Cora citation graph dataset
```

### 4.6 Time-Series Forecasting

```text
Forecast future multivariate weather observations from past sequences of the same length
```

## 5. Reference Datasets Used In The Project

| Task Type | Representative Datasets | Typical Source |
| --- | --- | --- |
| `image_classification` | Butterfly Image, Shopee-IET | Kaggle |
| `text_classification` | Ecommerce Text, Textual Entailment | Kaggle and public benchmarks |
| `tabular_classification` | Banana Quality, Software Defects | Kaggle |
| `tabular_clustering` | Smoker Status, Higher Education Students Performance | Kaggle and UCI |
| `tabular_regression` | Crab Age, Crop Price | Kaggle |
| `node_classification` | Cora, Citeseer | Planetoid |
| `ts_forecasting` | Weather, Electricity | TSLib-related sources |

## 6. Example Plan Files

The `example_plans/` directory contains raw planning artifacts and research-oriented references. These are not polished tutorials, but they are useful for understanding how the planning stage behaves.

| File | What It Is Useful For |
| --- | --- |
| `tabular_plans.md` | tabular planning examples |
| `image_plans.md` | image task planning examples |
| `text_plans.md` | text task planning examples |
| `node_plans.md` | graph task planning examples |
| `ts_plans.md` | forecasting planning examples |
| `plan_knowledge.md` | knowledge used to enrich planning |
| `plan_with_noises.md` | noisy plan examples |
| `plan-decom-examples.md` | plan decomposition examples |
| `prompt_sensitivity.md` | prompt variation studies |
| `scenario-examples.md` | scenario-oriented references |

## 7. Applying The Tool On Different Machines

On smaller local machines, especially CPU-only setups, the safest pattern is:

1. choose lighter backbone and Prompt Agent combinations,
2. keep system information enabled,
3. set moderate constraints such as `LightGBM` and a training-time budget,
4. start with `n_plans=1`.

## 8. Practical Boundaries

Do not treat the repository as a guarantee that every task will be solved with equal quality. Performance depends on:

1. task clarity,
2. dataset quality,
3. provider model choice,
4. local environment constraints,
5. how well the generated script matches the available dependencies.

## 9. Reading Continuation

- Read [04. Quickstart Tutorial](04_QUICKSTART_TUTORIAL.md) for a concrete successful path.
- Read [08. Task Types And Metrics](08_TASK_TYPES_AND_METRICS.md) for prompt and metric guidance.
- Read [11. Troubleshooting](11_TROUBLESHOOTING.md) for scenario-specific failure recovery.