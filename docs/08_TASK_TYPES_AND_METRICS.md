# 08. Task Types And Metrics

## 1. Supported Task Types

AutoML-Agent currently supports seven task types.

| Task Type | Problem Family | Primary Metric |
| --- | --- | --- |
| `image_classification` | supervised image classification | accuracy |
| `text_classification` | supervised text classification | accuracy |
| `tabular_classification` | supervised tabular classification | F1 |
| `tabular_regression` | supervised tabular regression | RMSLE |
| `tabular_clustering` | clustering with external evaluation when available | RI |
| `node_classification` | graph node classification | accuracy |
| `ts_forecasting` | time-series forecasting | RMSLE |

## 2. Why Task Type Matters

The selected task type influences:

1. prompt parsing,
2. plan-generation language,
3. primary metric defaults,
4. task-specific template selection,
5. how the Operation Agent frames the final code.

If the task type is wrong, the rest of the pipeline can drift even if the natural-language prompt is good.

## 3. Metric Semantics

### 3.1 Accuracy

Used for image, text, and node classification by default.

### 3.2 F1

Used for tabular classification, where class imbalance can make pure accuracy misleading.

### 3.3 RMSLE

Used for tabular regression and time-series forecasting as the primary planning anchor.

Generated scripts may still report additional metrics such as MAE, RMSE, or R² when those are more interpretable for the chosen implementation.

### 3.4 RI

Short for Rand Index. Used for tabular clustering when an evaluation label or benchmark convention exists.

## 4. Effective Prompting By Task Type

The best prompts usually include:

1. the target variable or forecasting objective,
2. the modality,
3. whether data is uploaded, local, or to be retrieved,
4. any strong operational constraint.

## 5. Example Prompts

### 5.1 Image Classification

```text
Classify butterfly species from the uploaded dataset of butterfly wing images
```

### 5.2 Text Classification

```text
Classify e-commerce product descriptions into Electronics, Household, Books, and Clothing categories
```

### 5.3 Tabular Classification

```text
Classify banana quality as Good or Bad based on size, weight, sweetness, softness, harvest time, ripeness, and acidity
```

### 5.4 Tabular Regression

```text
Predict crop price from soil composition, environmental factors, and crop management features in the Crop Price Prediction dataset
```

### 5.5 Tabular Clustering

```text
Group smoker status into clusters based on numerical health features and compare the clusters against the smoking label
```

### 5.6 Node Classification

```text
Predict paper category for each node in the Cora citation graph dataset
```

### 5.7 Time-Series Forecasting

```text
Forecast future weather observations from past multivariate weather sequences of the same size
```

## 6. Constraint Injection

The CLI lets you optionally attach constraints that are injected into later planning stages.

Useful examples:

1. preferred model or algorithm,
2. metric target,
3. training time budget,
4. inference latency budget.

These constraints help turn a generic prompt into a more operationally precise run.

## 7. Practical Guidance By Task Family

### 7.1 Classification Tasks

Include the label space whenever possible.

### 7.2 Regression Tasks

Include the target variable and the broad feature types.

### 7.3 Forecasting Tasks

Include horizon length, sequence length, and whether the task is univariate or multivariate.

### 7.4 Graph Tasks

Be explicit that the dataset is graph-structured and whether node features and labels are available.

## 8. Common Mistakes

1. using a regression task type for what is actually classification,
2. omitting the target variable from the natural-language prompt,
3. asking for very heavy models on a small local machine without constraints,
4. expecting the primary metric to be the only metric reported in the final script.

## 9. Reading Continuation

- Read [05. CLI Reference](05_CLI_REFERENCE.md) to see where task type is chosen in the wizard.
- Read [09. LLM Configuration](09_LLM_CONFIGURATION.md) if you want to bias the generator toward lighter or stronger models.
- Read [10. Examples And Applicability](10_EXAMPLES_AND_APPLICABILITY.md) for scenario-specific prompt examples.