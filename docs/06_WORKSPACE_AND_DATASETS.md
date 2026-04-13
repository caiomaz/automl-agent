# 06. Workspace And Datasets

## 1. Canonical Workspace Structure

AutoML-Agent relies on a fixed workspace layout defined in `utils.workspace`:

```text
agent_workspace/
├── datasets/
├── exp/
└── trained_models/
```

Each directory has a distinct role:

1. `datasets/` stores user uploads, URL downloads, and auto-retrieved datasets,
2. `exp/` stores generated scripts and experiment outputs,
3. `trained_models/` stores serialized pipelines and trained model artifacts.

## 2. Why The Workspace Contract Matters

The Operation Agent is explicitly instructed not to use arbitrary absolute paths such as `/app/`, `/data/`, or `/home/...` inside generated scripts.

Instead, generated code must stay inside the workspace contract above. This makes the repository predictable, easier to inspect, and safer to automate.

## 3. Dataset Entry Modes

AutoML-Agent supports three primary dataset workflows.

### 3.1 Local Path Or Upload

You provide a file or directory under `agent_workspace/datasets/` or anywhere else on disk.

The interactive CLI also discovers previously stored datasets in `agent_workspace/datasets/` and presents them as numbered choices.

### 3.2 Remote URL Download

You provide a direct URL or dataset-like remote path. The repository downloads it into `agent_workspace/datasets/` and extracts archives when possible.

### 3.3 Automatic Retrieval

If you leave the dataset blank, the agents try to resolve a suitable dataset automatically from the user requirements.

## 4. Retrieval Order

When dataset retrieval is needed, the repository can traverse sources in the following order depending on the requirement mode:

1. user upload or existing local file,
2. explicit user link,
3. direct search against Hugging Face,
4. Kaggle,
5. PyTorch dataset hubs,
6. TensorFlow datasets,
7. UCI,
8. OpenML,
9. infer-search backed by web search and document parsing.

## 5. Supported Format Patterns

### 5.1 Image Classification

Recommended structure:

```text
agent_workspace/datasets/butterfly_classification/
├── train/
│   ├── class_a/
│   └── class_b/
└── test/
    ├── class_a/
    └── class_b/
```

Supported image formats include `.jpg`, `.jpeg`, `.png`, `.gif`, and `.webp`.

### 5.2 Text Classification

Recommended structure:

```text
agent_workspace/datasets/ecommerce_text/
└── ecommerce_text.csv
```

CSV files usually contain label and text columns, either named or positional.

### 5.3 Tabular Classification And Regression

Recommended structure:

```text
agent_workspace/datasets/banana_quality/
└── banana_quality.csv
```

CSV and XLSX are the expected tabular formats.

### 5.4 Time Series Forecasting

Recommended structure:

```text
agent_workspace/datasets/weather_forecast/
└── weather.csv
```

Rows normally represent time steps, with a time index or timestamp column plus signal columns.

### 5.5 Node Classification

Recommended structure:

```text
agent_workspace/datasets/cora_nodes/
├── edges.csv
├── node_features.csv
└── node_labels.csv
```

### 5.6 Multimodal Or Paired Data

Recommended structure:

```text
agent_workspace/datasets/caption_data/
├── images/
└── captions.csv
```

## 6. URL Download Behavior

URL-based datasets are cached deterministically. The helper that computes the destination path hashes the URL and creates a stable directory name. This means:

1. the same URL maps to the same local dataset directory,
2. different URLs do not collide,
3. repeated runs can reuse downloaded data.

Archives are automatically extracted when they are valid ZIP or TAR files.

## 7. Kaggle Download Behavior

Kaggle downloads are cached under `agent_workspace/datasets/`. If the dataset already exists locally, the repository reuses the cached copy rather than downloading it again.

To use Kaggle reliably, configure either:

1. `KAGGLE_API_TOKEN` in `.env`, or
2. `~/.kaggle/kaggle.json` with the proper credentials.

## 8. Selecting Datasets In The CLI

When previously downloaded datasets are found, the CLI shows a numbered list. You can respond with:

1. a list number,
2. a raw file system path,
3. blank input to skip local data and move to URL or auto-retrieval mode.

If you enter a directory, the CLI expands it into the underlying files before handing the data path to the Agent Manager.

## 9. Practical Recommendations

1. Use a local path when reproducibility matters most.
2. Use a URL when the source is stable and shareable.
3. Use automatic retrieval when exploring or prototyping.
4. Keep your own datasets grouped by task under `agent_workspace/datasets/`.

## 10. Reading Continuation

- Read [07. Execution Pipeline And Artifacts](07_EXECUTION_PIPELINE_AND_ARTIFACTS.md) for what happens after the dataset is chosen.
- Read [08. Task Types And Metrics](08_TASK_TYPES_AND_METRICS.md) for task-specific expectations.
- Read [11. Troubleshooting](11_TROUBLESHOOTING.md) for dataset-related failures and recovery.