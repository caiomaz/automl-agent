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

## 10. Run-Namespaced Workspace (Phase 1)

Starting with Phase 1, each run creates an isolated namespace inside the canonical workspace:

```text
agent_workspace/
├── datasets/
│   ├── cache/              ← stable, shared across runs, keyed by URL hash
│   └── runs/<run_id>/      ← per-run dataset references
├── exp/
│   └── runs/<run_id>/      ← scripts, logs, manifests
└── trained_models/
    └── runs/<run_id>/      ← saved model artifacts
```

Key rules:

1. Each run gets a unique UUIDv4 ``run_id``.
2. The canonical flat directories (``datasets/``, ``exp/``, ``trained_models/``) are preserved for backward compatibility.
3. ``datasets/cache/`` is shared across runs and is never purged by default.
4. Two runs of the same task type never collide — each writes to its own namespace.
5. The cleanup mode (``preserve`` by default) never deletes previous run data.

The path helpers in ``utils.workspace`` produce these paths:

- ``run_datasets_dir(run_id)``
- ``run_exp_dir(run_id)``
- ``run_models_dir(run_id)``
- ``datasets_cache_dir()``
- ``ensure_run_workspace(run_id)``

## 10.1 Cleanup Modes

Before each new run starts, ``prepare_new_run(...)`` invokes ``utils.workspace.cleanup_workspace(mode)`` according to the ``--cleanup-mode`` flag (or interactive answer):

- ``preserve`` (default) — no destructive action; previous run subtrees stay in place.
- ``archive`` — every existing ``runs/<id>`` subtree under ``datasets/``, ``exp/``, and ``trained_models/`` is moved into ``agent_workspace/archive/<UTC-timestamp>/`` for cold storage.
- ``purge`` — those subtrees are deleted with ``shutil.rmtree``.

In every mode, ``datasets/cache/`` is preserved so remote downloads can be reused. The ledger of the new run records both ``run_cleanup_started`` and ``run_cleanup_completed`` events with the chosen mode and the affected ``run_id`` list.

## 10.2 Dataset Provenance

Whenever a dataset is bound to a run, ``utils.provenance.record_provenance(...)`` writes an entry to ``exp/runs/<run_id>/analyses/dataset_provenance.json`` (a JSON list) and emits a ``dataset_recorded`` event. The recorded modes are:

- ``manual-upload`` — a local path was passed via ``--data``.
- ``user-link`` — a URL was provided and downloaded into the workspace.
- ``auto-retrieval`` — fetched automatically by the Data Agent.

Each entry includes the original source, the resolved local path, an optional SHA-256 checksum (``compute_checksum``) and a free-form note. This forms the audit trail required by ADR-007.

## 11. Reading Continuation

- Read [07. Execution Pipeline And Artifacts](07_EXECUTION_PIPELINE_AND_ARTIFACTS.md) for what happens after the dataset is chosen.
- Read [08. Task Types And Metrics](08_TASK_TYPES_AND_METRICS.md) for task-specific expectations.
- Read [11. Troubleshooting](11_TROUBLESHOOTING.md) for dataset-related failures and recovery.