# Dataset Organization Guide

## Overview

The AutoML-Agent supports three primary data workflows:

1. **Manual Upload** — You provide local datasets (files/folders)
2. **Auto-Retrieval** — Agent searches online hubs (HuggingFace, Kaggle, UCI, OpenML, etc.)
3. **Direct Links** — You provide URLs for the agent to download

---

## 1. Manual Dataset Setup

### Location

Place your datasets in:
```
agent_workspace/datasets/
```

When you run the CLI, the agent automatically discovers all folders and files here and presents them as options.

### Supported Formats by Task Type

#### **Image Classification**
- **Structure**: Any folder containing image subfolders for each class or split

**Example 1: Class-based folders (for train/test split detection)**
```
agent_workspace/datasets/butterfly_classification/
├── train/
│   ├── monarch/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── swallowtail/
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
└── test/
    ├── monarch/
    │   └── ...
    └── swallowtail/
        └── ...
```

**Example 2: Direct class folders (no train/test split)**
```
agent_workspace/datasets/fruit_classifier/
├── apple/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── banana/
│   ├── img1.jpg
│   └── ...
└── orange/
    ├── img1.jpg
    └── ...
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`

#### **Text Classification**
- **Format**: CSV files with text and labels
- **Structure**: 
  - One or two columns (titleless) or named columns
  - First column: class label
  - Second column: text to classify

**Example: Single CSV file**
```
agent_workspace/datasets/ecommerce_text/
└── ecommerce_text.csv
```

Contents (titleless, comma-separated):
```
1,"This product is amazing"
0,"Terrible quality"
1,"Highly recommend"
```

**Example: Named columns**
```
agent_workspace/datasets/sentiment_analysis/
└── reviews.csv
```

Contents (with headers):
```
label,text
1,"Great service"
0,"Poor experience"
1,"Five stars"
```

**Example: Train/test split**
```
agent_workspace/datasets/news_classifier/
├── train.csv
└── test.csv
```

#### **Tabular Classification/Regression**
- **Format**: CSV or XLSX files with structured data
- **Structure**: Rows = instances, columns = features + target

**Example: Single file**
```
agent_workspace/datasets/banana_quality/
└── banana_quality.csv
```

Contents:
```
Size,Weight,Sweetness,Softness,HarvestTime,Ripeness,Acidity,Quality
5,150,8,4,95,6,0.5,Good
4,120,7,3,80,5,0.4,Bad
6,180,9,5,100,7,0.6,Good
```

**Example: Train/test split**
```
agent_workspace/datasets/diabetes_prediction/
├── train.csv
└── test.csv
```

#### **Time Series Forecasting**
- **Format**: CSV with time-indexed numerical data
- **Structure**: Rows = time steps, last column = target variable

**Example:**
```
agent_workspace/datasets/weather_forecast/
└── weather.csv
```

Contents:
```
Date,Temperature,Humidity,Pressure,OT
2023-01-01,5.5,72,1013.2,6.1
2023-01-02,6.2,70,1012.8,7.0
2023-01-03,5.8,75,1013.5,6.5
```

#### **Node Classification (Graphs)**
- **Format**: Graph structure files (edge lists, adjacency matrices, or GML/GraphML)

**Example: Edge list**
```
agent_workspace/datasets/cora_nodes/
├── edges.csv
├── node_features.csv
└── node_labels.csv
```

#### **Text-to-Image / Multimodal**
- **Format**: Paired data (images + text descriptions)

**Example:**
```
agent_workspace/datasets/caption_data/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── captions.csv
```

`captions.csv`:
```
image_id,caption
img1,"A red apple on a table"
img2,"Banana bunch in a bowl"
```

---

## 2. Selecting Datasets in the CLI

When you run the agent:

```bash
python -m cli
```

**Step 4: Data Selection**

```
Discovered datasets:
  1. agent_workspace/datasets/banana_quality
  2. agent_workspace/datasets/weather_forecast

Data path (number, path, or press Enter to skip): 1
```

### Options:

1. **Enter a number** → Selects from discovered datasets
2. **Enter a path** → Provide custom path (absolute or relative)
3. **Press Enter** → Prompts for an optional dataset URL

**If you skip the path**, the CLI asks for an optional URL:

```
Dataset URL (Kaggle, HuggingFace, direct link — or Enter to skip): https://example.com/data.zip
  → URL: https://example.com/data.zip
  The dataset will be downloaded to agent_workspace/datasets/ before launch.
```

If you also skip the URL, the agent will attempt auto-retrieval from online hubs.

---

## 3. Auto-Retrieval Workflow

When **no local dataset** is provided, the agent searches online in this order:

1. **HuggingFace Hub** — Popular NLP & CV datasets
2. **Kaggle** — Requires API credentials (`~/.kaggle/kaggle.json`)
3. **PyTorch Datasets** — torchvision, torchtext, torchdata, etc.
4. **TensorFlow Datasets** — tensorflow-datasets catalog
5. **UCI ML Repository** — Classic ML datasets
6. **OpenML** — Benchmark datasets
7. **Infer Search** — Web search + PDF mining (if RAP enabled)

### Configuration

**Make Kaggle available:**
1. Create account at kaggle.com
2. Go to **Account → API → Create New Token**
3. Drop `kaggle.json` in `~/.kaggle/`
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

**Enable RAP (Retrieval-Augmented Planning):**
In the CLI, when asked "RAP — retrieve external knowledge? (y/n)", answer `y`. This enables:
- Web search for datasets
- arXiv PDF mining
- Additional knowledge retrieval

---

## 4. User Requirements JSON Structure

When the agent needs a dataset, it checks the `user_requirements` JSON for dataset specifications:

```json
{
  "dataset": [
    {
      "name": "banana_quality",
      "source": "user-upload",
      "description": "Banana quality classification dataset"
    }
  ],
  "problem": {
    "downstream_task": "tabular_classification"
  }
}
```

### Dataset Source Types:

| Source | Behavior | Location |
|--------|----------|----------|
| `"user-upload"` | Uses `data_path` from CLI | `agent_workspace/datasets/` or custom path |
| `"user-link"` | Downloads from URL | `agent_workspace/datasets/<dataset_name>/` |
| `"direct-search"` | Auto-retrieves from hubs | HuggingFace → Kaggle → PyTorch → ... → UCI → OpenML |
| `"infer-search"` | Web search + PDF mining | Web/arXiv |

---

## 5. Data Flow Diagram

```
┌─────────────────────┐
│  Start Agent CLI    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ Discover agent_workspace/datasets/ │
│ Show options to user                │
└──────────┬──────────────────────────┘
           │
      ┌────┴────┐
      ▼         ▼
   (User Pick) (Skip/Auto-Retrieve)
      │             │
      ▼             ▼
┌─────────────┐  ┌──────────────────────┐
│ data_path   │  │ Call retrieve_datasets│
│ (local)     │  │ Search online hubs    │
└──────┬──────┘  └──────────┬───────────┘
       │                   │
       └───────┬───────────┘
               ▼
       ┌──────────────────┐
       │ DataAgent loads  │
       │ executes plan on │
       │ retrieved data   │
       └──────────────────┘
```

---

## 6. Data Persistence

- **Manual uploads**: Remain in `agent_workspace/datasets/` for future runs
- **URL downloads**: Downloaded and extracted to `agent_workspace/datasets/<dataset_name>/` — persisted for reuse
- **Auto-retrieved datasets**: Stored temporarily, typically cleaned up after execution
- **Generated code downloads**: The OperationAgent instructs generated code to save datasets to `agent_workspace/datasets/`
- **Generated outputs**: Saved in:
  - `agent_workspace/exp/` — Experiment results
  - `agent_workspace/trained_models/` — Trained models

---

## 7. Example: Complete Manual Setup

### Scenario: Tabular Regression for Crop Yield

**1. Prepare your data:**

```csv
Nitrogen,Phosphorus,Potassium,Temperature,Humidity,pH_Value,Rainfall,Crop,Yield
50,30,20,25.5,65,7.2,150,wheat,4500
60,35,25,24.0,68,7.0,160,wheat,4800
```

**2. Create folder structure:**

```bash
mkdir -p agent_workspace/datasets/crop_yield
```

**3. Copy dataset:**

```bash
# Single file
cp crop_data.csv agent_workspace/datasets/crop_yield/

# Or with train/test split
cp crop_train.csv agent_workspace/datasets/crop_yield/train.csv
cp crop_test.csv agent_workspace/datasets/crop_yield/test.csv
```

**4. Run agent:**

```bash
python -m cli
# Select from discovered: crop_yield
# Task: "Predict crop yield based on soil and climate parameters"
```

---

## 8. Troubleshooting

| Issue | Solution |
|-------|----------|
| Dataset not discovered | Check it's in `agent_workspace/datasets/`, not in parent directory |
| "File not found" in auto-retrieval | Ensure Kaggle API token is set up if searching Kaggle |
| Mixed data types in CSV | Ensure consistent formatting; agent will preprocess automatically |
| Image classes not recognized | Use folder names as class labels; avoid special characters |
| Memory error on large dataset | Consider splitting into train/test subsets |

---

## 9. Adding Your Own Dataset

```bash
# 1. Create dataset directory
mkdir -p agent_workspace/datasets/my_dataset

# 2. Add files
cp my_data.csv agent_workspace/datasets/my_dataset/

# 3. (Optional) Add metadata JSON
cat > agent_workspace/datasets/my_dataset/metadata.json << 'EOF'
{
  "name": "My Dataset",
  "task": "tabular_classification",
  "features": 10,
  "instances": 1000
}
EOF

# 4. Run agent
python -m cli
```

---

## 10. Advanced: Custom Data Loaders

If your dataset format isn't recognized, the agent can generate custom loading code based on your description. When prompted for task description, provide:

```
"Load dataset from images in subfolders, with folder names as class labels.
Train/test split: 80/20. Images are 224x224 JPEG format."
```

The agent will generate appropriate PyTorch/TensorFlow data loading code.
