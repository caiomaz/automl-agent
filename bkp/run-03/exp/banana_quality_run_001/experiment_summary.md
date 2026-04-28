# Banana Quality Classification - Experiment Summary

## Date
2026-04-13 13:19:04

## Dataset Information
- **Source**: Local File
- **Total Samples**: 1000
- **Features**: 7
- **Target**: quality

## Data Split
- **Training**: 70%
- **Validation**: 20%
- **Test**: 10%

## Model Configuration
- **Model Type**: LightGBM Classifier
- **Class Weight**: balanced

## Hyperparameters
- **num_leaves**: 34
- **learning_rate**: 0.09556428757689246
- **n_estimators**: 160
- **feature_fraction**: 0.8394633936788146

## Performance Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.97 |
| Precision | 0.9796 |
| Recall | 0.96 |
| F1-Score | 0.9697 |
| ROC-AUC | 0.998 |

## Execution Details
- **Random Seed**: 42
- **Optuna Trials**: 20
- **HPO Time Limit**: 180 seconds
