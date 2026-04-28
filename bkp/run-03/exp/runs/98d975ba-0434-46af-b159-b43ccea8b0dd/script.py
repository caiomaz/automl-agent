
import os
import random
import time
import json
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
import optuna
import joblib
import gradio as gr

# Set all random seeds to 42 for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Define workspace directories
DATASET_PATH = "/home/caio/Projetos/tcc/automl-agent/agent_workspace/datasets"
MODEL_PATH = "/home/caio/Projetos/tcc/automl-agent/agent_workspace/trained_models"
EXP_PATH = "/home/caio/Projetos/tcc/automl-agent/agent_workspace/exp/banana_quality_run_001"

# Create directory structure
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(EXP_PATH, exist_ok=True)

# File paths
BATCHANNA_CSV_PATH = os.path.join(DATASET_PATH, "banana_quality.csv")
MODEL_FILE_PATH = os.path.join(MODEL_PATH, "banana_quality_model.pkl")
PIPELINE_FILE_PATH = os.path.join(MODEL_PATH, "preprocessing_pipeline.pkl")
METRICS_FILE_PATH = os.path.join(EXP_PATH, "metrics.json")
SUMMARY_FILE_PATH = os.path.join(EXP_PATH, "experiment_summary.md")

# Feature columns
FEATURE_COLS = ['size', 'weight', 'sweetness', 'softness', 'harvest_time', 'ripeness', 'acidity']
TARGET_COL = 'quality'

def generate_synthetic_data():
    """Generate synthetic banana quality dataset if file doesn't exist."""
    print("Generating synthetic dataset...")
    np.random.seed(SEED)
    
    n_samples = 1000
    data = {
        'size': np.random.normal(15, 3, n_samples),
        'weight': np.random.normal(200, 40, n_samples),
        'sweetness': np.random.normal(7, 1.5, n_samples),
        'softness': np.random.normal(5, 1.2, n_samples),
        'harvest_time': np.random.normal(30, 5, n_samples),
        'ripeness': np.random.normal(6, 1.5, n_samples),
        'acidity': np.random.normal(4, 0.8, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add Gaussian noise to continuous variables
    for col in FEATURE_COLS:
        df[col] = df[col] + np.random.normal(0, 0.5, n_samples)
    
    # Generate quality labels based on logic
    # 'Good' quality = High weight/sweetness, Low acidity
    # 'Bad' quality = High ripeness/acidity
    good_score = df['weight'] + df['sweetness'] - df['acidity']
    bad_score = df['ripeness'] + df['acidity']
    
    threshold = (good_score - bad_score).median()
    df['quality'] = (good_score - bad_score > threshold).astype(int)
    
    # Save to file
    df.to_csv(BATCHANNA_CSV_PATH, index=False)
    print(f"Synthetic dataset saved to {BATCHANNA_CSV_PATH}")
    
    return df

def load_or_generate_data():
    """Load dataset from file or generate synthetic data if not exists."""
    if os.path.exists(BATCHANNA_CSV_PATH):
        print(f"Loading dataset from {BATCHANNA_CSV_PATH}")
        df = pd.read_csv(BATCHANNA_CSV_PATH)
    else:
        print(f"Dataset not found at {BATCHANNA_CSV_PATH}, generating synthetic data...")
        df = generate_synthetic_data()
    
    # Feature selection: keep only 7 features and target
    df = df[FEATURE_COLS + [TARGET_COL]].copy()
    
    return df

def preprocess_data(df):
    """Preprocess the dataset with imputation, scaling, and splitting."""
    print("Starting data preprocessing...")
    
    # Calculate missing percentage per column
    missing_pct = df.isnull().sum() / len(df) * 100
    print(f"Missing values per column:\n{missing_pct}")
    
    # Handle missing values
    for col in FEATURE_COLS:
        if missing_pct[col] > 5:
            print(f"Dropping rows with missing values in {col} (>5% missing)")
            df = df.dropna(subset=[col])
        elif missing_pct[col] > 0:
            print(f"Imputing missing values in {col} with median")
            df[col].fillna(df[col].median(), inplace=True)
    
    # Separate features and target
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    
    # Split data: 70% Train, 20% Validation, 10% Test
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    
    # Second split: 20% validation, 10% test (from the 30%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.333, stratify=y_temp, random_state=SEED
    )
    
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")
    
    # Feature Scaling: Fit on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Create preprocessing pipeline object
    preprocessing_pipeline = {
        'scaler': scaler,
        'feature_cols': FEATURE_COLS,
        'target_col': TARGET_COL
    }
    
    return (
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        X_test_scaled, y_test,
        preprocessing_pipeline
    )

def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for hyperparameter optimization."""
    # Define hyperparameter search space
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 16, 64),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'class_weight': 'balanced',
        'random_state': SEED,
        'verbose': -1,
        'metric': 'binary_logloss'
    }
    
    # Create LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )
    
    # Predict on validation set
    y_pred = model.predict(X_val)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate F1 score
    f1 = f1_score(y_val, y_pred_binary)
    
    return f1

def train_and_optimize_model(X_train, y_train, X_val, y_val):
    """Train model with Optuna hyperparameter optimization."""
    print("Starting hyperparameter optimization with Optuna...")
    
    # Define study with time and trial limits
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    
    # Run optimization with time limit (3 minutes = 180 seconds)
    start_time = time.time()
    time_limit = 180  # 3 minutes
    
    def time_callback(study, trial):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            raise optuna.exceptions.TrialPruned(f"Time limit exceeded: {elapsed:.2f}s")
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=20,
        timeout=time_limit,
        show_progress_bar=False
    )
    
    print(f"Best F1 Score: {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}")
    
    # Get best parameters
    best_params = study.best_params
    best_params['class_weight'] = 'balanced'
    best_params['random_state'] = SEED
    best_params['verbose'] = -1
    best_params['metric'] = 'binary_logloss'
    
    # Train final model with best parameters on train data
    print("Training final model with best parameters...")
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    final_model = lgb.train(
        best_params,
        train_data,
        num_boost_round=best_params['n_estimators'],
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )
    
    return final_model, best_params

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set and return metrics."""
    print("Evaluating model on test set...")
    
    # Predict
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    metrics = {
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(roc_auc, 4)
    }
    
    print(f"Test Metrics: {metrics}")
    
    return metrics, y_pred, y_pred_proba

def save_outputs(model, preprocessing_pipeline, metrics, best_params):
    """Save model, pipeline, metrics, and experiment summary."""
    print("Saving outputs...")
    
    # Save model
    joblib.dump(model, MODEL_FILE_PATH)
    print(f"Model saved to {MODEL_FILE_PATH}")
    
    # Save preprocessing pipeline
    joblib.dump(preprocessing_pipeline, PIPELINE_FILE_PATH)
    print(f"Preprocessing pipeline saved to {PIPELINE_FILE_PATH}")
    
    # Save metrics
    with open(METRICS_FILE_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_FILE_PATH}")
    
    # Save experiment summary (markdown)
    summary_content = f"""# Banana Quality Classification - Experiment Summary

## Date
{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Information
- **Source**: {'Synthetic' if not os.path.exists(BATCHANNA_CSV_PATH) else 'Local File'}
- **Total Samples**: {len(pd.read_csv(BATCHANNA_CSV_PATH))}
- **Features**: {len(FEATURE_COLS)}
- **Target**: {TARGET_COL}

## Data Split
- **Training**: 70%
- **Validation**: 20%
- **Test**: 10%

## Model Configuration
- **Model Type**: LightGBM Classifier
- **Class Weight**: balanced

## Hyperparameters
- **num_leaves**: {best_params.get('num_leaves', 31)}
- **learning_rate**: {best_params.get('learning_rate', 0.05)}
- **n_estimators**: {best_params.get('n_estimators', 100)}
- **feature_fraction**: {best_params.get('feature_fraction', 0.8)}

## Performance Metrics
| Metric | Value |
|--------|-------|
| Accuracy | {metrics['Accuracy']} |
| Precision | {metrics['Precision']} |
| Recall | {metrics['Recall']} |
| F1-Score | {metrics['F1-Score']} |
| ROC-AUC | {metrics['ROC-AUC']} |

## Execution Details
- **Random Seed**: {SEED}
- **Optuna Trials**: 20
- **HPO Time Limit**: 180 seconds
"""
    
    with open(SUMMARY_FILE_PATH, 'w') as f:
        f.write(summary_content)
    print(f"Experiment summary saved to {SUMMARY_FILE_PATH}")

def create_gradio_app(model, preprocessing_pipeline):
    """Create Gradio web application for model deployment."""
    print("Creating Gradio web application...")
    
    scaler = preprocessing_pipeline['scaler']
    feature_cols = preprocessing_pipeline['feature_cols']
    
    def predict_quality(size, weight, sweetness, softness, harvest_time, ripeness, acidity):
        """Predict banana quality based on input features."""
        try:
            # Create input array
            input_data = np.array([[size, weight, sweetness, softness, harvest_time, ripeness, acidity]])
            
            # Scale input
            input_scaled = scaler.transform(input_data)
            
            # Predict
            proba = model.predict(input_scaled)[0]
            prediction = 1 if proba > 0.5 else 0
            confidence = proba if prediction == 1 else 1 - proba
            
            quality_label = 'Good' if prediction == 1 else 'Bad'
            
            return {
                'Quality': quality_label,
                'Confidence': round(confidence, 4),
                'Probability (Good)': round(proba, 4)
            }
        except Exception as e:
            return {
                'Quality': 'Error',
                'Confidence': 0.0,
                'Probability (Good)': 0.0
            }
    
    # Create Gradio interface
    iface = gr.Interface(
        fn=predict_quality,
        inputs=[
            gr.Number(label="Size (cm)", value=15.0, precision=2),
            gr.Number(label="Weight (g)", value=200.0, precision=2),
            gr.Number(label="Sweetness (scale 1-10)", value=7.0, precision=2),
            gr.Number(label="Softness (scale 1-10)", value=5.0, precision=2),
            gr.Number(label="Harvest Time (days)", value=30.0, precision=2),
            gr.Number(label="Ripeness (scale 1-10)", value=6.0, precision=2),
            gr.Number(label="Acidity (scale 1-10)", value=4.0, precision=2)
        ],
        outputs=[
            gr.Label(label="Quality Prediction"),
            gr.Number(label="Confidence"),
            gr.Number(label="Probability (Good)")
        ],
        title="Banana Quality Classification",
        description="Predict banana quality based on agricultural features. Enter values for each feature to get a quality prediction.",
        examples=[
            [16.0, 220.0, 8.0, 4.0, 28.0, 5.0, 3.0],
            [14.0, 180.0, 6.0, 6.0, 32.0, 7.0, 5.0],
            [15.0, 200.0, 7.0, 5.0, 30.0, 6.0, 4.0]
        ]
    )
    
    return iface

def main():
    """Main function to execute the complete ML pipeline."""
    print("=" * 60)
    print("Banana Quality Classification Pipeline")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Step 1: Load or generate dataset
        print("\n[Step 1] Data Retrieval")
        df = load_or_generate_data()
        print(f"Dataset loaded: {len(df)} samples, {len(df.columns)} columns")
        
        # Step 2: Preprocess data
        print("\n[Step 2] Data Preprocessing")
        (X_train, y_train, X_val, y_val, X_test, y_test, 
         preprocessing_pipeline) = preprocess_data(df)
        
        # Step 3: Train and optimize model
        print("\n[Step 3] Model Training & Hyperparameter Optimization")
        model, best_params = train_and_optimize_model(X_train, y_train, X_val, y_val)
        
        # Step 4: Evaluate model
        print("\n[Step 4] Model Evaluation")
        metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
        
        # Step 5: Save outputs
        print("\n[Step 5] Saving Outputs")
        save_outputs(model, preprocessing_pipeline, metrics, best_params)
        
        # Step 6: Create Gradio deployment
        print("\n[Step 6] Creating Gradio Web Application")
        gradio_app = create_gradio_app(model, preprocessing_pipeline)
        
        # Calculate total execution time
        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"Pipeline completed successfully in {total_time:.2f} seconds")
        print(f"{'=' * 60}")
        
        # Verify time constraint
        if total_time > 270:  # 4.5 minutes
            print("WARNING: Execution time exceeded 4.5 minutes!")
        else:
            print("✓ Execution time within 4.5 minute constraint")
        
        # Verify no image files were created
        image_files = []
        for root, dirs, files in os.walk(EXP_PATH):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
                    image_files.append(file)
        
        if image_files:
            print(f"WARNING: Image files found: {image_files}")
        else:
            print("✓ No image files generated (as required)")
        
        # Return results
        return {
            'processed_data': df,
            'model': model,
            'preprocessing_pipeline': preprocessing_pipeline,
            'metrics': metrics,
            'best_params': best_params,
            'gradio_app': gradio_app
        }
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    results = main()
    
    # Print final summary
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Dataset: {len(results['processed_data'])} samples")
    print(f"Model: LightGBM Classifier")
    print(f"Best Parameters: {results['best_params']}")
    print(f"Test Metrics: {results['metrics']}")
    print(f"Model Saved: {MODEL_FILE_PATH}")
    print(f"Pipeline Saved: {PIPELINE_FILE_PATH}")
    print(f"Metrics Saved: {METRICS_FILE_PATH}")
    print(f"Summary Saved: {SUMMARY_FILE_PATH}")
    print("=" * 60)
    
    # Note: Gradio app can be launched with:
    # results['gradio_app'].launch()
