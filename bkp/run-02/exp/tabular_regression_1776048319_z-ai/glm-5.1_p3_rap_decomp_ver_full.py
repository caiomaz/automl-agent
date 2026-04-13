
import os
import sys
import random
import time
import json
import warnings
import traceback
import psutil

import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import shap
import gradio as gr

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATASET_PATH = "/home/caio/Projetos/tcc/automl-agent/agent_workspace/datasets"
TRAINED_MODELS_PATH = "/home/caio/Projetos/tcc/automl-agent/agent_workspace/trained_models"
EXP_PATH = "/home/caio/Projetos/tcc/automl-agent/agent_workspace/exp"

SOURCE_FILE = os.path.join(DATASET_PATH, "varshitanalluri_crop-price-prediction-dataset", "Crop_Yield_Prediction.csv")
RAW_SAVE_PATH = os.path.join(DATASET_PATH, "crop_price_raw.csv")

for path in [
    os.path.join(EXP_PATH, "data_audit"),
    os.path.join(EXP_PATH, "eda", "target_analysis"),
    os.path.join(EXP_PATH, "eda", "univariate"),
    os.path.join(EXP_PATH, "eda", "bivariate"),
    os.path.join(EXP_PATH, "evaluation", "residuals"),
    os.path.join(EXP_PATH, "evaluation", "feature_importance"),
    TRAINED_MODELS_PATH,
]:
    os.makedirs(path, exist_ok=True)


def find_target_column(df):
    for col in df.columns:
        if 'price' in col.lower():
            return col
    for col in df.columns:
        if col.lower() in ['crop_price', 'crop price', 'price', 'target', 'yield']:
            return col
    raise ValueError(f"Could not find target column. Available columns: {df.columns.tolist()}")


def step1_data_ingestion_and_audit():
    print("=" * 60)
    print("STEP 1: Data Ingestion and Audit")
    print("=" * 60)

    df = pd.read_csv(SOURCE_FILE)
    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    df.to_csv(RAW_SAVE_PATH, index=False)
    print(f"Saved raw copy to {RAW_SAVE_PATH}")

    target_col = find_target_column(df)
    print(f"Target column identified: '{target_col}'")

    audit_report = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "data_types": {col: str(df[col].dtype) for col in df.columns},
        "missing_values_per_column": {col: int(df[col].isnull().sum()) for col in df.columns},
        "total_missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_column": target_col,
    }

    audit_path = os.path.join(EXP_PATH, "data_audit", "initial_audit.json")
    with open(audit_path, 'w') as f:
        json.dump(audit_report, f, indent=2)
    print(f"Saved audit report to {audit_path}")

    return df, target_col


def step2_data_preprocessing(df, target_col):
    print("\n" + "=" * 60)
    print("STEP 2: Data Preprocessing")
    print("=" * 60)
    print(f"Initial shape: {df.shape}")

    # Drop columns with >80% missingness
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > 0.8].index.tolist()
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped columns with >80% missing: {cols_to_drop}")

    # Drop rows where target is missing
    before_rows = len(df)
    df = df.dropna(subset=[target_col])
    print(f"Dropped {before_rows - len(df)} rows with missing target")

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_features = [c for c in numeric_cols if c != target_col]
    categorical_features = [c for c in categorical_cols if c != target_col]

    # Impute numeric features with median
    for col in numeric_features:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Impute categorical features with mode or "Missing"
    for col in categorical_features:
        mode_val = df[col].mode()
        fill_val = mode_val[0] if len(mode_val) > 0 else "Missing"
        df[col] = df[col].fillna(fill_val)
    print("Imputed missing values")

    # Drop duplicate rows
    before_rows = len(df)
    df = df.drop_duplicates(keep='first')
    print(f"Dropped {before_rows - len(df)} duplicate rows")

    # Winsorize numeric features (NOT target)
    numeric_features = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != target_col]
    for col in numeric_features:
        try:
            col_vals = df[col].values.copy()
            if len(np.unique(col_vals)) > 2:
                winsorized = winsorize(col_vals, limits=[0.01, 0.01])
                df[col] = np.asarray(winsorized).astype(float)
        except Exception as e:
            print(f"Could not winsorize column {col}: {e}")
    print(f"Winsorized {len(numeric_features)} numeric features at 1st/99th percentiles")

    # Cast categorical columns to category dtype
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    print(f"Cast {len(categorical_cols)} categorical columns to category dtype")

    # Target Analysis - Skewness
    target_skew = float(skew(df[target_col].dropna()))
    log_transformed = False

    skewness_metrics = {
        "original_skewness": target_skew,
        "abs_original_skewness": float(abs(target_skew)),
        "threshold": 0.75,
        "log_transform_applied": False,
        "transformed_skewness": None,
    }

    if abs(target_skew) > 0.75:
        df[target_col] = np.log1p(df[target_col])
        transformed_skew = float(skew(df[target_col].dropna()))
        log_transformed = True
        skewness_metrics["log_transform_applied"] = True
        skewness_metrics["transformed_skewness"] = transformed_skew
        print(f"Applied log1p transformation. Original skewness: {target_skew:.4f}, Transformed: {transformed_skew:.4f}")
    else:
        print(f"No log transformation needed. Skewness: {target_skew:.4f}")

    skewness_path = os.path.join(EXP_PATH, "eda", "target_analysis", "skewness_metrics.json")
    with open(skewness_path, 'w') as f:
        json.dump(skewness_metrics, f, indent=2)
    print(f"Saved skewness metrics to {skewness_path}")
    print(f"Preprocessed shape: {df.shape}")

    return df, log_transformed


def step2_split_data(df, target_col):
    print("\n" + "=" * 60)
    print("STEP 2b: Data Splitting")
    print("=" * 60)

    # Check for temporal features (numeric columns that look temporal)
    temporal_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if col != target_col and df[col].dtype in [np.int64, np.float64, 'int64', 'float64']:
            if any(t in col_lower for t in ['date', 'time', 'year', 'month', 'day']):
                temporal_cols.append(col)

    if temporal_cols:
        print(f"Found temporal columns: {temporal_cols}")
        sort_col = temporal_cols[0]
        df = df.sort_values(by=sort_col).reset_index(drop=True)
        n = len(df)
        train_end = int(0.7 * n)
        val_end = int(0.9 * n)
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
    else:
        print("No temporal features found. Using random split.")
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=42)

    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

    train_df.to_csv(os.path.join(DATASET_PATH, "train.csv"), index=False)
    val_df.to_csv(os.path.join(DATASET_PATH, "val.csv"), index=False)
    test_df.to_csv(os.path.join(DATASET_PATH, "test.csv"), index=False)
    print("Saved train.csv, val.csv, test.csv")

    return train_df, val_df, test_df


def apply_feature_engineering(df, target_col, constant_features=None, id_features=None):
    df = df.copy()

    # Build case-insensitive column mapping
    col_map = {}
    for c in df.columns:
        col_map[c.lower()] = c

    # --- Interactions ---
    for nutrient in ['N', 'P', 'K']:
        n_key = nutrient.lower()
        r_key = 'rainfall'
        if n_key in col_map and r_key in col_map:
            df[f'{nutrient}_rainfall_interaction'] = df[col_map[n_key]] * df[col_map[r_key]]

    # Find temperature column
    temp_key = None
    for t in ['temperature', 'temp']:
        if t in col_map:
            temp_key = t
            break

    ph_key = 'ph'
    if ph_key in col_map and temp_key:
        df['pH_temp_interaction'] = df[col_map[ph_key]] * df[col_map[temp_key]]

    moist_key = 'moisture'
    if moist_key in col_map and temp_key:
        df['moisture_temp_interaction'] = df[col_map[moist_key]] * df[col_map[temp_key]]

    # --- Nutrient Ratios ---
    for num, den, ratio_name in [('N', 'P', 'N_P_ratio'), ('N', 'K', 'N_K_ratio'), ('P', 'K', 'P_K_ratio')]:
        num_key = num.lower()
        den_key = den.lower()
        if num_key in col_map and den_key in col_map:
            df[ratio_name] = df[col_map[num_key]] / (df[col_map[den_key]] + 1e-8)

    if 'n' in col_map and 'p' in col_map and 'k' in col_map:
        df['NPK_sum'] = df[col_map['n']] + df[col_map['p']] + df[col_map['k']]

    # --- Stress Index ---
    if temp_key and 'rainfall' in col_map:
        df['drought_index'] = df[col_map[temp_key]] / (df[col_map['rainfall']] + 1e-8)

    # --- Cyclical Encoding ---
    if 'month' in col_map:
        month_col = col_map['month']
        df['month_sin'] = np.sin(2 * np.pi * df[month_col] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df[month_col] / 12)
        df = df.drop(columns=[month_col])

    if 'day_of_year' in col_map:
        day_col = col_map['day_of_year']
        df['day_sin'] = np.sin(2 * np.pi * df[day_col] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df[day_col] / 365)
        df = df.drop(columns=[day_col])

    # --- Pruning ---
    cols_to_drop = []
    if constant_features:
        cols_to_drop.extend([c for c in constant_features if c in df.columns and c != target_col])
    if id_features:
        cols_to_drop.extend([c for c in id_features if c in df.columns and c != target_col])
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df


def step3_feature_engineering(train_df, val_df, test_df, target_col):
    print("\n" + "=" * 60)
    print("STEP 3: Feature Engineering")
    print("=" * 60)

    # Find constant features from training set
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    constant_features = []
    for col in numeric_cols:
        if col != target_col and train_df[col].std() == 0:
            constant_features.append(col)

    # Find identifier columns
    id_features = []
    for col in train_df.columns:
        if col != target_col and train_df[col].nunique() == len(train_df):
            id_features.append(col)

    print(f"Constant features: {constant_features}")
    print(f"Identifier columns: {id_features}")

    train_fe = apply_feature_engineering(train_df, target_col, constant_features, id_features)
    val_fe = apply_feature_engineering(val_df, target_col, constant_features, id_features)
    test_fe = apply_feature_engineering(test_df, target_col, constant_features, id_features)

    print(f"Train shape after FE: {train_fe.shape}")
    print(f"Val shape after FE: {val_fe.shape}")
    print(f"Test shape after FE: {test_fe.shape}")

    # Univariate statistics
    numeric_cols_fe = train_fe.select_dtypes(include=[np.number]).columns.tolist()
    univariate_stats = train_fe[numeric_cols_fe].describe().to_dict()
    skewness_dict = {}
    for col in numeric_cols_fe:
        skewness_dict[col] = float(skew(train_fe[col].dropna()))
    univariate_stats['skewness'] = skewness_dict

    univariate_path = os.path.join(EXP_PATH, "eda", "univariate", "univariate_stats.json")
    with open(univariate_path, 'w') as f:
        json.dump(univariate_stats, f, indent=2, default=str)
    print(f"Saved univariate stats to {univariate_path}")

    # Bivariate correlations
    corr_matrix = train_fe[numeric_cols_fe].corr().to_dict()
    bivariate_path = os.path.join(EXP_PATH, "eda", "bivariate", "correlation_matrix.json")
    with open(bivariate_path, 'w') as f:
        json.dump(corr_matrix, f, indent=2, default=str)
    print(f"Saved bivariate correlations to {bivariate_path}")

    # Save updated splits
    train_fe.to_csv(os.path.join(DATASET_PATH, "train.csv"), index=False)
    val_fe.to_csv(os.path.join(DATASET_PATH, "val.csv"), index=False)
    test_fe.to_csv(os.path.join(DATASET_PATH, "test.csv"), index=False)

    return train_fe, val_fe, test_fe


def step4_model_training(train_fe, val_fe, target_col):
    print("\n" + "=" * 60)
    print("STEP 4: Model Training and Optimization")
    print("=" * 60)

    training_start_time = time.time()

    X_train = train_fe.drop(columns=[target_col])
    y_train = train_fe[target_col]
    X_val = val_fe.drop(columns=[target_col])
    y_val = val_fe[target_col]

    cat_features = X_train.select_dtypes(include=['category']).columns.tolist()

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features, reference=train_data)

    # Fallback/Target Configuration
    fallback_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 35,
        'lambda_l1': 0.8,
        'lambda_l2': 1.2,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.75,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': SEED,
    }

    print("\nTraining fallback model...")
    fallback_start = time.time()
    fallback_model = lgb.train(
        fallback_params,
        train_data,
        num_boost_round=1500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    fallback_time = time.time() - fallback_start
    fallback_rmse = fallback_model.best_score['valid_0']['rmse']
    print(f"Fallback model - RMSE: {fallback_rmse:.6f}, Training time: {fallback_time:.2f}s")

    # Optuna Tuning
    print("\nStarting Optuna hyperparameter tuning...")
    optuna_start = time.time()

    best_optuna_rmse = float('inf')
    best_optuna_model = None
    optuna_success = False
    best_optuna_params = None

    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'seed': SEED,
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 5.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'max_depth': trial.suggest_int('max_depth', -1, 15),
        }
        n_estimators = trial.suggest_int('n_estimators', 100, 3000)

        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_scores = []

        for train_idx, val_idx in kf.split(X_train):
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_va = X_train.iloc[val_idx]
            y_va = y_train.iloc[val_idx]

            tr_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features)
            va_data = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_features, reference=tr_data)

            model = lgb.train(
                params,
                tr_data,
                num_boost_round=n_estimators,
                valid_sets=[va_data],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )

            y_pred = model.predict(X_va)
            rmse = np.sqrt(mean_squared_error(y_va, y_pred))
            cv_scores.append(rmse)

        return np.mean(cv_scores)

    try:
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=SEED))
        study.optimize(objective, timeout=1200)

        best_optuna_params = study.best_params.copy()
        print(f"Optuna best params: {best_optuna_params}")
        print(f"Optuna best CV RMSE: {study.best_value:.6f}")

        n_estimators_optuna = best_optuna_params.pop('n_estimators', 1500)
        final_optuna_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'seed': SEED,
        }
        final_optuna_params.update(best_optuna_params)

        best_optuna_model = lgb.train(
            final_optuna_params,
            train_data,
            num_boost_round=n_estimators_optuna,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        best_optuna_rmse = best_optuna_model.best_score['valid_0']['rmse']
        optuna_success = True

    except Exception as e:
        print(f"Optuna tuning failed: {e}")
        traceback.print_exc()

    optuna_time = time.time() - optuna_start

    # Select best model
    if optuna_success and best_optuna_rmse < fallback_rmse:
        final_model = best_optuna_model
        final_rmse = best_optuna_rmse
        model_source = "optuna"
        print(f"\nOptuna model selected (RMSE: {best_optuna_rmse:.6f} < Fallback RMSE: {fallback_rmse:.6f})")
    else:
        final_model = fallback_model
        final_rmse = fallback_rmse
        model_source = "fallback"
        if optuna_success:
            print(f"\nFallback model selected (RMSE: {fallback_rmse:.6f} <= Optuna RMSE: {best_optuna_rmse:.6f})")
        else:
            print(f"\nFallback model selected (Optuna failed)")

    total_training_time = time.time() - training_start_time

    # Save model
    model_path = os.path.join(TRAINED_MODELS_PATH, "crop_price_lgbm_model.txt")
    final_model.save_model(model_path)
    print(f"Saved model to {model_path}")

    # Save feature info for deployment
    feature_names = X_train.columns.tolist()
    feature_info = {
        "feature_names": feature_names,
        "categorical_features": cat_features,
        "model_source": model_source,
        "validation_rmse": float(final_rmse),
    }
    feature_info_path = os.path.join(TRAINED_MODELS_PATH, "feature_info.json")
    with open(feature_info_path, 'w') as f:
        json.dump(feature_info, f, indent=2)

    return final_model, X_train, y_train, X_val, y_val, cat_features, total_training_time, model_source


def step5_model_evaluation(final_model, train_fe, val_fe, test_fe, target_col, log_transformed, total_training_time, model_source):
    print("\n" + "=" * 60)
    print("STEP 5: Model Evaluation and Extraction")
    print("=" * 60)

    X_test = test_fe.drop(columns=[target_col])
    y_test = test_fe[target_col]

    y_pred = final_model.predict(X_test)

    # Inverse log1p if applied
    if log_transformed:
        y_test_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred)
    else:
        y_test_orig = y_test.values
        y_pred_orig = y_pred

    # Compute metrics
    rmse = float(np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)))
    mae = float(mean_absolute_error(y_test_orig, y_pred_orig))
    r2 = float(r2_score(y_test_orig, y_pred_orig))

    mask = y_test_orig != 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_test_orig[mask] - y_pred_orig[mask]) / y_test_orig[mask])) * 100)
    else:
        mape = float('inf')

    test_metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape,
        "model_source": model_source,
    }

    metrics_path = os.path.join(EXP_PATH, "evaluation", "test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Test Metrics: RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.6f}, MAPE={mape:.4f}%")
    print(f"Saved test metrics to {metrics_path}")

    # Residuals (in transformed space)
    residuals = y_pred - y_test.values
    residual_stats = {
        "mean": float(np.mean(residuals)),
        "variance": float(np.var(residuals)),
        "skewness": float(skew(residuals)),
    }

    residual_path = os.path.join(EXP_PATH, "evaluation", "residuals", "residual_stats.json")
    with open(residual_path, 'w') as f:
        json.dump(residual_stats, f, indent=2)
    print(f"Residual stats: mean={residual_stats['mean']:.6f}, variance={residual_stats['variance']:.6f}, skewness={residual_stats['skewness']:.6f}")

    # Feature Importance (split-based)
    importance_split = final_model.feature_importance(importance_type='split')
    feature_names = final_model.feature_name()

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_split': importance_split.astype(int),
    }).sort_values('importance_split', ascending=False)

    top20 = importance_df.head(20).to_dict('records')
    importance_path = os.path.join(EXP_PATH, "evaluation", "feature_importance", "top20_split_importance.json")
    with open(importance_path, 'w') as f:
        json.dump(top20, f, indent=2)
    print(f"Saved top 20 feature importance to {importance_path}")

    # SHAP values
    print("Computing SHAP values...")
    X_test_sample = X_test.iloc[:min(500, len(X_test))].copy()

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_mean_abs = np.abs(shap_values).mean(axis=0)
    shap_summary = pd.DataFrame({
        'feature': X_test_sample.columns.tolist(),
        'mean_abs_shap': shap_mean_abs.tolist(),
    }).sort_values('mean_abs_shap', ascending=False).to_dict('records')

    shap_path = os.path.join(EXP_PATH, "evaluation", "feature_importance", "shap_summary.json")
    with open(shap_path, 'w') as f:
        json.dump(shap_summary, f, indent=2, default=str)
    print(f"Saved SHAP summary to {shap_path}")

    shap_values_path = os.path.join(EXP_PATH, "evaluation", "feature_importance", "shap_values.npy")
    np.save(shap_values_path, shap_values)

    # Profiling
    X_single = X_test.iloc[:1]
    latencies = []
    for _ in range(100):
        start_time = time.time()
        final_model.predict(X_single)
        latencies.append((time.time() - start_time) * 1000)

    avg_latency = float(np.mean(latencies))

    process = psutil.Process(os.getpid())
    peak_memory_rss = float(process.memory_info().rss / (1024 * 1024))

    profiling = {
        "total_training_time_seconds": float(total_training_time),
        "peak_memory_rss_mb": peak_memory_rss,
        "avg_inference_latency_ms_per_sample": avg_latency,
        "model_source": model_source,
    }

    profiling_path = os.path.join(EXP_PATH, "evaluation", "model_profiling.json")
    with open(profiling_path, 'w') as f:
        json.dump(profiling, f, indent=2)
    print(f"Model profiling: Training time={total_training_time:.2f}s, Peak RSS={peak_memory_rss:.2f}MB, Avg latency={avg_latency:.4f}ms/sample")

    return test_metrics


def step6_deployment(final_model, train_fe, target_col, log_transformed, cat_features):
    print("\n" + "=" * 60)
    print("STEP 6: Model Deployment with Gradio")
    print("=" * 60)

    feature_info_path = os.path.join(TRAINED_MODELS_PATH, "feature_info.json")
    with open(feature_info_path, 'r') as f:
        feature_info = json.load(f)

    feature_names = feature_info['feature_names']
    cat_feats = feature_info['categorical_features']

    # Get sample values for defaults
    sample_row = train_fe.iloc[0]

    # Create Gradio inputs
    input_components = []
    for feat in feature_names:
        if feat in cat_feats:
            unique_vals = train_fe[feat].unique().tolist()
            unique_vals_str = [str(v) for v in unique_vals]
            input_components.append(gr.Dropdown(choices=unique_vals_str, label=feat, value=unique_vals_str[0]))
        else:
            val = float(sample_row[feat]) if feat in sample_row.index else 0.0
            input_components.append(gr.Number(label=feat, value=round(val, 4)))

    def predict(*inputs):
        input_dict = {}
        for i, feat in enumerate(feature_names):
            input_dict[feat] = inputs[i]

        input_df = pd.DataFrame([input_dict])

        # Cast categorical columns
        for col in cat_feats:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype('category')

        pred = final_model.predict(input_df)[0]

        if log_transformed:
            pred = np.expm1(pred)

        return float(pred)

    demo = gr.Interface(
        fn=predict,
        inputs=input_components,
        outputs=gr.Number(label="Predicted Crop Price"),
        title="Crop Price Prediction",
        description="Enter feature values to predict crop price using a LightGBM model.",
    )

    url_endpoint = demo.launch(share=False, server_name="0.0.0.0", server_port=7860)

    return url_endpoint, demo


def main():
    # Step 1: Data Ingestion and Audit
    df, target_col = step1_data_ingestion_and_audit()

    # Step 2: Data Preprocessing
    df, log_transformed = step2_data_preprocessing(df, target_col)

    # Step 2b: Data Splitting
    train_df, val_df, test_df = step2_split_data(df, target_col)

    # Step 3: Feature Engineering
    train_fe, val_fe, test_fe = step3_feature_engineering(train_df, val_df, test_df, target_col)

    # Step 4: Model Training
    final_model, X_train, y_train, X_val, y_val, cat_features, total_training_time, model_source = step4_model_training(train_fe, val_fe, target_col)

    # Step 5: Model Evaluation
    test_metrics = step5_model_evaluation(final_model, train_fe, val_fe, test_fe, target_col, log_transformed, total_training_time, model_source)

    # Step 6: Deployment
    url_endpoint, demo = step6_deployment(final_model, train_fe, target_col, log_transformed, cat_features)

    return {
        "test_metrics": test_metrics,
        "url_endpoint": url_endpoint,
    }


if __name__ == "__main__":
    results = main()
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Test Metrics: {results['test_metrics']}")
    print(f"Deployment URL: {results['url_endpoint']}")
