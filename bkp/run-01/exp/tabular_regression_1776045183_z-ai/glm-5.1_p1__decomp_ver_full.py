
# ─── Crab Age Prediction Pipeline — LightGBM + Optuna ───
import os, random, time, json, warnings, glob, zipfile, io
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import joblib
import shap
import gradio as gr

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATASET_PATH  = "/home/caio/Projetos/tcc/automl-agent/agent_workspace/datasets"
MODEL_PATH    = "/home/caio/Projetos/tcc/automl-agent/agent_workspace/trained_models/crab_age_regression"
EXP_PATH      = "/home/caio/Projetos/tcc/automl-agent/agent_workspace/exp"

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(EXP_PATH, exist_ok=True)

CATEGORICAL_FEATURES = ['sex']
NUMERICAL_FEATURES   = ['length', 'diameter', 'height', 'weight',
                        'shucked_weight', 'viscera_weight', 'shell_weight']
TARGET = 'age'
ENGINEERED_FEATURES = ['volume', 'surface_area',
                       'shucked_weight_ratio', 'viscera_weight_ratio',
                       'shell_weight_ratio']
ALL_NUMERICAL = NUMERICAL_FEATURES + ENGINEERED_FEATURES


# ═══════════════════════════════════════════════════════════════
# Custom sklearn transformer
# ═══════════════════════════════════════════════════════════════
class CrabPreprocessor(BaseEstimator, TransformerMixin):
    """Impute → clip outliers (1st/99th pct) → engineer features."""

    def __init__(self):
        self.num_imputer_ = None
        self.cat_imputer_ = None
        self.clip_bounds_ = None

    def fit(self, X, y=None):
        X = X.copy()
        self.num_imputer_ = SimpleImputer(strategy='median')
        self.num_imputer_.fit(X[NUMERICAL_FEATURES])
        self.cat_imputer_ = SimpleImputer(strategy='constant', fill_value='Missing')
        self.cat_imputer_.fit(X[CATEGORICAL_FEATURES])
        X_num = pd.DataFrame(self.num_imputer_.transform(X[NUMERICAL_FEATURES]),
                             columns=NUMERICAL_FEATURES, index=X.index)
        self.clip_bounds_ = {}
        for col in NUMERICAL_FEATURES:
            self.clip_bounds_[col] = (float(X_num[col].quantile(0.01)),
                                      float(X_num[col].quantile(0.99)))
        return self

    def transform(self, X):
        X = X.copy()
        X[NUMERICAL_FEATURES]   = self.num_imputer_.transform(X[NUMERICAL_FEATURES])
        X[CATEGORICAL_FEATURES] = self.cat_imputer_.transform(X[CATEGORICAL_FEATURES])
        for col in NUMERICAL_FEATURES:
            lo, hi = self.clip_bounds_[col]
            X[col] = X[col].clip(lower=lo, upper=hi)
        X['volume']               = X['length'] * X['diameter'] * X['height']
        X['surface_area']         = X['length'] * X['diameter']
        w = X['weight'].replace(0, np.nan)
        X['shucked_weight_ratio'] = (X['shucked_weight'] / w).fillna(0)
        X['viscera_weight_ratio'] = (X['viscera_weight'] / w).fillna(0)
        X['shell_weight_ratio']   = (X['shell_weight']   / w).fillna(0)
        return X


# ═══════════════════════════════════════════════════════════════
# 1. Data Ingestion — load from local CSV
# ═══════════════════════════════════════════════════════════════
def load_data():
    filepath = os.path.join(DATASET_PATH, "CrabAgePrediction.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    print(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Raw shape: {df.shape}  columns: {list(df.columns)}")

    # Standardise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Drop identifier / index columns
    for c in ['id', 'index', 'unnamed:_0']:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Validate
    required = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Available: {list(df.columns)}")

    df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
    for c in NUMERICAL_FEATURES:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    before = len(df)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Dropped {before - len(df)} duplicates → final shape {df.shape}")
    return df


# ═══════════════════════════════════════════════════════════════
# 2. Data Splitting — 70 / 20 / 10 stratified on sex
# ═══════════════════════════════════════════════════════════════
def split_data(df):
    train_df, temp_df = train_test_split(df, test_size=0.30,
                                         random_state=SEED, stratify=df['sex'])
    val_df, test_df   = train_test_split(temp_df, test_size=1/3,
                                         random_state=SEED, stratify=temp_df['sex'])
    for d in (train_df, val_df, test_df):
        d.reset_index(drop=True, inplace=True)
    n = len(df)
    print(f"Train {len(train_df)} ({100*len(train_df)/n:.1f}%)  "
          f"Val {len(val_df)} ({100*len(val_df)/n:.1f}%)  "
          f"Test {len(test_df)} ({100*len(test_df)/n:.1f}%)")
    return train_df, val_df, test_df


# ═══════════════════════════════════════════════════════════════
# 3. Preprocessing & Feature Engineering
# ═══════════════════════════════════════════════════════════════
def preprocess_data(train_df, val_df, test_df):
    preprocessor = CrabPreprocessor()
    preprocessor.fit(train_df)

    train_p = preprocessor.transform(train_df)
    val_p   = preprocessor.transform(val_df)
    test_p  = preprocessor.transform(test_df)

    col_tx = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), ALL_NUMERICAL),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
             CATEGORICAL_FEATURES)
        ], remainder='drop')
    col_tx.fit(train_p)

    X_train = col_tx.transform(train_p)
    X_val   = col_tx.transform(val_p)
    X_test  = col_tx.transform(test_p)

    y_train = train_df[TARGET].values
    y_val   = val_df[TARGET].values
    y_test  = test_df[TARGET].values

    ohe_cols = col_tx.named_transformers_['cat'].get_feature_names_out(
        CATEGORICAL_FEATURES).tolist()
    feature_names = ALL_NUMERICAL + ohe_cols
    print(f"Feature dim = {X_train.shape[1]}  ({len(ALL_NUMERICAL)} num + "
          f"{len(ohe_cols)} cat)")
    return (X_train, X_val, X_test, y_train, y_val, y_test,
            feature_names, preprocessor, col_tx)


# ═══════════════════════════════════════════════════════════════
# 4. Hyperparameter Optimisation — Optuna TPE, 100 trials, 5-fold CV
# ═══════════════════════════════════════════════════════════════
def run_optuna_hpo(X_train, y_train, X_val, y_val):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = dict(
            learning_rate     = trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            num_leaves        = trial.suggest_int('num_leaves', 15, 127),
            n_estimators      = trial.suggest_int('n_estimators', 200, 2000),
            reg_alpha         = trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
            reg_lambda        = trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
            min_child_samples = trial.suggest_int('min_child_samples', 5, 100),
            colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0),
            subsample         = trial.suggest_float('subsample', 0.5, 1.0),
            subsample_freq    = trial.suggest_int('subsample_freq', 1, 7),
            max_depth         = trial.suggest_int('max_depth', -1, 12),
            objective         = 'regression',
            metric            = 'mae',
            verbose           = -1,
            random_state      = SEED,
            n_jobs            = -1,
        )
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        fold_maes = []
        for tr_idx, va_idx in kf.split(X_train):
            m = lgb.LGBMRegressor(**params)
            m.fit(X_train[tr_idx], y_train[tr_idx],
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(0)])
            fold_maes.append(mean_absolute_error(y_train[va_idx],
                                                  m.predict(X_train[va_idx])))
        return np.mean(fold_maes)

    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print(f"\nBest CV MAE : {study.best_value:.4f}")
    print(f"Best params : {json.dumps(study.best_params, indent=2, default=str)}")
    return study


# ═══════════════════════════════════════════════════════════════
# 5. Train final model with best hyperparameters
# ═══════════════════════════════════════════════════════════════
def train_model(X_train, y_train, X_val, y_val, best_params):
    params = dict(
        learning_rate     = best_params['learning_rate'],
        num_leaves        = best_params['num_leaves'],
        n_estimators      = best_params['n_estimators'],
        reg_alpha         = best_params['reg_alpha'],
        reg_lambda        = best_params['reg_lambda'],
        min_child_samples = best_params['min_child_samples'],
        colsample_bytree  = best_params['colsample_bytree'],
        subsample         = best_params['subsample'],
        subsample_freq    = best_params['subsample_freq'],
        max_depth         = best_params['max_depth'],
        objective         = 'regression',
        metric            = 'mae',
        verbose           = -1,
        random_state      = SEED,
        n_jobs            = -1,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(0)])
    best_iter = getattr(model, 'best_iteration_', params['n_estimators'] - 1)
    print(f"Best iteration = {best_iter}  →  trees = {best_iter + 1}")
    return model


# ═══════════════════════════════════════════════════════════════
# 6. Evaluation, SHAP, Residual Analysis
# ═══════════════════════════════════════════════════════════════
def evaluate_model(model, X_test, y_test, feature_names):
    y_pred = model.predict(X_test)

    # ── Performance ──
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    performance = {'MAE': float(mae), 'RMSE': float(rmse), 'R2': float(r2)}
    print(f"MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    # ── SHAP ──
    print("Computing SHAP values …")
    explainer   = shap.TreeExplainer(model)
    X_test_df   = pd.DataFrame(X_test, columns=feature_names)
    shap_values = explainer.shap_values(X_test_df)

    shap_imp = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    print("Top-5 SHAP features:")
    print(shap_imp.head().to_string(index=False))

    pd.DataFrame(shap_values, columns=feature_names).to_csv(
        os.path.join(EXP_PATH, 'shap_values.csv'), index=False)
    shap_imp.to_csv(os.path.join(EXP_PATH, 'shap_importance.csv'), index=False)

    # ── Residual analysis ──
    residuals = y_test - y_pred
    res_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred,
                           'residual': residuals})
    res_df.to_csv(os.path.join(EXP_PATH, 'residuals.csv'), index=False)

    residual_metrics = {
        'mean_residual': float(np.mean(residuals)),
        'std_residual' : float(np.std(residuals)),
        'min_residual' : float(np.min(residuals)),
        'max_residual' : float(np.max(residuals)),
        'residuals_by_age_range': {}
    }
    for label, (lo, hi) in [('young_0-8', (0, 8)),
                             ('middle_8-12', (8, 12)),
                             ('old_12+', (12, 200))]:
        mask = (y_test >= lo) & (y_test < hi)
        if mask.sum() > 0:
            residual_metrics['residuals_by_age_range'][label] = {
                'mean_residual': float(np.mean(residuals[mask])),
                'std_residual' : float(np.std(residuals[mask])),
                'mae'          : float(np.mean(np.abs(residuals[mask]))),
                'count'        : int(mask.sum())
            }
    print(f"Mean residual={residual_metrics['mean_residual']:.4f}  "
          f"Std={residual_metrics['std_residual']:.4f}")

    # ── Complexity (FIXED: use io.BytesIO instead of non-existent joblib.dumps) ──
    buf = io.BytesIO()
    joblib.dump(model, buf)
    model_bytes = buf.tell()
    buf.close()

    n_inf = min(100, len(X_test))
    t0 = time.time()
    _ = model.predict(X_test[:n_inf])
    inf_ms = (time.time() - t0) / n_inf * 1000

    best_iter = getattr(model, 'best_iteration_', None)
    complexity = {
        'model_size_bytes'  : model_bytes,
        'model_size_mb'     : round(model_bytes / (1024**2), 4),
        'inference_time_ms' : round(inf_ms, 4),
        'best_n_estimators' : int(best_iter + 1) if best_iter is not None else int(model.n_estimators),
        'num_leaves'        : int(model.num_leaves),
        'max_depth'         : int(model.max_depth) if model.max_depth is not None else -1,
    }
    print(f"Model size={complexity['model_size_mb']:.4f} MB  "
          f"Inf latency={complexity['inference_time_ms']:.4f} ms/sample")

    return performance, complexity, residual_metrics


# ═══════════════════════════════════════════════════════════════
# 7. Artifact Serialization — full sklearn Pipeline → joblib
# ═══════════════════════════════════════════════════════════════
def prepare_model_for_deployment(train_df, y_train, best_params, final_model):
    best_iter = getattr(final_model, 'best_iteration_', best_params['n_estimators'] - 1)
    deploy_params = dict(
        learning_rate     = best_params['learning_rate'],
        num_leaves        = best_params['num_leaves'],
        n_estimators      = int(best_iter + 1),
        reg_alpha         = best_params['reg_alpha'],
        reg_lambda        = best_params['reg_lambda'],
        min_child_samples = best_params['min_child_samples'],
        colsample_bytree  = best_params['colsample_bytree'],
        subsample         = best_params['subsample'],
        subsample_freq    = best_params['subsample_freq'],
        max_depth         = best_params['max_depth'],
        objective         = 'regression',
        metric            = 'mae',
        verbose           = -1,
        random_state      = SEED,
        n_jobs            = -1,
    )

    pipeline = Pipeline([
        ('preprocessor', CrabPreprocessor()),
        ('encoder_scaler', ColumnTransformer(
            transformers=[
                ('num', RobustScaler(), ALL_NUMERICAL),
                ('cat', OneHotEncoder(sparse_output=False,
                                      handle_unknown='ignore'), CATEGORICAL_FEATURES)
            ], remainder='drop')),
        ('regressor', lgb.LGBMRegressor(**deploy_params))
    ])

    X_train_df = train_df.drop(columns=[TARGET])
    pipeline.fit(X_train_df, y_train)

    pipeline_path = os.path.join(MODEL_PATH, 'crab_age_predictor_pipeline.joblib')
    joblib.dump(pipeline, pipeline_path)
    print(f"Pipeline saved → {pipeline_path}")
    return pipeline


# ═══════════════════════════════════════════════════════════════
# 8. Gradio Web Application
# ═══════════════════════════════════════════════════════════════
def deploy_model(pipeline):
    def predict_age(sex, length, diameter, height, weight,
                    shucked_weight, viscera_weight, shell_weight):
        inp = pd.DataFrame({
            'sex': [sex],
            'length': [float(length)],
            'diameter': [float(diameter)],
            'height': [float(height)],
            'weight': [float(weight)],
            'shucked_weight': [float(shucked_weight)],
            'viscera_weight': [float(viscera_weight)],
            'shell_weight': [float(shell_weight)]
        })
        pred = pipeline.predict(inp)[0]
        return round(float(pred), 2)

    demo = gr.Interface(
        fn=predict_age,
        inputs=[
            gr.Dropdown(choices=['F', 'M', 'I'],
                        label='Sex (F=Female, M=Male, I=Infant)'),
            gr.Number(label='Length', value=1.0),
            gr.Number(label='Diameter', value=0.8),
            gr.Number(label='Height', value=0.3),
            gr.Number(label='Weight', value=10.0),
            gr.Number(label='Shucked Weight', value=4.0),
            gr.Number(label='Viscera Weight', value=2.0),
            gr.Number(label='Shell Weight', value=3.0),
        ],
        outputs=gr.Number(label='Predicted Age (months)'),
        title='🦀 Crab Age Predictor',
        description='Enter physical measurements of a crab to predict its age '
                    '(in months). Powered by LightGBM + Optuna-optimised pipeline.',
        examples=[
            ['M', 1.4, 1.1, 0.4, 25.0, 11.0, 5.0, 7.0],
            ['F', 1.2, 0.9, 0.3, 15.0, 6.5, 3.0, 4.5],
            ['I', 0.8, 0.6, 0.2, 5.0, 2.0, 1.0, 1.5],
        ],
    )

    url_endpoint = demo.launch(server_name="0.0.0.0", server_port=7860,
                               share=False)
    print(f"Gradio app running at {url_endpoint}")
    return url_endpoint


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    # ── Step 1: Load data ──
    print("=" * 60, "\nStep 1: Loading data …\n" + "=" * 60)
    df = load_data()

    # ── Step 2: Split data (70/20/10, stratified on sex) ──
    print("=" * 60, "\nStep 2: Splitting data …\n" + "=" * 60)
    train_df, val_df, test_df = split_data(df)

    # ── Step 3: Preprocess & feature engineering ──
    print("=" * 60, "\nStep 3: Preprocessing …\n" + "=" * 60)
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     feature_names, preprocessor, col_tx) = preprocess_data(train_df, val_df, test_df)

    processed_data = (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)

    # ── Step 4: Optuna HPO (100 trials × 5-fold CV) ──
    print("=" * 60, "\nStep 4: Optuna HPO (100 trials × 5-fold CV) …\n" + "=" * 60)
    study = run_optuna_hpo(X_train, y_train, X_val, y_val)
    best_params = study.best_params

    # ── Step 5: Train final model ──
    print("=" * 60, "\nStep 5: Training final model …\n" + "=" * 60)
    final_model = train_model(X_train, y_train, X_val, y_val, best_params)

    # ── Step 6: Evaluate on held-out test set ──
    print("=" * 60, "\nStep 6: Evaluation on test set …\n" + "=" * 60)
    model_performance, model_complexity, residual_metrics = evaluate_model(
        final_model, X_test, y_test, feature_names)

    # Save all experiment outputs
    for fname, obj in [
        ('evaluation_metrics.json',  model_performance),
        ('complexity_metrics.json',  model_complexity),
        ('residual_analysis.json',   residual_metrics),
        ('optuna_best_params.json',  study.best_params),
    ]:
        with open(os.path.join(EXP_PATH, fname), 'w') as f:
            json.dump(obj, f, indent=2, default=str)
    print(f"All experiment artefacts saved → {EXP_PATH}")

    # ── Step 7: Serialize full pipeline for deployment ──
    print("=" * 60, "\nStep 7: Building & saving deployment pipeline …\n" + "=" * 60)
    deployment_pipeline = prepare_model_for_deployment(
        train_df, y_train, best_params, final_model)

    # Sanity-check: pipeline vs standalone predictions
    y_pipe = deployment_pipeline.predict(test_df.drop(columns=[TARGET]))
    print(f"Pipeline test MAE = {mean_absolute_error(y_test, y_pipe):.4f}  "
          f"(should match standalone)")

    # ── Step 8: Deploy with Gradio ──
    print("=" * 60, "\nStep 8: Deploying Gradio demo …\n" + "=" * 60)
    url_endpoint = deploy_model(deployment_pipeline)

    return (processed_data, final_model, deployment_pipeline,
            url_endpoint, model_performance, model_complexity)


if __name__ == "__main__":
    (processed_data, model, deployable_model,
     url_endpoint, model_performance, model_complexity) = main()

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print("Model Performance on Test Set:", json.dumps(model_performance, indent=2))
    print("Model Complexity:", json.dumps(model_complexity, indent=2))
