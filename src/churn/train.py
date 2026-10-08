"""Model training, hyperparameter tuning, calibration, and artifact persistence."""

import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn.config import (
    METADATA_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    SEED,
    TARGET,
    TRAIN_DATA_PATH,
)
from churn.data import load_train
from churn.evaluate import evaluate_cv
from churn.features import (
    FeatureEngineer,
    build_pipeline,
    build_tree_preprocessor,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def compare_baseline_models(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    cv: int = 5,
    random_state: int = SEED,
) -> pd.DataFrame:
    """Run Experiment 1: Compare 5 baseline models with class weighting.

    Models evaluated: DummyClassifier, LogisticRegression, RandomForest, XGBoost, LightGBM.

    Parameters
    ----------
    X : pd.DataFrame
        Training features.
    y : pd.Series | np.ndarray
        Training labels.
    cv : int, optional
        CV folds, default 5.
    random_state : int, optional
        Random seed, default SEED (42).

    Returns
    -------
    pd.DataFrame
        Summary dataframe comparing mean metrics and std across all 5 models.
    """
    models = ["dummy", "logistic_regression", "random_forest", "xgboost", "lightgbm"]
    records: list[dict[str, Any]] = []

    # Calculate scale_pos_weight for XGBoost
    y_arr = np.asarray(y).astype(int)
    scale_pos = float((y_arr == 0).sum() / (y_arr == 1).sum())

    for model_name in models:
        kwargs: dict[str, Any] = {}
        if model_name == "xgboost":
            kwargs["scale_pos_weight"] = scale_pos

        pipe = build_pipeline(model_name, random_state=random_state, **kwargs)
        mean_m, std_m, _ = evaluate_cv(pipe, X, y_arr, n_splits=cv, random_state=random_state)

        record: dict[str, Any] = {"model": model_name, "stage": "baseline"}
        for k, v in mean_m.items():
            record[f"{k}_mean"] = round(v, 4)
            record[f"{k}_std"] = round(std_m[k], 4)
        records.append(record)

    return pd.DataFrame(records)


def compare_imbalance_strategies(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    top_models: list[str] | None = None,
    cv: int = 5,
    random_state: int = SEED,
) -> pd.DataFrame:
    """Run Experiment 2: Compare class imbalance strategies for top 2 models.

    Strategies compared:
    - none (unweighted)
    - class_weights / scale_pos_weight
    - SMOTE (applied strictly inside imblearn Pipeline on training folds)

    Parameters
    ----------
    X : pd.DataFrame
        Training features.
    y : pd.Series | np.ndarray
        Training labels.
    top_models : list[str] | None, optional
        Model names to test, default ['lightgbm', 'xgboost'].
    cv : int, optional
        CV folds, default 5.
    random_state : int, optional
        Random seed, default SEED (42).

    Returns
    -------
    pd.DataFrame
        Comparison table of imbalance strategies.
    """
    if top_models is None:
        top_models = ["lightgbm", "xgboost"]

    y_arr = np.asarray(y).astype(int)
    scale_pos = float((y_arr == 0).sum() / (y_arr == 1).sum())
    records: list[dict[str, Any]] = []

    for model_name in top_models:
        # 1. Strategy: None (unweighted)
        if model_name == "lightgbm":
            clf_none = LGBMClassifier(random_state=random_state, verbose=-1)
        elif model_name == "xgboost":
            clf_none = XGBClassifier(random_state=random_state, eval_metric="logloss")
        else:
            clf_none = LGBMClassifier(random_state=random_state, verbose=-1)

        pipe_none = Pipeline([
            ("feature_engineer", FeatureEngineer()),
            ("preprocessor", build_tree_preprocessor()),
            ("classifier", clf_none),
        ])
        mean_none, std_none, _ = evaluate_cv(pipe_none, X, y_arr, n_splits=cv, random_state=random_state)
        rec_none: dict[str, Any] = {"model": model_name, "strategy": "none"}
        for k, v in mean_none.items():
            rec_none[f"{k}_mean"] = round(v, 4)
            rec_none[f"{k}_std"] = round(std_none[k], 4)
        records.append(rec_none)

        # 2. Strategy: Class Weighting
        if model_name == "lightgbm":
            clf_weighted = LGBMClassifier(random_state=random_state, class_weight="balanced", verbose=-1)
        elif model_name == "xgboost":
            clf_weighted = XGBClassifier(random_state=random_state, scale_pos_weight=scale_pos, eval_metric="logloss")
        else:
            clf_weighted = LGBMClassifier(random_state=random_state, class_weight="balanced", verbose=-1)

        pipe_weighted = Pipeline([
            ("feature_engineer", FeatureEngineer()),
            ("preprocessor", build_tree_preprocessor()),
            ("classifier", clf_weighted),
        ])
        mean_wt, std_wt, _ = evaluate_cv(pipe_weighted, X, y_arr, n_splits=cv, random_state=random_state)
        rec_wt: dict[str, Any] = {"model": model_name, "strategy": "class_weights"}
        for k, v in mean_wt.items():
            rec_wt[f"{k}_mean"] = round(v, 4)
            rec_wt[f"{k}_std"] = round(std_wt[k], 4)
        records.append(rec_wt)

        # 3. Strategy: SMOTE (inside imblearn Pipeline)
        pipe_smote = ImbPipeline([
            ("feature_engineer", FeatureEngineer()),
            ("preprocessor", build_tree_preprocessor()),
            ("smote", SMOTE(random_state=random_state)),
            ("classifier", clf_none),
        ])
        mean_sm, std_sm, _ = evaluate_cv(pipe_smote, X, y_arr, n_splits=cv, random_state=random_state)
        rec_sm: dict[str, Any] = {"model": model_name, "strategy": "smote"}
        for k, v in mean_sm.items():
            rec_sm[f"{k}_mean"] = round(v, 4)
            rec_sm[f"{k}_std"] = round(std_sm[k], 4)
        records.append(rec_sm)

    return pd.DataFrame(records)


def run_feature_ablation(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    model_name: str = "lightgbm",
    cv: int = 5,
    random_state: int = SEED,
) -> dict[str, Any]:
    """Run Experiment 3: Feature ablation (with vs without engineered features).

    Parameters
    ----------
    X : pd.DataFrame
        Training features.
    y : pd.Series | np.ndarray
        Training labels.
    model_name : str, optional
        Model architecture to test, default 'lightgbm'.
    cv : int, optional
        CV folds, default 5.
    random_state : int, optional
        Random seed, default SEED (42).

    Returns
    -------
    dict[str, Any]
        Dictionary reporting with-features, without-features, and delta metrics.
    """
    y_arr = np.asarray(y).astype(int)

    # With engineered features
    pipe_with = build_pipeline(model_name, include_engineered=True, random_state=random_state)
    mean_with, std_with, _ = evaluate_cv(pipe_with, X, y_arr, n_splits=cv, random_state=random_state)

    # Without engineered features
    pipe_without = build_pipeline(model_name, include_engineered=False, random_state=random_state)
    mean_without, std_without, _ = evaluate_cv(pipe_without, X, y_arr, n_splits=cv, random_state=random_state)

    delta = {k: round(mean_with[k] - mean_without[k], 4) for k in mean_with}

    return {
        "model": model_name,
        "with_engineered_mean": {k: round(v, 4) for k, v in mean_with.items()},
        "without_engineered_mean": {k: round(v, 4) for k, v in mean_without.items()},
        "delta": delta,
    }


def tune_model_optuna(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    n_trials: int = 50,
    cv: int = 5,
    random_state: int = SEED,
) -> tuple[dict[str, Any], float, optuna.Study]:
    """Run Experiment 4: Hyperparameter optimization using Optuna TPE.

    Objective: Maximize mean CV PR-AUC.

    Parameters
    ----------
    model_name : str
        Model to tune ('lightgbm' or 'xgboost').
    X : pd.DataFrame
        Training features.
    y : pd.Series | np.ndarray
        Training labels.
    n_trials : int, optional
        Number of Optuna trials, default 50.
    cv : int, optional
        CV folds, default 5.
    random_state : int, optional
        Random seed, default SEED (42).

    Returns
    -------
    tuple[dict[str, Any], float, optuna.Study]
        Tuple of (best_params, best_pr_auc, study).
    """
    y_arr = np.asarray(y).astype(int)
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        if model_name == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 4.0),
                "random_state": random_state,
                "verbose": -1,
            }
            clf = LGBMClassifier(**params)
        elif model_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 4.0),
                "random_state": random_state,
                "eval_metric": "logloss",
            }
            clf = XGBClassifier(**params)
        else:
            raise ValueError(f"Tuning not implemented for model '{model_name}'.")

        pipe = Pipeline([
            ("feature_engineer", FeatureEngineer()),
            ("preprocessor", build_tree_preprocessor()),
            ("classifier", clf),
        ])

        mean_m, _, _ = evaluate_cv(pipe, X, y_arr, n_splits=cv, random_state=random_state)
        return float(mean_m["pr_auc"])

    study.optimize(objective, n_trials=n_trials)
    return study.best_params, float(study.best_value), study


def calibrate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    cv: int = 5,
    method: str = "isotonic",
    random_state: int = SEED,
) -> tuple[CalibratedClassifierCV, dict[str, float], dict[str, float]]:
    """Run Experiment 5: Isotonic calibration and compare Brier scores.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted or uncalibrated Pipeline.
    X : pd.DataFrame
        Training features.
    y : pd.Series | np.ndarray
        Training labels.
    cv : int, optional
        CV folds for calibration, default 5.
    method : str, optional
        Calibration method ('isotonic' or 'sigmoid'), default 'isotonic'.
    random_state : int, optional
        Random seed, default SEED (42).

    Returns
    -------
    tuple[CalibratedClassifierCV, dict[str, float], dict[str, float]]
        Tuple of (calibrated_model, uncalibrated_cv_metrics, calibrated_cv_metrics).
    """
    y_arr = np.asarray(y).astype(int)

    # Uncalibrated baseline metrics
    uncal_mean, uncal_std, _ = evaluate_cv(pipeline, X, y_arr, n_splits=cv, random_state=random_state)

    # Build calibrated pipeline wrapper
    calibrated_clf = CalibratedClassifierCV(
        estimator=pipeline,
        method=method,
        cv=cv,
    )
    cal_mean, cal_std, _ = evaluate_cv(calibrated_clf, X, y_arr, n_splits=cv, random_state=random_state)

    return calibrated_clf, uncal_mean, cal_mean


def run_all_experiments(
    output_dir: Path | str = REPORTS_DIR,
    n_trials: int = 50,
) -> dict[str, Any]:
    """Execute all Phase 4 experiments, log to MLflow, and persist comparison files.

    Outputs:
    - reports/model_comparison.csv
    - reports/best_params.json

    Parameters
    ----------
    output_dir : Path | str, optional
        Directory to save reports, default REPORTS_DIR.
    n_trials : int, optional
        Optuna trials per model, default 50.

    Returns
    -------
    dict[str, Any]
        Dictionary with all experiment results.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_df = load_train()
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET]

    # Initialize MLflow with local SQLite backend
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("bank_churn_modeling")

    # 1. Baseline Model Comparison
    print("--- Running Experiment 1: Baseline Comparison ---")
    df_baseline = compare_baseline_models(X, y)

    # 2. Imbalance Strategy Comparison
    print("--- Running Experiment 2: Imbalance Strategies ---")
    df_imbalance = compare_imbalance_strategies(X, y, top_models=["lightgbm", "xgboost"])

    # 3. Feature Ablation
    print("--- Running Experiment 3: Feature Ablation ---")
    ablation_results = run_feature_ablation(X, y, model_name="lightgbm")

    # 4. Optuna Hyperparameter Tuning
    print("--- Running Experiment 4: Optuna Tuning ---")
    best_lgbm_params, best_lgbm_prauc, _ = tune_model_optuna("lightgbm", X, y, n_trials=n_trials)
    best_xgb_params, best_xgb_prauc, _ = tune_model_optuna("xgboost", X, y, n_trials=n_trials)

    # Evaluate tuned models
    tuned_lgbm_pipe = Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("preprocessor", build_tree_preprocessor()),
        ("classifier", LGBMClassifier(**best_lgbm_params)),
    ])
    lgbm_mean, lgbm_std, _ = evaluate_cv(tuned_lgbm_pipe, X, y)

    tuned_xgb_pipe = Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("preprocessor", build_tree_preprocessor()),
        ("classifier", XGBClassifier(**best_xgb_params)),
    ])
    xgb_mean, xgb_std, _ = evaluate_cv(tuned_xgb_pipe, X, y)

    # Combine into model comparison table
    comparison_rows = df_baseline.to_dict(orient="records")
    rec_tuned_lgbm = {"model": "tuned_lightgbm", "stage": "tuned"}
    for k, v in lgbm_mean.items():
        rec_tuned_lgbm[f"{k}_mean"] = round(v, 4)
        rec_tuned_lgbm[f"{k}_std"] = round(lgbm_std[k], 4)
    comparison_rows.append(rec_tuned_lgbm)

    rec_tuned_xgb = {"model": "tuned_xgboost", "stage": "tuned"}
    for k, v in xgb_mean.items():
        rec_tuned_xgb[f"{k}_mean"] = round(v, 4)
        rec_tuned_xgb[f"{k}_std"] = round(xgb_std[k], 4)
    comparison_rows.append(rec_tuned_xgb)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv_path = out_path / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)

    # 5. Calibration
    print("--- Running Experiment 5: Calibration ---")
    winner_name = "lightgbm" if lgbm_mean["pr_auc"] >= xgb_mean["pr_auc"] else "xgboost"
    winner_pipe = tuned_lgbm_pipe if winner_name == "lightgbm" else tuned_xgb_pipe
    best_params = best_lgbm_params if winner_name == "lightgbm" else best_xgb_params

    _, uncal_mean, cal_mean = calibrate_model(winner_pipe, X, y, method="isotonic")

    # Persist best params
    best_params_path = out_path / "best_params.json"
    params_record = {
        "winning_model": winner_name,
        "best_params": best_params,
        "tuned_cv_pr_auc": float(lgbm_mean["pr_auc"] if winner_name == "lightgbm" else xgb_mean["pr_auc"]),
        "uncalibrated_brier_score": round(uncal_mean["brier_score"], 4),
        "calibrated_brier_score": round(cal_mean["brier_score"], 4),
    }
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(params_record, f, indent=2)

    # Log best run to MLflow
    with mlflow.start_run(run_name=f"best_{winner_name}"):
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "cv_pr_auc": params_record["tuned_cv_pr_auc"],
            "cv_roc_auc": lgbm_mean["roc_auc"] if winner_name == "lightgbm" else xgb_mean["roc_auc"],
            "uncalibrated_brier": params_record["uncalibrated_brier_score"],
            "calibrated_brier": params_record["calibrated_brier_score"],
        })

    return {
        "comparison_df": comparison_df,
        "imbalance_df": df_imbalance,
        "best_params": params_record,
        "ablation": ablation_results,
    }


def train_and_save_production_model(
    config: str = "default",
    tune: bool = False,
    model_name: str = "xgboost",
    model_path: Path | str = MODEL_PATH,
    metadata_path: Path | str = METADATA_PATH,
    random_state: int = SEED,
) -> dict[str, Any]:
    """Train, calibrate, and serialize production model and metadata.

    Parameters
    ----------
    config : str, optional
        Configuration profile name, default 'default'.
    tune : bool, optional
        Whether to re-run hyperparameter optimization, default False.
    model_name : str, optional
        Model family, default 'xgboost'.
    model_path : Path | str, optional
        Destination model file.
    metadata_path : Path | str, optional
        Destination metadata file.
    random_state : int, optional
        Random seed.

    Returns
    -------
    dict[str, Any]
        Metadata record of serialized model.
    """
    import datetime
    import hashlib
    import importlib.metadata

    import joblib

    from churn.business import optimal_threshold
    from churn.data import load_train, make_split

    print(f"[{datetime.datetime.now(datetime.UTC).isoformat()}] Starting production model training pipeline...")

    # Ensure train dataset exists
    if not TRAIN_DATA_PATH.exists():
        print("Processed train dataset missing. Running make_split()...")
        make_split(random_state=random_state)

    train_df = load_train()
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET]

    # Calculate data SHA-256 hash
    with open(TRAIN_DATA_PATH, "rb") as f:
        data_hash = hashlib.sha256(f.read()).hexdigest()

    # Determine hyperparameters
    if tune:
        print("Tuning hyperparameters via Optuna (50 trials)...")
        best_params, _, _ = tune_model_optuna(model_name, X, y, n_trials=50, random_state=random_state)
    elif (REPORTS_DIR / "best_params.json").exists():
        with open(REPORTS_DIR / "best_params.json", encoding="utf-8") as f:
            saved_info = json.load(f)
        best_params = saved_info.get("best_params", {})
    else:
        # Default tuned parameters for XGBoost
        best_params = {
            "n_estimators": 275,
            "learning_rate": 0.0566,
            "max_depth": 3,
            "min_child_weight": 3,
            "subsample": 0.812,
            "colsample_bytree": 0.714,
            "scale_pos_weight": 1.41,
        }

    # Build base pipeline
    base_pipeline = build_pipeline(model_name, random_state=random_state, **best_params)

    # Cross-validation performance check
    print("Evaluating 5-fold cross-validation...")
    mean_m, std_m, _ = evaluate_cv(base_pipeline, X, y, n_splits=5, random_state=random_state)

    # Wrap in Isotonic Calibration
    print("Fitting calibrated classifier on full training split...")
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="isotonic",
        cv=5,
    )
    calibrated_clf.fit(X, y)

    # Compute decision threshold
    opt_t = optimal_threshold()

    # Collect library versions
    lib_names = ["xgboost", "lightgbm", "scikit-learn", "pandas", "numpy", "joblib", "pydantic"]
    versions = {}
    for lib in lib_names:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = "unknown"

    # Save model artifact
    m_dest = Path(model_path)
    meta_dest = Path(metadata_path)
    m_dest.parent.mkdir(parents=True, exist_ok=True)
    meta_dest.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(calibrated_clf, m_dest)
    print(f"Serialized production model to {m_dest} ({m_dest.stat().st_size / 1024:.1f} KB)")

    metadata = {
        "model_architecture": f"CalibratedClassifierCV(estimator=Pipeline({model_name}), method='isotonic', cv=5)",
        "model_name": model_name,
        "config_profile": config,
        "training_timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "random_seed": random_state,
        "decision_threshold": round(opt_t, 4),
        "data_hash_sha256": data_hash,
        "training_sample_count": len(train_df),
        "features": list(X.columns),
        "cv_metrics_mean": {k: round(v, 4) for k, v in mean_m.items()},
        "cv_metrics_std": {k: round(v, 4) for k, v in std_m.items()},
        "hyperparameters": best_params,
        "library_versions": versions,
    }

    with open(meta_dest, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {meta_dest}")
    return metadata


def main() -> None:
    """CLI entrypoint for training the bank churn model."""
    import argparse

    parser = argparse.ArgumentParser(description="Bank Customer Churn Model Training CLI")
    parser.add_argument("--config", type=str, default="default", help="Config profile name")
    parser.add_argument("--tune", action="store_true", help="Re-run Optuna hyperparameter optimization")
    parser.add_argument("--model-name", type=str, default="xgboost", help="Model architecture")
    args = parser.parse_args()

    train_and_save_production_model(
        config=args.config,
        tune=args.tune,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
