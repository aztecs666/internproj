"""
Comprehensive Model Evaluation & Comparison
============================================
Loads all trained RF and XGBoost models, evaluates each on its corresponding
test split, compiles results, finds best split method, and generates summary.
"""

import sys
import os
import glob
import pickle
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from year_based_split import prepare_data, get_all_percentage_splits, SPLIT_CONFIGS

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RF_DIR = os.path.join(BASE_DIR, "Random Forest")
XGB_DIR = os.path.join(BASE_DIR, "XGBoost")
TESTING_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(TESTING_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASET_PATH = r"d:\Internproj\new dataset\engineered_ml_dataset.csv"


def load_rf_model(model_path):
    """Load a pickled RF model."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_xgb_model(model_path):
    """Load an XGBoost model from JSON."""
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }, y_pred


def parse_model_filename(filename, model_type):
    """Extract split_name and param_label from model filename.

    Format: rf_{split}_{label}.pkl or xgb_{split}_{label}.json
    where split is one of: 80_20, 60_40, 40_60, 20_80
    """
    base = os.path.splitext(filename)[0]
    prefix = "rf_" if model_type == "rf" else "xgb_"
    if not base.startswith(prefix):
        return None, None
    remainder = base[len(prefix) :]

    # Known splits
    known_splits = ["80_20", "60_40", "40_60", "20_80"]
    for split in known_splits:
        if remainder.startswith(split + "_"):
            param_label = remainder[len(split) + 1 :]
            return split, param_label
    return None, None


def run_evaluation():
    print("=" * 80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 80)

    # Load data
    print(f"\nLoading data from: {DATASET_PATH}")
    X, y, years = prepare_data(DATASET_PATH)
    splits = get_all_percentage_splits(years)
    print(f"Total samples: {len(y)}, Features: {X.shape[1]}")

    # Find all model files
    rf_models = glob.glob(os.path.join(RF_DIR, "rf_*.pkl"))
    xgb_models = glob.glob(os.path.join(XGB_DIR, "xgb_*.json"))
    print(f"\nFound {len(rf_models)} RF models, {len(xgb_models)} XGBoost models")

    all_results = []
    all_predictions = {}

    # Evaluate RF Models
    print(f"\n{'=' * 40} RANDOM FOREST {'=' * 40}")
    for i, model_path in enumerate(rf_models):
        filename = os.path.basename(model_path)
        split_name, param_label = parse_model_filename(filename, "rf")

        if split_name not in splits:
            print(f"  SKIP: {filename} (unknown split: {split_name})")
            continue

        _, test_idx = splits[split_name]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        print(f"\n[{i + 1}/{len(rf_models)}] Evaluating: {filename}")
        model = load_rf_model(model_path)
        metrics, y_pred = evaluate_model(model, X_test, y_test)

        print(f"  Split: {split_name} ({SPLIT_CONFIGS[split_name]['description']})")
        print(f"  Params: {param_label}")
        print(
            f"  MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}"
        )

        result = {
            "model_type": "RandomForest",
            "model_file": filename,
            "param_label": param_label,
            "split": split_name,
            "split_description": SPLIT_CONFIGS[split_name]["description"],
            "test_size": len(test_idx),
            **metrics,
        }
        all_results.append(result)

        key = f"rf_{split_name}_{param_label}"
        all_predictions[key] = {
            "model_type": "RF",
            "split": split_name,
            "label": param_label,
            "y_true": y_test.values,
            "y_pred": y_pred,
        }

    # Evaluate XGBoost Models
    print(f"\n{'=' * 40} XGBOOST {'=' * 40}")
    for i, model_path in enumerate(xgb_models):
        filename = os.path.basename(model_path)
        split_name, param_label = parse_model_filename(filename, "xgb")

        if split_name not in splits:
            print(f"  SKIP: {filename} (unknown split: {split_name})")
            continue

        _, test_idx = splits[split_name]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        print(f"\n[{i + 1}/{len(xgb_models)}] Evaluating: {filename}")
        model = load_xgb_model(model_path)
        metrics, y_pred = evaluate_model(model, X_test, y_test)

        print(f"  Split: {split_name} ({SPLIT_CONFIGS[split_name]['description']})")
        print(f"  Params: {param_label}")
        print(
            f"  MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}"
        )

        result = {
            "model_type": "XGBoost",
            "model_file": filename,
            "param_label": param_label,
            "split": split_name,
            "split_description": SPLIT_CONFIGS[split_name]["description"],
            "test_size": len(test_idx),
            **metrics,
        }
        all_results.append(result)

        key = f"xgb_{split_name}_{param_label}"
        all_predictions[key] = {
            "model_type": "XGB",
            "split": split_name,
            "label": param_label,
            "y_true": y_test.values,
            "y_pred": y_pred,
        }

    # Save All Results
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(RESULTS_DIR, "all_evaluation_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n{'=' * 80}")
    print(f"All evaluation results saved to: {results_path}")

    # Best Split Method Analysis
    print(f"\n{'=' * 40} BEST SPLIT METHOD ANALYSIS {'=' * 40}")
    for model_type in ["RandomForest", "XGBoost"]:
        type_df = results_df[results_df["model_type"] == model_type]
        split_avg = (
            type_df.groupby("split")
            .agg(
                avg_MAE=("MAE", "mean"),
                avg_RMSE=("RMSE", "mean"),
                avg_R2=("R2", "mean"),
                best_MAE=("MAE", "min"),
                worst_MAE=("MAE", "max"),
            )
            .sort_values("avg_MAE")
        )
        print(f"\n--- {model_type}: Split Performance ---")
        print(split_avg.to_string())

    overall_split = (
        results_df.groupby("split")
        .agg(avg_MAE=("MAE", "mean"), avg_R2=("R2", "mean"))
        .sort_values("avg_MAE")
    )
    best_split = overall_split.index[0]
    print(
        f"\n*** BEST OVERALL SPLIT: {best_split} ({SPLIT_CONFIGS[best_split]['description']}) ***"
    )
    print(f"    Average MAE: {overall_split.iloc[0]['avg_MAE']:.4f}")
    print(f"    Average R2:  {overall_split.iloc[0]['avg_R2']:.4f}")

    # Top 5 Models
    print(f"\n{'=' * 40} TOP 5 MODELS OVERALL {'=' * 40}")
    top5 = results_df.nsmallest(5, "MAE")
    print(
        top5[["model_type", "param_label", "split", "MAE", "RMSE", "R2"]].to_string(
            index=False
        )
    )

    rf_results = results_df[results_df["model_type"] == "RandomForest"]
    top5_rf = rf_results.nsmallest(5, "MAE")
    print(f"\n--- Top 5 Random Forest ---")
    print(top5_rf[["param_label", "split", "MAE", "RMSE", "R2"]].to_string(index=False))

    xgb_results = results_df[results_df["model_type"] == "XGBoost"]
    top5_xgb = xgb_results.nsmallest(5, "MAE")
    print(f"\n--- Top 5 XGBoost ---")
    print(
        top5_xgb[["param_label", "split", "MAE", "RMSE", "R2"]].to_string(index=False)
    )

    # Save Summary
    summary = {
        "total_models_evaluated": len(all_results),
        "rf_models": len(rf_models),
        "xgb_models": len(xgb_models),
        "best_overall_split": best_split,
        "best_overall_split_mae": float(overall_split.iloc[0]["avg_MAE"]),
        "best_overall_model": f"{top5.iloc[0]['model_type']}_{top5.iloc[0]['split']}_{top5.iloc[0]['param_label']}",
        "best_overall_mae": float(top5.iloc[0]["MAE"]),
        "top5_overall": top5[["model_type", "param_label", "split", "MAE"]].to_dict(
            orient="records"
        ),
        "top5_rf": top5_rf[["param_label", "split", "MAE"]].to_dict(orient="records"),
        "top5_xgb": top5_xgb[["param_label", "split", "MAE"]].to_dict(orient="records"),
    }
    summary_path = os.path.join(RESULTS_DIR, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    predictions_path = os.path.join(RESULTS_DIR, "predictions_cache.pkl")
    with open(predictions_path, "wb") as f:
        pickle.dump(all_predictions, f)
    print(f"Predictions cache saved to: {predictions_path}")

    return results_df, all_predictions


if __name__ == "__main__":
    run_evaluation()
