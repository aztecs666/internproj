"""
Visualizations for Walk-Forward + % Change Target Models
========================================================
Generates plots for the new walk-forward validation results.
Saves to D:\\Internproj\\Visualization\\v2_walkforward\\ (keeps old plots in parent folder)
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

# Paths
VIS_DIR = r"D:\Internproj\Visualization\v2_walkforward"
DATA_PATH = r"D:\Internproj\new dataset\engineered_ml_dataset.csv"
RF_RESULTS = r"D:\Internproj\ML Models\Random Forest\results\all_rf_results.csv"
XGB_RESULTS = r"D:\Internproj\ML Models\XGBoost\results\all_xgb_results.csv"
EVAL_RESULTS = r"D:\Internproj\ML Models\Testing\results\all_evaluation_results.csv"
PREDICTIONS_CACHE = r"D:\Internproj\ML Models\Testing\results\predictions_cache.pkl"

os.makedirs(VIS_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")


# =============================================================================
# 1. WALK-FORWARD FOLD COMPARISON
# =============================================================================
def plot_fold_comparison():
    """Compare all models grouped by walk-forward fold."""
    eval_df = pd.read_csv(EVAL_RESULTS)

    fig, axes = plt.subplots(2, 4, figsize=(24, 14))
    folds = sorted(eval_df["fold"].unique())

    for idx, fold in enumerate(folds):
        ax = axes[idx // 4, idx % 4]

        rf_fold = eval_df[
            (eval_df["model_type"] == "RandomForest") & (eval_df["fold"] == fold)
        ].nsmallest(10, "MAE")
        xgb_fold = eval_df[
            (eval_df["model_type"] == "XGBoost") & (eval_df["fold"] == fold)
        ].nsmallest(10, "MAE")

        rf_fold = rf_fold.copy()
        rf_fold["model"] = "RF"
        xgb_fold = xgb_fold.copy()
        xgb_fold["model"] = "XGB"

        combined = pd.concat([rf_fold, xgb_fold])
        combined["label"] = combined["param_label"].str[:18]

        colors = ["#2196F3" if m == "RF" else "#FF9800" for m in combined["model"]]
        ax.barh(range(len(combined)), combined["MAE"], color=colors)
        ax.set_yticks(range(len(combined)))
        ax.set_yticklabels(
            [f"{r['model']}:{r['label']}" for _, r in combined.iterrows()], fontsize=7
        )
        ax.set_xlabel("MAE ($) - lower = better", fontsize=9)
        fold_desc = combined.iloc[0]["fold_description"] if len(combined) > 0 else fold
        ax.set_title(f"{fold}\n{fold_desc}", fontsize=10, fontweight="bold")
        ax.invert_yaxis()

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2196F3", label="Random Forest"),
        Patch(facecolor="#FF9800", label="XGBoost"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=12)

    plt.suptitle(
        "Walk-Forward Fold Comparison: Top 10 Models Per Fold",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(VIS_DIR, "fold_comparison.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved: fold_comparison.png")


# =============================================================================
# 2. ABSOLUTE vs % CHANGE R2 COMPARISON
# =============================================================================
def plot_r2_comparison():
    """Compare R2 on absolute price vs R2 on % change."""
    eval_df = pd.read_csv(EVAL_RESULTS)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for idx, model_type in enumerate(["RandomForest", "XGBoost"]):
        ax = axes[idx]
        subset = eval_df[eval_df["model_type"] == model_type]

        fold_avg = (
            subset.groupby("fold")
            .agg(
                avg_R2=("R2", "mean"),
                avg_R2_pct=("R2_pct", "mean"),
            )
            .reset_index()
        )

        x = range(len(fold_avg))
        width = 0.35

        ax.bar(
            [i - width / 2 for i in x],
            fold_avg["avg_R2"],
            width,
            label="R2 (Absolute $)",
            color="#F44336",
            alpha=0.8,
        )
        ax.bar(
            [i + width / 2 for i in x],
            fold_avg["avg_R2_pct"],
            width,
            label="R2 (% Change)",
            color="#4CAF50",
            alpha=0.8,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(fold_avg["fold"], rotation=45, fontsize=9)
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel("R2 Score")
        ax.set_title(
            f"{model_type}: Absolute vs % Change R2", fontsize=14, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Does Target Transformation Fix the Negative R2?",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(VIS_DIR, "r2_comparison.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved: r2_comparison.png")


# =============================================================================
# 3. FEATURE IMPORTANCE (from best models)
# =============================================================================
def plot_feature_importance():
    """Extract and plot feature importance from best RF and XGB models."""
    import xgboost as xgb_lib
    import pickle

    eval_df = pd.read_csv(EVAL_RESULTS)
    best_xgb_row = (
        eval_df[eval_df["model_type"] == "XGBoost"].nsmallest(1, "MAE").iloc[0]
    )
    best_rf_row = (
        eval_df[eval_df["model_type"] == "RandomForest"].nsmallest(1, "MAE").iloc[0]
    )

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # XGBoost feature importance
    try:
        xgb_model_path = os.path.join(
            r"D:\Internproj\ML Models\XGBoost", best_xgb_row["model_file"]
        )
        xgb_model = xgb_lib.XGBRegressor()
        xgb_model.load_model(xgb_model_path)

        importances = xgb_model.feature_importances_
        feature_names = xgb_model.get_booster().feature_names
        if feature_names is None:
            df = pd.read_csv(DATA_PATH)
            df = df.drop(
                columns=[
                    "Date",
                    "Route",
                    "Price_USD",
                    "Price_PctChange",
                    "Price_Prev_Day",
                ],
                errors="ignore",
            )
            feature_names = list(df.columns)

        idx = np.argsort(importances)[-20:]
        axes[0].barh(range(len(idx)), importances[idx], color="#FF9800")
        axes[0].set_yticks(range(len(idx)))
        axes[0].set_yticklabels([feature_names[i] for i in idx], fontsize=8)
        axes[0].set_xlabel("Feature Importance")
        axes[0].set_title(
            f"XGBoost: Top 20 Features\n({best_xgb_row['model_file']})",
            fontsize=12,
            fontweight="bold",
        )
    except Exception as e:
        axes[0].text(0.5, 0.5, f"Error loading model:\n{e}", ha="center", va="center")

    # Random Forest feature importance
    try:
        rf_model_path = os.path.join(
            r"D:\Internproj\ML Models\Random Forest", best_rf_row["model_file"]
        )
        with open(rf_model_path, "rb") as f:
            rf_model = pickle.load(f)

        importances = rf_model.feature_importances_
        df = pd.read_csv(DATA_PATH)
        feature_names = [
            c
            for c in df.columns
            if c
            not in ["Date", "Route", "Price_USD", "Price_PctChange", "Price_Prev_Day"]
        ]

        idx = np.argsort(importances)[-20:]
        axes[1].barh(range(len(idx)), importances[idx], color="#2196F3")
        axes[1].set_yticks(range(len(idx)))
        axes[1].set_yticklabels([feature_names[i] for i in idx], fontsize=8)
        axes[1].set_xlabel("Feature Importance")
        axes[1].set_title(
            f"Random Forest: Top 20 Features\n({best_rf_row['model_file']})",
            fontsize=12,
            fontweight="bold",
        )
    except Exception as e:
        axes[1].text(0.5, 0.5, f"Error loading model:\n{e}", ha="center", va="center")

    plt.suptitle(
        "What Features Actually Matter? (Walk-Forward + % Change Target)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(VIS_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved: feature_importance.png")


# =============================================================================
# 4. WALK-FORWARD PERFORMANCE OVER TIME
# =============================================================================
def plot_walk_forward_progression():
    """Show how model performance evolves as training data expands."""
    eval_df = pd.read_csv(EVAL_RESULTS)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for idx, metric in enumerate(["MAE", "R2_pct"]):
        ax = axes[idx]

        for model_type in ["RandomForest", "XGBoost"]:
            subset = eval_df[eval_df["model_type"] == model_type]
            fold_avg = (
                subset.groupby("fold")
                .agg(
                    avg_metric=(metric, "mean"),
                    best_metric=(metric, "min") if metric == "MAE" else (metric, "max"),
                )
                .reset_index()
            )

            # Sort folds chronologically
            fold_order = [
                f"wf_{y}" for y in [2018, 2019, 2020, 2021, 2022, 2023, 2024, "2025_26"]
            ]
            fold_order = [f for f in fold_order if f in fold_avg["fold"].values]
            fold_avg = fold_avg.set_index("fold").loc[fold_order].reset_index()

            marker = "o" if model_type == "RandomForest" else "s"
            color = "#2196F3" if model_type == "RandomForest" else "#FF9800"
            ax.plot(
                fold_avg["fold"],
                fold_avg["avg_metric"],
                marker=marker,
                linewidth=2,
                label=f"{model_type} (avg)",
                color=color,
            )
            ax.plot(
                fold_avg["fold"],
                fold_avg["best_metric"],
                marker=marker,
                linewidth=2,
                linestyle="--",
                label=f"{model_type} (best)",
                color=color,
                alpha=0.6,
            )

        ax.set_xlabel("Walk-Forward Fold", fontsize=11)
        ylabel = (
            "MAE ($) - lower = better"
            if metric == "MAE"
            else "R2 (% Change) - higher = better"
        )
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(
            f"Walk-Forward Progression: {metric}", fontsize=14, fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Does More Training Data Improve Performance? (Expanding Window)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(VIS_DIR, "walk_forward_progression.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved: walk_forward_progression.png")


# =============================================================================
# 5. PREDICTED vs ACTUAL (Best Model per Type)
# =============================================================================
def plot_predicted_vs_actual():
    """Scatter plot of predicted vs actual prices for best models."""
    try:
        with open(PREDICTIONS_CACHE, "rb") as f:
            predictions = pickle.load(f)
    except FileNotFoundError:
        print("Predictions cache not found. Skipping plot.")
        return

    eval_df = pd.read_csv(EVAL_RESULTS)
    best_rf_row = (
        eval_df[eval_df["model_type"] == "RandomForest"].nsmallest(1, "MAE").iloc[0]
    )
    best_xgb_row = (
        eval_df[eval_df["model_type"] == "XGBoost"].nsmallest(1, "MAE").iloc[0]
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for idx, (model_type, best_row) in enumerate(
        [("RF", best_rf_row), ("XGB", best_xgb_row)]
    ):
        ax = axes[idx]
        key = f"{model_type.lower()}_{'xgb' if model_type == 'XGB' else model_type.lower()}_{best_row['fold']}_{best_row['param_label']}"
        # Try different key formats
        pred_data = None
        for k, v in predictions.items():
            if (
                k == f"rf_{best_row['fold']}_{best_row['param_label']}"
                and model_type == "RF"
            ):
                pred_data = v
                break
            if (
                k == f"xgb_{best_row['fold']}_{best_row['param_label']}"
                and model_type == "XGB"
            ):
                pred_data = v
                break

        if pred_data is None:
            ax.text(
                0.5,
                0.5,
                f"No predictions found for\n{model_type}",
                ha="center",
                va="center",
            )
            continue

        y_true = pred_data["y_true_abs"]
        y_pred = pred_data["y_pred_abs"]

        ax.scatter(
            y_true,
            y_pred,
            alpha=0.3,
            s=10,
            color="#2196F3" if model_type == "RF" else "#FF9800",
        )

        # Perfect prediction line
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=2, label="Perfect prediction")

        ax.set_xlabel("Actual Price ($)", fontsize=11)
        ax.set_ylabel("Predicted Price ($)", fontsize=11)
        ax.set_title(
            f"{model_type}: Predicted vs Actual\n({best_row['fold']}/{best_row['param_label']})",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Predicted vs Actual Container Prices", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(VIS_DIR, "predicted_vs_actual.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved: predicted_vs_actual.png")


# =============================================================================
# 6. FEATURE CORRELATION HEATMAP (Updated)
# =============================================================================
def plot_feature_correlation():
    """Show correlation of features with Price_PctChange target."""
    df = pd.read_csv(DATA_PATH)

    key_features = [
        "Price_PctChange",
        "Price_Momentum_7d",
        "Price_Momentum_30d",
        "Price_Acceleration",
        "Price_Volatility_7d",
        "Price_Volatility_30d",
        "Volatility_Regime",
        "Trend_Consistency",
        "Price_Distance_From_MA30",
        "Volatility_Spike",
        "Route_Mean_wind_speed",
        "Route_Max_cyclone_wind",
        "Route_Min_cyclone_dist",
        "Cyclone_Wind_Lag7d",
        "Cyclone_Wind_Forecast3d",
        "SWI",
        "Coastal_Threat",
        "Congestion_Risk",
        "Month",
        "Is_Peak_Season",
    ]

    available = [f for f in key_features if f in df.columns]
    corr_matrix = df[available].corr()

    fig, ax = plt.subplots(figsize=(14, 10))

    # Correlation with target only
    target_corr = corr_matrix["Price_PctChange"].drop("Price_PctChange").sort_values()
    colors = ["#E53935" if v < 0 else "#43A047" for v in target_corr]
    ax.barh(range(len(target_corr)), target_corr.values, color=colors)
    ax.set_yticks(range(len(target_corr)))
    ax.set_yticklabels(target_corr.index, fontsize=9)
    ax.set_xlabel("Correlation with Price_PctChange", fontsize=12)
    ax.set_title(
        "Feature Correlation with % Change Target (Stationary)",
        fontsize=14,
        fontweight="bold",
    )
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(VIS_DIR, "feature_correlation_pct.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved: feature_correlation_pct.png")


# =============================================================================
# 7. TOP 5 HYPERPARAMETER COMPARISON
# =============================================================================
def plot_top5_hyperparams():
    """Compare top 5 hyperparameter combos across folds."""
    eval_df = pd.read_csv(EVAL_RESULTS)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for idx, model_type in enumerate(["RandomForest", "XGBoost"]):
        ax = axes[idx]
        subset = eval_df[eval_df["model_type"] == model_type]

        top5_params = subset.groupby("param_label")["MAE"].mean().nsmallest(5).index

        for param in top5_params:
            param_data = subset[subset["param_label"] == param]
            fold_avg = param_data.groupby("fold")["MAE"].mean().reset_index()

            # Sort folds chronologically
            fold_order = [
                f"wf_{y}" for y in [2018, 2019, 2020, 2021, 2022, 2023, 2024, "2025_26"]
            ]
            fold_order = [f for f in fold_order if f in fold_avg["fold"].values]
            fold_avg = fold_avg.set_index("fold").loc[fold_order].reset_index()

            ax.plot(
                fold_avg["fold"],
                fold_avg["MAE"],
                marker="o",
                linewidth=2,
                label=param[:20],
            )

        ax.set_title(
            f"{model_type}: Top 5 Params Across Folds", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Walk-Forward Fold", fontsize=11)
        ax.set_ylabel("MAE ($) - lower = better", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Hyperparameter Stability Across Walk-Forward Folds",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(VIS_DIR, "top5_hyperparams.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved: top5_hyperparams.png")


# =============================================================================
# RUN ALL
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING VISUALIZATIONS (Walk-Forward + % Change Target)")
    print("=" * 60)

    plot_fold_comparison()
    plot_r2_comparison()
    plot_feature_importance()
    plot_walk_forward_progression()
    plot_predicted_vs_actual()
    plot_feature_correlation()
    plot_top5_hyperparams()

    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {VIS_DIR}")
    print("=" * 60)
