"""
Comprehensive Visualization for ML Model Comparison
=====================================================
Generates matplotlib plots for all RF and XGBoost model comparisons.
Saves all plots to the Visualization folder.
"""

import sys
import os
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTING_DIR = os.path.join(BASE_DIR, "Testing")
RESULTS_DIR = os.path.join(TESTING_DIR, "results")
VIS_DIR = os.path.join(BASE_DIR, "Visualization")
os.makedirs(VIS_DIR, exist_ok=True)

# Style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


# =============================================================================
# Load Data
# =============================================================================
def load_results():
    results_path = os.path.join(RESULTS_DIR, "all_evaluation_results.csv")
    preds_path = os.path.join(RESULTS_DIR, "predictions_cache.pkl")
    summary_path = os.path.join(RESULTS_DIR, "evaluation_summary.json")

    results_df = pd.read_csv(results_path)
    with open(preds_path, "rb") as f:
        predictions = pickle.load(f)
    with open(summary_path, "r") as f:
        summary = json.load(f)
    return results_df, predictions, summary


# =============================================================================
# Plot 1: MAE Comparison by Split Method (Grouped Bar Chart)
# =============================================================================
def plot_mae_by_split(results_df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for idx, model_type in enumerate(["RandomForest", "XGBoost"]):
        ax = axes[idx]
        type_df = results_df[results_df["model_type"] == model_type]

        # Pivot: rows=param_label, cols=split
        pivot = type_df.pivot_table(index="param_label", columns="split", values="MAE")

        # Sort by best overall MAE
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

        # Plot top 10 only for readability
        pivot.head(10).plot(kind="barh", ax=ax, width=0.8)
        ax.set_xlabel("MAE (lower is better)")
        ax.set_title(f"{model_type}: MAE by Split Method (Top 10 Params)")
        ax.legend(title="Split", loc="lower right")
        ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(VIS_DIR, "mae_by_split_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# Plot 2: Best Split Method Heatmap
# =============================================================================
def plot_split_heatmap(results_df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for idx, model_type in enumerate(["RandomForest", "XGBoost"]):
        ax = axes[idx]
        type_df = results_df[results_df["model_type"] == model_type]

        # Average MAE by split
        split_avg = type_df.groupby("split")["MAE"].mean()

        # Create simple bar chart
        colors = sns.color_palette("RdYlGn_r", len(split_avg))
        bars = ax.bar(split_avg.index, split_avg.values, color=colors)
        ax.set_ylabel("Average MAE")
        ax.set_title(f"{model_type}: Avg MAE by Split Method")
        ax.set_xlabel("Train/Test Split")

        # Add value labels
        for bar, val in zip(bars, split_avg.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    path = os.path.join(VIS_DIR, "best_split_method_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# Plot 3: Top 5 Models Actual vs Predicted
# =============================================================================
def plot_top5_actual_vs_predicted(predictions, summary):
    top5 = summary["top5_overall"]
    fig, axes = plt.subplots(len(top5), 1, figsize=(14, 4 * len(top5)))
    if len(top5) == 1:
        axes = [axes]

    for idx, model_info in enumerate(top5):
        ax = axes[idx]
        model_type = "rf" if model_info["model_type"] == "RandomForest" else "xgb"
        key = f"{model_type}_{model_info['split']}_{model_info['param_label']}"

        if key not in predictions:
            ax.text(0.5, 0.5, f"Data not found for {key}", ha="center", va="center")
            continue

        pred_data = predictions[key]
        y_true = pred_data["y_true"]
        y_pred = pred_data["y_pred"]

        sample_size = min(200, len(y_true))
        x_axis = range(sample_size)

        ax.plot(
            x_axis,
            y_true[:sample_size],
            label="Actual",
            color="black",
            linewidth=1.5,
            alpha=0.8,
        )
        ax.plot(
            x_axis,
            y_pred[:sample_size],
            label="Predicted",
            color="red",
            linewidth=1.2,
            linestyle="--",
            alpha=0.8,
        )

        mae = model_info["MAE"]
        ax.set_title(
            f"#{idx + 1}: {model_info['model_type']} | {model_info['param_label']} | Split: {model_info['split']} | MAE: {mae:.2f}"
        )
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Price USD")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Top 5 Models: Actual vs Predicted (First 200 Test Samples)",
        fontsize=14,
        y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(VIS_DIR, "top5_actual_vs_predicted.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# Plot 4: RF Pre-Pruning vs Post-Pruning Performance
# =============================================================================
def plot_rf_pruning_comparison(results_df):
    rf_df = results_df[results_df["model_type"] == "RandomForest"].copy()

    rf_df["prune_type"] = rf_df["param_label"].apply(
        lambda x: "Pre-Pruning"
        if "preprune" in x
        else ("Post-Pruning" if "postprune" in x else "Combined")
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, prune_type in enumerate(["Pre-Pruning", "Post-Pruning", "Combined"]):
        ax = axes[idx]
        subset = rf_df[rf_df["prune_type"] == prune_type]

        if subset.empty:
            ax.text(0.5, 0.5, f"No {prune_type} models", ha="center", va="center")
            continue

        # Group by label and split
        pivot = subset.pivot_table(index="param_label", columns="split", values="MAE")
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

        pivot.plot(kind="barh", ax=ax, width=0.8)
        ax.set_xlabel("MAE")
        ax.set_title(f"RF {prune_type}")
        ax.legend(title="Split", fontsize=8)
        ax.invert_yaxis()

    plt.suptitle("Random Forest: Pre-Pruning vs Post-Pruning vs Combined", fontsize=14)
    plt.tight_layout()
    path = os.path.join(VIS_DIR, "rf_pruning_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# Plot 5: XGBoost Learning Rate Impact
# =============================================================================
def plot_xgb_lr_impact(results_df):
    xgb_df = results_df[results_df["model_type"] == "XGBoost"].copy()

    # Extract learning rate from label
    def extract_lr(label):
        if "lr001" in label:
            return 0.01
        if "lr005" in label:
            return 0.05
        if "lr01" in label:
            return 0.1
        if "lr02" in label:
            return 0.2
        if "lr03" in label:
            return 0.3
        if "conservative" in label:
            return 0.05
        if "aggressive" in label:
            return 0.2
        if "balanced" in label:
            return 0.1
        if "deep_slow" in label:
            return 0.01
        return None

    xgb_df["learning_rate"] = xgb_df["param_label"].apply(extract_lr)
    xgb_df = xgb_df.dropna(subset=["learning_rate"])

    fig, ax = plt.subplots(figsize=(12, 6))

    for split in xgb_df["split"].unique():
        subset = xgb_df[xgb_df["split"] == split].groupby("learning_rate")["MAE"].mean()
        ax.plot(
            subset.index,
            subset.values,
            marker="o",
            linewidth=2,
            label=f"Split: {split}",
        )

    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Average MAE")
    ax.set_title("XGBoost: Learning Rate Impact on Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    path = os.path.join(VIS_DIR, "xgb_learning_rate_impact.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# Plot 6: RF vs XGBoost Head-to-Head
# =============================================================================
def plot_rf_vs_xgb(results_df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ["MAE", "RMSE", "R2"]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        rf_avg = (
            results_df[results_df["model_type"] == "RandomForest"]
            .groupby("split")[metric]
            .mean()
        )
        xgb_avg = (
            results_df[results_df["model_type"] == "XGBoost"]
            .groupby("split")[metric]
            .mean()
        )

        x = np.arange(len(rf_avg))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2,
            rf_avg.values,
            width,
            label="Random Forest",
            color="steelblue",
        )
        bars2 = ax.bar(
            x + width / 2, xgb_avg.values, width, label="XGBoost", color="darkorange"
        )

        ax.set_xlabel("Split Method")
        ax.set_ylabel(metric)
        ax.set_title(f"RF vs XGBoost: {metric}")
        ax.set_xticks(x)
        ax.set_xticklabels(rf_avg.index, rotation=45)
        ax.legend()

    plt.suptitle("Random Forest vs XGBoost: Head-to-Head Comparison", fontsize=14)
    plt.tight_layout()
    path = os.path.join(VIS_DIR, "rf_vs_xgb_head_to_head.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# Plot 7: Training Time Comparison
# =============================================================================
def plot_training_time(results_df):
    # Note: training_time not in evaluation results, but we can check RF/XGB results
    rf_results_path = os.path.join(
        BASE_DIR, "Random Forest", "results", "all_rf_results.csv"
    )
    xgb_results_path = os.path.join(
        BASE_DIR, "XGBoost", "results", "all_xgb_results.csv"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if os.path.exists(rf_results_path):
        rf_res = pd.read_csv(rf_results_path)
        pivot = rf_res.pivot_table(
            index="label", columns="split", values="train_time_sec"
        )
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index].head(10)
        pivot.plot(kind="barh", ax=axes[0], width=0.8)
        axes[0].set_xlabel("Training Time (seconds)")
        axes[0].set_title("Random Forest: Training Time")
        axes[0].legend(title="Split", fontsize=8)
        axes[0].invert_yaxis()

    if os.path.exists(xgb_results_path):
        xgb_res = pd.read_csv(xgb_results_path)
        pivot = xgb_res.pivot_table(
            index="label", columns="split", values="train_time_sec"
        )
        pivot = pivot.loc[pivot.mean(axis=1).sort_values().index].head(10)
        pivot.plot(kind="barh", ax=axes[1], width=0.8)
        axes[1].set_xlabel("Training Time (seconds)")
        axes[1].set_title("XGBoost: Training Time")
        axes[1].legend(title="Split", fontsize=8)
        axes[1].invert_yaxis()

    plt.suptitle("Training Time Comparison", fontsize=14)
    plt.tight_layout()
    path = os.path.join(VIS_DIR, "training_time_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# Plot 8: Residual Distribution for Top Models
# =============================================================================
def plot_residual_distribution(predictions, summary):
    top5 = summary["top5_overall"]
    fig, axes = plt.subplots(1, len(top5), figsize=(4 * len(top5), 5))
    if len(top5) == 1:
        axes = [axes]

    for idx, model_info in enumerate(top5):
        ax = axes[idx]
        model_type = "rf" if model_info["model_type"] == "RandomForest" else "xgb"
        key = f"{model_type}_{model_info['split']}_{model_info['param_label']}"

        if key not in predictions:
            continue

        pred_data = predictions[key]
        residuals = pred_data["y_true"] - pred_data["y_pred"]

        ax.hist(residuals, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", linewidth=2)
        ax.set_xlabel("Residual (Actual - Predicted)")
        ax.set_ylabel("Count")
        ax.set_title(
            f"#{idx + 1}: {model_info['param_label']}\nMAE={model_info['MAE']:.2f}"
        )

    plt.suptitle("Residual Distribution: Top 5 Models", fontsize=14)
    plt.tight_layout()
    path = os.path.join(VIS_DIR, "residual_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# Plot 9: Model Count & Coverage
# =============================================================================
def plot_model_coverage(results_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Count by model type and split
    counts = results_df.groupby(["model_type", "split"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", ax=axes[0], width=0.8)
    axes[0].set_title("Number of Models Trained")
    axes[0].set_ylabel("Count")
    axes[0].set_xlabel("Model Type")
    axes[0].legend(title="Split")

    # Best model per split
    best_per_split = results_df.loc[results_df.groupby("split")["MAE"].idxmin()]
    best_per_split = best_per_split[["split", "model_type", "param_label", "MAE"]]
    axes[1].axis("off")
    table = axes[1].table(
        cellText=best_per_split.values,
        colLabels=best_per_split.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    axes[1].set_title("Best Model per Split")

    plt.suptitle("Model Training Coverage", fontsize=14)
    plt.tight_layout()
    path = os.path.join(VIS_DIR, "model_coverage.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# Plot 10: Comprehensive Summary Dashboard
# =============================================================================
def plot_summary_dashboard(results_df, summary):
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

    # 1. Best Split Method
    ax1 = fig.add_subplot(gs[0, 0])
    split_avg = results_df.groupby("split")["MAE"].mean().sort_values()
    colors = ["green" if i == 0 else "steelblue" for i in range(len(split_avg))]
    ax1.bar(split_avg.index, split_avg.values, color=colors)
    ax1.set_title("Best Split Method (Avg MAE)")
    ax1.set_ylabel("MAE")
    for i, (idx, val) in enumerate(split_avg.items()):
        ax1.text(i, val + 0.5, f"{val:.1f}", ha="center", fontsize=9)

    # 2. RF vs XGB
    ax2 = fig.add_subplot(gs[0, 1])
    type_avg = results_df.groupby("model_type")["MAE"].mean()
    ax2.bar(type_avg.index, type_avg.values, color=["steelblue", "darkorange"])
    ax2.set_title("RF vs XGBoost (Avg MAE)")
    ax2.set_ylabel("MAE")

    # 3. Top 5 Models
    ax3 = fig.add_subplot(gs[0, 2])
    top5 = summary["top5_overall"]
    labels = [
        f"{m['model_type'][:2]}_{m['split']}_{m['param_label'][:10]}" for m in top5
    ]
    maes = [m["MAE"] for m in top5]
    ax3.barh(labels, maes, color="teal")
    ax3.set_title("Top 5 Models by MAE")
    ax3.set_xlabel("MAE")
    ax3.invert_yaxis()

    # 4. MAE Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    rf_maes = results_df[results_df["model_type"] == "RandomForest"]["MAE"]
    xgb_maes = results_df[results_df["model_type"] == "XGBoost"]["MAE"]
    ax4.hist(rf_maes, bins=30, alpha=0.6, label="RF", color="steelblue")
    ax4.hist(xgb_maes, bins=30, alpha=0.6, label="XGB", color="darkorange")
    ax4.set_title("MAE Distribution")
    ax4.set_xlabel("MAE")
    ax4.legend()

    # 5. R2 by Split
    ax5 = fig.add_subplot(gs[1, 1])
    r2_by_split = results_df.groupby("split")["R2"].mean().sort_values(ascending=False)
    ax5.bar(r2_by_split.index, r2_by_split.values, color="teal")
    ax5.set_title("Avg R2 by Split")
    ax5.set_ylabel("R2")

    # 6. Summary Text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    summary_text = (
        f"OVERALL SUMMARY\n"
        f"{'=' * 30}\n"
        f"Total Models: {summary['total_models_evaluated']}\n"
        f"RF Models: {summary['rf_models']}\n"
        f"XGB Models: {summary['xgb_models']}\n\n"
        f"Best Split: {summary['best_overall_split']}\n"
        f"Best Split MAE: {summary['best_overall_split_mae']:.2f}\n\n"
        f"Best Model: {summary['best_overall_model']}\n"
        f"Best MAE: {summary['best_overall_mae']:.2f}"
    )
    ax6.text(
        0.1,
        0.9,
        summary_text,
        transform=ax6.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    # 7-9. Placeholder for more metrics
    for i in range(3):
        ax = fig.add_subplot(gs[2, i])
        ax.axis("off")

    plt.suptitle(
        "COMPREHENSIVE MODEL EVALUATION DASHBOARD", fontsize=16, fontweight="bold"
    )
    path = os.path.join(VIS_DIR, "summary_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# Main
# =============================================================================
def generate_all_visualizations():
    print("=" * 80)
    print("GENERATING ALL VISUALIZATIONS")
    print("=" * 80)

    results_df, predictions, summary = load_results()
    print(f"Loaded {len(results_df)} evaluation results")

    print(f"\n--- Generating Plots ---")
    plot_mae_by_split(results_df)
    plot_split_heatmap(results_df)
    plot_top5_actual_vs_predicted(predictions, summary)
    plot_rf_pruning_comparison(results_df)
    plot_xgb_lr_impact(results_df)
    plot_rf_vs_xgb(results_df)
    plot_training_time(results_df)
    plot_residual_distribution(predictions, summary)
    plot_model_coverage(results_df)
    plot_summary_dashboard(results_df, summary)

    print(f"\n{'=' * 80}")
    print(f"All visualizations saved to: {VIS_DIR}")
    print(f"Total plots generated: 10")


if __name__ == "__main__":
    generate_all_visualizations()
