"""
Enhanced Visualizations - Split-wise, Feature Correlation, Cross-Validation
============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
VIS_DIR = r"D:\Internproj\Visualization"
DATA_PATH = r"D:\Internproj\new dataset\engineered_ml_dataset.csv"
RF_RESULTS = r"D:\Internproj\ML Models\Random Forest\results\all_rf_results.csv"
XGB_RESULTS = r"D:\Internproj\ML Models\XGBoost\results\all_xgb_results.csv"
EVAL_RESULTS = r"D:\Internproj\ML Models\Testing\results\all_evaluation_results.csv"

plt.style.use("seaborn-v0_8-whitegrid")


# =============================================================================
# 1. SPLIT-WISE TRAINING COMPARISON
# =============================================================================
def plot_split_wise_comparison():
    """Compare all models grouped by train/test split."""
    rf = pd.read_csv(RF_RESULTS)
    xgb = pd.read_csv(XGB_RESULTS)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    splits = ["80_20", "60_40", "40_60", "20_80"]
    split_labels = [
        "Train 80% / Test 20%",
        "Train 60% / Test 40%",
        "Train 40% / Test 60%",
        "Train 20% / Test 80%",
    ]

    for idx, (split, label) in enumerate(zip(splits, split_labels)):
        ax = axes[idx // 2, idx % 2]

        rf_split = rf[rf["split"] == split].nlargest(10, "MAE")  # worst 10
        xgb_split = xgb[xgb["split"] == split].nlargest(10, "MAE")

        # Actually get top 10 by best MAE (lowest)
        rf_split = rf[rf["split"] == split].nsmallest(10, "MAE")
        xgb_split = xgb[xgb["split"] == split].nsmallest(10, "MAE")

        rf_split = rf_split.copy()
        rf_split["model"] = "RF"
        xgb_split = xgb_split.copy()
        xgb_split["model"] = "XGB"

        combined = pd.concat([rf_split, xgb_split])
        combined["label"] = combined["label"].str[:15]

        colors = ["#2196F3" if m == "RF" else "#FF9800" for m in combined["model"]]
        bars = ax.barh(range(len(combined)), combined["MAE"], color=colors)
        ax.set_yticks(range(len(combined)))
        ax.set_yticklabels(
            [f"{r['model']}:{r['label']}" for _, r in combined.iterrows()], fontsize=8
        )
        ax.set_xlabel("MAE (lower = better)")
        ax.set_title(
            f"{label}\nTrain={list(combined[combined['model'] == 'RF']['train_size'])[0] if len(combined) else '?'} / Test={list(combined[combined['model'] == 'RF']['test_size'])[0] if len(combined) else '?'}"
        )
        ax.invert_yaxis()

    plt.suptitle("Top 10 Models per Train/Test Split", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(VIS_DIR, "split_wise_comparison.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved: split_wise_comparison.png")


# =============================================================================
# 2. FEATURE CORRELATION HEATMAP
# =============================================================================
def plot_feature_correlation():
    """Show correlation of features with target Price_USD."""
    df = pd.read_csv(DATA_PATH)

    # Select key numeric features
    key_features = [
        "Price_USD",
        "Route_Mean_wind_speed",
        "Route_Mean_slp",
        "Route_Max_cyclone_wind",
        "Route_Min_cyclone_dist",
        "Route_cyclone_active",
        "Dest_LSCI",
        "Origin_LSCI",
        "Route_Mean_LSCI",
        "SWI",
        "Coastal_Threat",
        "Congestion_Risk",
        "Price_Momentum_7d",
        "Price_Momentum_30d",
        "Price_Volatility_7d",
        "Price_Volatility_30d",
        "Price_Zscore_30d",
        "Rolling_7d_Cyclone_Days",
        "LSCI_30d_Delta",
        "Is_Peak_Season",
        "Month",
    ]

    available = [f for f in key_features if f in df.columns]
    corr_matrix = df[available].corr()

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Full correlation heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=axes[0],
        vmin=-1,
        vmax=1,
        annot_kws={"size": 7},
    )
    axes[0].set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    axes[0].tick_params(labelsize=8)

    # Correlation with target only
    target_corr = corr_matrix["Price_USD"].drop("Price_USD").sort_values()
    colors = ["#E53935" if v < 0 else "#43A047" for v in target_corr]
    axes[1].barh(range(len(target_corr)), target_corr.values, color=colors)
    axes[1].set_yticks(range(len(target_corr)))
    axes[1].set_yticklabels(target_corr.index, fontsize=9)
    axes[1].set_xlabel("Correlation with Price_USD")
    axes[1].set_title("Feature Correlation with Target", fontsize=14, fontweight="bold")
    axes[1].axvline(x=0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(
        os.path.join(VIS_DIR, "feature_correlation.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved: feature_correlation.png")


# =============================================================================
# 3. CROSS-VALIDATION PERFORMANCE BY SPLIT
# =============================================================================
def plot_cross_validation():
    """Show how each model performs across different splits (CV-style)."""
    eval_df = pd.read_csv(EVAL_RESULTS)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # RF cross-split performance
    rf = eval_df[eval_df["model_type"] == "RandomForest"]
    top_rf_params = rf.groupby("param_label")["MAE"].mean().nsmallest(5).index

    for param in top_rf_params:
        rf_param = rf[rf["param_label"] == param]
        axes[0].plot(
            rf_param["split"],
            rf_param["MAE"],
            marker="o",
            linewidth=2,
            label=param[:20],
        )

    axes[0].set_title("RF: Top 5 Models Across Splits", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Train/Test Split")
    axes[0].set_ylabel("MAE")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, alpha=0.3)

    # XGB cross-split performance
    xgb = eval_df[eval_df["model_type"] == "XGBoost"]
    top_xgb_params = xgb.groupby("param_label")["MAE"].mean().nsmallest(5).index

    for param in top_xgb_params:
        xgb_param = xgb[xgb["param_label"] == param]
        axes[1].plot(
            xgb_param["split"],
            xgb_param["MAE"],
            marker="s",
            linewidth=2,
            label=param[:20],
        )

    axes[1].set_title(
        "XGBoost: Top 5 Models Across Splits", fontsize=14, fontweight="bold"
    )
    axes[1].set_xlabel("Train/Test Split")
    axes[1].set_ylabel("MAE")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        "Cross-Validation: Model Stability Across Train/Test Splits",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(VIS_DIR, "cross_validation_performance.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved: cross_validation_performance.png")


# =============================================================================
# 4. TRAIN SIZE vs MAE SCATTER
# =============================================================================
def plot_train_size_vs_mae():
    """Show relationship between training data size and model performance."""
    eval_df = pd.read_csv(EVAL_RESULTS)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for idx, model_type in enumerate(["RandomForest", "XGBoost"]):
        ax = axes[idx]
        subset = eval_df[eval_df["model_type"] == model_type]

        # Map splits to approximate train sizes
        split_sizes = {"20_80": 1006, "40_60": 2496, "60_40": 4500, "80_20": 6520}
        subset = subset.copy()
        subset["train_size"] = subset["split"].map(split_sizes)

        splits = (
            subset.groupby("split")
            .agg(
                train_size=("train_size", "first"),
                avg_MAE=("MAE", "mean"),
                best_MAE=("MAE", "min"),
            )
            .reset_index()
        )

        # Individual points
        colors = {
            "80_20": "#2196F3",
            "60_40": "#4CAF50",
            "40_60": "#FF9800",
            "20_80": "#F44336",
        }
        for _, row in subset.iterrows():
            ax.scatter(
                row["train_size"],
                row["MAE"],
                alpha=0.3,
                s=30,
                color=colors.get(row["split"], "gray"),
            )

        # Best per split
        ax.plot(
            splits["train_size"],
            splits["best_MAE"],
            "k--",
            linewidth=2,
            label="Best MAE",
        )
        ax.scatter(
            splits["train_size"],
            splits["best_MAE"],
            s=100,
            c="red",
            zorder=5,
            edgecolors="black",
            linewidth=2,
        )

        for _, row in splits.iterrows():
            ax.annotate(
                row["split"],
                (row["train_size"], row["best_MAE"]),
                textcoords="offset points",
                xytext=(10, 10),
                fontsize=9,
            )

        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("MAE (lower = better)")
        ax.set_title(
            f"{model_type}: Training Size vs Performance",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Does More Training Data Help?", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(VIS_DIR, "train_size_vs_mae.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved: train_size_vs_mae.png")


# =============================================================================
# 5. FEATURE IMPORTANCE (from best model)
# =============================================================================
def plot_feature_importance():
    """Extract and plot feature importance from best XGBoost model."""
    import xgboost as xgb
    import pickle

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    # XGBoost feature importance
    try:
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(
            r"D:\Internproj\ML Models\XGBoost\xgb_20_80_conservative.json"
        )

        importances = xgb_model.feature_importances_
        feature_names = xgb_model.get_booster().feature_names
        if feature_names is None:
            # Need to get from data
            df = pd.read_csv(DATA_PATH)
            df = df.drop(columns=["Date", "Route", "Price_USD"], errors="ignore")
            feature_names = list(df.columns)

        idx = np.argsort(importances)[-20:]  # Top 20
        axes[0].barh(range(len(idx)), importances[idx], color="#FF9800")
        axes[0].set_yticks(range(len(idx)))
        axes[0].set_yticklabels([feature_names[i] for i in idx], fontsize=8)
        axes[0].set_xlabel("Feature Importance")
        axes[0].set_title("XGBoost: Top 20 Features", fontsize=14, fontweight="bold")
    except Exception as e:
        axes[0].text(0.5, 0.5, f"Error loading model:\n{e}", ha="center", va="center")

    # Random Forest feature importance
    try:
        with open(
            r"D:\Internproj\ML Models\Random Forest\rf_20_80_preprune_leaf8.pkl", "rb"
        ) as f:
            rf_model = pickle.load(f)

        importances = rf_model.feature_importances_
        df = pd.read_csv(DATA_PATH)
        feature_names = [
            c for c in df.columns if c not in ["Date", "Route", "Price_USD"]
        ]

        idx = np.argsort(importances)[-20:]  # Top 20
        axes[1].barh(range(len(idx)), importances[idx], color="#2196F3")
        axes[1].set_yticks(range(len(idx)))
        axes[1].set_yticklabels([feature_names[i] for i in idx], fontsize=8)
        axes[1].set_xlabel("Feature Importance")
        axes[1].set_title(
            "Random Forest: Top 20 Features", fontsize=14, fontweight="bold"
        )
    except Exception as e:
        axes[1].text(0.5, 0.5, f"Error loading model:\n{e}", ha="center", va="center")

    plt.suptitle("What Features Actually Matter?", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(VIS_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved: feature_importance.png")


# =============================================================================
# 6. TRAIN/TEST SPLIT DISTRIBUTION
# =============================================================================
def plot_split_distribution():
    """Show data distribution across train/test splits."""
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    splits = {
        "80_20": {"train": (2015, 2022), "test": (2023, 2026)},
        "60_40": {"train": (2015, 2020), "test": (2021, 2026)},
        "40_60": {"train": (2015, 2018), "test": (2019, 2026)},
        "20_80": {"train": (2015, 2016), "test": (2017, 2026)},
    }

    for idx, (split_name, bounds) in enumerate(splits.items()):
        ax = axes[idx // 2, idx % 2]

        train_start, train_end = bounds["train"]
        test_start, test_end = bounds["test"]

        train_data = df[(df["Year"] >= train_start) & (df["Year"] <= train_end)]
        test_data = df[(df["Year"] >= test_start) & (df["Year"] <= test_end)]

        years = sorted(df["Year"].unique())
        counts = [len(df[df["Year"] == y]) for y in years]
        colors = [
            "#4CAF50" if train_start <= y <= train_end else "#F44336" for y in years
        ]

        ax.bar(years, counts, color=colors, alpha=0.8)
        ax.axvline(x=test_start - 0.5, color="black", linewidth=2, linestyle="--")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Samples")
        ax.set_title(
            f"{split_name}: Train({train_start}-{train_end}) / Test({test_start}-{test_end})\n"
            f"Train={len(train_data)} ({100 * len(train_data) / len(df):.0f}%) / Test={len(test_data)} ({100 * len(test_data) / len(df):.0f}%)",
            fontsize=11,
        )
        ax.legend(["Train/Test Split", "Train", "Test"], loc="upper right")

        # Color legend manually
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#4CAF50", label="Train"),
            Patch(facecolor="#F44336", label="Test"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

    plt.suptitle(
        "Train/Test Split Distribution Over Time", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(VIS_DIR, "split_distribution.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved: split_distribution.png")


# =============================================================================
# RUN ALL
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING ENHANCED VISUALIZATIONS")
    print("=" * 60)

    plot_split_wise_comparison()
    plot_feature_correlation()
    plot_cross_validation()
    plot_train_size_vs_mae()
    plot_feature_importance()
    plot_split_distribution()

    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {VIS_DIR}")
    print("=" * 60)
