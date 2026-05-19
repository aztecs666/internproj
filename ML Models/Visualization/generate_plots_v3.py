"""
Visualizations for Current Top 5 Models
=========================================
Generates plots focused on the top 5 RF and top 5 XGB models.
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import xgboost as xgb_lib

# Paths
VIS_DIR = r"D:\Internproj\Visualization"
DATA_PATH = r"D:\Internproj\new dataset\engineered_ml_dataset.csv"
EVAL_RESULTS = r"D:\Internproj\ML Models\Testing\results\all_evaluation_results.csv"
RF_RESULTS = r"D:\Internproj\ML Models\Random Forest\results\all_rf_results.csv"
XGB_RESULTS = r"D:\Internproj\ML Models\XGBoost\results\all_xgb_results.csv"
PREDICTIONS_CACHE = r"D:\Internproj\ML Models\Testing\results\predictions_cache.pkl"

os.makedirs(VIS_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")


# =============================================================================
# 1. TOP 5 MODELS: MAE BAR CHART
# =============================================================================
def plot_top5_mae():
    eval_df = pd.read_csv(EVAL_RESULTS)

    # Top 5 overall
    top5 = eval_df.nsmallest(5, "MAE")

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = []
    for _, row in top5.iterrows():
        if row["model_type"] == "RandomForest":
            colors.append("#2196F3")
        else:
            colors.append("#FF9800")

    bars = ax.barh(
        range(len(top5)), top5["MAE"], color=colors, edgecolor="black", linewidth=0.5
    )

    for i, (_, row) in enumerate(top5.iterrows()):
        ax.text(
            row["MAE"] + 5,
            i,
            f"${row['MAE']:.0f}  (R2={row['R2']:.4f})",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_yticks(range(len(top5)))
    ax.set_yticklabels(
        [
            f"{r['model_type']}: {r['param_label']} ({r['split']})"
            for _, r in top5.iterrows()
        ],
        fontsize=10,
    )
    ax.set_xlabel("MAE ($) — lower is better", fontsize=12)
    ax.set_title("Top 5 Models Overall (Best MAE)", fontsize=16, fontweight="bold")
    ax.invert_yaxis()

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2196F3", label="Random Forest"),
        Patch(facecolor="#FF9800", label="XGBoost"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "01_top5_mae.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 01_top5_mae.png")


# =============================================================================
# 2. TOP 5 RF vs TOP 5 XGB HEAD-TO-HEAD
# =============================================================================
def plot_rf_vs_xgb_top5():
    eval_df = pd.read_csv(EVAL_RESULTS)

    top5_rf = eval_df[eval_df["model_type"] == "RandomForest"].nsmallest(5, "MAE")
    top5_xgb = eval_df[eval_df["model_type"] == "XGBoost"].nsmallest(5, "MAE")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # RF Top 5
    ax = axes[0]
    ax.barh(
        range(len(top5_rf)),
        top5_rf["MAE"],
        color="#2196F3",
        edgecolor="black",
        linewidth=0.5,
    )
    for i, (_, row) in enumerate(top5_rf.iterrows()):
        ax.text(row["MAE"] + 2, i, f"${row['MAE']:.0f}", va="center", fontsize=9)
    ax.set_yticks(range(len(top5_rf)))
    ax.set_yticklabels([r["param_label"] for _, r in top5_rf.iterrows()], fontsize=9)
    ax.set_xlabel("MAE ($)")
    ax.set_title("Random Forest: Top 5", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    # XGB Top 5
    ax = axes[1]
    ax.barh(
        range(len(top5_xgb)),
        top5_xgb["MAE"],
        color="#FF9800",
        edgecolor="black",
        linewidth=0.5,
    )
    for i, (_, row) in enumerate(top5_xgb.iterrows()):
        ax.text(row["MAE"] + 2, i, f"${row['MAE']:.0f}", va="center", fontsize=9)
    ax.set_yticks(range(len(top5_xgb)))
    ax.set_yticklabels([r["param_label"] for _, r in top5_xgb.iterrows()], fontsize=9)
    ax.set_xlabel("MAE ($)")
    ax.set_title("XGBoost: Top 5", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    plt.suptitle(
        "Top 5 Models: Random Forest vs XGBoost", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(VIS_DIR, "02_rf_vs_xgb_top5.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved: 02_rf_vs_xgb_top5.png")


# =============================================================================
# 3. SPLIT-WISE PERFORMANCE (all 4 splits, top models)
# =============================================================================
def plot_split_comparison():
    eval_df = pd.read_csv(EVAL_RESULTS)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    splits = ["80_20", "60_40", "40_60", "20_80"]
    split_labels = [
        "Train 80% / Test 20%",
        "Train 60% / Test 40%",
        "Train 40% / Test 60%",
        "Train 20% / Test 80%",
    ]

    for idx, (split, label) in enumerate(zip(splits, split_labels)):
        ax = axes[idx // 2, idx % 2]

        rf_split = eval_df[
            (eval_df["model_type"] == "RandomForest") & (eval_df["split"] == split)
        ].nsmallest(5, "MAE")
        xgb_split = eval_df[
            (eval_df["model_type"] == "XGBoost") & (eval_df["split"] == split)
        ].nsmallest(5, "MAE")

        rf_split = rf_split.copy()
        rf_split["model"] = "RF"
        xgb_split = xgb_split.copy()
        xgb_split["model"] = "XGB"

        combined = pd.concat([rf_split, xgb_split])
        colors = ["#2196F3" if m == "RF" else "#FF9800" for m in combined["model"]]

        ax.barh(
            range(len(combined)),
            combined["MAE"],
            color=colors,
            edgecolor="black",
            linewidth=0.3,
        )
        ax.set_yticks(range(len(combined)))
        ax.set_yticklabels(
            [f"{r['model']}:{r['param_label'][:18]}" for _, r in combined.iterrows()],
            fontsize=8,
        )
        ax.set_xlabel("MAE ($)")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.invert_yaxis()

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2196F3", label="Random Forest"),
        Patch(facecolor="#FF9800", label="XGBoost"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=12)

    plt.suptitle("Top Models Per Train/Test Split", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(VIS_DIR, "03_split_comparison.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved: 03_split_comparison.png")


# =============================================================================
# 4. ACTUAL vs PREDICTED (best XGB + best RF)
# =============================================================================
def plot_actual_vs_predicted():
    try:
        with open(PREDICTIONS_CACHE, "rb") as f:
            predictions = pickle.load(f)
    except FileNotFoundError:
        print("Predictions cache not found. Skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Best XGB: sub06_col10 on 80_20
    xgb_key = "xgb_80_20_sub06_col10"
    if xgb_key in predictions:
        pred = predictions[xgb_key]
        y_true = pred["y_true"]
        y_pred = pred["y_pred"]

        axes[0].scatter(y_true, y_pred, alpha=0.2, s=8, color="#FF9800")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        axes[0].plot(lims, lims, "r--", linewidth=2, label="Perfect prediction")
        axes[0].set_xlabel("Actual Price ($)", fontsize=11)
        axes[0].set_ylabel("Predicted Price ($)", fontsize=11)
        axes[0].set_title(
            "Best XGBoost: sub06_col10 (80/20)\nMAE=$217, R2=0.97",
            fontsize=12,
            fontweight="bold",
        )
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Best RF: preprune_leaf8 on 80_20
    rf_key = "rf_80_20_preprune_leaf8"
    if rf_key in predictions:
        pred = predictions[rf_key]
        y_true = pred["y_true"]
        y_pred = pred["y_pred"]

        axes[1].scatter(y_true, y_pred, alpha=0.2, s=8, color="#2196F3")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        axes[1].plot(lims, lims, "r--", linewidth=2, label="Perfect prediction")
        axes[1].set_xlabel("Actual Price ($)", fontsize=11)
        axes[1].set_ylabel("Predicted Price ($)", fontsize=11)
        axes[1].set_title(
            "Best RF: preprune_leaf8 (80/20)\nMAE=$260, R2=0.95",
            fontsize=12,
            fontweight="bold",
        )
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.suptitle("Predicted vs Actual Container Prices", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(VIS_DIR, "04_actual_vs_predicted.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved: 04_actual_vs_predicted.png")


# =============================================================================
# 5. FEATURE IMPORTANCE (best XGB + best RF)
# =============================================================================
def plot_feature_importance():
    eval_df = pd.read_csv(EVAL_RESULTS)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Best XGB
    try:
        xgb_model = xgb_lib.XGBRegressor()
        xgb_model.load_model(
            r"D:\Internproj\ML Models\XGBoost\xgb_80_20_sub06_col10.json"
        )
        importances = xgb_model.feature_importances_
        feature_names = xgb_model.get_booster().feature_names
        if feature_names is None:
            df = pd.read_csv(DATA_PATH)
            df = df.drop(columns=["Date", "Route", "Price_USD"], errors="ignore")
            feature_names = list(df.columns)

        idx = np.argsort(importances)[-20:]
        axes[0].barh(
            range(len(idx)),
            importances[idx],
            color="#FF9800",
            edgecolor="black",
            linewidth=0.3,
        )
        axes[0].set_yticks(range(len(idx)))
        axes[0].set_yticklabels([feature_names[i] for i in idx], fontsize=8)
        axes[0].set_xlabel("Feature Importance")
        axes[0].set_title(
            "XGBoost: Top 20 Features\n(sub06_col10, 80/20 split)",
            fontsize=12,
            fontweight="bold",
        )
    except Exception as e:
        axes[0].text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center")

    # Best RF
    try:
        with open(
            r"D:\Internproj\ML Models\Random Forest\rf_80_20_preprune_leaf8.pkl", "rb"
        ) as f:
            rf_model = pickle.load(f)

        importances = rf_model.feature_importances_
        df = pd.read_csv(DATA_PATH)
        feature_names = [
            c for c in df.columns if c not in ["Date", "Route", "Price_USD"]
        ]

        idx = np.argsort(importances)[-20:]
        axes[1].barh(
            range(len(idx)),
            importances[idx],
            color="#2196F3",
            edgecolor="black",
            linewidth=0.3,
        )
        axes[1].set_yticks(range(len(idx)))
        axes[1].set_yticklabels([feature_names[i] for i in idx], fontsize=8)
        axes[1].set_xlabel("Feature Importance")
        axes[1].set_title(
            "Random Forest: Top 20 Features\n(preprune_leaf8, 80/20 split)",
            fontsize=12,
            fontweight="bold",
        )
    except Exception as e:
        axes[1].text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center")

    plt.suptitle("What Features Matter Most?", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(VIS_DIR, "05_feature_importance.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved: 05_feature_importance.png")


# =============================================================================
# 6. FEATURE CORRELATION WITH TARGET
# =============================================================================
def plot_feature_correlation():
    df = pd.read_csv(DATA_PATH)

    key_features = [
        "Price_USD",
        "Price_Lag_7d",
        "Price_Lag_30d",
        "Price_Momentum_7d",
        "Price_Momentum_30d",
        "Price_Volatility_7d",
        "Price_Volatility_30d",
        "Price_Zscore_30d",
        "Route_Mean_wind_speed",
        "Route_Max_cyclone_wind",
        "Route_Min_cyclone_dist",
        "Cyclone_Wind_Lag7d",
        "Cyclone_Wind_Lag30d",
        "Cyclone_Wind_Forecast3d",
        "SWI",
        "Coastal_Threat",
        "Congestion_Risk",
        "Dest_LSCI",
        "Month",
        "Is_Peak_Season",
    ]

    available = [f for f in key_features if f in df.columns]
    corr = df[available].corr()["Price_USD"].drop("Price_USD").sort_values()

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#E53935" if v < 0 else "#43A047" for v in corr]
    ax.barh(
        range(len(corr)), corr.values, color=colors, edgecolor="black", linewidth=0.3
    )
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index, fontsize=9)
    ax.set_xlabel("Correlation with Price_USD", fontsize=12)
    ax.set_title(
        "Feature Correlation with Target Price", fontsize=14, fontweight="bold"
    )
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(VIS_DIR, "06_feature_correlation.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved: 06_feature_correlation.png")


# =============================================================================
# 7. SPLIT DISTRIBUTION (train/test by year)
# =============================================================================
def plot_split_distribution():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year

    splits = {
        "80/20": {"train": (2015, 2022), "test": (2023, 2026)},
        "60/40": {"train": (2015, 2020), "test": (2021, 2026)},
        "40/60": {"train": (2015, 2018), "test": (2019, 2026)},
        "20/80": {"train": (2015, 2016), "test": (2017, 2026)},
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for idx, (split_name, bounds) in enumerate(splits.items()):
        ax = axes[idx // 2, idx % 2]
        train_start, train_end = bounds["train"]
        test_start, test_end = bounds["test"]

        years = sorted(df["Year"].unique())
        counts = [len(df[df["Year"] == y]) for y in years]
        colors = [
            "#4CAF50" if train_start <= y <= train_end else "#F44336" for y in years
        ]

        ax.bar(years, counts, color=colors, alpha=0.8, edgecolor="black", linewidth=0.3)
        ax.axvline(x=test_start - 0.5, color="black", linewidth=2, linestyle="--")

        train_n = len(df[(df["Year"] >= train_start) & (df["Year"] <= train_end)])
        test_n = len(df[(df["Year"] >= test_start) & (df["Year"] <= test_end)])

        ax.set_xlabel("Year")
        ax.set_ylabel("Samples")
        ax.set_title(
            f"{split_name}: Train({train_start}-{train_end}) / Test({test_start}-{test_end})\n"
            f"Train={train_n} / Test={test_n}",
            fontsize=11,
            fontweight="bold",
        )

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#4CAF50", label="Train"),
            Patch(facecolor="#F44336", label="Test"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

    plt.suptitle("Train/Test Split Distribution", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(VIS_DIR, "07_split_distribution.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved: 07_split_distribution.png")


# =============================================================================
# 8. TOP 5 CROSS-SPLIT PERFORMANCE
# =============================================================================
def plot_top5_cross_split():
    eval_df = pd.read_csv(EVAL_RESULTS)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for idx, model_type in enumerate(["RandomForest", "XGBoost"]):
        ax = axes[idx]
        subset = eval_df[eval_df["model_type"] == model_type]
        top5_params = subset.groupby("param_label")["MAE"].mean().nsmallest(5).index

        for param in top5_params:
            param_data = subset[subset["param_label"] == param]
            split_avg = param_data.groupby("split")["MAE"].mean().reset_index()

            # Sort splits
            split_order = ["80_20", "60_40", "40_60", "20_80"]
            split_avg = (
                split_avg.set_index("split")
                .reindex([s for s in split_order if s in split_avg.index])
                .reset_index()
            )

            ax.plot(
                split_avg["split"],
                split_avg["MAE"],
                marker="o",
                linewidth=2,
                label=param[:20],
            )

        ax.set_title(
            f"{model_type}: Top 5 Across Splits", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Train/Test Split")
        ax.set_ylabel("MAE ($)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "How Do Best Params Perform Across Different Splits?",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(VIS_DIR, "08_top5_cross_split.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved: 08_top5_cross_split.png")


# =============================================================================
# RUN ALL
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING VISUALIZATIONS (Top 5 Models)")
    print("=" * 60)

    plot_top5_mae()
    plot_rf_vs_xgb_top5()
    plot_split_comparison()
    plot_actual_vs_predicted()
    plot_feature_importance()
    plot_feature_correlation()
    plot_split_distribution()
    plot_top5_cross_split()

    print(f"\n{'=' * 60}")
    print(f"All visualizations saved to: {VIS_DIR}")
    print("=" * 60)
