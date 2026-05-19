"""
Correlation Analysis for Engineered ML Dataset
Analyzes feature correlations, multicollinearity, and target correlations.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Configuration
INPUT_FILE = r"d:\Internproj\new dataset\engineered_ml_dataset.csv"
OUTPUT_DIR = r"d:\Internproj\Analysis\correlation_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print("=" * 70)
print("CORRELATION ANALYSIS - ENGINEERED ML DATASET")
print("=" * 70)
df = pd.read_csv(INPUT_FILE)
df["Date"] = pd.to_datetime(df["Date"])

print(f"\nDataset Shape: {df.shape}")
print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Routes: {df['Route'].unique()}")

# Identify numeric columns (exclude Date, Route)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric Features: {len(numeric_cols)}")

# =========================================================================
# 1. CORRELATION WITH TARGET (Price_USD)
# =========================================================================
print("\n" + "=" * 70)
print("1. CORRELATION WITH TARGET (Price_USD)")
print("=" * 70)

correlations = (
    df[numeric_cols].corr()["Price_USD"].drop("Price_USD").sort_values(ascending=False)
)

print("\n--- TOP 20 POSITIVE CORRELATIONS ---")
print(correlations.head(20).to_string())

print("\n--- TOP 20 NEGATIVE CORRELATIONS ---")
print(correlations.tail(20).to_string())

# Features with significant correlation (|r| > 0.1)
sig_corr = correlations[abs(correlations) > 0.1]
print(f"\nFeatures with |r| > 0.1: {len(sig_corr)}")
print(sig_corr.to_string())

# Save to CSV
correlations.to_csv(
    os.path.join(OUTPUT_DIR, "target_correlations.csv"), header=["correlation"]
)

# =========================================================================
# 2. FULL CORRELATION MATRIX
# =========================================================================
print("\n" + "=" * 70)
print("2. CORRELATION MATRIX STATISTICS")
print("=" * 70)

corr_matrix = df[numeric_cols].corr()

# Statistics
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
all_corrs = upper_tri.values.flatten()
all_corrs = all_corrs[~np.isnan(all_corrs)]

print(f"\nCorrelation Statistics (off-diagonal):")
print(f"  Mean:   {np.mean(all_corrs):.4f}")
print(f"  Median: {np.median(all_corrs):.4f}")
print(f"  Std:    {np.std(all_corrs):.4f}")
print(f"  Min:    {np.min(all_corrs):.4f}")
print(f"  Max:    {np.max(all_corrs):.4f}")

# =========================================================================
# 3. MULTICOLLINEARITY ANALYSIS
# =========================================================================
print("\n" + "=" * 70)
print("3. MULTICOLLINEARITY ANALYSIS (|r| > 0.8)")
print("=" * 70)

high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.8:
            high_corr_pairs.append(
                {
                    "Feature_1": corr_matrix.columns[i],
                    "Feature_2": corr_matrix.columns[j],
                    "Correlation": corr_val,
                }
            )

high_corr_df = pd.DataFrame(high_corr_pairs).sort_values("Correlation", ascending=False)
print(f"\nTotal high-correlation pairs (|r| > 0.8): {len(high_corr_df)}")
print("\n--- TOP 30 HIGHLY CORRELATED PAIRS ---")
print(high_corr_df.head(30).to_string())

high_corr_df.to_csv(os.path.join(OUTPUT_DIR, "high_correlation_pairs.csv"), index=False)

# =========================================================================
# 4. FEATURE GROUP ANALYSIS
# =========================================================================
print("\n" + "=" * 70)
print("4. FEATURE GROUP ANALYSIS")
print("=" * 70)

# Categorize features
feature_groups = {
    "LSCI": [c for c in numeric_cols if "LSCI" in c],
    "Weather_Air": [c for c in numeric_cols if c.startswith("air_")],
    "Weather_SLP": [
        c for c in numeric_cols if c.startswith("slp_") and not c.startswith("cyclone_")
    ],
    "Weather_Wind": [c for c in numeric_cols if "wind_speed" in c],
    "Cyclone": [c for c in numeric_cols if "cyclone" in c],
    "Engineered": [
        c
        for c in numeric_cols
        if c
        in [
            "SWI",
            "Coastal_Threat",
            "Month",
            "Week_of_Year",
            "Is_Peak_Season",
            "Price_Momentum_7d",
            "Price_Momentum_30d",
            "Price_Volatility_7d",
            "Price_Volatility_30d",
            "Price_Zscore_30d",
            "Rolling_7d_Cyclone_Days",
            "Congestion_Risk",
            "LSCI_30d_Delta",
        ]
    ],
    "Route_Aggregate": [c for c in numeric_cols if c.startswith("Route_")],
}

for group_name, features in feature_groups.items():
    if features:
        group_corr = correlations[features].sort_values(ascending=False)
        print(f"\n--- {group_name} ({len(features)} features) ---")
        print(group_corr.to_string())

# =========================================================================
# 5. CORRELATION HEATMAP (FULL MATRIX)
# =========================================================================
print("\n" + "=" * 70)
print("5. GENERATING VISUALIZATIONS")
print("=" * 70)

# 5a. Full heatmap with all numeric features
fig, ax = plt.subplots(figsize=(24, 20))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap="RdBu_r",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
    vmin=-1,
    vmax=1,
    ax=ax,
)
ax.set_title(
    "Full Correlation Matrix (Engineered ML Dataset)", fontsize=16, fontweight="bold"
)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "correlation_heatmap_full.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.close()
print("  Saved: correlation_heatmap_full.png")

# 5b. Target-focused heatmap (top 30 correlated features)
top_features = (
    correlations.head(15).index.tolist() + correlations.tail(15).index.tolist()
)
top_features = list(set(top_features))
top_features.append("Price_USD")

fig, ax = plt.subplots(figsize=(16, 14))
top_corr_matrix = df[top_features].corr()
mask = np.triu(np.ones_like(top_corr_matrix, dtype=bool))
sns.heatmap(
    top_corr_matrix,
    mask=mask,
    cmap="RdBu_r",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.7},
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".2f",
    ax=ax,
)
ax.set_title(
    "Top 30 Features Correlated with Price_USD", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "correlation_heatmap_top30.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.close()
print("  Saved: correlation_heatmap_top30.png")

# 5c. Engineered features heatmap
eng_features = feature_groups["Engineered"] + ["Price_USD"]
if len(eng_features) > 1:
    fig, ax = plt.subplots(figsize=(12, 10))
    eng_corr_matrix = df[eng_features].corr()
    sns.heatmap(
        eng_corr_matrix,
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".3f",
        ax=ax,
    )
    ax.set_title(
        "Engineered Features Correlation Matrix", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "correlation_heatmap_engineered.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("  Saved: correlation_heatmap_engineered.png")

# =========================================================================
# 6. ROUTE-SPECIFIC CORRELATIONS
# =========================================================================
print("\n" + "=" * 70)
print("6. ROUTE-SPECIFIC CORRELATIONS WITH TARGET")
print("=" * 70)

route_corr_results = {}
for route in df["Route"].unique():
    route_df = df[df["Route"] == route]
    route_corr = (
        route_df[numeric_cols]
        .corr()["Price_USD"]
        .drop("Price_USD")
        .sort_values(ascending=False)
    )
    route_corr_results[route] = route_corr

    print(f"\n--- {route} ---")
    print("Top 10:")
    print(route_corr.head(10).to_string())
    print("Bottom 10:")
    print(route_corr.tail(10).to_string())

# Save route correlations
route_corr_df = pd.DataFrame(route_corr_results)
route_corr_df.to_csv(os.path.join(OUTPUT_DIR, "route_specific_correlations.csv"))

# =========================================================================
# 7. ZERO/NEAR-ZERO VARIANCE FEATURES
# =========================================================================
print("\n" + "=" * 70)
print("7. ZERO/NEAR-ZERO VARIANCE FEATURES")
print("=" * 70)

low_var_features = []
for col in numeric_cols:
    if col == "Price_USD":
        continue
    unique_count = df[col].nunique()
    zero_pct = (df[col] == 0).mean() * 100
    if unique_count <= 2 or zero_pct > 95:
        low_var_features.append(
            {
                "Feature": col,
                "Unique_Values": unique_count,
                "Zero_Percentage": round(zero_pct, 2),
                "Std": df[col].std(),
            }
        )

low_var_df = pd.DataFrame(low_var_features)
print(f"\nFeatures with <=2 unique values or >95% zeros: {len(low_var_df)}")
print(low_var_df.to_string())

# =========================================================================
# 8. KEY INSIGHTS SUMMARY
# =========================================================================
print("\n" + "=" * 70)
print("8. KEY INSIGHTS SUMMARY")
print("=" * 70)

print("\n--- TARGET CORRELATION INSIGHTS ---")
print(
    f"Strongest positive correlate: {correlations.index[0]} (r={correlations.iloc[0]:.4f})"
)
print(
    f"Strongest negative correlate: {correlations.index[-1]} (r={correlations.iloc[-1]:.4f})"
)

strong_pos = correlations[correlations > 0.3]
strong_neg = correlations[correlations < -0.3]
print(f"\nFeatures with |r| > 0.3 to Price_USD: {len(strong_pos) + len(strong_neg)}")
print(f"  Positive (>0.3): {len(strong_pos)}")
print(f"  Negative (<-0.3): {len(strong_neg)}")

print("\n--- MULTICOLLINEARITY INSIGHTS ---")
print(f"Feature pairs with |r| > 0.8: {len(high_corr_df)}")
print(
    f"Feature pairs with |r| > 0.9: {len(high_corr_df[abs(high_corr_df['Correlation']) > 0.9])}"
)

# Most connected features (appear in most high-correlation pairs)
if len(high_corr_df) > 0:
    all_high_features = list(high_corr_df["Feature_1"]) + list(
        high_corr_df["Feature_2"]
    )
    feature_freq = pd.Series(all_high_features).value_counts()
    print("\nMost connected features (in high-corr pairs):")
    print(feature_freq.head(10).to_string())

print("\n--- LOW VARIANCE WARNING ---")
print(f"Features that may be candidates for removal: {len(low_var_df)}")
if len(low_var_df) > 0:
    print("Consider removing features with >95% zeros as they add noise.")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print(f"Output saved to: {OUTPUT_DIR}")
print("=" * 70)
