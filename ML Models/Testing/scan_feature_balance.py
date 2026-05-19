"""Scan all 80_20 models for feature balance across categories."""

import pickle, json, numpy as np, pandas as pd, xgboost as xgb, sys, os

sys.path.append(r"D:\Internproj\ML Models")
from year_based_split import prepare_data

DATASET = r"d:\Internproj\new dataset\engineered_ml_dataset.csv"
X, y, years = prepare_data(DATASET)
feature_names = list(X.columns)


def categorize_importance(feature_names, importances):
    total = importances.sum()
    if total == 0:
        return {}
    cats = {
        "price_lag": 0,
        "weather": 0,
        "economic": 0,
        "route": 0,
        "season": 0,
        "other": 0,
    }
    for fn, imp in zip(feature_names, importances):
        fn_lower = fn.lower()
        if any(
            k in fn_lower
            for k in ["lag", "price_momentum", "price_volatility", "price_zscore"]
        ):
            cats["price_lag"] += imp
        elif any(
            k in fn_lower
            for k in [
                "wind",
                "cyclone",
                "slp",
                "swi",
                "coastal",
                "threat",
                "congestion",
                "air_",
                "prate_",
                "vwnd_",
                "uwnd_",
            ]
        ):
            cats["weather"] += imp
        elif "lsci" in fn_lower:
            cats["economic"] += imp
        elif fn_lower.startswith("route_"):
            cats["route"] += imp
        elif any(k in fn_lower for k in ["month", "week", "peak"]):
            cats["season"] += imp
        else:
            cats["other"] += imp
    return {k: v / total * 100 for k, v in cats.items()}


# Load R2 lookup from evaluation results
eval_df = pd.read_csv(
    r"D:\Internproj\ML Models\Testing\results\all_evaluation_results.csv"
)
r2_lookup = {}
for _, row in eval_df.iterrows():
    key = f"{row['model_file']}"
    r2_lookup[key] = row["R2"]

# Scan XGBoost 80_20
xgb_dir = r"D:\Internproj\ML Models\XGBoost"
xgb_files = [
    f for f in os.listdir(xgb_dir) if f.startswith("xgb_80_20_") and f.endswith(".json")
]

results = []
for fname in sorted(xgb_files):
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(xgb_dir, fname))
    imps = model.feature_importances_
    cats = categorize_importance(feature_names, imps)
    label = fname.replace("xgb_80_20_", "").replace(".json", "")
    r2 = r2_lookup.get(fname, None)
    results.append(("XGB", label, cats, fname, r2))

# Scan RF 80_20
rf_dir = r"D:\Internproj\ML Models\Random Forest"
rf_files = [
    f for f in os.listdir(rf_dir) if f.startswith("rf_80_20_") and f.endswith(".pkl")
]

for fname in sorted(rf_files):
    with open(os.path.join(rf_dir, fname), "rb") as f:
        model = pickle.load(f)
    imps = model.feature_importances_
    cats = categorize_importance(feature_names, imps)
    label = fname.replace("rf_80_20_", "").replace(".pkl", "")
    r2 = r2_lookup.get(fname, None)
    results.append(("RF", label, cats, fname, r2))

# Sort by price_lag dominance (lowest = most balanced)
results.sort(key=lambda x: x[2].get("price_lag", 100))

print(
    f"{'Type':<5} {'Label':<30} {'R2':<8} {'P/Lag%':<9} {'Weather%':<9} {'Econ%':<8} {'Route%':<8} {'Season%':<8}"
)
print("-" * 95)
for mtype, label, cats, fname, r2 in results[:30]:
    r2_str = f"{r2:.4f}" if r2 is not None else "N/A"
    print(
        f"{mtype:<5} {label:<30} {r2_str:<8} {cats['price_lag']:<9.1f} {cats['weather']:<9.1f} {cats['economic']:<8.1f} {cats['route']:<8.1f} {cats['season']:<8.1f}"
    )

print()
print("=" * 95)
print("TOP 5 MOST BALANCED MODELS")
print("=" * 95)
for mtype, label, cats, fname, r2 in results[:5]:
    r2_str = f"{r2:.4f}" if r2 is not None else "N/A"
    print(f"\n{mtype} {label}  (R2={r2_str})")
    print(f"  Price/Lag: {cats['price_lag']:.1f}%")
    print(f"  Weather:   {cats['weather']:.1f}%")
    print(f"  Economic:  {cats['economic']:.1f}%")
    print(f"  Route:     {cats['route']:.1f}%")
    print(f"  Season:    {cats['season']:.1f}%")
    print(f"  Other:     {cats['other']:.1f}%")
