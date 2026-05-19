import pandas as pd
import numpy as np


def prepare_data(csv_path):
    """
    Loads engineered dataset and prepares X and y (Price_USD).
    """
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Target variable
    y = df["Price_USD"]

    # Extract year for time series splits
    years = df["Date"].dt.year

    # Drop target and non-feature columns
    features_to_drop = ["Date", "Price_USD"]
    X = df.drop(columns=features_to_drop)

    # One-hot encode the categorical Route
    if "Route" in X.columns:
        X = pd.get_dummies(X, columns=["Route"], drop_first=False)

    # Fill any remaining NaNs with 0
    X = X.fillna(0)

    return X, y, years


# =============================================================================
# Percentage-Based Time-Series Train/Test Splits
# =============================================================================
SPLIT_CONFIGS = {
    "80_20": {
        "train_end_year": 2022,
        "test_start_year": 2023,
        "description": "Train 2015-2022, Test 2023-2026",
    },
    "60_40": {
        "train_end_year": 2020,
        "test_start_year": 2021,
        "description": "Train 2015-2020, Test 2021-2026",
    },
    "40_60": {
        "train_end_year": 2018,
        "test_start_year": 2019,
        "description": "Train 2015-2018, Test 2019-2026",
    },
    "20_80": {
        "train_end_year": 2016,
        "test_start_year": 2017,
        "description": "Train 2015-2016, Test 2017-2026",
    },
}


def get_percentage_split(years, split_name):
    """
    Returns train and test indices for a named percentage split.
    """
    config = SPLIT_CONFIGS[split_name]
    train_idx = np.where(years <= config["train_end_year"])[0]
    test_idx = np.where(years >= config["test_start_year"])[0]
    return train_idx, test_idx


def get_all_percentage_splits(years):
    """
    Returns dict of all percentage splits: {split_name: (train_idx, test_idx)}
    """
    splits = {}
    for name in SPLIT_CONFIGS:
        splits[name] = get_percentage_split(years, name)
    return splits


if __name__ == "__main__":
    X, y, yrs = prepare_data(r"d:\Internproj\new dataset\engineered_ml_dataset.csv")
    print(f"Total features: {X.shape[1]}")
    print(f"Total samples: {len(y)}")
    print(f"\n--- Percentage-Based Splits ---")
    pct_splits = get_all_percentage_splits(yrs)
    for name, (tr, te) in pct_splits.items():
        desc = SPLIT_CONFIGS[name]["description"]
        print(
            f"{name} ({desc}): Train={len(tr)} ({100 * len(tr) / len(y):.1f}%), Test={len(te)} ({100 * len(te) / len(y):.1f}%)"
        )
