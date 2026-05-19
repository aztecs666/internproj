import pandas as pd
import numpy as np
import os


def engineer_features(input_path, output_path):
    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)

    # 1. Basic formatting
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Route", "Date"]).reset_index(drop=True)

    print("Generating Features...")

    # -------------------------------------------------------------
    # 1. Weather "Hazard & Threat" Interactions
    # -------------------------------------------------------------
    epsilon = 1e-6
    if "Route_Mean_wind_speed" in df.columns and "Route_Mean_slp" in df.columns:
        df["SWI"] = df["Route_Mean_wind_speed"] / (df["Route_Mean_slp"] + epsilon)

    if (
        "Route_Max_cyclone_wind" in df.columns
        and "Route_Min_cyclone_dist" in df.columns
    ):
        cyclone_wind = df["Route_Max_cyclone_wind"].fillna(0)
        cyclone_dist = df["Route_Min_cyclone_dist"].fillna(500)
        df["Coastal_Threat"] = cyclone_wind / (cyclone_dist + 1)

    # -------------------------------------------------------------
    # 2. Seasonality Blocks
    # -------------------------------------------------------------
    df["Month"] = df["Date"].dt.month
    df["Week_of_Year"] = df["Date"].dt.isocalendar().week
    df["Is_Peak_Season"] = df["Month"].isin([8, 9, 10]).astype(int)

    # -------------------------------------------------------------
    # 3. Time-Series Features (grouped by Route)
    # -------------------------------------------------------------
    grouped = df.groupby("Route")
    epsilon = 1e-6

    # Price Lags
    df["Price_Lag_7d"] = grouped["Price_USD"].shift(7)
    df["Price_Lag_30d"] = grouped["Price_USD"].shift(30)

    # Internal lagged prices for momentum calculation
    price_lag_1d = grouped["Price_USD"].shift(1)
    price_lag_7d = df["Price_Lag_7d"]
    price_lag_14d = grouped["Price_USD"].shift(14)
    price_lag_30d = df["Price_Lag_30d"]

    # Price Momentum
    df["Price_Momentum_7d"] = (price_lag_1d - price_lag_7d) / (price_lag_7d + epsilon)
    df["Price_Momentum_30d"] = (price_lag_7d - price_lag_30d) / (
        price_lag_14d + epsilon
    )

    # Price Volatility
    df["Price_Volatility_7d"] = (
        grouped["Price_USD"]
        .transform(
            lambda x: x.shift(1).pct_change().rolling(window=7, min_periods=1).std()
        )
        .fillna(0)
    )
    df["Price_Volatility_30d"] = (
        grouped["Price_USD"]
        .transform(
            lambda x: x.shift(1).pct_change().rolling(window=30, min_periods=1).std()
        )
        .fillna(0)
    )

    # Z-score of lagged price
    lag_7d_mean = price_lag_7d.rolling(window=30, min_periods=1).mean()
    lag_7d_std = price_lag_7d.rolling(window=30, min_periods=1).std()
    df["Price_Zscore_30d"] = (
        (price_lag_7d - lag_7d_mean) / (lag_7d_std + epsilon)
    ).fillna(0)

    # Consecutive Hazard Days
    if "Route_cyclone_active" in df.columns:
        df["Rolling_7d_Cyclone_Days"] = grouped["Route_cyclone_active"].transform(
            lambda x: x.rolling(window=7, min_periods=1).sum()
        )

    # -------------------------------------------------------------
    # 4. Lagged Weather Features (past weather affects current prices)
    # -------------------------------------------------------------
    if "Route_Max_cyclone_wind" in df.columns:
        df["Cyclone_Wind_Lag7d"] = grouped["Route_Max_cyclone_wind"].shift(7).fillna(0)
        df["Cyclone_Wind_Lag14d"] = (
            grouped["Route_Max_cyclone_wind"].shift(14).fillna(0)
        )
        df["Cyclone_Wind_Lag30d"] = (
            grouped["Route_Max_cyclone_wind"].shift(30).fillna(0)
        )

    if "Route_cyclone_active" in df.columns:
        df["Cyclone_Active_Lag7d"] = grouped["Route_cyclone_active"].shift(7).fillna(0)
        df["Cyclone_Active_Lag14d"] = (
            grouped["Route_cyclone_active"].shift(14).fillna(0)
        )
        df["Cyclone_Active_Lag30d"] = (
            grouped["Route_cyclone_active"].shift(30).fillna(0)
        )

    if "Route_Mean_wind_speed" in df.columns:
        df["Wind_Speed_Lag7d"] = grouped["Route_Mean_wind_speed"].shift(7).fillna(0)
        df["Wind_Speed_Lag14d"] = grouped["Route_Mean_wind_speed"].shift(14).fillna(0)
        df["Wind_Speed_Lag30d"] = grouped["Route_Mean_wind_speed"].shift(30).fillna(0)

    # -------------------------------------------------------------
    # 5. Forward Weather Features (weather forecast proxies)
    # -------------------------------------------------------------
    if "Route_Max_cyclone_wind" in df.columns:
        df["Cyclone_Wind_Forecast3d"] = (
            grouped["Route_Max_cyclone_wind"].shift(-3).fillna(0)
        )
        df["Cyclone_Wind_Forecast7d"] = (
            grouped["Route_Max_cyclone_wind"].shift(-7).fillna(0)
        )

    if "Route_cyclone_active" in df.columns:
        df["Cyclone_Active_Forecast3d"] = (
            grouped["Route_cyclone_active"].shift(-3).fillna(0)
        )

    if "Route_Mean_wind_speed" in df.columns:
        df["Wind_Speed_Forecast7d"] = (
            grouped["Route_Mean_wind_speed"].shift(-7).fillna(0)
        )

    # -------------------------------------------------------------
    # 6. Economic "Vulnerability" Interactions
    # -------------------------------------------------------------
    if "Route_Max_cyclone_wind" in df.columns and "Dest_LSCI" in df.columns:
        wind = df["Route_Max_cyclone_wind"].fillna(0)
        lsci = df["Dest_LSCI"].fillna(df["Dest_LSCI"].mean()) + epsilon
        df["Congestion_Risk"] = wind / lsci

    if "Route_Mean_LSCI" in df.columns:
        df["LSCI_30d_Delta"] = df["Route_Mean_LSCI"] - grouped["Route_Mean_LSCI"].shift(
            30
        )
        df["LSCI_30d_Delta"] = df["LSCI_30d_Delta"].fillna(0)

    # -------------------------------------------------------------
    # Clean up any remaining NaNs
    # -------------------------------------------------------------
    df = df.fillna(0)

    # -------------------------------------------------------------
    # Save the output
    # -------------------------------------------------------------
    df.to_csv(output_path, index=False)
    print(f"Success! Engineered dataset saved to: {output_path}")
    print(f"Total features created: {len(df.columns)}")


if __name__ == "__main__":
    INPUT_FILE = r"D:\testing\data\final_ml_dataset.csv"
    OUTPUT_FILE = r"D:\testing\data\engineered_ml_dataset.csv"

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    if os.path.exists(INPUT_FILE):
        engineer_features(INPUT_FILE, OUTPUT_FILE)
    else:
        print(f"Error: Input file {INPUT_FILE} not found.")
