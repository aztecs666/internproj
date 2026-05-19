"""
Data Cleaning + Proper Feature Engineering Pipeline
Fixes:
1. Missing data sentinels (0K temps, 0Pa SLP) -> NaN
2. Lag feature bug (identical distributions) -> proper route-grouped shifts
3. Recomputes all engineered features from clean base
4. Handles route-specific zone zeros correctly (structural NaN, not 0)
5. Proper temporal-aware feature computation
"""

import os
import pandas as pd
import numpy as np

INPUT_PATH = r'D:\Internproj\new dataset\engineered_ml_dataset.csv'
OUTPUT_DIR = r'D:\Internproj\New ML'
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'clean_engineered_dataset.csv')

EPS = 1e-8

print('Loading raw dataset...')
df = pd.read_csv(INPUT_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Route', 'Date']).reset_index(drop=True)
print(f'Loaded: {df.shape}')

# ========================================================================
# STEP 1: Clean missing-data sentinels
# ========================================================================
print('\n[1] Cleaning missing-data sentinels...')

# Identify rows where temperature is 0K (physically impossible)
air_cols = [c for c in df.columns if c.startswith('air_') or c == 'Route_Mean_air']
slp_cols = [c for c in df.columns if c.startswith('slp_') or c == 'Route_Mean_slp']
wind_cols = [c for c in df.columns if c.startswith('wind_speed_') or c == 'Route_Mean_wind_speed']

# Flag rows with impossible temps (0K)
bad_temp_mask = (df['Route_Mean_air'] == 0) | (df['air_South_China_Sea'] == 0)
bad_slp_mask = (df['Route_Mean_slp'] == 0) | (df['slp_South_China_Sea'] == 0)

bad_rows = df[bad_temp_mask | bad_slp_mask]
print(f'  Rows with impossible temps/SLP: {len(bad_rows)}')
if len(bad_rows) > 0:
    print(f'  Date range of bad rows: {bad_rows["Date"].min()} to {bad_rows["Date"].max()}')
    # These are early 2015 rows where weather data wasn't available
    # Replace all weather-related columns with NaN for these rows
    for col in air_cols + slp_cols + wind_cols:
        df.loc[bad_temp_mask | bad_slp_mask, col] = np.nan
    print('  Replaced impossible weather values with NaN')

# ========================================================================
# STEP 2: Handle route-specific structural zeros
# ========================================================================
print('\n[2] Handling route-specific structural zeros...')

# Define which zones each route traverses
route_zones = {
    'XSICFENE_FarEast_NorthEurope': ['South_China_Sea', 'Indian_Ocean', 'Arabian_Sea', 'Red_Sea',
                                      'Mediterranean_Sea', 'North_Atlantic', 'North_Sea'],
    'XSICNEFE_NorthEurope_FarEast': ['South_China_Sea', 'Indian_Ocean', 'Arabian_Sea', 'Red_Sea',
                                      'Mediterranean_Sea', 'North_Atlantic', 'North_Sea'],
    'XSICFEUW_FarEast_USWestCoast': ['South_China_Sea', 'North_Pacific_West', 'North_Pacific_East'],
    'XSICUWFE_USWestCoast_FarEast': ['South_China_Sea', 'North_Pacific_West', 'North_Pacific_East'],
}

# For cyclone features, zeros are valid (no cyclone present)
# But for weather features, zeros in non-traversed zones should be NaN
all_zones = ['South_China_Sea', 'Indian_Ocean', 'Arabian_Sea', 'Red_Sea',
             'Mediterranean_Sea', 'North_Atlantic', 'North_Sea',
             'North_Pacific_West', 'North_Pacific_East']

for route in df['Route'].unique():
    route_mask = df['Route'] == route
    traversed = route_zones.get(route, [])
    not_traversed = [z for z in all_zones if z not in traversed]
    for zone in not_traversed:
        for prefix in ['air_', 'slp_', 'wind_speed_']:
            col = f'{prefix}{zone}'
            if col in df.columns:
                # These should be NaN, not 0
                df.loc[route_mask, col] = np.nan
    print(f'  {route}: {len(not_traversed)} non-traversed zones set to NaN')

# ========================================================================
# STEP 3: Recompute engineered features correctly
# ========================================================================
print('\n[3] Recomputing engineered features...')

# --- Seasonality (from Date) ---
df['Month'] = df['Date'].dt.month
df['Week_of_Year'] = df['Date'].dt.isocalendar().week.astype(int)
df['Is_Peak_Season'] = df['Month'].isin([8, 9, 10]).astype(int)
print('  Seasonality features recomputed')

# --- Price Time-Series Features (per route) ---
print('  Price time-series features...')
routes = df['Route'].unique()

for route in routes:
    mask = df['Route'] == route
    route_df = df.loc[mask].copy()
    route_idx = route_df.index

    # Lags
    df.loc[route_idx, 'Price_Lag_7d'] = route_df['Price_USD'].shift(7)
    df.loc[route_idx, 'Price_Lag_30d'] = route_df['Price_USD'].shift(30)

    # Price_Momentum_7d = (Price_t-1 - Price_t-7) / (Price_t-7 + eps)
    price_1d_ago = route_df['Price_USD'].shift(1)
    price_7d_ago = route_df['Price_USD'].shift(7)
    df.loc[route_idx, 'Price_Momentum_7d'] = (price_1d_ago - price_7d_ago) / (price_7d_ago + EPS)

    # Price_Momentum_30d = (Price_t-7 - Price_t-30) / (Price_t-14 + eps)
    price_14d_ago = route_df['Price_USD'].shift(14)
    price_30d_ago = route_df['Price_USD'].shift(30)
    df.loc[route_idx, 'Price_Momentum_30d'] = (price_7d_ago - price_30d_ago) / (price_14d_ago + EPS)

    # Price_Volatility_7d: std of daily pct changes over 7-day rolling window
    pct_change = route_df['Price_USD'].pct_change()
    df.loc[route_idx, 'Price_Volatility_7d'] = pct_change.rolling(window=7, min_periods=2).std()

    # Price_Volatility_30d: std of daily pct changes over 30-day rolling window
    df.loc[route_idx, 'Price_Volatility_30d'] = pct_change.rolling(window=30, min_periods=5).std()

    # Price_Zscore_30d = (Price_t-7 - 30d_mean) / (30d_std + eps)
    rolling_mean_30 = route_df['Price_USD'].shift(7).rolling(window=30, min_periods=5).mean()
    rolling_std_30 = route_df['Price_USD'].shift(7).rolling(window=30, min_periods=5).std()
    df.loc[route_idx, 'Price_Zscore_30d'] = (price_7d_ago - rolling_mean_30) / (rolling_std_30 + EPS)

    # Rolling_7d_Cyclone_Days: count of days with active cyclones in past 7 days
    df.loc[route_idx, 'Rolling_7d_Cyclone_Days'] = route_df['Route_cyclone_active'].rolling(window=7, min_periods=1).sum()

print('  Price time-series features recomputed')

# --- Lagged Weather Features (per route, PROPER SHIFT) ---
print('  Lagged weather features...')
for route in routes:
    mask = df['Route'] == route
    route_df = df.loc[mask].copy()
    route_idx = route_df.index

    # Cyclone Wind Lags
    df.loc[route_idx, 'Cyclone_Wind_Lag7d'] = route_df['Route_Max_cyclone_wind'].shift(7)
    df.loc[route_idx, 'Cyclone_Wind_Lag14d'] = route_df['Route_Max_cyclone_wind'].shift(14)
    df.loc[route_idx, 'Cyclone_Wind_Lag30d'] = route_df['Route_Max_cyclone_wind'].shift(30)

    # Cyclone Active Lags
    df.loc[route_idx, 'Cyclone_Active_Lag7d'] = route_df['Route_cyclone_active'].shift(7)
    df.loc[route_idx, 'Cyclone_Active_Lag14d'] = route_df['Route_cyclone_active'].shift(14)
    df.loc[route_idx, 'Cyclone_Active_Lag30d'] = route_df['Route_cyclone_active'].shift(30)

    # Wind Speed Lags
    df.loc[route_idx, 'Wind_Speed_Lag7d'] = route_df['Route_Mean_wind_speed'].shift(7)
    df.loc[route_idx, 'Wind_Speed_Lag14d'] = route_df['Route_Mean_wind_speed'].shift(14)
    df.loc[route_idx, 'Wind_Speed_Lag30d'] = route_df['Route_Mean_wind_speed'].shift(30)

print('  Lagged weather features recomputed (proper per-route shifts)')

# --- Forward Weather (Forecast Proxies) ---
print('  Forward weather features...')
for route in routes:
    mask = df['Route'] == route
    route_df = df.loc[mask].copy()
    route_idx = route_df.index

    df.loc[route_idx, 'Cyclone_Wind_Forecast3d'] = route_df['Route_Max_cyclone_wind'].shift(-3)
    df.loc[route_idx, 'Cyclone_Wind_Forecast7d'] = route_df['Route_Max_cyclone_wind'].shift(-7)
    df.loc[route_idx, 'Cyclone_Active_Forecast3d'] = route_df['Route_cyclone_active'].shift(-3)
    df.loc[route_idx, 'Wind_Speed_Forecast7d'] = route_df['Route_Mean_wind_speed'].shift(-7)

print('  Forward weather features recomputed')

# --- Hazard Interactions ---
print('  Hazard interactions...')
df['SWI'] = df['Route_Mean_wind_speed'] / (df['Route_Mean_slp'] + EPS)
df['Coastal_Threat'] = df['Route_Max_cyclone_wind'] / (df['Route_Min_cyclone_dist'] + 1)
print('  Hazard interactions recomputed')

# --- Economic Interactions ---
print('  Economic interactions...')
df['Congestion_Risk'] = df['Route_Max_cyclone_wind'] / (df['Dest_LSCI'] + EPS)

# LSCI_30d_Delta: change in Route_Mean_LSCI over past 30 days
for route in routes:
    mask = df['Route'] == route
    route_df = df.loc[mask].copy()
    route_idx = route_df.index
    lsci_30d_ago = route_df['Route_Mean_LSCI'].shift(30)
    df.loc[route_idx, 'LSCI_30d_Delta'] = route_df['Route_Mean_LSCI'] - lsci_30d_ago

print('  Economic interactions recomputed')

# ========================================================================
# STEP 4: Data Quality Report
# ========================================================================
print('\n[4] Data Quality Report')
print(f'  Final shape: {df.shape}')

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(f'  Columns with NaN: {len(missing)}')
for col, count in missing.head(15).items():
    pct = count / len(df) * 100
    print(f'    {col}: {count} ({pct:.1f}%)')

# Verify lag fix
print('\n[5] Verifying lag fix...')
from scipy import stats
a = df['Cyclone_Wind_Lag7d'].dropna()
b = df['Cyclone_Wind_Lag14d'].dropna()
c = df['Cyclone_Wind_Lag30d'].dropna()
ks_7_14 = stats.ks_2samp(a, b)
ks_14_30 = stats.ks_2samp(b, c)
print(f'  Lag7d vs Lag14d KS: stat={ks_7_14.statistic:.4f}, p={ks_7_14.pvalue:.4f}')
print(f'  Lag14d vs Lag30d KS: stat={ks_14_30.statistic:.4f}, p={ks_14_30.pvalue:.4f}')
if ks_7_14.statistic > 0.01:
    print('  SUCCESS: Lag distributions are now different!')
else:
    print('  WARNING: Lag distributions still look similar')

# ========================================================================
# STEP 5: Save
# ========================================================================
print(f'\n[6] Saving to {OUTPUT_PATH}...')
df.to_csv(OUTPUT_PATH, index=False)
print('Done!')
