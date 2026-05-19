"""
Test: Does the selected model actually capture weather-driven price changes?
=============================================================================
Tests whether high volatility weather periods correlate with:
1. Higher prediction errors (model struggling)
2. Actual price movements (weather does affect prices)
3. Model's feature sensitivity to weather inputs
"""

import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from year_based_split import prepare_data, get_all_percentage_splits

MODEL_PATH = r"D:\Internproj\selectedmodel\xgb_80_20_lr001_depth5.json"
DATASET_PATH = r"d:\Internproj\new dataset\engineered_ml_dataset.csv"
RAW_DATASET_PATH = r"d:\Internproj\new dataset\final_ml_dataset.csv"
OUTPUT_DIR = r"D:\Internproj\Analysis\data design"

# Load model
print("Loading selected model...")
model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

# Load data
print("Loading datasets...")
X, y, years = prepare_data(DATASET_PATH)
splits = get_all_percentage_splits(years)
train_idx, test_idx = splits["80_20"]

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

# Load raw data for weather context
raw_df = pd.read_csv(RAW_DATASET_PATH)
raw_df['Date'] = pd.to_datetime(raw_df['Date'])
raw_test = raw_df.iloc[test_idx].copy()

# Get predictions
print("Generating predictions...")
y_pred = model.predict(X_test)

# Create results dataframe
results = raw_test[['Date', 'Route']].copy()
results['Actual_Price'] = y_test.values
results['Predicted_Price'] = y_pred
results['Error'] = np.abs(results['Actual_Price'] - results['Predicted_Price'])
results['Pct_Error'] = results['Error'] / results['Actual_Price'] * 100

# ============================================================
# TEST 1: Identify High Weather Volatility Periods
# ============================================================
print("\n" + "="*80)
print("TEST 1: HIGH WEATHER VOLATILITY ANALYSIS")
print("="*80)

# Define weather volatility indicators
weather_cols = ['Route_Mean_wind_speed', 'Route_Max_cyclone_wind', 
                'Route_Mean_slp', 'Route_cyclone_active']

available_weather_cols = [c for c in weather_cols if c in raw_test.columns]

# Calculate weather volatility score (normalized)
raw_test_vol = raw_test.copy()
weather_volatility = np.zeros(len(raw_test_vol))

for col in available_weather_cols:
    if col in raw_test_vol.columns:
        values = raw_test_vol[col].fillna(0).values
        if values.std() > 0:
            normalized = (values - values.mean()) / values.std()
            weather_volatility += np.abs(normalized)

results['Weather_Volatility'] = weather_volatility

# Classify periods
vol_threshold_high = np.percentile(weather_volatility, 75)
vol_threshold_low = np.percentile(weather_volatility, 25)

results['Volatility_Regime'] = 'Normal'
results.loc[weather_volatility >= vol_threshold_high, 'Volatility_Regime'] = 'High'
results.loc[weather_volatility <= vol_threshold_low, 'Volatility_Regime'] = 'Low'

# Compare model performance across regimes
print("\n--- Model Performance by Weather Volatility Regime ---")
for regime in ['Low', 'Normal', 'High']:
    subset = results[results['Volatility_Regime'] == regime]
    if len(subset) > 0:
        mae = mean_absolute_error(subset['Actual_Price'], subset['Predicted_Price'])
        avg_error_pct = subset['Pct_Error'].mean()
        avg_actual = subset['Actual_Price'].mean()
        avg_predicted = subset['Predicted_Price'].mean()
        print(f"\n  {regime} Volatility ({len(subset)} records):")
        print(f"    MAE: ${mae:.2f}")
        print(f"    Avg % Error: {avg_error_pct:.2f}%")
        print(f"    Avg Actual Price: ${avg_actual:.2f}")
        print(f"    Avg Predicted Price: ${avg_predicted:.2f}")
        print(f"    Price Difference: ${avg_predicted - avg_actual:.2f}")

# ============================================================
# TEST 2: Cyclone Event Analysis
# ============================================================
print("\n" + "="*80)
print("TEST 2: CYCLONE EVENT ANALYSIS")
print("="*80)

if 'Route_cyclone_active' in raw_test.columns:
    cyclone_days = results[raw_test['Route_cyclone_active'] == 1]
    no_cyclone_days = results[raw_test['Route_cyclone_active'] == 0]
    
    print(f"\n--- Model Performance: Cyclone Days vs No Cyclone Days ---")
    print(f"\n  Cyclone Days ({len(cyclone_days)} records):")
    if len(cyclone_days) > 0:
        mae_cyc = mean_absolute_error(cyclone_days['Actual_Price'], cyclone_days['Predicted_Price'])
        print(f"    MAE: ${mae_cyc:.2f}")
        print(f"    Avg Actual Price: ${cyclone_days['Actual_Price'].mean():.2f}")
        print(f"    Avg Predicted Price: ${cyclone_days['Predicted_Price'].mean():.2f}")
        print(f"    Avg % Error: {cyclone_days['Pct_Error'].mean():.2f}%")
    
    print(f"\n  No Cyclone Days ({len(no_cyclone_days)} records):")
    if len(no_cyclone_days) > 0:
        mae_no = mean_absolute_error(no_cyclone_days['Actual_Price'], no_cyclone_days['Predicted_Price'])
        print(f"    MAE: ${mae_no:.2f}")
        print(f"    Avg Actual Price: ${no_cyclone_days['Actual_Price'].mean():.2f}")
        print(f"    Avg Predicted Price: ${no_cyclone_days['Predicted_Price'].mean():.2f}")
        print(f"    Avg % Error: {no_cyclone_days['Pct_Error'].mean():.2f}%")

# ============================================================
# TEST 3: Extreme Wind Events
# ============================================================
print("\n" + "="*80)
print("TEST 3: EXTREME WIND EVENTS ANALYSIS")
print("="*80)

if 'Route_Mean_wind_speed' in raw_test.columns:
    wind_threshold = raw_test['Route_Mean_wind_speed'].quantile(0.90)
    high_wind = results[raw_test['Route_Mean_wind_speed'] >= wind_threshold]
    normal_wind = results[raw_test['Route_Mean_wind_speed'] < wind_threshold]
    
    print(f"\n--- Model Performance: High Wind (Top 10%) vs Normal Wind ---")
    print(f"\n  High Wind Days ({len(high_wind)} records):")
    if len(high_wind) > 0:
        mae_high = mean_absolute_error(high_wind['Actual_Price'], high_wind['Predicted_Price'])
        print(f"    MAE: ${mae_high:.2f}")
        print(f"    Avg Actual Price: ${high_wind['Actual_Price'].mean():.2f}")
        print(f"    Avg Predicted Price: ${high_wind['Predicted_Price'].mean():.2f}")
        print(f"    Avg % Error: {high_wind['Pct_Error'].mean():.2f}%")
    
    print(f"\n  Normal Wind Days ({len(normal_wind)} records):")
    if len(normal_wind) > 0:
        mae_norm = mean_absolute_error(normal_wind['Actual_Price'], normal_wind['Predicted_Price'])
        print(f"    MAE: ${mae_norm:.2f}")
        print(f"    Avg Actual Price: ${normal_wind['Actual_Price'].mean():.2f}")
        print(f"    Avg Predicted Price: ${normal_wind['Predicted_Price'].mean():.2f}")
        print(f"    Avg % Error: {normal_wind['Pct_Error'].mean():.2f}%")

# ============================================================
# TEST 4: Price Movement Prediction Accuracy
# ============================================================
print("\n" + "="*80)
print("TEST 4: PRICE MOVEMENT DIRECTION ACCURACY")
print("="*80)

# Calculate day-over-day price changes
results_sorted = results.sort_values(['Route', 'Date']).copy()
results_sorted['Actual_Change'] = results_sorted.groupby('Route')['Actual_Price'].diff()
results_sorted['Predicted_Change'] = results_sorted.groupby('Route')['Predicted_Price'].diff()

# Remove NaN rows
valid_changes = results_sorted.dropna(subset=['Actual_Change', 'Predicted_Change']).copy()

# Direction accuracy (did we predict the right direction of price change?)
valid_changes['Actual_Direction'] = np.sign(valid_changes['Actual_Change'])
valid_changes['Predicted_Direction'] = np.sign(valid_changes['Predicted_Change'])
valid_changes['Direction_Correct'] = valid_changes['Actual_Direction'] == valid_changes['Predicted_Direction']

direction_accuracy = valid_changes['Direction_Correct'].mean() * 100
print(f"\n  Price Direction Accuracy: {direction_accuracy:.1f}%")
print(f"  (Model correctly predicts whether price goes up or down)")

# Direction accuracy during high volatility
if 'Weather_Volatility' in valid_changes.columns:
    high_vol_changes = valid_changes[valid_changes['Weather_Volatility'] >= vol_threshold_high]
    low_vol_changes = valid_changes[valid_changes['Weather_Volatility'] <= vol_threshold_low]
    
    if len(high_vol_changes) > 0:
        high_vol_acc = high_vol_changes['Direction_Correct'].mean() * 100
        print(f"\n  Direction Accuracy (High Weather Volatility): {high_vol_acc:.1f}%")
    
    if len(low_vol_changes) > 0:
        low_vol_acc = low_vol_changes['Direction_Correct'].mean() * 100
        print(f"  Direction Accuracy (Low Weather Volatility): {low_vol_acc:.1f}%")

# ============================================================
# TEST 5: Feature Importance (Weather vs Economic)
# ============================================================
print("\n" + "="*80)
print("TEST 5: FEATURE IMPORTANCE ANALYSIS")
print("="*80)

importance = model.feature_importances_
feature_names = X.columns.tolist()

# Categorize features
def categorize_feature(name):
    if any(x in name for x in ['air', 'slp', 'wind_speed', 'prate']):
        return 'Weather'
    elif 'cyclone' in name.lower():
        return 'Cyclone'
    elif any(x in name for x in ['LSCI', 'lsci']):
        return 'Economic'
    elif any(x in name for x in ['Price_Lag', 'Price_Momentum', 'Price_Volatility', 'Price_Zscore']):
        return 'Price History'
    elif any(x in name for x in ['Month', 'Week', 'Season', 'SWI', 'Coastal', 'Congestion']):
        return 'Derived/Seasonal'
    elif 'Route' in name:
        return 'Route'
    else:
        return 'Other'

feat_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance,
    'Category': [categorize_feature(f) for f in feature_names]
})

# Aggregate by category
cat_importance = feat_df.groupby('Category')['Importance'].sum().sort_values(ascending=False)
print("\n--- Feature Importance by Category ---")
for cat, imp in cat_importance.items():
    print(f"  {cat}: {imp:.4f} ({imp*100:.1f}%)")

# Top 15 features
print("\n--- Top 15 Most Important Features ---")
top15 = feat_df.nlargest(15, 'Importance')
for _, row in top15.iterrows():
    print(f"  {row['Feature'][:40]:40s} | {row['Category']:15s} | {row['Importance']:.4f}")

# Weather dominance check
weather_total = cat_importance.get('Weather', 0) + cat_importance.get('Cyclone', 0)
economic_total = cat_importance.get('Economic', 0) + cat_importance.get('Price History', 0)
print(f"\n--- Category Totals ---")
print(f"  Weather + Cyclone: {weather_total:.4f} ({weather_total*100:.1f}%)")
print(f"  Economic + Price:  {economic_total:.4f} ({economic_total*100:.1f}%)")

# ============================================================
# TEST 6: Monthly Performance Breakdown
# ============================================================
print("\n" + "="*80)
print("TEST 6: MONTHLY PERFORMANCE BREAKDOWN")
print("="*80)

results['Month'] = results['Date'].dt.month
monthly_perf = results.groupby('Month').agg(
    MAE=('Error', 'mean'),
    Avg_Actual=('Actual_Price', 'mean'),
    Avg_Predicted=('Predicted_Price', 'mean'),
    Avg_Error_Pct=('Pct_Error', 'mean'),
    Count=('Error', 'count')
).round(2)

print("\n--- Model Performance by Month ---")
print(monthly_perf.to_string())

# Peak season performance (Aug-Oct)
peak_months = results[results['Month'].isin([8, 9, 10])]
off_peak = results[~results['Month'].isin([8, 9, 10])]

if len(peak_months) > 0 and len(off_peak) > 0:
    peak_mae = mean_absolute_error(peak_months['Actual_Price'], peak_months['Predicted_Price'])
    off_peak_mae = mean_absolute_error(off_peak['Actual_Price'], off_peak['Predicted_Price'])
    print(f"\n  Peak Season (Aug-Oct) MAE: ${peak_mae:.2f}")
    print(f"  Off-Peak Season MAE: ${off_peak_mae:.2f}")

print("\n" + "="*80)
print("WEATHER VOLATILITY TEST COMPLETE")
print("="*80)
