import pandas as pd
import numpy as np
from pathlib import Path
import json

# Define datasets to analyze
datasets = {
    "final_ml_dataset": r"D:\Internproj\new dataset\final_ml_dataset.csv",
    "engineered_ml_dataset": r"D:\Internproj\new dataset\engineered_ml_dataset.csv",
    "xeneta_combined": r"D:\Internproj\Index\xeneta\xeneta_xsi_combined.csv",
    "xeneta_simple": r"D:\Internproj\Index\xeneta\xeneta_xsi_simple.csv",
    "weather_zones_2015": r"D:\Internproj\new dataset\weather_zones_2015.csv",
    "XSICFENE_FarEast_NorthEurope": r"D:\Internproj\Index\xeneta\XSICFENE_FarEast_NorthEurope.csv",
    "XSICNEFE_NorthEurope_FarEast": r"D:\Internproj\Index\xeneta\XSICNEFE_NorthEurope_FarEast.csv",
    "XSICFEUW_FarEast_USWestCoast": r"D:\Internproj\Index\xeneta\XSICFEUW_FarEast_USWestCoast.csv",
    "XSICUWFE_USWestCoast_FarEast": r"D:\Internproj\Index\xeneta\XSICUWFE_USWestCoast_FarEast.csv",
}

results = {}

for name, path in datasets.items():
    try:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {name}")
        print(f"{'='*60}")
        
        df = pd.read_csv(path)
        
        # Basic Info
        print(f"\n--- BASIC INFO ---")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Date column: {'Date' if 'Date' in df.columns else 'date' if 'date' in df.columns else 'None'}")
        
        # Parse dates if present
        date_col = 'Date' if 'Date' in df.columns else 'date' if 'date' in df.columns else None
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
        
        # Numeric columns analysis
        numeric_df = df.select_dtypes(include=[np.number])
        print(f"\n--- NUMERIC COLUMNS ({len(numeric_df.columns)}) ---")
        
        if not numeric_df.empty:
            desc = numeric_df.describe().T
            desc['skewness'] = numeric_df.skew()
            desc['kurtosis'] = numeric_df.kurtosis()
            desc['missing'] = numeric_df.isnull().sum()
            desc['missing_pct'] = (numeric_df.isnull().sum() / len(df)) * 100
            print(desc.to_string())
        
        # Categorical columns
        cat_df = df.select_dtypes(include=['object'])
        if not cat_df.empty:
            print(f"\n--- CATEGORICAL COLUMNS ({len(cat_df.columns)}) ---")
            for col in cat_df.columns:
                if col == date_col:
                    continue
                print(f"\n{col}:")
                print(f"  Unique values: {df[col].nunique()}")
                print(f"  Top values:")
                print(df[col].value_counts().head(5).to_string().replace('\n', '\n    '))
        
        # Special analysis for Price_USD
        if 'Price_USD' in df.columns:
            print(f"\n--- TARGET VARIABLE: Price_USD ---")
            price = df['Price_USD'].dropna()
            print(f"Mean: ${price.mean():.2f}")
            print(f"Median: ${price.median():.2f}")
            print(f"Std: ${price.std():.2f}")
            print(f"Min: ${price.min():.2f}")
            print(f"Max: ${price.max():.2f}")
            print(f"Range: ${price.max() - price.min():.2f}")
            
            # Distribution shape
            skew = price.skew()
            if abs(skew) < 0.5:
                dist_shape = "Approximately symmetric"
            elif skew > 0:
                dist_shape = "Right-skewed (positive)"
            else:
                dist_shape = "Left-skewed (negative)"
            print(f"Distribution: {dist_shape} (skew={skew:.3f})")
            
            # Quartiles
            q1, q2, q3 = price.quantile([0.25, 0.5, 0.75])
            print(f"Q1: ${q1:.2f}, Q2 (Median): ${q2:.2f}, Q3: ${q3:.2f}")
            print(f"IQR: ${q3-q1:.2f}")
        
        # Store summary
        results[name] = {
            "shape": df.shape,
            "columns": len(df.columns),
            "numeric_columns": len(numeric_df.columns),
            "missing_total": df.isnull().sum().sum(),
            "date_range": f"{df[date_col].min()} to {df[date_col].max()}" if date_col else None
        }
        
    except Exception as e:
        print(f"ERROR analyzing {name}: {e}")

print("\n" + "="*60)
print("SUMMARY OF ALL DATASETS")
print("="*60)
for name, info in results.items():
    print(f"\n{name}:")
    for k, v in info.items():
        print(f"  {k}: {v}")
