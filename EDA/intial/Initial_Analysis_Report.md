# Initial Data Analysis Report
## D:​Internproj​EDA​intial

**Analysis Date:** April 26, 2026
**Analyst:** AI Assistant
**Datasets Analyzed:** 9 files

---

## Executive Summary

This report provides an initial statistical analysis of all datasets found in the project. The primary dataset is the **Xeneta Shipping Index (XSI)** covering 4 major shipping routes from 2015-2026, enriched with weather data, cyclone information, and LSCI (Liner Shipping Connectivity Index) metrics.

---

## 1. PRIMARY DATASET: Xeneta Shipping Index (XSI)

### 1.1 Dataset Overview
- **File:** `xeneta_xsi_combined.csv`
- **Shape:** 9,800 rows × 3 columns
- **Date Range:** January 5, 2015 - April 15, 2026 (11+ years)
- **Routes:** 4 international shipping routes

### 1.2 Route Distribution
| Route | Records | Date Range | Mean Price | Median Price | Std Dev |
|-------|---------|------------|------------|--------------|---------|
| XSICFENE (FarEast→NorthEurope) | 2,829 | 2015-2026 | $3,526 | $1,785 | $3,667 |
| XSICNEFE (NorthEurope→FarEast) | 2,829 | 2015-2026 | $3,526 | $1,785 | $3,667 |
| XSICFEUW (FarEast→USWestCoast) | 2,071 | 2018-2026 | $4,485 | $2,913 | $3,355 |
| XSICUWFE (USWestCoast→FarEast) | 2,071 | 2018-2026 | $4,485 | $2,913 | $3,355 |

### 1.3 Price Distribution Analysis

#### Overall (All Routes Combined):
- **Mean:** $3,931.71
- **Median:** $2,196.00
- **Standard Deviation:** $3,569.94
- **Minimum:** $312.00
- **Maximum:** $15,185.00
- **Range:** $14,873.00
- **Distribution Shape:** **Right-skewed (positive)** - Skewness = 1.465
- **Quartiles:** Q1=$1,581, Q2=$2,196, Q3=$5,178
- **IQR:** $3,596.75

#### Key Insights:
1. **Highly right-skewed distribution** - Most prices cluster in the $1,000-$3,000 range, with a long tail extending to $15,000+
2. **Mean >> Median** ($3,932 vs $2,196) confirms strong right skew
3. **Massive volatility** - Standard deviation ($3,570) is nearly equal to the mean
4. **Bimodal tendency** - Suggests distinct market regimes (normal vs. crisis pricing)

---

## 2. ML DATASETS

### 2.1 Final ML Dataset (`final_ml_dataset.csv`)
- **Shape:** 9,800 rows × 76 columns
- **Numeric Features:** 74
- **Categorical Features:** 1 (Route)
- **Missing Values:** 369,216 total (significant missing data issue!)
- **Date Range:** 2015-2026

#### Missing Data Pattern:
- **Critical Issue:** Many cyclone and weather features have 40-100% missing values
- Cyclone_slp_Red_Sea: 100% missing
- Cyclone features for North Atlantic: 100% missing
- Route-level cyclone features: ~60-90% missing
- Weather zone features: ~44% missing (systematic pattern)

#### Key Feature Distributions:

**LSCI (Liner Shipping Connectivity Index):**
- Origin_LSCI: Mean=56.07, Std=16.96, Range=0-81.8
- Dest_LSCI: Mean=56.07, Std=16.96 (identical - same routes)
- Route_Mean_LSCI: Mean=56.07, Std=15.10
- Distribution: **Left-skewed** with peak around 55

**Weather Features:**
- Route_Mean_air: Mean=292.79K, Std=3.41K
- Route_Mean_slp: Mean=101,309 Pa, Std=209
- Route_Mean_wind_speed: Mean=2.65 m/s, Std=0.86

**Cyclone Features:**
- Route_Max_cyclone_wind: Mean=17.83, Std=28.72 (highly zero-inflated)
- Route_cyclone_active: Mean=0.44 (binary, 44% active days)
- Highly zero-inflated distributions - most days have no cyclone activity

### 2.2 Engineered ML Dataset (`engineered_ml_dataset.csv`)
- **Shape:** 9,800 rows × 104 columns
- **Numeric Features:** 102
- **Missing Values:** 0 (all imputed!)
- **Date Range:** 2015-2026

#### New Engineered Features:
- **Temporal Features:** Month, Week_of_Year, Is_Peak_Season
- **Lag Features:** Price_Lag_7d, Price_Lag_30d
- **Momentum Features:** Price_Momentum_7d, Price_Momentum_30d
- **Volatility Features:** Price_Volatility_7d, Price_Volatility_30d
- **Rolling Features:** Rolling_7d_Cyclone_Days
- **Forecast Features:** Cyclone_Wind_Forecast3d/7d
- **Risk Metrics:** Congestion_Risk, LSCI_30d_Delta, Coastal_Threat

#### Distribution Insights:
- **Price_Lag features:** Similar distribution to target (as expected)
- **Price_Momentum:** Mean≈0, Std≈0.18, approximately symmetric
- **Price_Volatility:** Right-skewed, mean=0.036-0.046
- **Is_Peak_Season:** Binary, 25.4% peak season days
- **Month/Week:** Uniform-ish distribution across time
- **Coastal_Threat:** Mean=4.08, highly right-skewed (most days=0)

---

## 3. WEATHER DATA (`weather_zones_2015.csv`)
- **Shape:** 365 rows × 37 columns
- **Coverage:** Full year 2015, daily granularity
- **Zones:** 9 maritime weather zones
- **Variables per zone:** air temperature, slp, uwnd, vwnd

### 3.1 Temperature (air_*) Distributions
| Zone | Mean (K) | Std | Min | Max | Skewness |
|------|----------|-----|-----|-----|----------|
| Arabian Sea | 300.38 | 2.22 | 295.5 | 304.0 | -0.54 |
| Indian Ocean | 298.17 | 1.19 | 296.0 | 300.3 | -0.29 |
| Mediterranean Sea | 290.57 | 6.31 | 278.4 | 301.2 | +0.03 |
| North Atlantic | 287.73 | 3.70 | 281.3 | 293.7 | +0.06 |
| North Pacific East | 288.63 | 2.81 | 284.6 | 293.8 | +0.37 |
| North Pacific West | 286.83 | 5.58 | 277.4 | 295.8 | +0.02 |
| North Sea | 281.53 | 4.24 | 272.9 | 290.7 | +0.17 |
| Red Sea | 299.32 | 5.60 | 285.5 | 307.3 | -0.48 |
| South China Sea | 297.92 | 1.86 | 293.2 | 300.4 | -0.87 |

**Insights:**
- **Hottest:** Arabian Sea (300.4K / 27.2°C avg)
- **Coolest:** North Sea (281.5K / 8.4°C avg)
- **Most Variable:** Mediterranean Sea (σ=6.31K) - seasonal extremes
- **Most Stable:** Indian Ocean (σ=1.19K) - maritime climate buffering

### 3.2 Sea Level Pressure (slp_*) Distributions
| Zone | Mean (Pa) | Std | Range |
|------|-----------|-----|-------|
| Arabian Sea | 101,052 | 349 | 100,217-101,815 |
| Indian Ocean | 101,443 | 206 | 100,876-101,871 |
| Mediterranean Sea | 101,744 | 555 | 100,033-103,446 |
| North Atlantic | 101,635 | 293 | 100,672-102,695 |
| North Sea | 101,212 | 985 | 97,381-103,520 |

**Insights:**
- **Highest Pressure:** Mediterranean Sea
- **Lowest Pressure:** Arabian Sea
- **Most Variable:** North Sea (σ=985) - storm track influence
- **Least Variable:** Indian Ocean (σ=206) - stable subtropical high

### 3.3 Wind Components (uwnd_*, vwnd_*)
- **Zonal winds (uwnd):** Range from -6 to +8 m/s
- **Meridional winds (vwnd):** Range from -7 to +14 m/s
- **Indian Ocean:** Strong consistent easterlies (uwnd mean=-3.0 m/s)
- **North Sea:** Highly variable winds (σ≈3.5 m/s both components)

---

## 4. CORRELATION ANALYSIS

### 4.1 Key Feature Correlations with Price_USD
From the engineered dataset heatmap:

| Feature | Correlation with Price_USD | Interpretation |
|---------|---------------------------|----------------|
| Origin_LSCI | -0.25 | Higher connectivity = lower prices (economies of scale) |
| Dest_LSCI | -0.25 | Same as above |
| Route_Mean_LSCI | -0.29 | Strongest negative correlation |
| Route_Mean_air | +0.07 | Weak positive |
| Route_Mean_slp | +0.07 | Weak positive |
| Route_Mean_wind_speed | +0.01 | Negligible |
| Route_Max_cyclone_wind | +0.01 | Weak positive |
| Route_cyclone_active | +0.05 | Slight positive |

### 4.2 Feature Intercorrelations
- **LSCI variables highly correlated (0.59-0.89)** - expected, same underlying metric
- **Weather features moderately correlated (0.12-0.46)** - regional climate patterns
- **Cyclone features strongly correlated (0.71)** - wind and active status linked

---

## 5. CRITICAL DATA QUALITY ISSUES

### 5.1 Missing Data Problem
**Severity: HIGH**
- Final ML dataset has 369,216 missing values
- Many cyclone features 85-100% missing
- North Pacific features 59% missing (only available from 2018)
- **Recommendation:** Use engineered dataset (no missing values) for modeling

### 5.2 Distribution Issues
1. **Price right-skewness:** Consider log-transform for modeling
2. **Zero-inflation in cyclone data:** 56-99% zeros depending on zone
3. **Bimodal LSCI:** Multiple peaks suggest different port tiers

### 5.3 Temporal Considerations
- **US West Coast routes start in 2018** - 3 years less data than Europe routes
- **COVID spike visible** - 2021-2022 prices spike to $14,000+
- **Seasonal patterns** present - peak season flag captures some of this

---

## 6. SUMMARY STATISTICS TABLE

### Target Variable: Price_USD
| Statistic | Value |
|-----------|-------|
| Count | 9,800 |
| Mean | $3,931.71 |
| Median | $2,196.00 |
| Std Dev | $3,569.94 |
| Min | $312.00 |
| 25th Percentile | $1,581.00 |
| 75th Percentile | $5,177.75 |
| Max | $15,185.00 |
| Skewness | 1.465 (right-skewed) |
| Kurtosis | 1.015 (platykurtic) |

### Dataset Comparison
| Dataset | Rows | Columns | Missing Values | Date Range |
|---------|------|---------|----------------|------------|
| final_ml_dataset | 9,800 | 76 | 369,216 | 2015-2026 |
| engineered_ml_dataset | 9,800 | 104 | 0 | 2015-2026 |
| xeneta_combined | 9,800 | 3 | 0 | 2015-2026 |
| xeneta_simple | 2,829 | 2 | 0 | 2015-2026 |
| weather_zones_2015 | 365 | 37 | 0 | 2015 only |

---

## 7. RECOMMENDATIONS

1. **Use engineered_ml_dataset** for modeling - no missing values, rich features
2. **Log-transform Price_USD** to handle right skewness
3. **Consider route-specific models** - different price levels and volatilities
4. **Feature selection needed** - 102 features, many highly correlated
5. **Validate cyclone imputation** - understand how zeros were filled
6. **Investigate LSCI=0 values** - may be data quality issue or true zeros
7. **Time series cross-validation** - respect temporal ordering

---

## Appendix: Files Generated
- `01_price_distribution_by_route.png`
- `02_overall_price_analysis.png`
- `03_price_timeseries.png`
- `04_weather_distributions.png`
- `05_correlation_heatmap.png`
- `06_missing_data_pattern.png`
- `07_lsci_distributions.png`
- `initial_analysis.py` (analysis script)
- `generate_visualizations.py` (visualization script)
