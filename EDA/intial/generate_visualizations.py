import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
fig_dir = Path(r"D:\Internproj\EDA\intial")
fig_dir.mkdir(parents=True, exist_ok=True)

# Load datasets
final_df = pd.read_csv(r"D:\Internproj\new dataset\final_ml_dataset.csv")
eng_df = pd.read_csv(r"D:\Internproj\new dataset\engineered_ml_dataset.csv")
xeneta_df = pd.read_csv(r"D:\Internproj\Index\xeneta\xeneta_xsi_combined.csv")
weather_df = pd.read_csv(r"D:\Internproj\new dataset\weather_zones_2015.csv")

# Parse dates
for df, col in [(final_df, 'Date'), (eng_df, 'Date'), (xeneta_df, 'Date'), (weather_df, 'date')]:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# =====================
# FIGURE 1: Price Distributions by Route
# =====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Price_USD Distribution by Route', fontsize=16, fontweight='bold')

routes = xeneta_df['Route'].unique()
for idx, route in enumerate(routes):
    ax = axes[idx//2, idx%2]
    data = xeneta_df[xeneta_df['Route'] == route]['Price_USD']
    ax.hist(data, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${data.mean():.0f}')
    ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${data.median():.0f}')
    ax.set_title(route.replace('_', '\n'), fontsize=10)
    ax.set_xlabel('Price (USD)')
    ax.set_ylabel('Frequency')
    ax.legend()

plt.tight_layout()
plt.savefig(fig_dir / "01_price_distribution_by_route.png", dpi=300, bbox_inches='tight')
plt.close()

# =====================
# FIGURE 2: Overall Price Distribution (All Routes Combined)
# =====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(xeneta_df['Price_USD'], bins=60, alpha=0.7, color='coral', edgecolor='black')
axes[0].axvline(xeneta_df['Price_USD'].mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: ${xeneta_df["Price_USD"].mean():.0f}')
axes[0].axvline(xeneta_df['Price_USD'].median(), color='green', linestyle='--', linewidth=2, 
                label=f'Median: ${xeneta_df["Price_USD"].median():.0f}')
axes[0].set_title('Overall Price Distribution (All Routes)', fontweight='bold')
axes[0].set_xlabel('Price (USD)')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Box plot by route
xeneta_df.boxplot(column='Price_USD', by='Route', ax=axes[1])
axes[1].set_title('Price Distribution by Route (Box Plot)', fontweight='bold')
axes[1].set_xlabel('Route')
axes[1].set_ylabel('Price (USD)')
axes[1].tick_params(axis='x', rotation=45)

plt.suptitle('')
plt.tight_layout()
plt.savefig(fig_dir / "02_overall_price_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

# =====================
# FIGURE 3: Time Series of Prices
# =====================
fig, ax = plt.subplots(figsize=(14, 6))

for route in routes:
    route_data = xeneta_df[xeneta_df['Route'] == route].sort_values('Date')
    ax.plot(route_data['Date'], route_data['Price_USD'], label=route.replace('_', ' '), alpha=0.8)

ax.set_title('Price Time Series by Route (2015-2026)', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / "03_price_timeseries.png", dpi=300, bbox_inches='tight')
plt.close()

# =====================
# FIGURE 4: Weather Variables Distribution (2015)
# =====================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Weather Zone Distributions (2015)', fontsize=16, fontweight='bold')

weather_vars = ['air_South_China_Sea', 'air_Indian_Ocean', 'air_Mediterranean_Sea',
                'uwnd_South_China_Sea', 'uwnd_Indian_Ocean', 'uwnd_Mediterranean_Sea']

for idx, var in enumerate(weather_vars):
    ax = axes[idx//3, idx%3]
    data = weather_df[var].dropna()
    ax.hist(data, bins=30, alpha=0.7, color='teal', edgecolor='black')
    ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.1f}')
    ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data.median():.1f}')
    ax.set_title(var.replace('_', ' '))
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.legend()

plt.tight_layout()
plt.savefig(fig_dir / "04_weather_distributions.png", dpi=300, bbox_inches='tight')
plt.close()

# =====================
# FIGURE 5: Feature Correlation Heatmap (Top Features)
# =====================
fig, ax = plt.subplots(figsize=(12, 10))

# Select key features for correlation
key_features = ['Price_USD', 'Origin_LSCI', 'Dest_LSCI', 'Route_Mean_LSCI', 
                'Route_Mean_air', 'Route_Mean_slp', 'Route_Mean_wind_speed',
                'Route_Max_cyclone_wind', 'Route_cyclone_active']

corr_matrix = eng_df[key_features].corr()
im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

# Add colorbar
plt.colorbar(im, ax=ax, label='Correlation')

# Set ticks
ax.set_xticks(np.arange(len(key_features)))
ax.set_yticks(np.arange(len(key_features)))
ax.set_xticklabels([f.replace('_', '\n') for f in key_features], rotation=45, ha='right')
ax.set_yticklabels([f.replace('_', '\n') for f in key_features])

# Add correlation values
for i in range(len(key_features)):
    for j in range(len(key_features)):
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                      ha="center", va="center", color="black", fontsize=9)

ax.set_title('Feature Correlation Heatmap (Engineered Dataset)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(fig_dir / "05_correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# =====================
# FIGURE 6: Missing Data Pattern
# =====================
fig, ax = plt.subplots(figsize=(14, 8))

missing_data = final_df.isnull().sum()
missing_data = missing_data[missing_data > 0].sort_values(ascending=True)

if len(missing_data) > 0:
    colors = ['red' if x > 5000 else 'orange' if x > 1000 else 'yellow' for x in missing_data.values]
    ax.barh(range(len(missing_data)), missing_data.values, color=colors)
    ax.set_yticks(range(len(missing_data)))
    ax.set_yticklabels([idx.replace('_', ' ') for idx in missing_data.index], fontsize=8)
    ax.set_xlabel('Number of Missing Values')
    ax.set_title('Missing Data Pattern (Final ML Dataset)', fontsize=14, fontweight='bold')
    ax.axvline(x=len(final_df)*0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend()
else:
    ax.text(0.5, 0.5, 'No Missing Data', ha='center', va='center', fontsize=16)

plt.tight_layout()
plt.savefig(fig_dir / "06_missing_data_pattern.png", dpi=300, bbox_inches='tight')
plt.close()

# =====================
# FIGURE 7: LSCI Distribution
# =====================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

lsci_vars = ['Origin_LSCI', 'Dest_LSCI', 'Route_Mean_LSCI']
colors = ['skyblue', 'lightgreen', 'salmon']

for idx, var in enumerate(lsci_vars):
    data = eng_df[var].dropna()
    axes[idx].hist(data, bins=40, alpha=0.7, color=colors[idx], edgecolor='black')
    axes[idx].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.1f}')
    axes[idx].axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data.median():.1f}')
    axes[idx].set_title(f'{var} Distribution')
    axes[idx].set_xlabel('LSCI Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()

plt.suptitle('LSCI (Liner Shipping Connectivity Index) Distributions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / "07_lsci_distributions.png", dpi=300, bbox_inches='tight')
plt.close()

print("All visualizations generated successfully!")
print(f"\nFiles saved to: {fig_dir}")
print("\nGenerated files:")
for f in sorted(fig_dir.glob("*.png")):
    print(f"  - {f.name}")
