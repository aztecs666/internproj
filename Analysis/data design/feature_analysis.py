"""
Feature Analysis Visualization
Shows all features with their descriptions and categories.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

# Load data
df = pd.read_csv(r"D:\Internproj\new dataset\final_ml_dataset.csv")
eng_df = pd.read_csv(r"D:\Internproj\new dataset\engineered_ml_dataset.csv")

fig = plt.figure(figsize=(15.5, 8.5))
fig.patch.set_facecolor('#1a1a2e')

gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25,
                      left=0.05, right=0.95, top=0.93, bottom=0.04)

fig.suptitle('FEATURE LIST & FEATURE ANALYSIS',
             fontsize=20, fontweight='bold', color='#e94560',
             fontfamily='monospace', y=0.98)

# ============================================================
# TOP LEFT: Raw Feature Categories
# ============================================================
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('#1a1a2e')

cat_counts = {
    'Target (Price)': 1,
    'Identifiers': 2,
    'LSCI': 3,
    'Weather\n(temperature)': 11,
    'Weather\n(pressure)': 11,
    'Weather\n(wind speed)': 11,
    'Cyclone\n(wind)': 11,
    'Cyclone\n(pressure)': 11,
    'Cyclone\n(distance)': 11,
    'Cyclone\n(active flag)': 11,
    'Route Aggregates': 4,
}

colors = ['#e94560', '#666', '#533483',
          '#0f3460', '#0f3460', '#0f3460',
          '#16213e', '#16213e', '#16213e', '#16213e',
          '#14ffec']

bars = ax1.barh(range(len(cat_counts)), list(cat_counts.values()),
                color=colors, edgecolor='white', linewidth=0.5, alpha=0.9)
ax1.set_yticks(range(len(cat_counts)))
ax1.set_yticklabels(list(cat_counts.keys()), fontsize=8, color='white', fontfamily='monospace')
ax1.set_xlabel('Number of Features', fontsize=10, color='#ccc', fontfamily='monospace')
ax1.set_title('RAW FEATURE CATEGORIES (76 Total)', fontsize=12, fontweight='bold',
              color='#14ffec', fontfamily='monospace')
ax1.tick_params(colors='#aaa')
ax1.spines['bottom'].set_color('#444')
ax1.spines['left'].set_color('#444')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

for i, (val, bar) in enumerate(zip(cat_counts.values(), bars)):
    ax1.text(val + 0.3, i, str(val), va='center', fontsize=9,
             color='white', fontweight='bold', fontfamily='monospace')
ax1.invert_yaxis()

# ============================================================
# TOP RIGHT: Engineered Feature Categories
# ============================================================
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('#1a1a2e')

eng_categories = {
    'Weather Hazard\nInteractions': 2,
    'Seasonality': 3,
    'Price Time-Series': 7,
    'Lagged Weather': 6,
    'Forward Weather': 4,
    'Economic\nInteractions': 2,
}

pie_colors = ['#e94560', '#0f3460', '#533483', '#0d7377', '#16213e', '#14ffec']
wedges, texts, autotexts = ax2.pie(
    eng_categories.values(), labels=None,
    autopct='%1.0f%%', startangle=90, colors=pie_colors,
    textprops={'color': 'white', 'fontsize': 9},
    wedgeprops={'edgecolor': 'white', 'linewidth': 1}
)

ax2.legend(eng_categories.keys(), loc='center left', bbox_to_anchor=(-0.15, -0.2),
           fontsize=8, facecolor='#1a1a2e', edgecolor='#444',
           labelcolor='white', ncol=3)
ax2.set_title('ENGINEERED FEATURES (24 Created)', fontsize=12, fontweight='bold',
              color='#e94560', fontfamily='monospace')

# ============================================================
# MIDDLE FULL: Feature Descriptions Table (replaces correlation)
# ============================================================
ax3 = fig.add_subplot(gs[1, :])
ax3.set_facecolor('#1a1a2e')
ax3.axis('off')
ax3.set_title('KEY FEATURES & WHAT THEY REPRESENT', fontsize=14, fontweight='bold',
              color='#14ffec', fontfamily='monospace', pad=10)

feature_descriptions = [
    # (Feature Name, Category, Description)
    ("Price_USD", "Target", "Shipping rate in USD (what we predict)"),
    ("Route_Mean_LSCI", "LSCI", "Avg logistics index of origin & destination"),
    ("Origin_LSCI / Dest_LSCI", "LSCI", "Liner shipping connectivity score per region"),
    ("Route_Mean_air", "Weather", "Avg air temp across route's sea zones (K)"),
    ("Route_Mean_slp", "Weather", "Avg sea-level pressure across zones (hPa)"),
    ("Route_Mean_wind_speed", "Weather", "Avg wind speed from u/v components (m/s)"),
    ("Route_Mean_prate", "Weather", "Avg precipitation rate across zones (kg/m2/s)"),
    ("Route_Max_cyclone_wind", "Cyclone", "Max cyclone wind speed on route (kt)"),
    ("Route_Min_cyclone_slp", "Cyclone", "Min sea-level pressure from cyclones (hPa)"),
    ("Route_Min_cyclone_dist", "Cyclone", "Closest cyclone distance to land (km)"),
    ("Route_cyclone_active", "Cyclone", "Binary: 1 if any cyclone active on route"),
    ("air_*_SeaName", "Weather", "Air temp per individual sea zone (9 zones)"),
    ("slp_*_SeaName", "Weather", "Sea-level pressure per zone"),
    ("wind_speed_*_SeaName", "Weather", "Wind speed per zone (sqrt(u^2+v^2))"),
    ("cyclone_wind_*_SeaName", "Cyclone", "Max cyclone wind per zone"),
    ("cyclone_active_*_SeaName", "Cyclone", "Cyclone activity flag per zone"),
    ("SWI", "Engineered", "Storm Weather Index = wind_speed / slp"),
    ("Coastal_Threat", "Engineered", "Cyclone wind / distance_to_land"),
    ("Is_Peak_Season", "Engineered", "1 if month is Aug-Oct (peak shipping)"),
    ("Month / Week_of_Year", "Engineered", "Temporal features for seasonality"),
    ("Price_Lag_7d / _30d", "Engineered", "Price 7/30 days ago (per route)"),
    ("Price_Momentum_7d / _30d", "Engineered", "Rate of price change over window"),
    ("Price_Volatility_7d / _30d", "Engineered", "Std dev of price % changes"),
    ("Price_Zscore_30d", "Engineered", "How far price is from 30d mean (std units)"),
    ("Rolling_7d_Cyclone_Days", "Engineered", "Count of cyclone days in past week"),
    ("Cyclone_Wind_Lag7d/14d/30d", "Engineered", "Past cyclone wind speeds"),
    ("Cyclone_Active_Lag7d/14d/30d", "Engineered", "Past cyclone activity flags"),
    ("Wind_Speed_Lag7d/14d/30d", "Engineered", "Past wind speeds on route"),
    ("Cyclone_Wind_Forecast3d/7d", "Engineered", "Future cyclone wind (proxy)"),
    ("Congestion_Risk", "Engineered", "Cyclone wind / destination LSCI"),
    ("LSCI_30d_Delta", "Engineered", "Change in route LSCI over 30 days"),
]

# Draw as a table-like layout
cat_colors = {
    'Target': '#e94560',
    'LSCI': '#533483',
    'Weather': '#0f3460',
    'Cyclone': '#16213e',
    'Engineered': '#0d7377',
}

y_start = 0.95
line_h = 0.029
col1_x = 0.01
col2_x = 0.35
col3_x = 0.52

# Column headers
ax3.text(col1_x, y_start, 'FEATURE', fontsize=9, color='#e94560',
         fontweight='bold', fontfamily='monospace', va='top', transform=ax3.transAxes)
ax3.text(col2_x, y_start, 'CATEGORY', fontsize=9, color='#e94560',
         fontweight='bold', fontfamily='monospace', va='top', transform=ax3.transAxes)
ax3.text(col3_x, y_start, 'DESCRIPTION', fontsize=9, color='#e94560',
         fontweight='bold', fontfamily='monospace', va='top', transform=ax3.transAxes)

# Draw line
ax3.plot([col1_x, 0.98], [y_start - 0.015, y_start - 0.015],
         color='#e94560', linewidth=1, transform=ax3.transAxes)

# Feature rows (split into two columns if needed)
mid = len(feature_descriptions) // 2 + 1
for col_offset, start_idx, end_idx, x_off in [
    (0, 0, mid, 0),
    (1, mid, len(feature_descriptions), 0.5)
]:
    y = y_start - 0.04
    for feat, cat, desc in feature_descriptions[start_idx:end_idx]:
        color = cat_colors.get(cat, '#aaa')
        # Feature name
        ax3.text(col1_x + x_off, y, feat[:28], fontsize=6, color='white',
                 fontfamily='monospace', va='top', transform=ax3.transAxes)
        # Category
        ax3.text(col2_x + x_off, y, cat, fontsize=6, color=color,
                 fontweight='bold', fontfamily='monospace', va='top', transform=ax3.transAxes)
        # Description
        ax3.text(col3_x + x_off - 0.15, y, desc[:40], fontsize=6, color='#ccc',
                 fontfamily='monospace', va='top', transform=ax3.transAxes)
        y -= line_h

# Legend
legend_elements = [mpatches.Patch(facecolor=color, label=cat)
                   for cat, color in cat_colors.items()]
ax3.legend(handles=legend_elements, loc='lower center',
           ncol=5, fontsize=8, facecolor='#1a1a2e',
           edgecolor='#444', labelcolor='white',
           bbox_to_anchor=(0.5, -0.12))

# ============================================================
# BOTTOM: Feature Type Distribution + Sea Zones
# ============================================================
ax4 = fig.add_subplot(gs[2, 0])
ax4.set_facecolor('#1a1a2e')

feature_types = {
    'Lagged': ['_Lag7d', '_Lag14d', '_Lag30d'],
    'Forecast': ['_Forecast3d', '_Forecast7d'],
    'Interaction': ['SWI', 'Coastal_Threat', 'Congestion_Risk'],
    'Rolling': ['_Volatility', '_Zscore', '_Rolling'],
    'Route Agg': ['Route_Mean_', 'Route_Max_', 'Route_Min_', 'Route_cyclone_'],
    'Zone Level': ['_Arabian_', '_Indian_', '_Mediterranean_', '_North_Atlantic_',
                   '_North_Sea_', '_Red_Sea_', '_South_China_', '_North_Pacific_'],
}

type_counts = {}
eng_cols = eng_df.columns.tolist()
for type_name, patterns in feature_types.items():
    count = sum(1 for col in eng_cols if any(p in col for p in patterns))
    type_counts[type_name] = count

bars = ax4.bar(range(len(type_counts)), list(type_counts.values()),
               color=['#e94560', '#0f3460', '#533483', '#0d7377', '#14ffec', '#ff6b35'],
               edgecolor='white', linewidth=1, alpha=0.9)
ax4.set_xticks(range(len(type_counts)))
ax4.set_xticklabels(list(type_counts.keys()), fontsize=8, color='white', fontfamily='monospace')
ax4.set_ylabel('Count', fontsize=10, color='#ccc', fontfamily='monospace')
ax4.set_title('FEATURE TYPE DISTRIBUTION', fontsize=12, fontweight='bold',
              color='#14ffec', fontfamily='monospace')
ax4.tick_params(colors='#aaa')
ax4.spines['bottom'].set_color('#444')
ax4.spines['left'].set_color('#444')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

for bar, val in zip(bars, type_counts.values()):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(val), ha='center', va='bottom', fontsize=10,
             color='white', fontweight='bold', fontfamily='monospace')

# ============================================================
# BOTTOM RIGHT: Sea Zone Coverage
# ============================================================
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_facecolor('#1a1a2e')
ax5.axis('off')
ax5.set_title('SEA ZONES IN DATASET (9 Zones)', fontsize=12, fontweight='bold',
              color='#e94560', fontfamily='monospace', pad=10)

zones = [
    ("Arabian Sea", "5-25N, 50-75E", "FE-NE, FE-US"),
    ("Indian Ocean", "30S-5N, 50-100E", "FE-NE"),
    ("Mediterranean Sea", "30-45N, 0-35E", "FE-NE"),
    ("North Sea", "50-65N, 0-15E", "FE-NE"),
    ("Red Sea", "12-30N, 32-45E", "FE-NE"),
    ("South China Sea", "0-25N, 100-120E", "All Routes"),
    ("North Pacific West", "20-55N, 120-180E", "FE-US"),
    ("North Pacific East", "20-55N, 180-240E", "FE-US"),
    ("North Atlantic", "20-60N, 290-350E", "FE-NE"),
]

y = 0.88
for zone, coords, routes in zones:
    ax5.text(0.05, y, f"● {zone}", fontsize=8, color='#14ffec',
             fontweight='bold', fontfamily='monospace', va='top', transform=ax5.transAxes)
    ax5.text(0.45, y, coords, fontsize=7, color='#aaa',
             fontfamily='monospace', va='top', transform=ax5.transAxes)
    ax5.text(0.75, y, routes, fontsize=7, color='#ff6b35',
             fontfamily='monospace', va='top', transform=ax5.transAxes)
    y -= 0.095

# Headers
ax5.text(0.45, 0.97, 'Coordinates', fontsize=7, color='#666',
         fontfamily='monospace', va='top', transform=ax5.transAxes)
ax5.text(0.75, 0.97, 'Used By Routes', fontsize=7, color='#666',
         fontfamily='monospace', va='top', transform=ax5.transAxes)

plt.savefig(r"D:\Internproj\Analysis\data design\02_feature_analysis.png", dpi=300,
            bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("Feature Analysis visualization saved!")
print(f"Engineered dataset: {eng_df.shape[0]} records x {eng_df.shape[1]} features")
