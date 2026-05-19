"""
Basic Data Design Visualization
Shows data sources, processing pipeline, and dataset structure for the ML project.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

# Load actual data stats
df = pd.read_csv(r"D:\Internproj\new dataset\final_ml_dataset.csv")
n_records = len(df)
n_features = len(df.columns)
n_routes = df['Route'].nunique()
date_min = df['Date'].min()
date_max = df['Date'].max()

fig, axes = plt.subplots(1, 2, figsize=(15.5, 8), gridspec_kw={'width_ratios': [1.1, 1]})
fig.patch.set_facecolor('#1a1a2e')

# ============================================================
# LEFT PANEL: Data Sources & Pipeline Flow
# ============================================================
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.set_facecolor('#1a1a2e')
ax1.axis('off')
ax1.set_title('DATA SOURCES & PROCESSING PIPELINE', fontsize=16, fontweight='bold',
              color='#e94560', pad=15, fontfamily='monospace')

colors = {
    'xeneta': '#e94560',
    'noaa': '#0f3460',
    'ibtracs': '#16213e',
    'lsci': '#533483',
    'intermediate': '#0d7377',
    'final': '#14ffec',
    'arrow': '#e94560',
    'text': '#ffffff'
}

# --- Row 1: Data Sources ---
source_y = 8.5
sources = [
    ("XENETA XSI", "Shipping Rates", "4 Routes\nDaily Prices (USD)", colors['xeneta']),
    ("NOAA Weather", "Gridded Climate Data", "5 Variables\n9 Sea Zones\n2015-2025", colors['noaa']),
    ("IBTrACS/JTWC", "Cyclone Database", "Wind, Pressure\nDistance to Land\nPer-zone Tracking", colors['intermediate']),
    ("UNCTAD LSCI", "Logistics Index", "Country Scores\nQuarterly Updates\n9 Countries", colors['lsci']),
]

for i, (name, desc, detail, color) in enumerate(sources):
    x = 1.1 + i * 2.4
    # Main box
    box = FancyBboxPatch((x - 0.9, source_y - 0.7), 1.8, 1.8,
                          boxstyle="round,pad=0.15", facecolor=color, alpha=0.9,
                          edgecolor='white', linewidth=2)
    ax1.add_patch(box)
    ax1.text(x, source_y + 0.7, name, ha='center', va='center',
             fontsize=9, fontweight='bold', color='white', fontfamily='monospace')
    ax1.text(x, source_y + 0.1, desc, ha='center', va='center',
             fontsize=7, color='#ddd', fontfamily='monospace')
    ax1.text(x, source_y - 0.45, detail, ha='center', va='center',
             fontsize=6.5, color='#ccc', fontfamily='monospace', linespacing=1.3)

# --- Row 2: Processing Steps ---
proc_y = 5.5
ax1.text(5, proc_y + 1.0, '▼  PROCESSING PIPELINE  ▼', ha='center', va='center',
         fontsize=12, color='#e94560', fontweight='bold', fontfamily='monospace')

steps = [
    ("Weather Zone\nAggregation", "Grid lat/lon →\n9 Sea Zones"),
    ("Wind Speed\nCalculation", "uwnd, vwnd →\nwind_speed"),
    ("Route-Sea\nMapping", "4 Routes linked\nto relevant seas"),
    ("LSCI Country\nGrouping", "Avg scores by\nFE / NE / US"),
    ("Cyclone\nIntegration", "IBTrACS →\nper-zone flags"),
    ("Date\nAlignment", "All sources →\ndaily frequency"),
]

for i, (title, desc) in enumerate(steps):
    x = 0.7 + i * 1.55
    box = FancyBboxPatch((x, proc_y - 0.9), 1.3, 1.3,
                          boxstyle="round,pad=0.12", facecolor='#0d7377', alpha=0.85,
                          edgecolor='#14ffec', linewidth=1.5)
    ax1.add_patch(box)
    ax1.text(x + 0.65, proc_y, title, ha='center', va='center',
             fontsize=7.5, fontweight='bold', color='white', fontfamily='monospace')
    ax1.text(x + 0.65, proc_y - 0.6, desc, ha='center', va='center',
             fontsize=6, color='#ccc', fontfamily='monospace', linespacing=1.3)

# --- Row 3: Final Dataset ---
final_y = 2.8
ax1.text(5, final_y + 1.0, '▼  FINAL ML DATASET  ▼', ha='center', va='center',
         fontsize=12, color='#14ffec', fontweight='bold', fontfamily='monospace')

final_box = FancyBboxPatch((1.5, final_y - 1.2), 7, 1.6,
                           boxstyle="round,pad=0.2", facecolor='#14ffec', alpha=0.2,
                           edgecolor='#14ffec', linewidth=2.5)
ax1.add_patch(final_box)

stats = [
    (f"{n_records:,}", "Records"),
    (f"{n_features}", "Features"),
    (f"{n_routes}", "Routes"),
    (f"{date_min[:4]}-{date_max[:4]}", "Period"),
]

for i, (val, label) in enumerate(stats):
    x = 2.5 + i * 1.8
    ax1.text(x, final_y + 0.1, val, ha='center', va='center',
             fontsize=18, fontweight='bold', color='#14ffec', fontfamily='monospace')
    ax1.text(x, final_y - 0.5, label, ha='center', va='center',
             fontsize=9, color='#ccc', fontfamily='monospace')

# Arrows between sections
for y_start, y_end in [(7.7, 6.6), (4.5, 3.8)]:
    ax1.annotate('', xy=(5, y_end), xytext=(5, y_start),
                 arrowprops=dict(arrowstyle='->', color='#e94560', lw=2.5))

# Connecting arrows from sources to processing
for i in range(4):
    x_src = 1.1 + i * 2.4
    x_proc = 0.7 + (i * 1.55) + 0.65
    ax1.annotate('', xy=(x_proc, proc_y + 0.4), xytext=(x_src, source_y - 0.8),
                 arrowprops=dict(arrowstyle='->', color='#e94560', lw=1.5, alpha=0.6))


# ============================================================
# RIGHT PANEL: Dataset Structure & Route Info
# ============================================================
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.set_facecolor('#1a1a2e')
ax2.axis('off')
ax2.set_title('DATASET STRUCTURE & FEATURE BREAKDOWN', fontsize=16,
              fontweight='bold', color='#14ffec', pad=15, fontfamily='monospace')

# --- Feature Categories ---
categories = [
    ("TARGET", ["Price_USD"], "#e94560"),
    ("LSCI (3)", ["Origin_LSCI", "Dest_LSCI", "Route_Mean_LSCI"], "#533483"),
    ("WEATHER (21)", ["Route_Mean_air", "Route_Mean_slp", "Route_Mean_wind_speed",
                      "air_* × 9 zones", "slp_* × 9 zones", "wind_speed_* × 9 zones"], "#0f3460"),
    ("CYCLONE (29)", ["Route_Max_cyclone_wind", "Route_Min_cyclone_slp",
                       "Route_Min_cyclone_dist", "Route_cyclone_active",
                       "cyclone_wind_* × 9 zones", "cyclone_slp_* × 9",
                       "cyclone_dist_* × 9", "cyclone_active_* × 9"], "#16213e"),
    ("IDENTIFIERS (2)", ["Date", "Route"], "#444"),
]

cat_y = 9.2
for name, features, color in categories:
    box = FancyBboxPatch((0.3, cat_y - 0.15 - len(features) * 0.22), 9.4,
                          len(features) * 0.22 + 0.3,
                          boxstyle="round,pad=0.1", facecolor=color, alpha=0.75,
                          edgecolor='white', linewidth=1)
    ax2.add_patch(box)
    ax2.text(0.5, cat_y, name, ha='left', va='top',
             fontsize=9, fontweight='bold', color='white', fontfamily='monospace')
    for j, feat in enumerate(features):
        ax2.text(1.2, cat_y - 0.15 - j * 0.22, feat, ha='left', va='top',
                 fontsize=7.5, color='#ccc', fontfamily='monospace')
    cat_y -= len(features) * 0.22 + 0.65

# --- Route Information ---
route_y = 1.0
ax2.text(5, route_y + 0.6, 'ROUTES IN DATASET', ha='center', va='center',
         fontsize=10, fontweight='bold', color='#e94560', fontfamily='monospace')

routes = [
    ("Far East → North Europe", "7 seas: SCS, Indian, Arabian, Red, Med, N.Atl, N.Sea"),
    ("North Europe → Far East", "Same 7 seas (return route)"),
    ("Far East → US West Coast", "3 seas: SCS, N.Pacific West, N.Pacific East"),
    ("US West Coast → Far East", "Same 3 seas (return route)"),
]

for i, (route, seas) in enumerate(routes):
    y = route_y - i * 0.4
    ax2.text(0.8, y, f"• {route}", ha='left', va='center',
             fontsize=8, fontweight='bold', color='#14ffec', fontfamily='monospace')
    ax2.text(5.5, y, seas, ha='left', va='center',
             fontsize=7, color='#aaa', fontfamily='monospace')

plt.tight_layout()
plt.savefig(r"D:\Internproj\Analysis\data design\01_basic_data_design.png", dpi=300,
            bbox_inches='tight', facecolor='#1a1a2e')
plt.close()
print("Basic Data Design visualization saved!")
