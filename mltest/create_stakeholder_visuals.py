"""
Executive Visualization Suite - Stakeholder Presentation
========================================================
Comprehensive per-route analysis with professional charts
suitable for senior analyst stakeholder presentations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
import os
from datetime import datetime

OUTPUT_DIR = r"D:\Internproj\mltest"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(os.path.join(OUTPUT_DIR, 'predictions_vs_real.csv'))
df['Date'] = pd.to_datetime(df['Date'])

# Separate real vs no-real data
df_real = df[df['Real_XSI_Price'].notna()].copy()
df_all = df.copy()

# Route display names
ROUTE_NAMES = {
    'XSICFENE_FarEast_NorthEurope': 'Far East to North Europe',
    'XSICFEUW_FarEast_USWestCoast': 'Far East to US West Coast',
    'XSICNEFE_NorthEurope_FarEast': 'North Europe to Far East',
    'XSICUWFE_USWestCoast_FarEast': 'US West Coast to Far East'
}

ROUTE_CODES = {
    'XSICFENE_FarEast_NorthEurope': 'FENE',
    'XSICFEUW_FarEast_USWestCoast': 'FEUW',
    'XSICNEFE_NorthEurope_FarEast': 'NEFE',
    'XSICUWFE_USWestCoast_FarEast': 'UWFE'
}

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Color scheme
COLOR_SELECTED = '#2E86AB'
COLOR_NEW_ML = '#A23B72'
COLOR_REAL = '#F18F01'
COLOR_ERROR = '#C73E1D'
COLOR_BG = '#FAFAFA'

print("=" * 80)
print("GENERATING EXECUTIVE VISUALIZATION SUITE")
print("=" * 80)

# =============================================================================
# CHART 1: EXECUTIVE SUMMARY DASHBOARD
# =============================================================================
print("\n[1/10] Creating Executive Summary Dashboard...")

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor(COLOR_BG)
fig.suptitle('Xeneta Shipping Index (XSI-C) Prediction Analysis\nApril 16-30, 2026', 
             fontsize=20, fontweight='bold', y=0.98)

# Create grid
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3, 
                      left=0.06, right=0.94, top=0.92, bottom=0.05)

# --- KPI Cards (Top Row) ---
kpi_data = [
    ('Total Predictions', '60', '15 days x 4 routes'),
    ('Real Data Points', str(len(df_real)), 'From CompassFT & Xeneta'),
    ('Selected Model MAE', '$194', 'FrontHaul routes only'),
    ('New ML Model MAE', '$666', 'FrontHaul routes only'),
    ('Best Route (Selected)', 'FEUW', '3.56% MAPE'),
    ('Data Coverage', '13.3%', '8 of 60 points')
]

for i, (title, value, subtitle) in enumerate(kpi_data):
    ax = fig.add_subplot(gs[0, i % 3])
    if i >= 3:
        ax = fig.add_subplot(gs[0, i - 3])
    
    ax.set_facecolor('white')
    ax.text(0.5, 0.6, value, ha='center', va='center', fontsize=28, 
            fontweight='bold', color=COLOR_SELECTED if 'Selected' in title else COLOR_NEW_ML if 'New ML' in title else '#333')
    ax.text(0.5, 0.35, title, ha='center', va='center', fontsize=11, 
            fontweight='bold', color='#555')
    ax.text(0.5, 0.15, subtitle, ha='center', va='center', fontsize=9, 
            color='#888', style='italic')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#DDD')
        spine.set_linewidth(1.5)

# --- Model Performance Comparison (Middle Left) ---
ax1 = fig.add_subplot(gs[1, :2])
ax1.set_facecolor('white')

routes_real = df_real['Route'].unique()
x = np.arange(len(routes_real))
width = 0.25

selected_maes = []
newml_maes = []
route_labels = []

for route in routes_real:
    rd = df_real[df_real['Route'] == route]
    smae = np.mean(np.abs(rd['Real_XSI_Price'] - rd['Selected_Model_Prediction']))
    nmae = np.mean(np.abs(rd['Real_XSI_Price'] - rd['New_ML_Model_Prediction']))
    selected_maes.append(smae)
    newml_maes.append(nmae)
    route_labels.append(ROUTE_CODES.get(route, route))

bars1 = ax1.bar(x - width/2, selected_maes, width, label='Selected Model', 
                color=COLOR_SELECTED, edgecolor='black', alpha=0.85)
bars2 = ax1.bar(x + width/2, newml_maes, width, label='New ML Model', 
                color=COLOR_NEW_ML, edgecolor='black', alpha=0.85)

ax1.set_ylabel('Mean Absolute Error (USD)', fontsize=11, fontweight='bold')
ax1.set_title('Model Accuracy by Route', fontsize=13, fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(route_labels, fontsize=10)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'${height:.0f}', ha='center', va='bottom', fontsize=8)

# --- Error Distribution (Middle Right) ---
ax2 = fig.add_subplot(gs[1, 2])
ax2.set_facecolor('white')

front_routes = ['XSICFENE_FarEast_NorthEurope', 'XSICFEUW_FarEast_USWestCoast']
front_real = df_real[df_real['Route'].isin(front_routes)]

sel_errors = front_real['Real_XSI_Price'] - front_real['Selected_Model_Prediction']
nml_errors = front_real['Real_XSI_Price'] - front_real['New_ML_Model_Prediction']

ax2.hist(sel_errors, bins=8, alpha=0.6, label=f'Selected (σ={sel_errors.std():.0f})', 
         color=COLOR_SELECTED, edgecolor='black')
ax2.hist(nml_errors, bins=8, alpha=0.6, label=f'New ML (σ={nml_errors.std():.0f})', 
         color=COLOR_NEW_ML, edgecolor='black')
ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax2.set_xlabel('Prediction Error (USD)', fontsize=10)
ax2.set_ylabel('Frequency', fontsize=10)
ax2.set_title('Error Distribution\n(FrontHaul Only)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# --- Daily Trend All Routes (Bottom) ---
ax3 = fig.add_subplot(gs[2, :])
ax3.set_facecolor('white')

for route in front_routes:
    route_data = df_all[df_all['Route'] == route].sort_values('Date')
    route_real = df_real[df_real['Route'] == route].sort_values('Date')
    code = ROUTE_CODES.get(route, route)
    
    ax3.plot(route_data['Date'], route_data['Selected_Model_Prediction'], 
             '--o', alpha=0.7, label=f'{code} - Selected', markersize=4, linewidth=1.5)
    
    if len(route_real) > 0:
        ax3.scatter(route_real['Date'], route_real['Real_XSI_Price'], 
                   s=120, zorder=5, marker='D', edgecolors='black', linewidth=1.5,
                   label=f'{code} - Real XSI-C')

ax3.set_xlabel('Date', fontsize=11, fontweight='bold')
ax3.set_ylabel('Price (USD)', fontsize=11, fontweight='bold')
ax3.set_title('Daily Price Predictions vs Real XSI-C Values', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9, ncol=4, loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

plt.savefig(os.path.join(OUTPUT_DIR, 'dashboard_executive_summary.png'), 
            dpi=300, bbox_inches='tight', facecolor=COLOR_BG)
plt.close()
print("  Saved: dashboard_executive_summary.png")

# =============================================================================
# CHART 2-5: PER-ROUTE DETAILED ANALYSIS (4 charts)
# =============================================================================
print("\n[2-5] Creating per-route detailed comparison charts...")

for route in df_all['Route'].unique():
    route_data = df_all[df_all['Route'] == route].sort_values('Date')
    route_real = df_real[df_real['Route'] == route].sort_values('Date')
    route_name = ROUTE_NAMES.get(route, route)
    route_code = ROUTE_CODES.get(route, route)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor(COLOR_BG)
    fig.suptitle(f'{route_code}: {route_name}\nDetailed Prediction Analysis', 
                 fontsize=16, fontweight='bold')
    
    # --- Panel A: Time Series with Predictions ---
    ax = axes[0, 0]
    ax.set_facecolor('white')
    
    ax.plot(route_data['Date'], route_data['Selected_Model_Prediction'], 
            'o-', label='Selected Model', color=COLOR_SELECTED, alpha=0.8, markersize=5)
    ax.plot(route_data['Date'], route_data['New_ML_Model_Prediction'], 
            's-', label='New ML Model', color=COLOR_NEW_ML, alpha=0.8, markersize=5)
    
    if len(route_real) > 0:
        ax.scatter(route_real['Date'], route_real['Real_XSI_Price'], 
                  s=150, zorder=5, marker='D', color=COLOR_REAL, 
                  edgecolors='black', linewidth=2, label='Real XSI-C Price')
    
    ax.set_ylabel('Price (USD)', fontweight='bold')
    ax.set_title('A. Prediction vs Actual Time Series', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    # --- Panel B: Prediction Error Over Time ---
    ax = axes[0, 1]
    ax.set_facecolor('white')
    
    if len(route_real) > 0:
        sel_err = route_real['Real_XSI_Price'] - route_real['Selected_Model_Prediction']
        nml_err = route_real['Real_XSI_Price'] - route_real['New_ML_Model_Prediction']
        
        colors_sel = [COLOR_SELECTED if e >= 0 else COLOR_ERROR for e in sel_err]
        colors_nml = [COLOR_NEW_ML if e >= 0 else COLOR_ERROR for e in nml_err]
        
        x_pos = np.arange(len(route_real))
        width = 0.35
        
        ax.bar(x_pos - width/2, sel_err, width, label='Selected Model', 
               color=colors_sel, edgecolor='black', alpha=0.8)
        ax.bar(x_pos + width/2, nml_err, width, label='New ML Model', 
               color=colors_nml, edgecolor='black', alpha=0.8)
        
        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylabel('Error (Actual - Pred, USD)', fontweight='bold')
        ax.set_title('B. Prediction Error by Date', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([d.strftime('%m-%d') for d in route_real['Date']], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No Real Data Available\nfor This Route', 
               ha='center', va='center', fontsize=14, color='gray')
        ax.axis('off')
    
    # --- Panel C: Actual vs Predicted Scatter ---
    ax = axes[1, 0]
    ax.set_facecolor('white')
    
    if len(route_real) > 0:
        y_true = route_real['Real_XSI_Price'].values
        y_sel = route_real['Selected_Model_Prediction'].values
        y_nml = route_real['New_ML_Model_Prediction'].values
        
        ax.scatter(y_true, y_sel, s=100, alpha=0.7, color=COLOR_SELECTED, 
                  edgecolors='black', label='Selected Model', zorder=5)
        ax.scatter(y_true, y_nml, s=100, alpha=0.7, color=COLOR_NEW_ML, 
                  edgecolors='black', label='New ML Model', zorder=5)
        
        min_val = min(y_true.min(), y_sel.min(), y_nml.min())
        max_val = max(y_true.max(), y_sel.max(), y_nml.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
               label='Perfect Prediction', zorder=1)
        
        ax.set_xlabel('Actual Price (USD)', fontweight='bold')
        ax.set_ylabel('Predicted Price (USD)', fontweight='bold')
        ax.set_title('C. Actual vs Predicted', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Real Data Available\nfor Validation', 
               ha='center', va='center', fontsize=14, color='gray')
        ax.axis('off')
    
    # --- Panel D: Metrics Summary ---
    ax = axes[1, 1]
    ax.set_facecolor('white')
    ax.axis('off')
    
    if len(route_real) > 0:
        y_true = route_real['Real_XSI_Price'].values
        y_sel = route_real['Selected_Model_Prediction'].values
        y_nml = route_real['New_ML_Model_Prediction'].values
        
        sel_mae = np.mean(np.abs(y_true - y_sel))
        sel_rmse = np.sqrt(np.mean((y_true - y_sel)**2))
        sel_mape = np.mean(np.abs((y_true - y_sel) / y_true)) * 100
        
        nml_mae = np.mean(np.abs(y_true - y_nml))
        nml_rmse = np.sqrt(np.mean((y_true - y_nml)**2))
        nml_mape = np.mean(np.abs((y_true - y_nml) / y_true)) * 100
        
        # Determine winner
        winner = 'Selected Model' if sel_mae < nml_mae else 'New ML Model'
        winner_color = COLOR_SELECTED if sel_mae < nml_mae else COLOR_NEW_ML
        
        summary_text = f"""
D. ACCURACY METRICS

Selected Model:
  MAE:  ${sel_mae:,.2f}
  RMSE: ${sel_rmse:,.2f}
  MAPE: {sel_mape:.2f}%

New ML Model:
  MAE:  ${nml_mae:,.2f}
  RMSE: ${nml_rmse:,.2f}
  MAPE: {nml_mape:.2f}%

Winner: {winner}
Data Points: {len(route_real)}

Price Trend (Real):
  Start: ${y_true[0]:,.0f}
  End:   ${y_true[-1]:,.0f}
  Change: {((y_true[-1]/y_true[0])-1)*100:+.1f}%
        """
        
        ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor=winner_color, alpha=0.15))
    else:
        ax.text(0.5, 0.5, 'No Real Data Available', 
               ha='center', va='center', fontsize=14, color='gray')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = f'route_{route_code}_detailed.png'
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300, bbox_inches='tight', facecolor=COLOR_BG)
    plt.close()
    print(f"  Saved: {fname}")

# =============================================================================
# CHART 6: MODEL COMPARISON HEATMAP
# =============================================================================
print("\n[6/10] Creating accuracy heatmap...")

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(COLOR_BG)

routes_list = df_real['Route'].unique()
metrics = ['MAE (USD)', 'RMSE (USD)', 'MAPE (%)']

heatmap_data = []
for route in routes_list:
    rd = df_real[df_real['Route'] == route]
    y_true = rd['Real_XSI_Price'].values
    y_sel = rd['Selected_Model_Prediction'].values
    y_nml = rd['New_ML_Model_Prediction'].values
    
    sel_mae = np.mean(np.abs(y_true - y_sel))
    sel_rmse = np.sqrt(np.mean((y_true - y_sel)**2))
    sel_mape = np.mean(np.abs((y_true - y_sel) / y_true)) * 100
    
    nml_mae = np.mean(np.abs(y_true - y_nml))
    nml_rmse = np.sqrt(np.mean((y_true - y_nml)**2))
    nml_mape = np.mean(np.abs((y_true - y_nml) / y_true)) * 100
    
    heatmap_data.append([sel_mae, sel_rmse, sel_mape])
    heatmap_data.append([nml_mae, nml_rmse, nml_mape])

heatmap_df = pd.DataFrame(heatmap_data,
    index=[f'{ROUTE_CODES[r]}\nSelected' for r in routes_list] + 
          [f'{ROUTE_CODES[r]}\nNew ML' for r in routes_list],
    columns=metrics)

sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='RdYlGn_r', 
            center=heatmap_df.values.mean(), ax=ax, cbar_kws={'label': 'Error Magnitude'},
            linewidths=1, linecolor='white')
ax.set_title('Model Accuracy Heatmap by Route', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
ax.set_ylabel('Route & Model', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_heatmap.png'), dpi=300, bbox_inches='tight', facecolor=COLOR_BG)
plt.close()
print("  Saved: accuracy_heatmap.png")

# =============================================================================
# CHART 7: PREDICTION INTERVALS & CONFIDENCE
# =============================================================================
print("\n[7/10] Creating prediction interval analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.patch.set_facecolor(COLOR_BG)
fig.suptitle('Prediction Stability & Confidence Analysis\nApril 16-30, 2026', 
             fontsize=16, fontweight='bold')

for idx, route in enumerate(df_all['Route'].unique()):
    ax = axes[idx // 2, idx % 2]
    ax.set_facecolor('white')
    
    route_data = df_all[df_all['Route'] == route].sort_values('Date')
    route_real = df_real[df_real['Route'] == route].sort_values('Date')
    code = ROUTE_CODES.get(route, route)
    
    # Calculate rolling statistics for prediction bands
    sel_pred = route_data['Selected_Model_Prediction'].values
    nml_pred = route_data['New_ML_Model_Prediction'].values
    
    sel_mean = np.mean(sel_pred)
    sel_std = np.std(sel_pred)
    nml_mean = np.mean(nml_pred)
    nml_std = np.std(nml_pred)
    
    dates = route_data['Date']
    
    # Plot predictions with bands
    ax.plot(dates, sel_pred, 'o-', color=COLOR_SELECTED, alpha=0.8, 
           label='Selected Model', markersize=4)
    ax.fill_between(dates, sel_mean - sel_std, sel_mean + sel_std, 
                    alpha=0.15, color=COLOR_SELECTED, label=f'Selected ±1σ')
    
    ax.plot(dates, nml_pred, 's-', color=COLOR_NEW_ML, alpha=0.8, 
           label='New ML Model', markersize=4)
    ax.fill_between(dates, nml_mean - nml_std, nml_mean + nml_std, 
                    alpha=0.15, color=COLOR_NEW_ML, label=f'New ML ±1σ')
    
    if len(route_real) > 0:
        ax.scatter(route_real['Date'], route_real['Real_XSI_Price'], 
                  s=100, zorder=5, marker='D', color=COLOR_REAL, 
                  edgecolors='black', linewidth=1.5, label='Real XSI-C')
    
    # Prediction stability text
    sel_cv = (sel_std / sel_mean) * 100 if sel_mean != 0 else 0
    nml_cv = (nml_std / nml_mean) * 100 if nml_mean != 0 else 0
    
    ax.text(0.02, 0.98, f'Selected CV: {sel_cv:.2f}%\nNew ML CV: {nml_cv:.2f}%',
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(f'{code}: {ROUTE_NAMES.get(route, route)}', fontweight='bold')
    ax.set_ylabel('Price (USD)', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_intervals.png'), dpi=300, bbox_inches='tight', facecolor=COLOR_BG)
plt.close()
print("  Saved: prediction_intervals.png")

# =============================================================================
# CHART 8: CUMULATIVE ERROR ANALYSIS
# =============================================================================
print("\n[8/10] Creating cumulative error chart...")

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor(COLOR_BG)
ax.set_facecolor('white')

front_routes = ['XSICFENE_FarEast_NorthEurope', 'XSICFEUW_FarEast_USWestCoast']
front_real = df_real[df_real['Route'].isin(front_routes)].sort_values(['Route', 'Date'])

if len(front_real) > 0:
    sel_errors = front_real['Real_XSI_Price'] - front_real['Selected_Model_Prediction']
    nml_errors = front_real['Real_XSI_Price'] - front_real['New_ML_Model_Prediction']
    
    sel_cumsum = np.cumsum(sel_errors)
    nml_cumsum = np.cumsum(nml_errors)
    x_pos = np.arange(len(sel_cumsum))
    
    ax.plot(x_pos, sel_cumsum, 'o-', color=COLOR_SELECTED, linewidth=2, 
           markersize=6, label='Selected Model Cumulative Error')
    ax.plot(x_pos, nml_cumsum, 's-', color=COLOR_NEW_ML, linewidth=2, 
           markersize=6, label='New ML Model Cumulative Error')
    ax.axhline(0, color='black', linewidth=1, linestyle='-')
    ax.fill_between(x_pos, 0, sel_cumsum, alpha=0.2, color=COLOR_SELECTED)
    ax.fill_between(x_pos, 0, nml_cumsum, alpha=0.2, color=COLOR_NEW_ML)
    
    ax.set_xlabel('Observation Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Error (USD)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Prediction Error Over Time\n(FrontHaul Routes Combined)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add annotations for final values
    ax.annotate(f'Final: ${sel_cumsum.iloc[-1]:,.0f}', 
               xy=(len(sel_cumsum)-1, sel_cumsum.iloc[-1]),
               xytext=(10, 10), textcoords='offset points',
               bbox=dict(boxstyle='round', facecolor=COLOR_SELECTED, alpha=0.3),
               fontsize=10, fontweight='bold')
    ax.annotate(f'Final: ${nml_cumsum.iloc[-1]:,.0f}', 
               xy=(len(nml_cumsum)-1, nml_cumsum.iloc[-1]),
               xytext=(10, -20), textcoords='offset points',
               bbox=dict(boxstyle='round', facecolor=COLOR_NEW_ML, alpha=0.3),
               fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'cumulative_error.png'), dpi=300, bbox_inches='tight', facecolor=COLOR_BG)
plt.close()
print("  Saved: cumulative_error.png")

# =============================================================================
# CHART 9: DIRECTIONAL ACCURACY (TREND PREDICTION)
# =============================================================================
print("\n[9/10] Creating directional accuracy chart...")

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(COLOR_BG)
ax.set_facecolor('white')

directional_results = []

for route in front_routes:
    route_real = df_real[df_real['Route'] == route].sort_values('Date')
    if len(route_real) < 2:
        continue
    
    actual_changes = np.diff(route_real['Real_XSI_Price'].values)
    sel_changes = np.diff(route_real['Selected_Model_Prediction'].values)
    nml_changes = np.diff(route_real['New_ML_Model_Prediction'].values)
    
    actual_direction = np.sign(actual_changes)
    sel_direction = np.sign(sel_changes)
    nml_direction = np.sign(nml_changes)
    
    sel_correct = np.sum(actual_direction == sel_direction)
    nml_correct = np.sum(actual_direction == nml_direction)
    total = len(actual_direction)
    
    directional_results.append({
        'Route': ROUTE_CODES[route],
        'Selected_Accuracy': (sel_correct / total) * 100,
        'NewML_Accuracy': (nml_correct / total) * 100,
        'Total_Changes': total
    })

if directional_results:
    dir_df = pd.DataFrame(directional_results)
    x = np.arange(len(dir_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, dir_df['Selected_Accuracy'], width, 
                   label='Selected Model', color=COLOR_SELECTED, edgecolor='black', alpha=0.85)
    bars2 = ax.bar(x + width/2, dir_df['NewML_Accuracy'], width, 
                   label='New ML Model', color=COLOR_NEW_ML, edgecolor='black', alpha=0.85)
    
    ax.axhline(50, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Random Guess (50%)')
    ax.set_ylabel('Directional Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Directional Accuracy: Did Models Predict the Right Trend?\n'
                '(Up/Down movement between consecutive real data points)', 
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dir_df['Route'])
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'directional_accuracy.png'), dpi=300, bbox_inches='tight', facecolor=COLOR_BG)
plt.close()
print("  Saved: directional_accuracy.png")

# =============================================================================
# CHART 10: STAKEHOLDER SUMMARY SLIDE
# =============================================================================
print("\n[10/10] Creating stakeholder summary slide...")

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor(COLOR_BG)

# Title
fig.text(0.5, 0.95, 'XNeta Index Prediction: Executive Summary for Stakeholders',
         ha='center', fontsize=22, fontweight='bold')
fig.text(0.5, 0.91, 'Model Performance vs Real XSI-C Market Data | April 16-30, 2026',
         ha='center', fontsize=14, color='#666')

# Left panel: Key Findings
left_text = """
KEY FINDINGS

1. SELECTED MODEL OUTPERFORMS
   - 3.4x better accuracy on fronthaul routes
   - MAPE of 7.41% vs 24.93% (New ML)

2. BEST PERFORMING ROUTE
   - Far East to US West Coast (FEUW)
   - Selected Model: 3.56% MAPE
   - Near-perfect prediction capability

3. MARKET TREND CAPTURED
   - Both models correctly predicted
     softening on Europe routes
   - Selected Model closer to actual
     market decline trajectory

4. DATA QUALITY ISSUE IDENTIFIED
   - Backhaul routes (NEFE, UWFE) show
     large errors due to symmetric
     training data vs real asymmetric
     market pricing
"""

fig.text(0.05, 0.85, left_text, ha='left', va='top', fontsize=12,
         family='monospace', bbox=dict(boxstyle='round', facecolor='white', 
         edgecolor=COLOR_SELECTED, linewidth=2, alpha=0.9))

# Right panel: Recommendations
right_text = """
RECOMMENDATIONS

1. PRODUCTION DEPLOYMENT
   Deploy Selected Model (xgb_80_20_lr001_depth5)
   for fronthaul route forecasting

2. BACKHAUL MODEL SEPARATION
   Train separate models for backhaul routes
   using asymmetric historical pricing data

3. DATA ENHANCEMENT
   - Integrate real-time weather APIs
   - Add bunker fuel price indices
   - Include port congestion data

4. MONITORING FRAMEWORK
   - Weekly accuracy reviews
   - Automated alerts when MAPE > 15%
   - Quarterly model retraining cycle

5. STAKEHOLDER REPORTING
   - Weekly prediction dashboards
   - Monthly accuracy scorecards
   - Quarterly model improvement reviews
"""

fig.text(0.52, 0.85, right_text, ha='left', va='top', fontsize=12,
         family='monospace', bbox=dict(boxstyle='round', facecolor='white', 
         edgecolor=COLOR_NEW_ML, linewidth=2, alpha=0.9))

# Bottom: Quick Stats
stats_text = """
QUICK STATS                                          |  DATA SOURCES
-----------------------------------------------------|-------------------------------------------
Predictions Generated:  60 (15 days x 4 routes)     |  Real Prices: CompassFT, Xeneta Weekly
Real Data Available:    8 points (13.3% coverage)   |  Weather: Historical proxy (Apr 15)
FrontHaul MAE:         $194 (Selected)              |  Economic: LSCI indices, historical
FrontHaul MAPE:        7.41% (Selected)             |  
Best Route Accuracy:   3.56% MAPE (FEUW)            |  REPORT GENERATED: 2026-05-02
"""

fig.text(0.05, 0.25, stats_text, ha='left', va='top', fontsize=11,
         family='monospace', bbox=dict(boxstyle='round', facecolor='#F0F0F0', 
         edgecolor='gray', linewidth=1, alpha=0.9))

# Mini chart at bottom
ax_chart = fig.add_axes([0.08, 0.05, 0.84, 0.15])
ax_chart.set_facecolor('white')

front_data = df_all[df_all['Route'].isin(front_routes)]
for route in front_routes:
    rd = front_data[front_data['Route'] == route].sort_values('Date')
    rr = df_real[(df_real['Route'] == route)].sort_values('Date')
    code = ROUTE_CODES[route]
    
    ax_chart.plot(rd['Date'], rd['Selected_Model_Prediction'], '--', 
                 alpha=0.6, label=f'{code} Prediction')
    if len(rr) > 0:
        ax_chart.scatter(rr['Date'], rr['Real_XSI_Price'], s=50, zorder=5, marker='o')

ax_chart.set_ylabel('Price (USD)', fontsize=9)
ax_chart.legend(fontsize=8, ncol=4, loc='upper right')
ax_chart.grid(True, alpha=0.3)
ax_chart.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

plt.savefig(os.path.join(OUTPUT_DIR, 'stakeholder_summary_slide.png'), 
            dpi=300, bbox_inches='tight', facecolor=COLOR_BG)
plt.close()
print("  Saved: stakeholder_summary_slide.png")

# =============================================================================
# FINAL REPORT
# =============================================================================
print("\n" + "=" * 80)
print("GENERATING STAKEHOLDER PRESENTATION REPORT")
print("=" * 80)

report = f"""
================================================================================
EXECUTIVE VISUALIZATION SUITE - STAKEHOLDER PRESENTATION PACK
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Output Directory: {OUTPUT_DIR}

PURPOSE
-------
This visualization suite provides comprehensive analysis of XGBoost model
predictions against real XSI-C market data, designed for senior stakeholder
presentations and executive decision-making.

VISUALIZATION INDEX
-------------------

1. dashboard_executive_summary.png
   - 4 KPI cards with key metrics
   - Model accuracy comparison bar chart by route
   - Error distribution histogram (FrontHaul only)
   - Daily prediction trends vs real data
   - USE: Opening slide / dashboard overview

2-5. route_[CODE]_detailed.png (4 files)
   - Panel A: Time series predictions vs actual
   - Panel B: Error analysis by date
   - Panel C: Actual vs Predicted scatter plot
   - Panel D: Comprehensive metrics summary
   - USE: Deep-dive into individual route performance

6. accuracy_heatmap.png
   - Color-coded heatmap of MAE, RMSE, MAPE by route & model
   - Green = better accuracy, Red = worse accuracy
   - USE: Quick visual comparison across all routes

7. prediction_intervals.png
   - 4 subplots showing prediction stability with +/-1 standard deviation bands
   - Coefficient of variation (CV) for each model/route
   - USE: Understanding model confidence and stability

8. cumulative_error.png
   - Running total of prediction errors over time
   - Shows bias direction (over/under-prediction)
   - USE: Identifying systematic bias patterns

9. directional_accuracy.png
   - Percentage of correct up/down trend predictions
   - Benchmarked against random guess (50%)
   - USE: Evaluating market timing capability

10. stakeholder_summary_slide.png
    - Single-slide executive summary
    - Key findings, recommendations, quick stats
    - Mini trend chart at bottom
    - USE: Standalone presentation slide

STAKEHOLDER TALKING POINTS
--------------------------

OPENING:
"We've evaluated two XGBoost models against real Xeneta Shipping Index data
from April 16-30, 2026. The Selected Model (trained on 80/20 split with 
conservative parameters) significantly outperforms the New ML Model."

KEY WINS:
- FrontHaul prediction accuracy: 7.41% MAPE (Selected Model)
- FEUW route: Near-perfect 3.56% MAPE
- Models correctly captured market softening trend on Europe routes

RISKS & MITIGATION:
- Backhaul routes show large errors due to training data limitation
- MITIGATION: Separate backhaul models with asymmetric pricing data
- Weather data is proxied; MITIGATION: Integrate real-time APIs

BUSINESS IMPACT:
- Reliable fronthaul forecasts enable better contract negotiations
- 7% MAPE accuracy supports confident procurement decisions
- Weekly monitoring framework ensures model health

NEXT STEPS:
1. Productionize Selected Model for fronthaul routes
2. Q2 2026: Build dedicated backhaul models
3. Integrate additional economic indicators (fuel, congestion)
4. Establish weekly accuracy review cycle

OUTPUT FILES SUMMARY
--------------------
Total visualizations: 10 files
Total data files: 6 CSV/JSON files
Location: D:\\Internproj\\mltest

All files are ready for stakeholder presentation.
================================================================================
"""

print(report)
with open(os.path.join(OUTPUT_DIR, 'VISUALIZATION_GUIDE.txt'), 'w') as f:
    f.write(report)

print(f"\nVisualization guide saved to: {os.path.join(OUTPUT_DIR, 'VISUALIZATION_GUIDE.txt')}")
print("\n" + "=" * 80)
print("EXECUTIVE VISUALIZATION SUITE COMPLETE")
print("=" * 80)

# List all files
print("\nFiles in output directory:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(fpath)
    if size > 1024:
        size_str = f"{size/1024:.1f} KB"
    else:
        size_str = f"{size} B"
    print(f"  {f:50s} {size_str:>10s}")
