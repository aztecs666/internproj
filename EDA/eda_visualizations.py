"""
Comprehensive EDA Visualizations for ML Shipping Datasets
Generates plots for:
  1. final_ml_dataset.csv (9800 x 76)
  2. engineered_ml_dataset.csv (9800 x 104)
Output: D:\Internproj\EDA\plots\
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_theme(style='whitegrid', font_scale=1.0)
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
})

# ─── Config ───────────────────────────────────────────────────────────────────
BASE = r'D:\Internproj\new dataset'
OUT  = r'D:\Internproj\EDA\plots'
os.makedirs(OUT, exist_ok=True)

DATASETS = [
    {'name': 'final_ml_dataset',   'file': os.path.join(BASE, 'final_ml_dataset.csv')},
    {'name': 'engineered_ml_dataset', 'file': os.path.join(BASE, 'engineered_ml_dataset.csv')},
]

TARGET = 'Price_USD'
ROUTE  = 'Route'
DATE   = 'Date'

# Feature groups (by prefix)
WEATHER_PREFIXES = ['air_', 'slp_', 'wind_speed_', 'cyclone_wind_', 'cyclone_slp_',
                    'cyclone_dist_', 'cyclone_active_']

# ─── Helpers ──────────────────────────────────────────────────────────────────
def save(fig, slug, tag):
    path = os.path.join(OUT, f'{slug}_{tag}.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  [OK] {path}')
    return path

def numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_feature_groups(df):
    """Group columns by prefix for organized plotting."""
    groups = {}
    for col in numeric_cols(df):
        if col == TARGET:
            continue
        prefix = col.split('_')[0] if '_' in col else col
        key = prefix
        groups.setdefault(key, []).append(col)
    return groups

def top_corr_with_target(df, target, n=20):
    corr = df[numeric_cols(df)].corr()[target].abs().sort_values(ascending=False)
    return corr.drop(target).head(n).index.tolist()

# ─── Plot Functions ───────────────────────────────────────────────────────────

def plot_overview_text(df, slug):
    path = os.path.join(OUT, f'{slug}_00_overview.txt')
    with open(path, 'w') as f:
        f.write(f'Dataset: {slug}\n')
        f.write(f'Shape: {df.shape}\n')
        f.write(f'Numeric columns: {len(numeric_cols(df))}\n')
        f.write(f'Categorical columns: {len(df.select_dtypes(include=["object"]).columns.tolist())}\n\n')
        f.write('=== Dtypes ===\n')
        f.write(df.dtypes.to_string())
        f.write('\n\n=== Missing Values ===\n')
        m = df.isnull().sum()
        f.write(m[m > 0].to_string())
        f.write(f'\n\nTotal missing: {m.sum()}\n')
        f.write('\n=== Describe (numeric) ===\n')
        f.write(df.describe().to_string())
        f.write('\n\n=== Describe (categorical) ===\n')
        f.write(df.describe(include=['object']).to_string())
        f.write('\n\n=== Route Value Counts ===\n')
        if ROUTE in df.columns:
            f.write(df[ROUTE].value_counts().to_string())
    print(f'  [OK] {path}')

def plot_target_distribution(df, slug):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    # Histogram + KDE
    sns.histplot(df[TARGET], kde=True, ax=axes[0], color='#2196F3', bins=50, edgecolor='none')
    axes[0].set_title(f'{TARGET} Distribution')
    axes[0].set_xlabel('')
    # Box plot
    sns.boxplot(x=df[TARGET], ax=axes[1], color='#2196F3')
    axes[1].set_title(f'{TARGET} Box Plot')
    # Log scale histogram
    sns.histplot(np.log1p(df[TARGET]), kde=True, ax=axes[2], color='#4CAF50', bins=50, edgecolor='none')
    axes[2].set_title(f'log(1+{TARGET}) Distribution')
    axes[2].set_xlabel('')
    fig.suptitle(f'{slug} - Target Variable Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return save(fig, slug, '01_target_distribution')

def plot_route_analysis(df, slug):
    if ROUTE not in df.columns:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Box plot by route
    sns.boxplot(x=ROUTE, y=TARGET, data=df, ax=axes[0], palette='Set2')
    axes[0].tick_params(axis='x', rotation=25)
    axes[0].set_title(f'{TARGET} by Route')
    axes[0].set_xlabel('')
    # Violin plot
    sns.violinplot(x=ROUTE, y=TARGET, data=df, ax=axes[1], palette='Set2', inner='quartile')
    axes[1].tick_params(axis='x', rotation=25)
    axes[1].set_title(f'{TARGET} by Route (Violin)')
    axes[1].set_xlabel('')
    # Route counts
    counts = df[ROUTE].value_counts()
    axes[2].barh(counts.index, counts.values, color='#2196F3')
    axes[2].set_title('Route Sample Counts')
    axes[2].set_xlabel('Count')
    fig.suptitle(f'{slug} - Route Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return save(fig, slug, '02_route_analysis')

def plot_route_stats_table(df, slug):
    if ROUTE not in df.columns:
        return
    stats = df.groupby(ROUTE)[TARGET].agg(['mean', 'median', 'std', 'min', 'max', 'count']).round(2)
    stats.columns = ['Mean', 'Median', 'Std', 'Min', 'Max', 'Count']
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('tight')
    ax.axis('off')
    tbl = ax.table(cellText=stats.values, colLabels=stats.columns,
                   rowLabels=stats.index, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.8)
    ax.set_title(f'{slug} - {TARGET} Statistics by Route', fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    return save(fig, slug, '02b_route_stats_table')

def plot_missing_values(df, slug):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    # Bar chart of missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    if len(missing) > 0:
        axes[0].barh(missing.index, missing.values, color='#FF5722')
        axes[0].set_title('Missing Values by Column')
        axes[0].set_xlabel('Missing Count')
    # Missing percentage bar
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
    missing_pct = missing_pct[missing_pct > 0]
    if len(missing_pct) > 0:
        colors = ['#4CAF50' if v < 25 else '#FFC107' if v < 50 else '#FF5722' for v in missing_pct.values]
        axes[1].barh(missing_pct.index, missing_pct.values, color=colors)
        axes[1].set_title('Missing Values (% of total)')
        axes[1].set_xlabel('Percentage %')
    fig.suptitle(f'{slug} - Missing Value Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return save(fig, slug, '03_missing_values')

def plot_correlation_heatmap(df, slug, max_feat=25):
    top_feats = top_corr_with_target(df, TARGET, n=max_feat)
    cols_to_plot = [TARGET] + top_feats
    corr = df[cols_to_plot].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                mask=mask, ax=ax, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8}, annot_kws={'size': 7})
    ax.set_title(f'{slug} - Correlation Heatmap (Top {max_feat} Features with {TARGET})',
                 fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    return save(fig, slug, '04_correlation_heatmap')

def plot_target_correlation_bar(df, slug, n=20):
    top_feats = top_corr_with_target(df, TARGET, n=n)
    corr_vals = df[numeric_cols(df)].corr()[TARGET].loc[top_feats].sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#2196F3' if v >= 0 else '#FF5722' for v in corr_vals.values]
    ax.barh(corr_vals.index, corr_vals.values, color=colors, edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel(f'Correlation with {TARGET}')
    ax.set_title(f'{slug} - Top {n} Features Correlated with {TARGET}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return save(fig, slug, '05_target_correlation_bar')

def plot_feature_distributions_by_group(df, slug):
    """Distribution plots grouped by feature prefix."""
    groups = get_feature_groups(df)
    paths = []
    for prefix, cols in groups.items():
        if len(cols) == 0:
            continue
        # Take up to 9 cols per figure
        for chunk_start in range(0, len(cols), 9):
            chunk = cols[chunk_start:chunk_start+9]
            n = len(chunk)
            ncols = min(3, n)
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
            axes = axes.flatten() if n > 1 else [axes]
            for i, col in enumerate(chunk):
                data = df[col].dropna()
                if data.empty:
                    axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(col)
                    continue
                sns.histplot(data, kde=True, ax=axes[i], color='#2196F3', bins=30, edgecolor='none')
                axes[i].set_title(f'{col}\n(mean={data.mean():.2f}, std={data.std():.2f})', fontsize=8)
                axes[i].set_xlabel('')
            for j in range(i+1, len(axes)):
                axes[j].axis('off')
            fig.suptitle(f'{slug} - {prefix} Features Distribution', fontsize=12, fontweight='bold', y=1.02)
            plt.tight_layout()
            tag = f'06_dist_{prefix}_part{chunk_start//9 + 1}'
            paths.append(save(fig, slug, tag))
    return paths

def plot_boxplots_by_feature_group(df, slug):
    """Box plots grouped by feature prefix to show outliers."""
    groups = get_feature_groups(df)
    paths = []
    for prefix, cols in groups.items():
        if len(cols) == 0:
            continue
        # Standardize for comparison: z-score
        fig, ax = plt.subplots(figsize=(max(10, len(cols)*0.8), 6))
        data_for_plot = {}
        for c in cols:
            s = df[c].dropna()
            if len(s) > 0 and s.std() > 0:
                data_for_plot[c] = (s - s.mean()) / s.std()
        if data_for_plot:
            df_plot = pd.DataFrame(data_for_plot)
            sns.boxplot(data=df_plot, ax=ax, palette='Set2', fliersize=2, linewidth=0.8)
            ax.tick_params(axis='x', rotation=90)
            ax.set_ylabel('Z-Score')
            ax.set_title(f'{slug} - {prefix} Features (Z-Scored Box Plots)', fontsize=11, fontweight='bold')
            plt.tight_layout()
            paths.append(save(fig, slug, f'07_box_{prefix}'))
        else:
            plt.close(fig)
    return paths

def plot_time_series(df, slug):
    if DATE not in df.columns:
        return
    dft = df.copy()
    dft[DATE] = pd.to_datetime(dft[DATE], errors='coerce')
    dft = dft.dropna(subset=[DATE, TARGET]).sort_values(DATE)
    if dft.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # Raw time series
    for route in dft[ROUTE].unique():
        sub = dft[dft[ROUTE] == route].set_index(DATE)[TARGET]
        axes[0, 0].plot(sub.index, sub.values, label=route, alpha=0.7, linewidth=0.8)
    axes[0, 0].set_title(f'{TARGET} Over Time (by Route)')
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].set_xlabel('')

    # Monthly mean
    monthly = dft.set_index(DATE)[TARGET].resample('M').mean()
    axes[0, 1].plot(monthly.index, monthly.values, color='#E91E63', linewidth=2)
    axes[0, 1].fill_between(monthly.index, monthly.values, alpha=0.2, color='#E91E63')
    axes[0, 1].set_title(f'Monthly Mean {TARGET}')
    axes[0, 1].set_xlabel('')

    # Weekly mean
    weekly = dft.set_index(DATE)[TARGET].resample('W').mean()
    axes[1, 0].plot(weekly.index, weekly.values, color='#9C27B0', linewidth=1.5)
    axes[1, 0].set_title(f'Weekly Mean {TARGET}')
    axes[1, 0].set_xlabel('')

    # Rolling 30-day mean
    daily = dft.set_index(DATE)[TARGET]
    rolling = daily.rolling(30).mean()
    axes[1, 1].plot(daily.index, daily.values, alpha=0.3, color='#BDBDBD', linewidth=0.5, label='Daily')
    axes[1, 1].plot(rolling.index, rolling.values, color='#FF5722', linewidth=2, label='30-day Rolling')
    axes[1, 1].set_title(f'{TARGET} with 30-Day Rolling Mean')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_xlabel('')

    fig.suptitle(f'{slug} - Time Series Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return save(fig, slug, '08_time_series')

def plot_scatter_top_features(df, slug, n=6):
    top_feats = top_corr_with_target(df, TARGET, n=n)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()
    for i, feat in enumerate(top_feats):
        corr_val = df[numeric_cols(df)].corr()[TARGET][feat]
        # Sample for performance
        sub = df[[feat, TARGET]].dropna()
        sample = sub.sample(min(2000, len(sub)), random_state=42)
        sns.scatterplot(data=sample, x=feat, y=TARGET, ax=axes[i], alpha=0.3, s=10, color='#2196F3')
        # Add regression line
        sns.regplot(data=sample, x=feat, y=TARGET, ax=axes[i], scatter=False,
                    color='#FF5722', line_kws={'linewidth': 2})
        axes[i].set_title(f'{feat}\nr = {corr_val:.3f}', fontsize=9)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    fig.suptitle(f'{slug} - {TARGET} vs Top {n} Correlated Features', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return save(fig, slug, '09_scatter_top_features')

def plot_pairplot_top(df, slug, n=5):
    top_feats = top_corr_with_target(df, TARGET, n=n)
    cols = [TARGET] + top_feats[:4]
    sample = df[cols].dropna()
    sample = sample.sample(min(1500, len(sample)), random_state=42)
    g = sns.pairplot(sample, diag_kind='kde', plot_kws={'alpha': 0.4, 's': 10},
                     diag_kws={'fill': True}, height=2.5, corner=False)
    g.fig.suptitle(f'{slug} - Pair Plot (Top Features)', fontsize=13, fontweight='bold', y=1.02)
    return save(g.fig, slug, '10_pairplot_top')

def plot_feature_group_target_relationship(df, slug):
    """Aggregate feature groups vs target - mean by route."""
    if ROUTE not in df.columns:
        return
    groups = get_feature_groups(df)
    paths = []
    for prefix, cols in groups.items():
        if len(cols) == 0:
            continue
        # Take first 8 cols from group
        plot_cols = cols[:8]
        fig, ax = plt.subplots(figsize=(12, 6))
        route_means = df.groupby(ROUTE)[plot_cols].mean()
        route_means.plot(kind='bar', ax=ax, width=0.8)
        ax.tick_params(axis='x', rotation=20)
        ax.set_title(f'{slug} - Mean {prefix} Features by Route', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
        ax.set_ylabel('Mean Value')
        plt.tight_layout()
        paths.append(save(fig, slug, f'11_route_{prefix}_means'))
    return paths

def plot_cyclone_analysis(df, slug):
    """Specific analysis for cyclone features."""
    cyclone_cols = [c for c in numeric_cols(df) if 'cyclone' in c.lower()]
    if not cyclone_cols:
        return

    # Cyclone active count by route
    active_cols = [c for c in cyclone_cols if 'active' in c]
    if active_cols and ROUTE in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        df.groupby(ROUTE)[active_cols].sum().plot(kind='bar', ax=ax, width=0.8)
        ax.tick_params(axis='x', rotation=20)
        ax.set_title(f'{slug} - Cyclone Active Events by Route', fontsize=11, fontweight='bold')
        ax.set_ylabel('Total Active Events')
        ax.legend(fontsize=7)
        plt.tight_layout()
        save(fig, slug, '12_cyclone_active_by_route')

    # Cyclone wind vs target
    wind_cols = [c for c in cyclone_cols if 'cyclone_wind' in c]
    if wind_cols:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for i, col in enumerate(wind_cols[:2]):
            sub = df[[col, TARGET]].dropna()
            sample = sub.sample(min(2000, len(sub)), random_state=42)
            sns.scatterplot(data=sample, x=col, y=TARGET, ax=axes[i], alpha=0.3, s=10, color='#FF5722')
            corr = df[numeric_cols(df)].corr()[TARGET][col]
            axes[i].set_title(f'{col} vs {TARGET}\nr = {corr:.3f}')
        plt.tight_layout()
        save(fig, slug, '12b_cyclone_wind_vs_target')

def plot_lsci_analysis(df, slug):
    """LSCI feature analysis."""
    lsci_cols = [c for c in numeric_cols(df) if 'lsci' in c.lower()]
    if not lsci_cols:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # LSCI distribution
    for col in lsci_cols[:3]:
        data = df[col].dropna()
        if not data.empty:
            sns.kdeplot(data, ax=axes[0], label=col, fill=True, alpha=0.3)
    axes[0].set_title('LSCI Feature Distributions')
    axes[0].legend(fontsize=8)
    # LSCI vs target
    for col in lsci_cols[:3]:
        sub = df[[col, TARGET]].dropna()
        sample = sub.sample(min(2000, len(sub)), random_state=42)
        corr = df[numeric_cols(df)].corr()[TARGET].get(col, 0)
        axes[1].scatter(sample[col], sample[TARGET], alpha=0.3, s=10, label=f'{col} (r={corr:.3f})')
    axes[1].set_title(f'LSCI vs {TARGET}')
    axes[1].legend(fontsize=7)
    fig.suptitle(f'{slug} - LSCI Analysis', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    return save(fig, slug, '13_lsci_analysis')

def plot_weather_zones_comparison(df, slug):
    """Compare weather features across different sea zones."""
    zone_cols = {}
    for col in numeric_cols(df):
        parts = col.split('_')
        if len(parts) >= 2:
            prefix = parts[0]
            zone = '_'.join(parts[1:])
            if prefix in ['air', 'slp', 'wind_speed']:
                zone_cols.setdefault(prefix, {})[zone] = col

    paths = []
    for prefix, zones in zone_cols.items():
        if not zones:
            continue
        fig, ax = plt.subplots(figsize=(12, 5))
        for zone_name, col in sorted(zones.items())[:10]:
            data = df[col].dropna()
            if not data.empty:
                ax.plot(range(len(data)), data.values, label=zone_name, alpha=0.7, linewidth=0.5)
        ax.set_title(f'{slug} - {prefix} Across Sea Zones', fontsize=11, fontweight='bold')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel(prefix)
        ax.legend(fontsize=6, loc='upper right', ncol=2)
        plt.tight_layout()
        paths.append(save(fig, slug, f'14_weather_{prefix}_zones'))
    return paths

def plot_summary_statistics(df, slug):
    """Summary statistics visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # 1. Feature count by group
    groups = get_feature_groups(df)
    group_counts = {k: len(v) for k, v in groups.items()}
    axes[0, 0].bar(group_counts.keys(), group_counts.values(), color='#2196F3')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_title('Feature Count by Group')
    axes[0, 0].set_ylabel('Number of Features')
    # 2. Missing values distribution
    missing_pct = (df.isnull().sum() / len(df) * 100)
    bins = [0, 1, 10, 25, 50, 75, 100]
    labels = ['0-1%', '1-10%', '10-25%', '25-50%', '50-75%', '75-100%']
    cut = pd.cut(missing_pct, bins=bins, labels=labels)
    axes[0, 1].bar(cut.value_counts().index, cut.value_counts().values, color='#FF9800')
    axes[0, 1].tick_params(axis='x', rotation=30)
    axes[0, 1].set_title('Missing Value Distribution')
    axes[0, 1].set_ylabel('Feature Count')
    # 3. Target skewness/kurtosis
    from scipy import stats
    skew = df[TARGET].skew()
    kurt = df[TARGET].kurtosis()
    axes[1, 0].bar(['Skewness', 'Kurtosis'], [skew, kurt], color=['#4CAF50', '#E91E63'])
    axes[1, 0].set_title(f'{TARGET} Distribution Statistics')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].text(0, skew, f'{skew:.2f}', ha='center', va='bottom', fontweight='bold')
    axes[1, 0].text(1, kurt, f'{kurt:.2f}', ha='center', va='bottom', fontweight='bold')
    # 4. Correlation distribution
    corr_matrix = df[numeric_cols(df)].corr()
    upper_vals = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    axes[1, 1].hist(upper_vals, bins=50, color='#9C27B0', edgecolor='white', alpha=0.8)
    axes[1, 1].set_title('Pairwise Correlation Distribution')
    axes[1, 1].set_xlabel('Correlation Coefficient')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].axvline(0, color='black', linewidth=0.8)
    fig.suptitle(f'{slug} - Summary Statistics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return save(fig, slug, '15_summary_statistics')

# ─── Main ─────────────────────────────────────────────────────────────────────
def run_eda(slug, filepath):
    print(f'\n{"="*70}')
    print(f'  EDA: {slug}')
    print(f'{"="*70}')

    df = pd.read_csv(filepath)
    print(f'  Loaded: {df.shape[0]} rows x {df.shape[1]} columns')

    # 0. Overview text
    plot_overview_text(df, slug)

    # 1. Target distribution
    plot_target_distribution(df, slug)

    # 2. Route analysis
    plot_route_analysis(df, slug)
    plot_route_stats_table(df, slug)

    # 3. Missing values
    plot_missing_values(df, slug)

    # 4. Correlation
    plot_correlation_heatmap(df, slug, max_feat=25)
    plot_target_correlation_bar(df, slug, n=20)

    # 5. Feature distributions by group
    plot_feature_distributions_by_group(df, slug)

    # 6. Box plots by group
    plot_boxplots_by_feature_group(df, slug)

    # 7. Time series
    plot_time_series(df, slug)

    # 8. Scatter top features
    plot_scatter_top_features(df, slug, n=6)

    # 9. Pairplot
    plot_pairplot_top(df, slug, n=5)

    # 10. Feature group vs route
    plot_feature_group_target_relationship(df, slug)

    # 11. Cyclone analysis
    plot_cyclone_analysis(df, slug)

    # 12. LSCI analysis
    plot_lsci_analysis(df, slug)

    # 13. Weather zones comparison
    plot_weather_zones_comparison(df, slug)

    # 14. Summary statistics
    plot_summary_statistics(df, slug)

    print(f'\n  EDA complete for {slug}')

def main():
    print(f'Output directory: {OUT}')
    print(f'Generating EDA visualizations...\n')

    for ds in DATASETS:
        run_eda(ds['name'], ds['file'])

    # Final summary
    files = os.listdir(OUT)
    pngs = [f for f in files if f.endswith('.png')]
    txts = [f for f in files if f.endswith('.txt')]
    print(f'\n{"="*70}')
    print(f'  ALL EDA COMPLETE')
    print(f'{"="*70}')
    print(f'  Total plots generated: {len(pngs)}')
    print(f'  Total text files: {len(txts)}')
    print(f'  Output directory: {OUT}')
    print(f'\n  Plot files:')
    for f in sorted(pngs):
        print(f'    - {f}')

if __name__ == '__main__':
    main()
