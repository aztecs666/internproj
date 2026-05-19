import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
file_path = r"D:\Internproj\new dataset\engineered_ml_dataset.csv"
df = pd.read_csv(file_path)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()[:10]}...")  # Show first 10 column names

# Select numeric columns only
numeric_df = df.select_dtypes(include=[np.number])
print(f"\nNumeric columns: {len(numeric_df.columns)}")

# Calculate correlation matrix
correlation_matrix = numeric_df.corr()

# Create a large figure for the heatmap
plt.figure(figsize=(24, 20))

# Generate heatmap
sns.heatmap(
    correlation_matrix,
    annot=False,  # Don't annotate to keep it readable with many features
    cmap='RdBu_r',  # Red-Blue diverging colormap
    center=0,
    square=True,
    fmt='.2f',
    cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
    linewidths=0.1
)

plt.title('Feature Correlation Heatmap\nEngineered ML Dataset', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('')
plt.ylabel('')
plt.xticks(rotation=45, ha='right', fontsize=6)
plt.yticks(rotation=0, fontsize=6)
plt.tight_layout()

# Save the figure
output_path = r"D:\Internproj\Analysis\correlation_heatmap.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nHeatmap saved to: {output_path}")

# Also create a smaller heatmap with top correlated features
# Find features most correlated with Price_USD (target variable)
if 'Price_USD' in correlation_matrix.columns:
    price_correlations = correlation_matrix['Price_USD'].abs().sort_values(ascending=False)
    top_features = price_correlations.head(20).index.tolist()
    
    plt.figure(figsize=(14, 12))
    top_corr_matrix = correlation_matrix.loc[top_features, top_features]
    
    sns.heatmap(
        top_corr_matrix,
        annot=True,
        cmap='RdBu_r',
        center=0,
        square=True,
        fmt='.2f',
        cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
        linewidths=0.5
    )
    
    plt.title('Top 20 Features Correlation Heatmap\n(Most correlated with Price_USD)', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    output_path2 = r"D:\Internproj\Analysis\top_features_heatmap.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Top features heatmap saved to: {output_path2}")

plt.close('all')
print("\nHeatmap generation complete!")
