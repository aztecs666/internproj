import sys
import os
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from year_based_split import prepare_data

def evaluate():
    print("Loading test data...")
    dataset_path = r"d:\Internproj\new dataset\engineered_ml_dataset.csv"
    X, y, years = prepare_data(dataset_path)
    
    # Get final holdout indices (2024+)
    test_idx = np.where(years >= 2024)[0]
    
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    
    print(f"Test Set Size: {len(y_test)}")
    
    # Load Models
    rf_path = r"d:\Internproj\ML Models\Random Forest\best_rf_model.pkl"
    xgb_path = r"d:\Internproj\ML Models\XGBoost\best_xgb_model.json"
    
    with open(rf_path, 'rb') as f:
        rf_model = pickle.load(f)
        
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(xgb_path)
    
    # Predict
    print("Generating predictions...")
    y_pred_rf = rf_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)
    
    # Calculate Metrics
    results = {
        'Random Forest': {
            'MAE': mean_absolute_error(y_test, y_pred_rf),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
            'R2': r2_score(y_test, y_pred_rf)
        },
        'XGBoost': {
            'MAE': mean_absolute_error(y_test, y_pred_xgb),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
            'R2': r2_score(y_test, y_pred_xgb)
        }
    }
    
    results_df = pd.DataFrame(results).T
    results_path = os.path.join(os.path.dirname(__file__), "test_results.csv")
    results_df.to_csv(results_path)
    print(f"Saved evaluation metrics to {results_path}")
    
    # Visualization 1: Actual vs Predicted (first 100 points for clarity)
    plt.figure(figsize=(14, 7))
    sample_size = min(100, len(y_test))
    x_axis = range(sample_size)
    
    plt.plot(x_axis, y_test.values[:sample_size], label='Actual Price', color='black', linewidth=2)
    plt.plot(x_axis, y_pred_rf[:sample_size], label='Random Forest', linestyle='--', color='blue')
    plt.plot(x_axis, y_pred_xgb[:sample_size], label='XGBoost', linestyle='-.', color='orange')
    
    plt.title('Actual vs Predicted Shipping Prices (Test Set Sample)')
    plt.xlabel('Time (Indices)')
    plt.ylabel('Price_USD')
    plt.legend()
    plt.grid(True)
    viz1_path = r"d:\Internproj\Visualization\actual_vs_predicted_test.png"
    plt.savefig(viz1_path)
    print(f"Saved visualization to {viz1_path}")
    plt.close()
    
    # Visualization 2: Feature Importance (from XGBoost)
    # Extract top 15 features
    importances = xgb_model.feature_importances_
    indices = np.argsort(importances)[-15:] # Top 15
    features = X.columns[indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], color='teal', align='center')
    plt.yticks(range(len(indices)), features)
    plt.title('Top 15 Feature Importances (XGBoost)')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    viz2_path = r"d:\Internproj\Visualization\xgb_feature_importance.png"
    plt.savefig(viz2_path)
    print(f"Saved visualization to {viz2_path}")
    plt.close()
    
if __name__ == "__main__":
    evaluate()
