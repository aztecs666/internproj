import pandas as pd
import xgboost as xgb
import numpy as np

df = pd.read_csv(r'D:\Internproj\new dataset\engineered_ml_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])
model = xgb.Booster()
model.load_model(r'D:\Internproj\selectedmodel\xgb_80_20_lr001_depth5.json')
df = pd.get_dummies(df, columns=['Route'], prefix='Route')

feature_names = model.feature_names
test_dates = pd.to_datetime(['2026-04-13','2026-04-14','2026-04-15'])
test_data = df[df['Date'].isin(test_dates)]

results = []
for _, row in test_data.iterrows():
    date_str = row['Date'].strftime('%Y-%m-%d')
    actual = row['Price_USD']
    features = row[feature_names].values.reshape(1, -1)
    dmatrix = xgb.DMatrix(features, feature_names=feature_names)
    predicted = float(model.predict(dmatrix)[0])
    error = abs(actual - predicted)
    error_pct = error / actual * 100
    for r in ['XSICFENE_FarEast_NorthEurope','XSICNEFE_NorthEurope_FarEast','XSICFEUW_FarEast_USWestCoast','XSICUWFE_USWestCoast_FarEast']:
        if row['Route_' + r]:
            route_name = r
            break
    results.append((date_str, route_name, actual, predicted, error, error_pct))
    
for date_str, route_name, actual, predicted, error, error_pct in results:
    print(f"{date_str} | {route_name:34s} | Actual=${actual:7.0f} | Pred=${predicted:7.0f} | Err=${error:5.0f} ({error_pct:.1f}%)")

print()
all_features = test_data[feature_names].values
all_actuals = test_data['Price_USD'].values
dmatrix = xgb.DMatrix(all_features, feature_names=feature_names)
all_preds = model.predict(dmatrix)
mae = np.mean(np.abs(all_actuals - all_preds))
rmse = np.sqrt(np.mean((all_actuals - all_preds)**2))
ss_res = np.sum((all_actuals - all_preds)**2)
ss_tot = np.sum((all_actuals - np.mean(all_actuals))**2)
r2 = 1 - (ss_res / ss_tot)
print(f"MAE:  ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")
print(f"R2:   {r2:.4f}")
