import pandas as pd
import numpy as np

df = pd.read_csv(r'D:\Internproj\New ML\experiment_route_flags\test_predictions.csv')
models = ['RF_Predicted', 'XGB_Predicted']

print('=' * 130)
hdr = f"{'Route':45s} {'Model':12s} {'MAE':>10s} {'RMSE':>10s} {'R²':>8s} {'MAPE':>8s} {'Bias':>8s} {'Samples':>7s}"
print(hdr)
print('=' * 130)

all_results = []
for route in sorted(df['Route'].unique()):
    rdf = df[df['Route'] == route]
    actual = rdf['Actual'].values
    
    for model in models:
        pred = rdf[model].values
        errors = actual - pred
        pct_errors = (errors / actual) * 100
        
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))
        ss_res = float(np.sum(errors**2))
        ss_tot = float(np.sum((actual - np.mean(actual))**2))
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        mape = float(np.mean(np.abs(pct_errors)))
        bias = float(np.mean(pct_errors))
        
        name = route.replace('XSIC', '').replace('_', ' ')
        
        m = 'RandomForest' if model == 'RF_Predicted' else 'XGBoost'
        print(f'{name:45s} {m:12s} {mae:>10.0f} {rmse:>10.0f} {r2:>8.3f} {mape:>7.1f}% {bias:>7.1f}% {len(actual):>7d}')
        all_results.append({'route': route, 'model': model, 'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape, 'bias': bias, 'n': len(actual)})
    print('-' * 130)

print()
print('OVERALL (all routes combined):')
for model in models:
    pred = df[model].values
    actual = df['Actual'].values
    errors = actual - pred
    pct_errors = (errors / actual) * 100
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((actual - np.mean(actual))**2))
    r2 = 1 - (ss_res / ss_tot)
    mape = float(np.mean(np.abs(pct_errors)))
    bias = float(np.mean(pct_errors))
    m = 'RandomForest' if model == 'RF_Predicted' else 'XGBoost'
    print(f'  {m:15s}  MAE={mae:>8.0f}  RMSE={rmse:>8.0f}  R²={r2:>7.3f}  MAPE={mape:>5.1f}%  Bias={bias:>6.1f}%')

print()
print('=== BEST ROUTE PER MODEL (by R²) ===')
for model in models:
    m = 'RandomForest' if model == 'RF_Predicted' else 'XGBoost'
    route_results = [r for r in all_results if r['model'] == model]
    best = max(route_results, key=lambda x: x['r2'])
    worst = min(route_results, key=lambda x: x['r2'])
    print(f'\n{m}:')
    print(f'  BEST:  {best["route"]} — R²={best["r2"]:.3f}, MAE=${best["mae"]:.0f}, MAPE={best["mape"]:.1f}%')
    print(f'  WORST: {worst["route"]} — R²={worst["r2"]:.3f}, MAE=${worst["mae"]:.0f}, MAPE={worst["mape"]:.1f}%')

print()
print('=== XGBoost (selected model) ranked by performance ===')
xgb_results = sorted([r for r in all_results if r['model'] == 'XGB_Predicted'], key=lambda x: x['r2'], reverse=True)
for i, r in enumerate(xgb_results):
    name = r['route'].replace('XSIC', '').replace('_', ' ')
    print(f'  {i+1}. {name:40s} R²={r["r2"]:.3f}  MAE=${r["mae"]:>6.0f}  MAPE={r["mape"]:>5.1f}%  Bias={r["bias"]:>+5.1f}%')
