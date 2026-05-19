import pandas as pd
import xgboost as xgb
import numpy as np

df = pd.read_csv(r'D:\Internproj\new dataset\engineered_ml_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])
model = xgb.Booster()
model.load_model(r'D:\Internproj\selectedmodel\xgb_80_20_lr001_depth5.json')
df = pd.get_dummies(df, columns=['Route'], prefix='Route')
feature_names = model.feature_names

# Get the most recent known data (April 15) as template
latest = df[df['Date'] == '2026-04-15'].copy()

# Get actual prices for lag calculation
price_map = df.groupby('Date')['Price_USD'].first().to_dict()

# April 9 prices (for Price_Lag_7d on April 16)
apr9 = df[df['Date'] == '2026-04-09']
# April 10 prices (for Price_Lag_7d on April 17)
apr10 = df[df['Date'] == '2026-04-10']
# April 11 doesn't exist in dataset
apr11 = None

# March 17 prices (for Price_Lag_30d on April 16)
mar17 = df[df['Date'] == '2026-03-17']
# March 18 prices (for Price_Lag_30d on April 17)
mar18 = df[df['Date'] == '2026-03-18']
# March 19 prices (for Price_Lag_30d on April 18)
mar19 = df[df['Date'] == '2026-03-19']

print("=== Available lag prices ===")
print("April 9:", len(apr9), "rows")
print("April 10:", len(apr10), "rows")
print("April 11:", len(apr11) if apr11 else 0, "rows")
print("March 17:", len(mar17), "rows")
print("March 18:", len(mar18), "rows")
print("March 19:", len(mar19), "rows")

# Check if we can construct features for April 16
print("\n=== Constructing April 16 prediction ===")

# For each route, create a prediction row
routes = ['XSICFENE_FarEast_NorthEurope', 'XSICNEFE_NorthEurope_FarEast',
          'XSICFEUW_FarEast_USWestCoast', 'XSICUWFE_USWestCoast_FarEast']

for route in routes:
    # Get the template from April 15 for this route
    template = latest[latest['Route_' + route] == 1].copy()
    if len(template) == 0:
        print(f"No template for {route}")
        continue
    
    # Modify date-dependent features
    template['Month'] = 4  # April
    template['Week_of_Year'] = 16  # Week 16
    template['Is_Peak_Season'] = 0  # April is not peak
    
    # Get Price_Lag_7d from April 9
    apr9_route = apr9[apr9['Route_' + route] == 1]
    if len(apr9_route) > 0:
        template['Price_Lag_7d'] = apr9_route['Price_USD'].values[0]
    
    # Get Price_Lag_30d from March 17
    mar17_route = mar17[mar17['Route_' + route] == 1]
    if len(mar17_route) > 0:
        template['Price_Lag_30d'] = mar17_route['Price_USD'].values[0]
    
    # Note: We're using April 15 weather/indices as proxy for April 16
    # In production, these would come from real-time APIs
    
    features = template[feature_names].values.reshape(1, -1)
    dmatrix = xgb.DMatrix(features, feature_names=feature_names)
    predicted = float(model.predict(dmatrix)[0])
    
    print(f"  {route}: ${predicted:.0f}")

print("\n=== Constructing April 17 prediction ===")
for route in routes:
    template = latest[latest['Route_' + route] == 1].copy()
    if len(template) == 0:
        continue
    
    template['Month'] = 4
    template['Week_of_Year'] = 16
    template['Is_Peak_Season'] = 0
    
    # Price_Lag_7d from April 10
    apr10_route = apr10[apr10['Route_' + route] == 1]
    if len(apr10_route) > 0:
        template['Price_Lag_7d'] = apr10_route['Price_USD'].values[0]
    
    # Price_Lag_30d from March 18
    mar18_route = mar18[mar18['Route_' + route] == 1]
    if len(mar18_route) > 0:
        template['Price_Lag_30d'] = mar18_route['Price_USD'].values[0]
    
    features = template[feature_names].values.reshape(1, -1)
    dmatrix = xgb.DMatrix(features, feature_names=feature_names)
    predicted = float(model.predict(dmatrix)[0])
    
    print(f"  {route}: ${predicted:.0f}")

print("\n=== Constructing April 18 prediction ===")
# April 18 needs Price_Lag_7d from April 11 (doesn't exist in dataset)
# Use April 10 as closest available
for route in routes:
    template = latest[latest['Route_' + route] == 1].copy()
    if len(template) == 0:
        continue
    
    template['Month'] = 4
    template['Week_of_Year'] = 16
    template['Is_Peak_Season'] = 0
    
    # Price_Lag_7d from April 10 (closest available, April 11 missing)
    apr10_route = apr10[apr10['Route_' + route] == 1]
    if len(apr10_route) > 0:
        template['Price_Lag_7d'] = apr10_route['Price_USD'].values[0]
    
    # Price_Lag_30d from March 19
    mar19_route = mar19[mar19['Route_' + route] == 1]
    if len(mar19_route) > 0:
        template['Price_Lag_30d'] = mar19_route['Price_USD'].values[0]
    
    features = template[feature_names].values.reshape(1, -1)
    dmatrix = xgb.DMatrix(features, feature_names=feature_names)
    predicted = float(model.predict(dmatrix)[0])
    
    print(f"  {route}: ${predicted:.0f}")

print("\n=== Limitations ===")
print("1. April 11 data missing - used April 10 for April 18 lag")
print("2. Weather/indices used April 15 values as proxy")
print("3. Actual prices for April 16-18 not in dataset (not yet recorded)")
