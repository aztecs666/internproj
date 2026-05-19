import pandas as pd
import numpy as np

df = pd.read_csv(r'D:\Internproj\New ML\experiment_route_flags\dataset_with_flags.csv')
df['Date'] = pd.to_datetime(df['Date'])

print('=== Route-level Price Statistics ===')
for route in sorted(df['Route'].unique()):
    rdf = df[df['Route'] == route]
    prices = rdf['Price_USD']
    name = route.replace('XSIC', '').replace('_', ' ')
    cv = prices.std() / prices.mean() * 100
    print(f'\n{name:40s} Count={len(prices):>5d}  Mean=${prices.mean():>8.0f}  Std=${prices.std():>6.0f}  '
          f'Min=${prices.min():>6.0f}  Max=${prices.max():>7.0f}  Volatility={cv:>3.0f}%')
