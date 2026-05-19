import pandas as pd
import numpy as np
import os
import glob

# Zone mapping
grid_to_zone = {}
for lat in np.arange(-90, 90.1, 2.5):
    for lon in np.arange(0, 360, 2.5):
        # We assign each grid point to a single zone (priority based on order if overlaps)
        if 5 <= lat <= 25 and 50 <= lon <= 75:
            grid_to_zone[(lat, lon)] = "Arabian Sea"
        elif -30 <= lat <= 5 and 50 <= lon <= 100:
            grid_to_zone[(lat, lon)] = "Indian Ocean"
        elif 30 <= lat <= 45 and (lon >= 355 or lon <= 35):
            grid_to_zone[(lat, lon)] = "Mediterranean Sea"
        elif 50 <= lat <= 65 and (lon >= 355 or lon <= 15):
            grid_to_zone[(lat, lon)] = "North Sea"
        elif 12 <= lat <= 30 and 32.5 <= lon <= 45:
            grid_to_zone[(lat, lon)] = "Red Sea"
        elif 0 <= lat <= 25 and 100 <= lon <= 120:
            grid_to_zone[(lat, lon)] = "South China Sea"
        elif 20 <= lat <= 55 and 120 <= lon <= 180:
            grid_to_zone[(lat, lon)] = "North Pacific West"
        elif 20 <= lat <= 55 and 180 <= lon <= 240:
            grid_to_zone[(lat, lon)] = "North Pacific East"
        elif 20 <= lat <= 60 and 290 <= lon <= 350:
            grid_to_zone[(lat, lon)] = "North Atlantic"

# Print some mapping to verify
print("Mapped grids:", len(grid_to_zone))

RAW_DIR = "d:/Internproj/weatherdata/datasets/noaa_raw"
VARIABLES = ["air", "prate", "slp", "uwnd", "vwnd"]

def process_year(year):
    print(f"Processing {year}...")
    zone_data_dfs = []
    
    for var in VARIABLES:
        fpath = os.path.join(RAW_DIR, var, f"{var}.{year}.csv")
        if not os.path.exists(fpath):
            return None
        
        print(f" Reading {fpath}...")
        df = pd.read_csv(fpath)
        
        # Add 'zone' column by mapping
        # Convert lat, lon to tuple
        s_tuples = pd.Series(list(zip(df['lat'], df['lon'])))
        df['zone'] = s_tuples.map(grid_to_zone)
        
        # Drop rows without zone
        df = df.dropna(subset=['zone'])
        
        if df.empty:
            continue
            
        # Group by date and zone, then average
        agg_df = df.groupby(['date', 'zone'])['value'].mean().reset_index()
        # Pivot so that each zone is a column
        pivot_df = agg_df.pivot(index='date', columns='zone', values='value').reset_index()
        pivot_df.columns.name = None # Remove the 'zone' name from index
        
        # Add variable prefix to columns
        rename_dict = {c: f"{var}_{c.replace(' ', '_')}" for c in pivot_df.columns if c != 'date'}
        pivot_df = pivot_df.rename(columns=rename_dict)
        
        zone_data_dfs.append(pivot_df)

    if not zone_data_dfs:
        return None
        
    # Merge all variables for this year
    merged = zone_data_dfs[0]
    for i in range(1, len(zone_data_dfs)):
        merged = pd.merge(merged, zone_data_dfs[i], on='date', how='outer')
        
    return merged

# Test with just 2015
res = process_year(2015)
if res is not None:
    res.to_csv("d:/Internproj/new dataset/weather_zones_2015.csv", index=False)
    print("Test output saved.")
