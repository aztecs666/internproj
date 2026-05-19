import os
import pandas as pd
import numpy as np

# -------------------------------------------------------------------------
# Define configurations
# -------------------------------------------------------------------------
NEW_DATASET_DIR = "d:/Internproj/new dataset"
XENETA_FILE = "d:/Internproj/Index/xeneta/xeneta_xsi_combined.csv"
LSCI_FILE = "d:/Internproj/oprational costs/unctad_lsci/unctad_lsci_2015_2024.csv"
NOAA_RAW_DIR = "d:/Internproj/weatherdata/datasets/noaa_raw"
VARIABLES = ["air", "prate", "slp", "uwnd", "vwnd"]

# Zone mapping from lat/lon to Sea
grid_to_zone = {}
for lat in np.arange(-90, 90.1, 2.5):
    for lon in np.arange(0, 360, 2.5):
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

def get_zone(lat, lon):
    if 5 <= lat <= 25 and 50 <= lon <= 75:
        return "Arabian Sea"
    elif -30 <= lat <= 5 and 50 <= lon <= 100:
        return "Indian Ocean"
    elif 30 <= lat <= 45 and (lon >= 355 or lon <= 35):
        return "Mediterranean Sea"
    elif 50 <= lat <= 65 and (lon >= 355 or lon <= 15):
        return "North Sea"
    elif 12 <= lat <= 30 and 32.5 <= lon <= 45:
        return "Red Sea"
    elif 0 <= lat <= 25 and 100 <= lon <= 120:
        return "South China Sea"
    elif 20 <= lat <= 55 and 120 <= lon <= 180:
        return "North Pacific West"
    elif 20 <= lat <= 55 and 180 <= lon <= 240:
        return "North Pacific East"
    elif 20 <= lat <= 60 and 290 <= lon <= 350:
        return "North Atlantic"
    return None

# Route specific mappings
FE_COUNTRIES = ["China", "Japan", "South Korea", "Hong Kong SAR", "Malaysia", "Vietnam", "Thailand", "Singapore", "Indonesia"]
NE_COUNTRIES = ["United Kingdom", "Germany", "France", "Netherlands", "Belgium"]
UW_COUNTRIES = ["USA"]

ROUTE_SEAS = {
    "XSICFENE_FarEast_NorthEurope": ["South China Sea", "Indian Ocean", "Arabian Sea", "Red Sea", "Mediterranean Sea", "North Atlantic", "North Sea"],
    "XSICNEFE_NorthEurope_FarEast": ["South China Sea", "Indian Ocean", "Arabian Sea", "Red Sea", "Mediterranean Sea", "North Atlantic", "North Sea"],
    "XSICFEUW_FarEast_USWestCoast": ["South China Sea", "North Pacific West", "North Pacific East"],
    "XSICUWFE_USWestCoast_FarEast": ["South China Sea", "North Pacific West", "North Pacific East"]
}

ROUTE_LSCI_COUNTRIES = {
    "XSICFENE_FarEast_NorthEurope": (FE_COUNTRIES, NE_COUNTRIES),
    "XSICNEFE_NorthEurope_FarEast": (NE_COUNTRIES, FE_COUNTRIES),
    "XSICFEUW_FarEast_USWestCoast": (FE_COUNTRIES, UW_COUNTRIES),
    "XSICUWFE_USWestCoast_FarEast": (UW_COUNTRIES, FE_COUNTRIES)
}

# -------------------------------------------------------------------------
# 1. Processing Weather (NOAA -> Zones)
# -------------------------------------------------------------------------
print("Processing NOAA Weather data...")
yearly_weather_dfs = []

# Using 2015-2025 range
for year in range(2015, 2026):
    print(f"  Extracting zones for year {year}...")
    zone_data_dfs = []
    
    for var in VARIABLES:
        fpath = os.path.join(NOAA_RAW_DIR, var, f"{var}.{year}.csv")
        if not os.path.exists(fpath):
            continue
        
        df = pd.read_csv(fpath)
        
        s_tuples = pd.Series(list(zip(df['lat'], df['lon'])))
        df['zone'] = s_tuples.map(grid_to_zone)
        df = df.dropna(subset=['zone'])
        
        if df.empty:
            continue
            
        agg_df = df.groupby(['date', 'zone'])['value'].mean().reset_index()
        pivot_df = agg_df.pivot(index='date', columns='zone', values='value').reset_index()
        pivot_df.columns.name = None
        
        rename_dict = {c: f"{var}_{c.replace(' ', '_')}" for c in pivot_df.columns if c != 'date'}
        pivot_df = pivot_df.rename(columns=rename_dict)
        
        zone_data_dfs.append(pivot_df)

    if not zone_data_dfs:
        continue
        
    merged_year = zone_data_dfs[0]
    for i in range(1, len(zone_data_dfs)):
        merged_year = pd.merge(merged_year, zone_data_dfs[i], on='date', how='outer')
        
    # Calculate wind speed from uwnd and vwnd
    for zone in set(grid_to_zone.values()):
        zone_fmt = zone.replace(' ', '_')
        uwnd_col = f"uwnd_{zone_fmt}"
        vwnd_col = f"vwnd_{zone_fmt}"
        if uwnd_col in merged_year.columns and vwnd_col in merged_year.columns:
            merged_year[f"wind_speed_{zone_fmt}"] = np.sqrt(merged_year[uwnd_col]**2 + merged_year[vwnd_col]**2)
            merged_year.drop(columns=[uwnd_col, vwnd_col], inplace=True)
            
    yearly_weather_dfs.append(merged_year)

weather_df = pd.concat(yearly_weather_dfs, ignore_index=True)
weather_df['date'] = pd.to_datetime(weather_df['date'].str.slice(0, 10))

# -------------------------------------------------------------------------
# 2. Processing LSCI
# -------------------------------------------------------------------------
print("Processing LSCI data...")
lsci = pd.read_csv(LSCI_FILE)

# Map Quarters to Dates
def q_to_date(row):
    year = int(row['Year'])
    q = row['Quarter']
    if q == 'Q1': return f"{year}-01-01"
    if q == 'Q2': return f"{year}-04-01"
    if q == 'Q3': return f"{year}-07-01"
    if q == 'Q4': return f"{year}-10-01"
    return f"{year}-01-01" # Annual

lsci['date'] = pd.to_datetime(lsci.apply(q_to_date, axis=1))
lsci = lsci[['date', 'Country', 'LSCI_Score']]

# Average per country group
def get_avg_lsci(countries, name):
    d = lsci[lsci['Country'].isin(countries)].groupby('date')['LSCI_Score'].mean().reset_index()
    d.rename(columns={'LSCI_Score': f"Avg_LSCI_{name}"}, inplace=True)
    return d

fe_lsci = get_avg_lsci(FE_COUNTRIES, "FarEast")
ne_lsci = get_avg_lsci(NE_COUNTRIES, "NorthEurope")
uw_lsci = get_avg_lsci(UW_COUNTRIES, "USWestCoast")

lsci_all = fe_lsci.merge(ne_lsci, on='date', how='outer').merge(uw_lsci, on='date', how='outer')
lsci_all = lsci_all.sort_values('date').fillna(method='ffill') # Forward fill quarterly to daily effectively? 
# Wait, we need to reindex to daily first
full_dates = pd.DataFrame({'date': pd.date_range(start='2015-01-01', end='2025-12-31')})
lsci_daily = full_dates.merge(lsci_all, on='date', how='left').fillna(method='ffill')

# -------------------------------------------------------------------------
# 2.5. Processing Cyclones (IBTrACS)
# -------------------------------------------------------------------------
print("Processing Cyclone data...")
IBTRACS_FILE = "d:/Internproj/weatherdata/jtwc_cyclones/_ibtracs_cache.csv"
if os.path.exists(IBTRACS_FILE):
    cycles_df = pd.read_csv(IBTRACS_FILE, low_memory=False)
    cycles_df['date'] = pd.to_datetime(cycles_df['ISO_TIME'].str.slice(0, 10))
    cycles_df = cycles_df[cycles_df['date'].dt.year >= 2015].copy()
    
    cycles_df['WMO_WIND'] = pd.to_numeric(cycles_df['WMO_WIND'], errors='coerce')
    cycles_df['WMO_PRES'] = pd.to_numeric(cycles_df['WMO_PRES'], errors='coerce')
    cycles_df['DIST2LAND'] = pd.to_numeric(cycles_df['DIST2LAND'], errors='coerce')
    cycles_df['zone'] = cycles_df.apply(lambda row: get_zone(row['LAT'], row['LON']), axis=1)
    
    cycles_df = cycles_df.dropna(subset=['zone'])
    
    if not cycles_df.empty:
        c_agg = cycles_df.groupby(['date', 'zone']).agg(
            cyclone_active=('LAT', 'count'),
            cyclone_wind=('WMO_WIND', 'max'),
            cyclone_slp=('WMO_PRES', 'min'),
            cyclone_dist=('DIST2LAND', 'min')
        ).reset_index()
        c_agg['cyclone_active'] = 1
        
        c_wind = c_agg.pivot(index='date', columns='zone', values='cyclone_wind').reset_index()
        c_slp = c_agg.pivot(index='date', columns='zone', values='cyclone_slp').reset_index()
        c_dist = c_agg.pivot(index='date', columns='zone', values='cyclone_dist').reset_index()
        c_act = c_agg.pivot(index='date', columns='zone', values='cyclone_active').reset_index()
        
        c_wind.columns = [c if c == 'date' else f"cyclone_wind_{c.replace(' ', '_')}" for c in c_wind.columns]
        c_slp.columns = [c if c == 'date' else f"cyclone_slp_{c.replace(' ', '_')}" for c in c_slp.columns]
        c_dist.columns = [c if c == 'date' else f"cyclone_dist_{c.replace(' ', '_')}" for c in c_dist.columns]
        c_act.columns = [c if c == 'date' else f"cyclone_active_{c.replace(' ', '_')}" for c in c_act.columns]
        
        cyclone_daily = full_dates.merge(c_wind, on='date', how='left')\
                                  .merge(c_slp, on='date', how='left')\
                                  .merge(c_dist, on='date', how='left')\
                                  .merge(c_act, on='date', how='left')
    else:
        cyclone_daily = full_dates.copy()
else:
    cyclone_daily = full_dates.copy()

# -------------------------------------------------------------------------
# 3. Assemble Final Dataset
# -------------------------------------------------------------------------
print("Merging Xeneta, Weather, and LSCI...")
xeneta_df = pd.read_csv(XENETA_FILE)
xeneta_df['Date'] = pd.to_datetime(xeneta_df['Date'])
xeneta_df.rename(columns={'Date': 'date'}, inplace=True)

# We want route specific datasets. We'll append route specific data to the combined xeneta df 
# but only for the seas and LSCIs that match the route.

output_rows = []
for idx, row in xeneta_df.iterrows():
    route = row['Route']
    dt = row['date']
    
    out_row = {"Date": dt, "Route": route, "Price_USD": row['Price_USD']}
    
    # 1. Fetch LSCI
    lsci_row = lsci_daily[lsci_daily['date'] == dt]
    if not lsci_row.empty:
        origin_grp, dest_grp = ROUTE_LSCI_COUNTRIES[route]
        
        # Name matching
        if origin_grp == FE_COUNTRIES: orig_name = "FarEast"
        elif origin_grp == NE_COUNTRIES: orig_name = "NorthEurope"
        else: orig_name = "USWestCoast"
            
        if dest_grp == FE_COUNTRIES: dest_name = "FarEast"
        elif dest_grp == NE_COUNTRIES: dest_name = "NorthEurope"
        else: dest_name = "USWestCoast"
        
        out_row['Origin_LSCI'] = lsci_row[f"Avg_LSCI_{orig_name}"].values[0]
        out_row['Dest_LSCI'] = lsci_row[f"Avg_LSCI_{dest_name}"].values[0]
        out_row['Route_Mean_LSCI'] = (out_row['Origin_LSCI'] + out_row['Dest_LSCI']) / 2.0
        
    # 2. Fetch Weather
    weather_row = weather_df[weather_df['date'] == dt]
    if not weather_row.empty:
        w_curr = weather_row.iloc[0]
        seas = ROUTE_SEAS[route]
        # Aggregate variables across all relevant seas to get route mean weather
        for var in ['air', 'prate', 'slp', 'wind_speed']:
            vals = []
            for sea in seas:
                sea_fmt = sea.replace(' ', '_')
                col = f"{var}_{sea_fmt}"
                if col in w_curr:
                    v = w_curr[col]
                    if not pd.isna(v):
                        vals.append(v)
            if vals:
                out_row[f"Route_Mean_{var}"] = np.mean(vals)
                
            # Also keep individual seas for fine-grained
            for sea in seas:
                sea_fmt = sea.replace(' ', '_')
                col = f"{var}_{sea_fmt}"
                if col in w_curr:
                    out_row[col] = w_curr[col]

    # 3. Fetch Cyclones
    if not cyclone_daily.empty:
        c_row = cyclone_daily[cyclone_daily['date'] == dt]
        if not c_row.empty:
            c_curr = c_row.iloc[0]
            seas = ROUTE_SEAS[route]
            
            # Aggregate cyclone vars across all relevant seas to get route level features
            # Max wind, Min slp, Min dist, and whether any zone was active
            wind_vals = []
            slp_vals = []
            dist_vals = []
            active = 0
            
            for sea in seas:
                sea_fmt = sea.replace(' ', '_')
                wind_col = f"cyclone_wind_{sea_fmt}"
                slp_col = f"cyclone_slp_{sea_fmt}"
                dist_col = f"cyclone_dist_{sea_fmt}"
                act_col = f"cyclone_active_{sea_fmt}"
                
                if wind_col in c_curr and not pd.isna(c_curr[wind_col]):
                    wind_vals.append(c_curr[wind_col])
                    out_row[wind_col] = c_curr[wind_col]
                else:
                    out_row[wind_col] = 0
                    
                if slp_col in c_curr and not pd.isna(c_curr[slp_col]):
                    slp_vals.append(c_curr[slp_col])
                    out_row[slp_col] = c_curr[slp_col]
                else:
                    out_row[slp_col] = np.nan
                    
                if dist_col in c_curr and not pd.isna(c_curr[dist_col]):
                    dist_vals.append(c_curr[dist_col])
                    out_row[dist_col] = c_curr[dist_col]
                else:
                    out_row[dist_col] = np.nan
                    
                if act_col in c_curr and not pd.isna(c_curr[act_col]):
                    if c_curr[act_col] == 1:
                        active = 1
                    out_row[act_col] = c_curr[act_col]
                else:
                    out_row[act_col] = 0
                    
            if wind_vals: out_row['Route_Max_cyclone_wind'] = np.max(wind_vals)
            else: out_row['Route_Max_cyclone_wind'] = 0
            
            if slp_vals: out_row['Route_Min_cyclone_slp'] = np.min(slp_vals)
            else: out_row['Route_Min_cyclone_slp'] = np.nan
            
            if dist_vals: out_row['Route_Min_cyclone_dist'] = np.min(dist_vals)
            else: out_row['Route_Min_cyclone_dist'] = np.nan
            
            out_row['Route_cyclone_active'] = active

    output_rows.append(out_row)

final_df = pd.DataFrame(output_rows)
out_path = os.path.join(NEW_DATASET_DIR, "final_ml_dataset.csv")
final_df.to_csv(out_path, index=False)
print(f"Done! Final dataset saved to {out_path} with {len(final_df)} records.")
