# UNCTAD Liner Shipping Connectivity Index (LSCI) Data

## About LSCI
The Liner Shipping Connectivity Index (LSCI) is published by UNCTAD (United Nations Conference on Trade and Development) 
to measure how well a country is connected to global liner shipping networks.

## Data Source
- **Organization**: UNCTAD (UN Trade and Development)
- **URL**: https://unctadstat.unctad.org/datacentre/dataviewer/US.LSCI
- **Partner**: MDS Transmodal
- **Methodology**: Based on 6 components:
  1. Number of scheduled ship calls per week
  2. Total deployed capacity (TEU)
  3. Number of liner shipping services
  4. Number of shipping companies
  5. Size of largest vessel
  6. Number of direct country connections

## Coverage
- **Time Period**: 2015-2024
- **Frequency**: 
  - Annual data: 2015-2019
  - Quarterly data: 2020-2024
- **Countries**: 20 major economies
- **Total Records": 500+

## CSV File Structure
- `Year`: Year of measurement
- `Quarter`: Quarter (Q1/Q2/Q3/Q4) or 'Annual'
- `Country`: Country/Economy name
- `LSCI_Score`: LSCI index value
- `Data_Type`: 'Annual' or 'Quarterly'

## Index Interpretation
- Higher values = better connectivity
- Q1 2023 = 100 (reference base for recent data)
- Pre-2024 data uses different base year (2006 = 100)

## Top Performers (2024)
1. China - 128.3
2. Singapore - 69.8
3. South Korea - 64.8
4. Hong Kong SAR - 61.8
5. Malaysia - 54.9

## Usage
```python
import pandas as pd

# Load data
df = pd.read_csv('unctad_lsci_2015_2024.csv')

# Filter by year
2024_data = df[(df['Year'] == 2024) & (df['Quarter'] == 'Q4')]

# Plot trends for a country
china_data = df[df['Country'] == 'China']
```

## Citation
UNCTAD (2024). Liner Shipping Connectivity Index. 
United Nations Conference on Trade and Development.
Available at: https://unctadstat.unctad.org/

## Updates
Data is updated quarterly by UNCTAD (March, June, September, December)
