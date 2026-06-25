"""
00_generate_raw_data.py
-------------------------------------------------------------------
Generates a realistic MESSY dataset that simulates what you would
download from Kaggle/UCI for a "House Listings" problem.

In a REAL project: skip this script. Just download a CSV from
Kaggle/UCI and put it in the data/ folder instead.

This script exists ONLY so this demo has something messy to clean.
Run once to produce data/raw_house_listings.csv
-------------------------------------------------------------------
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 500

# ---- Base realistic features ----
area_sqft = np.random.normal(1800, 600, n).round(0)
bedrooms = np.random.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.25, 0.35, 0.2, 0.1])
bathrooms = np.random.choice([1, 2, 3, 4], n, p=[0.2, 0.4, 0.3, 0.1])
age_years = np.random.randint(0, 50, n)
distance_to_city_km = np.abs(np.random.normal(15, 10, n)).round(1)
neighborhood = np.random.choice(["Downtown", "Suburb", "Rural", "Uptown"], n, p=[0.3, 0.4, 0.15, 0.15])

# Price loosely depends on the above (so feature engineering has real signal)
base_price = (
    area_sqft * 120
    + bedrooms * 8000
    + bathrooms * 5000
    - age_years * 600
    - distance_to_city_km * 900
    + np.random.normal(0, 15000, n)
)
price = np.abs(base_price).round(0)

# An extra near-duplicate column on purpose, to demonstrate
# MULTICOLLINEARITY removal later (area_sqft_m2 is just area_sqft * 0.0929)
area_sqft_m2 = (area_sqft * 0.0929).round(1)

df = pd.DataFrame({
    "area_sqft": area_sqft,
    "area_sqft_m2": area_sqft_m2,   # redundant w/ area_sqft -> multicollinearity demo
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "age_years": age_years,
    "distance_to_city_km": distance_to_city_km,
    "neighborhood": neighborhood,
    "price": price,
})

# ---- Inject messiness, like a real-world dataset ----

# 1. Missing values at DIFFERENT rates per column, deliberately spanning
#    all three buckets from the PDF's decision matrix: <5%, 5-20%, >20%
df.loc[df.sample(frac=0.03, random_state=1).index, "age_years"] = np.nan            # <5%  -> drop rows
df.loc[df.sample(frac=0.12, random_state=2).index, "area_sqft"] = np.nan            # 5-20% -> median
df.loc[df.sample(frac=0.15, random_state=3).index, "distance_to_city_km"] = np.nan  # 5-20% -> group/KNN
df.loc[df.sample(frac=0.25, random_state=4).index, "bathrooms"] = np.nan            # >20% -> KNN
df.loc[df.sample(frac=0.03, random_state=5).index, "neighborhood"] = np.nan         # <5%  -> drop rows

# 2. Outliers (a few absurd values, like data-entry errors)
outlier_idx = np.random.choice(df.index, size=8, replace=False)
df.loc[outlier_idx[:4], "area_sqft"] = np.random.choice([15000, 18000, 50, 30], 4)
df.loc[outlier_idx[4:], "price"] = df.loc[outlier_idx[4:], "price"] * np.random.choice([6, 8], 4)

# 3. Inconsistent text casing/whitespace (very common real-world mess)
df["neighborhood"] = df["neighborhood"].apply(
    lambda x: x.lower() if pd.notna(x) and np.random.rand() < 0.15 else x
)
sample_idx = df.sample(5, random_state=6).index
df.loc[sample_idx, "neighborhood"] = df.loc[sample_idx, "neighborhood"].apply(
    lambda x: f"  {x}  " if pd.notna(x) else x
)

# 4. Duplicate rows (common in scraped/merged data)
dup_rows = df.sample(6, random_state=7)
df = pd.concat([df, dup_rows], ignore_index=True)

df.to_csv("data/raw_house_listings.csv", index=False)
print(f"Generated data/raw_house_listings.csv -> {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())
