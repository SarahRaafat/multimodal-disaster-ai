#!/usr/bin/env python3
import pandas as pd, numpy as np

# Load your test split
df = pd.read_csv("text_data/test.csv")

# Add random coordinates in US bounding box
df["lat"] = np.random.uniform(25, 50, size=len(df))     # latitudes ~ US
df["lon"] = np.random.uniform(-125, -65, size=len(df))  # longitudes ~ US

# Save as new split with geo columns
df.to_csv("text_data/test_geo.csv", index=False)

print("[done] Wrote test_geo.csv with fake lat/lon columns")
