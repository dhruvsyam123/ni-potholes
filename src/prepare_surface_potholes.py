#!/usr/bin/env python3
"""
Prepare official Surface Defects data, filtered to actual POTHOLE-coded defects,
for a more physically-grounded predictive model.

Uses the same 200m grid + 14-day recurrence proxy as the enquiries model,
but now on authority-logged pothole work items (the real physical events).
"""
import argparse
import os
import pandas as pd
import numpy as np
import pyproj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/surface_defects/surface_defects_2016.csv")
    parser.add_argument("--out", default="data/processed/surface_potholes_2016_prepared.csv")
    parser.add_argument("--grid_res", type=int, default=200)
    parser.add_argument("--radius", type=float, default=200.0)
    parser.add_argument("--horizon", type=int, default=14)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("Loading surface defects...")
    df = pd.read_csv(args.input, low_memory=False)
    df["RECORDED_DATE"] = pd.to_datetime(df["RECORDED_DATE"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["RECORDED_DATE", "EASTING", "NORTHING"])

    # Filter to true physical potholes (the coded ones)
    poth_mask = df["DEFECT_DETAIL"].str.contains("POTHOLE", case=False, na=False)
    df = df[poth_mask].copy()
    print(f"Filtered to {len(df):,} official POTHOLE-coded defects")

    # Coords + transform (same CRS)
    df["easting"] = df["EASTING"].astype(float)
    df["northing"] = df["NORTHING"].astype(float)
    transformer = pyproj.Transformer.from_crs("EPSG:29902", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(df["easting"].values, df["northing"].values)
    df["lon"] = lon
    df["lat"] = lat

    # Admin features (align names with previous where possible)
    df["division"] = df["DIVISION"].astype("category")
    df["section_office"] = df["SECTION_OFFICE"].astype("category")
    df["defect_detail"] = df["DEFECT_DETAIL"].astype("category")  # the physical code

    # Time features
    dt = df["RECORDED_DATE"]
    df["date_recorded"] = dt
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["dayofweek"] = dt.dt.dayofweek
    df["hour"] = dt.dt.hour
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_winter_spring"] = df["month"].isin([1,2,3,4,11,12]).astype(int)
    df["days_since_apr2016"] = (dt - pd.Timestamp("2016-04-01")).dt.total_seconds() / 86400.0

    # Grid
    df["grid_x"] = (df["easting"] // args.grid_res).astype(int)
    df["grid_y"] = (df["northing"] // args.grid_res).astype(int)
    df["grid_cell"] = df["grid_x"].astype(str) + "_" + df["grid_y"].astype(str)

    # Recurrence label (will another official pothole defect be logged nearby soon?)
    print("Computing recurrence label on official pothole defects (this may take a minute for 95k rows)...")
    df = df.sort_values("date_recorded").reset_index(drop=True)
    times = df["date_recorded"].values.astype("datetime64[ns]")
    east = df["easting"].values
    north = df["northing"].values
    n = len(df)
    labels = np.zeros(n, dtype=np.int8)
    horizon = np.timedelta64(args.horizon, "D")
    rad2 = args.radius ** 2

    for i in range(n):
        t_cutoff = times[i] + horizon
        j = i + 1
        while j < n and times[j] <= t_cutoff:
            de = east[j] - east[i]
            dn = north[j] - north[i]
            if (de*de + dn*dn) <= rad2:
                labels[i] = 1
                break
            j += 1
    df["repeat_within_14d_200m"] = labels

    pos = labels.sum()
    print(f"Positive rate (another pothole defect logged in window): {pos} / {n} = {100*pos/n:.2f}%")

    keep = [
        "date_recorded", "lon", "lat", "easting", "northing",
        "defect_detail", "division", "section_office",
        "grid_cell", "grid_x", "grid_y",
        "year", "month", "dayofweek", "hour", "is_weekend", "is_winter_spring", "days_since_apr2016",
        "repeat_within_14d_200m",
    ]
    out = df[keep].copy()
    out.to_csv(args.out, index=False)
    print(f"\nWrote prepared official pothole defects: {args.out} ({len(out):,} rows)")

    # Also save a 2017 version if provided for future test
    # (user can call with 2017 input separately or we handle in training)

if __name__ == "__main__":
    main()
