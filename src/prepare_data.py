#!/usr/bin/env python3
"""
Prepare NI Pothole Enquiries data for modeling.

- Loads 2016 (primary) +/- other years.
- Cleans junk coords.
- Adds lon/lat (EPSG:29902 Irish Grid -> WGS84).
- Adds time features.
- Assigns stable 200m grid cell IDs.
- Computes recurrence label: for each report, was there another report
  within ~200m in the next 14 days? (proxy for "this one is likely part
  of a fast-decaying or poorly-fixed cluster").
- Outputs cleaned CSV + a small metadata summary for the processed set.

This is deliberately lightweight (pure pandas + pyproj + stdlib) so it runs
immediately on a 16GB M1 MacBook Air with the project venv.
"""
import argparse
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import pyproj


def load_raw(paths):
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df["source_file"] = os.path.basename(p)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def clean_and_add_coords(df, crs_from="EPSG:29902", crs_to="EPSG:4326"):
    # Drop junk
    df = df[(df["EASTING"] > 100) & (df["NORTHING"] > 100)].copy()

    # Parse date
    df["date_recorded"] = pd.to_datetime(df["DATE_RECORDED"], format="%d/%m/%Y %H:%M:%S")

    # Transform coords
    transformer = pyproj.Transformer.from_crs(crs_from, crs_to, always_xy=True)
    lon, lat = transformer.transform(df["EASTING"].values, df["NORTHING"].values)
    df["lon"] = lon
    df["lat"] = lat

    # Keep original projected too (sometimes more stable for distance)
    df["easting"] = df["EASTING"].astype(float)
    df["northing"] = df["NORTHING"].astype(float)

    # Basic clean
    df["enquiry_category"] = df["ENQUIRY_CATEGORY"].astype("category")
    df["division"] = df["DIVISION"].astype("category")
    df["client_office"] = df["CLIENT_OFFICE_NAME"].astype("category")

    # Drop the original mixed-case cols we no longer need raw
    return df


def add_time_features(df):
    df = df.sort_values("date_recorded").reset_index(drop=True)
    dt = df["date_recorded"]
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek  # 0=Mon ... 6=Sun
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    # Simple winter proxy (high risk months in NI)
    df["is_winter_spring"] = df["month"].isin([1, 2, 3, 4, 11, 12]).astype(int)
    df["days_since_apr2016"] = (dt - pd.Timestamp("2016-04-01")).dt.total_seconds() / 86400.0
    return df


def assign_grid_cells(df, res_m=200):
    """Stable grid cell ID at ~res_m resolution using the original projected coords."""
    df["grid_x"] = (df["easting"] // res_m).astype(int)
    df["grid_y"] = (df["northing"] // res_m).astype(int)
    df["grid_cell"] = df["grid_x"].astype(str) + "_" + df["grid_y"].astype(str)
    df["grid_cell"] = df["grid_cell"].astype("category")
    return df


def add_recurrence_label(df, radius_m=200.0, horizon_days=14):
    """
    For each report, label whether >=1 OTHER report occurred
    within 'radius_m' (euclidean on Irish Grid meters) and within
    the next 'horizon_days' AFTER this report's timestamp.

    This is our main binary target for "high chance this complaint
    is part of a cluster that will keep generating reports soon".
    Useful for "solve quicker" prioritization.
    """
    # Work with numpy for speed on small N
    times = df["date_recorded"].values.astype("datetime64[ns]")
    east = df["easting"].values
    north = df["northing"].values
    n = len(df)

    # Precompute time deltas in days (upper triangle only for speed, but brute is fine)
    # We will do a simple double loop with early break since sorted by time.
    labels = np.zeros(n, dtype=np.int8)
    horizon = np.timedelta64(horizon_days, "D")

    for i in range(n):
        t0 = times[i]
        t_cutoff = t0 + horizon
        # Only look forward
        j = i + 1
        while j < n and times[j] <= t_cutoff:
            de = east[j] - east[i]
            dn = north[j] - north[i]
            dist = (de * de + dn * dn) ** 0.5
            if dist <= radius_m:
                labels[i] = 1
                break
            j += 1

    df["repeat_within_14d_200m"] = labels
    df["repeat_within_14d_200m"] = df["repeat_within_14d_200m"].astype("int8")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=["data/raw/pothole_enquiries_2016.csv"],
                        help="One or more raw CSVs (2016 primary)")
    parser.add_argument("--out", default="data/processed/potholes_2016_prepared.csv")
    parser.add_argument("--grid_res", type=int, default=200)
    parser.add_argument("--radius", type=float, default=200.0)
    parser.add_argument("--horizon", type=int, default=14)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("Loading raw...")
    raw = load_raw(args.inputs)
    print(f"  Loaded {len(raw)} rows from {len(args.inputs)} file(s)")

    print("Cleaning + coord transform + time + grid...")
    df = clean_and_add_coords(raw)
    df = add_time_features(df)
    df = assign_grid_cells(df, res_m=args.grid_res)
    print(f"  After clean: {len(df)} valid rows, {df['grid_cell'].nunique()} unique {args.grid_res}m cells")

    print(f"Computing recurrence label (radius={args.radius}m, horizon={args.horizon}d)...")
    df = add_recurrence_label(df, radius_m=args.radius, horizon_days=args.horizon)
    pos = df["repeat_within_14d_200m"].sum()
    print(f"  Positive rate (will have repeat in window): {pos} / {len(df)} = {100*pos/len(df):.2f}%")

    # Minimal columns for modeling + audit
    keep = [
        "date_recorded", "lon", "lat", "easting", "northing",
        "enquiry_category", "division", "client_office",
        "grid_cell", "grid_x", "grid_y",
        "year", "month", "dayofweek", "hour", "is_weekend", "is_winter_spring", "days_since_apr2016",
        "repeat_within_14d_200m",
        "source_file",
    ]
    out_df = df[keep].copy()

    out_df.to_csv(args.out, index=False)
    print(f"\nWrote processed: {args.out} ({len(out_df)} rows)")

    # Quick stats file
    stats = {
        "n_rows": int(len(out_df)),
        "n_unique_grid_cells": int(out_df["grid_cell"].nunique()),
        "positive_rate": float(out_df["repeat_within_14d_200m"].mean()),
        "date_min": str(out_df["date_recorded"].min()),
        "date_max": str(out_df["date_recorded"].max()),
        "grid_res_m": args.grid_res,
        "recurrence_radius_m": args.radius,
        "recurrence_horizon_d": args.horizon,
    }
    import json
    with open(args.out.replace(".csv", "_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("Wrote companion stats json.")


if __name__ == "__main__":
    main()
