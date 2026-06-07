#!/usr/bin/env python3
"""
Train the primary pothole recurrence model on official DfI surface-defect data.

This is the stronger operational model for cost saving because it uses
authority-logged pothole defects rather than citizen enquiries. The target is:

  will another official pothole defect be recorded within 200 m in the next 14d?

Important details:
- Raw 2016 and 2017 files are combined before labels are computed, so reports
  near the fiscal-year boundary can see future records in the next file.
- The final horizon window is excluded from evaluation/training because labels
  are censored there.
- Features use only information available at the defect recorded time.
"""
import argparse
import json
import os
import re
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
import pyproj
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DAY_NS = 24 * 60 * 60 * 1_000_000_000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/raw/surface_defects/surface_defects_2016.csv",
            "data/raw/surface_defects/surface_defects_2017.csv",
        ],
    )
    parser.add_argument("--model_out", default="models/surface_recurrence_best.joblib")
    parser.add_argument("--results_out", default="models/surface_recurrence_best_results.json")
    parser.add_argument("--radius_m", type=float, default=200.0)
    parser.add_argument("--horizon_days", type=int, default=14)
    parser.add_argument("--grid_res_m", type=int, default=200)
    parser.add_argument("--train_end", default="2017-03-31")
    parser.add_argument("--val_end", default="2017-09-30")
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def route_code(section_name):
    if not isinstance(section_name, str) or not section_name.strip():
        return "UNKNOWN"
    return section_name.strip().split()[0].upper()


def road_class(route):
    # Section tokens often look like 7020B0093_03 or 7020A0042_25.
    match = re.search(r"([ABCUM])\d", route)
    return match.group(1) if match else "UNKNOWN"


def pothole_code(defect_detail):
    if not isinstance(defect_detail, str):
        return "UNKNOWN"
    match = re.search(r"\(([^)]+)\)", defect_detail)
    return match.group(1).upper() if match else "UNKNOWN"


def load_surface_potholes(paths):
    frames = []
    for path in paths:
        df = pd.read_csv(path, low_memory=False)
        df["source_file"] = os.path.basename(path)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["date_recorded"] = pd.to_datetime(
        df["RECORDED_DATE"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    df = df.dropna(subset=["date_recorded", "EASTING", "NORTHING", "DEFECT_DETAIL"])
    df = df[df["DEFECT_DETAIL"].str.contains("POTHOLE", case=False, na=False)].copy()
    df = df[(df["EASTING"] > 100) & (df["NORTHING"] > 100)].copy()

    df["easting"] = df["EASTING"].astype(float)
    df["northing"] = df["NORTHING"].astype(float)
    transformer = pyproj.Transformer.from_crs("EPSG:29902", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(df["easting"].values, df["northing"].values)
    df["lon"] = lon
    df["lat"] = lat

    df["defect_detail"] = df["DEFECT_DETAIL"].astype(str).str.strip()
    df["division"] = df["DIVISION"].astype(str).str.strip()
    df["section_office"] = df["SECTION_OFFICE"].astype(str).str.strip()
    df["section_name"] = df["SECTION_NAME"].astype(str).str.strip()
    df["route_code"] = df["section_name"].map(route_code)
    df["road_class"] = df["route_code"].map(road_class)
    df["pothole_code"] = df["defect_detail"].map(pothole_code)
    df["surface_material"] = df["defect_detail"].str.split().str[0].fillna("UNKNOWN")

    df = df.sort_values("date_recorded").reset_index(drop=True)
    return df


def add_time_and_grid_features(df, grid_res_m):
    dt = df["date_recorded"]
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month.astype(str)
    df["dayofweek"] = dt.dt.dayofweek.astype(str)
    df["hour"] = dt.dt.hour
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["is_winter_spring"] = dt.dt.month.isin([1, 2, 3, 4, 11, 12]).astype(int)
    df["days_since_start"] = (dt - dt.min()).dt.total_seconds() / 86400.0
    df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12.0)
    df["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7.0)
    df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24.0)

    df["grid_x"] = (df["easting"] // grid_res_m).astype(int)
    df["grid_y"] = (df["northing"] // grid_res_m).astype(int)
    df["grid_cell"] = df["grid_x"].astype(str) + "_" + df["grid_y"].astype(str)
    return df


def build_cell_index(df):
    times = df["date_recorded"].values.astype("datetime64[ns]").astype("int64")
    east = df["easting"].values
    north = df["northing"].values
    grid_x = df["grid_x"].values
    grid_y = df["grid_y"].values

    groups = {}
    grouped = defaultdict(list)
    for i, key in enumerate(zip(grid_x, grid_y)):
        grouped[key].append(i)

    for key, idx in grouped.items():
        arr = np.asarray(idx, dtype=np.int64)
        groups[key] = {
            "idx": arr,
            "times": times[arr],
            "east": east[arr],
            "north": north[arr],
        }
    return groups, times, east, north, grid_x, grid_y


def add_spatial_temporal_features_and_label(df, radius_m, horizon_days):
    groups, times, east, north, grid_x, grid_y = build_cell_index(df)
    n = len(df)
    radius2 = radius_m * radius_m
    horizon_ns = horizon_days * DAY_NS
    windows = [3, 7, 14, 30]

    label = np.zeros(n, dtype=np.int8)
    near_counts = {d: np.zeros(n, dtype=np.int16) for d in windows}
    cell_counts = {d: np.zeros(n, dtype=np.int16) for d in windows}

    for i in range(n):
        if i and i % 50_000 == 0:
            print(f"  spatial/temporal scan: {i:,}/{n:,}")

        gx = grid_x[i]
        gy = grid_y[i]
        t0 = times[i]
        e0 = east[i]
        n0 = north[i]

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                group = groups.get((gx + dx, gy + dy))
                if group is None:
                    continue

                gt = group["times"]
                ge = group["east"]
                gn = group["north"]

                future_lo = np.searchsorted(gt, t0, side="right")
                future_hi = np.searchsorted(gt, t0 + horizon_ns, side="right")
                if future_hi > future_lo and label[i] == 0:
                    de = ge[future_lo:future_hi] - e0
                    dn = gn[future_lo:future_hi] - n0
                    if np.any((de * de + dn * dn) <= radius2):
                        label[i] = 1

                past_hi = np.searchsorted(gt, t0, side="left")
                if past_hi == 0:
                    continue
                for days in windows:
                    past_lo = np.searchsorted(gt, t0 - days * DAY_NS, side="left")
                    if past_hi <= past_lo:
                        continue
                    if dx == 0 and dy == 0:
                        cell_counts[days][i] += past_hi - past_lo
                    de = ge[past_lo:past_hi] - e0
                    dn = gn[past_lo:past_hi] - n0
                    near_counts[days][i] += int(np.sum((de * de + dn * dn) <= radius2))

    df[f"repeat_within_{horizon_days}d_{int(radius_m)}m"] = label
    for days in windows:
        df[f"nearby_{int(radius_m)}m_lag_{days}d"] = near_counts[days]
        df[f"same_grid_lag_{days}d"] = cell_counts[days]
    return df


def add_section_lag_features(df):
    times = df["date_recorded"].values.astype("datetime64[ns]").astype("int64")
    for name in ["section_name", "route_code"]:
        for days in [30, 90]:
            df[f"{name}_lag_{days}d"] = 0

        for _, idx in df.groupby(name, sort=False).groups.items():
            arr = np.asarray(list(idx), dtype=np.int64)
            t = times[arr]
            positions = np.arange(len(arr))
            for days in [30, 90]:
                left = np.searchsorted(t, t - days * DAY_NS, side="left")
                df.loc[arr, f"{name}_lag_{days}d"] = positions - left
    return df


def temporal_masks(df, train_end, val_end, horizon_days):
    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)
    censor_cutoff = df["date_recorded"].max() - pd.Timedelta(days=horizon_days)
    usable = df["date_recorded"] <= censor_cutoff
    train = usable & (df["date_recorded"] <= train_end)
    val = usable & (df["date_recorded"] > train_end) & (df["date_recorded"] <= val_end)
    test = usable & (df["date_recorded"] > val_end)
    return train, val, test, censor_cutoff


def metric_block(y, proba):
    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
    }


def threshold_table(y, proba, thresholds):
    rows = []
    for threshold in thresholds:
        pred = proba >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "flagged_rate": float(np.mean(pred)),
                "break_even_avoided_cost_multiple_if_effective_100pct": (
                    float(1.0 / precision) if precision > 0 else None
                ),
                "break_even_avoided_cost_multiple_if_effective_50pct": (
                    float(1.0 / (0.5 * precision)) if precision > 0 else None
                ),
            }
        )
    return rows


def fit_temperature(y_val, raw_val):
    """One-parameter calibration for HGB logits; keeps ranking unchanged."""
    best_temp = 1.0
    best_brier = brier_score_loss(y_val, expit(raw_val))
    for temp in np.linspace(0.6, 2.5, 39):
        proba = expit(raw_val / temp)
        score = brier_score_loss(y_val, proba)
        if score < best_brier:
            best_brier = score
            best_temp = float(temp)
    return best_temp


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    print("Loading and preparing raw surface-defect potholes...")
    df = load_surface_potholes(args.inputs)
    df = add_time_and_grid_features(df, args.grid_res_m)
    print(
        f"  {len(df):,} pothole-coded rows, "
        f"{df['date_recorded'].min()} to {df['date_recorded'].max()}"
    )

    print("Computing leakage-free spatial/temporal features and recurrence labels...")
    df = add_spatial_temporal_features_and_label(df, args.radius_m, args.horizon_days)
    df = add_section_lag_features(df)

    target_col = f"repeat_within_{args.horizon_days}d_{int(args.radius_m)}m"
    train_m, val_m, test_m, censor_cutoff = temporal_masks(
        df, args.train_end, args.val_end, args.horizon_days
    )
    print("Temporal split:")
    print(f"  train <= {args.train_end}: {int(train_m.sum()):,}")
    print(f"  val   <= {args.val_end}:   {int(val_m.sum()):,}")
    print(f"  test  >  {args.val_end}:   {int(test_m.sum()):,}")
    print(f"  excluded final censored rows after {censor_cutoff}: {int((~(train_m | val_m | test_m)).sum()):,}")

    low_card_cats = [
        "defect_detail",
        "division",
        "section_office",
        "surface_material",
        "pothole_code",
        "road_class",
        "month",
        "dayofweek",
    ]
    logreg_cats = low_card_cats + ["route_code"]
    radius_tag = int(args.radius_m)
    num_cols = [
        "lon",
        "lat",
        "easting",
        "northing",
        "grid_x",
        "grid_y",
        "hour",
        "is_weekend",
        "is_winter_spring",
        "days_since_start",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "hour_sin",
        "hour_cos",
        f"nearby_{radius_tag}m_lag_3d",
        f"nearby_{radius_tag}m_lag_7d",
        f"nearby_{radius_tag}m_lag_14d",
        f"nearby_{radius_tag}m_lag_30d",
        "same_grid_lag_3d",
        "same_grid_lag_7d",
        "same_grid_lag_14d",
        "same_grid_lag_30d",
        "section_name_lag_30d",
        "section_name_lag_90d",
        "route_code_lag_30d",
        "route_code_lag_90d",
    ]

    for col in set(low_card_cats + logreg_cats):
        df[col] = df[col].astype(str)

    y = df[target_col].astype(int).values
    thresholds = [0.40, 0.50, 0.60, 0.70, 0.80]

    model_specs = {
        "hgb": {
            "features": low_card_cats + num_cols,
            "pipeline": Pipeline(
                [
                    (
                        "preproc",
                        ColumnTransformer(
                            [
                                (
                                    "cat",
                                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                    low_card_cats,
                                ),
                                ("num", StandardScaler(), num_cols),
                            ],
                            verbose_feature_names_out=False,
                        ),
                    ),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            max_iter=350,
                            learning_rate=0.045,
                            max_leaf_nodes=31,
                            l2_regularization=0.05,
                            min_samples_leaf=30,
                            random_state=args.random_state,
                            early_stopping=True,
                            validation_fraction=0.1,
                        ),
                    ),
                ]
            ),
        },
        "logreg_route_sparse": {
            "features": logreg_cats + num_cols,
            "pipeline": Pipeline(
                [
                    (
                        "preproc",
                        ColumnTransformer(
                            [
                                (
                                    "cat",
                                    OneHotEncoder(
                                        handle_unknown="ignore",
                                        min_frequency=20,
                                        sparse_output=True,
                                    ),
                                    logreg_cats,
                                ),
                                ("num", StandardScaler(), num_cols),
                            ]
                        ),
                    ),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            C=0.5,
                            solver="saga",
                            n_jobs=-1,
                            random_state=args.random_state,
                        ),
                    ),
                ]
            ),
        },
    }

    results = {}
    fitted = {}
    for name, spec in model_specs.items():
        features = spec["features"]
        pipe = spec["pipeline"]
        print(f"\nTraining {name}...")
        pipe.fit(df.loc[train_m, features], y[train_m.values])
        fitted[name] = pipe

        model_result = {}
        for split_name, mask in [("train", train_m), ("val", val_m), ("test", test_m)]:
            proba = pipe.predict_proba(df.loc[mask, features])[:, 1]
            model_result[split_name] = metric_block(y[mask.values], proba)
            print(
                f"  {split_name}: ROC={model_result[split_name]['roc_auc']:.3f} "
                f"PR={model_result[split_name]['pr_auc']:.3f} "
                f"Brier={model_result[split_name]['brier']:.3f} "
                f"pos={model_result[split_name]['positive_rate']:.3f} "
                f"n={model_result[split_name]['n']:,}"
            )

        model_result["thresholds_on_test"] = threshold_table(
            y[test_m.values], pipe.predict_proba(df.loc[test_m, features])[:, 1], thresholds
        )

        if name == "hgb":
            raw_val = pipe.decision_function(df.loc[val_m, features])
            raw_test = pipe.decision_function(df.loc[test_m, features])
            temp = fit_temperature(y[val_m.values], raw_val)
            calibrated_test = expit(raw_test / temp)
            model_result["temperature_calibration"] = {
                "temperature": float(temp),
                "val_brier_uncalibrated": float(
                    brier_score_loss(y[val_m.values], expit(raw_val))
                ),
                "test_brier_calibrated": float(
                    brier_score_loss(y[test_m.values], calibrated_test)
                ),
                "thresholds_on_test_calibrated": threshold_table(
                    y[test_m.values], calibrated_test, thresholds
                ),
            }

        results[name] = model_result

    best_name = max(results, key=lambda n: results[n]["val"]["pr_auc"])
    print(f"\nBest model by validation PR-AUC: {best_name}")

    best_features = model_specs[best_name]["features"]
    final_pipe = model_specs[best_name]["pipeline"]
    train_val = train_m | val_m
    final_pipe.fit(df.loc[train_val, best_features], y[train_val.values])

    artifact = {
        "pipeline": final_pipe,
        "features": best_features,
        "target": target_col,
        "radius_m": args.radius_m,
        "horizon_days": args.horizon_days,
        "grid_res_m": args.grid_res_m,
        "train_end": args.train_end,
        "val_end": args.val_end,
        "censor_cutoff": str(censor_cutoff),
        "model_name": best_name,
        "cost_rule": (
            "Fast/permanent intervention is economically justified when "
            "p_repeat * intervention_effectiveness * avoided_repeat_cost "
            "> incremental_fast_repair_cost."
        ),
    }
    joblib.dump(artifact, args.model_out)

    summary = {
        "data": {
            "inputs": args.inputs,
            "rows_pothole_coded": int(len(df)),
            "date_min": str(df["date_recorded"].min()),
            "date_max": str(df["date_recorded"].max()),
            "target_positive_rate_all_rows": float(df[target_col].mean()),
            "target": target_col,
            "radius_m": args.radius_m,
            "horizon_days": args.horizon_days,
            "note": "Final horizon window excluded from train/val/test to avoid censored labels.",
        },
        "split": {
            "train_end": args.train_end,
            "val_end": args.val_end,
            "censor_cutoff": str(censor_cutoff),
            "train_n": int(train_m.sum()),
            "val_n": int(val_m.sum()),
            "test_n": int(test_m.sum()),
        },
        "models": results,
        "selected_model": best_name,
        "selection_metric": "validation_pr_auc",
        "saved_model": args.model_out,
        "cost_function": {
            "decision_rule": "act if p_repeat * effectiveness * avoided_repeat_cost > incremental_fast_repair_cost",
            "threshold_formula": "p_repeat > incremental_fast_repair_cost / (effectiveness * avoided_repeat_cost)",
            "interpretation": (
                "The data does not contain repair costs, so the monetary function is "
                "parameterized. Threshold rows include the break-even avoided-cost "
                "multiple implied by observed precision."
            ),
        },
    }

    with open(args.results_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved model -> {args.model_out}")
    print(f"Saved results -> {args.results_out}")


if __name__ == "__main__":
    main()
