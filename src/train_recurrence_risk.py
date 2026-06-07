#!/usr/bin/env python3
"""
Train a lightweight recurrence-risk model for NI pothole enquiries (2016 focus).

Target (proxy for "solve quicker / prevent further decay"):
  repeat_within_14d_200m  --  did this report get followed by another report
  inside 200 m (Irish Grid) within the next 14 days?

This is observable purely from the enquiry log. High predicted probability
means: "this location just generated a complaint that is likely to generate
more complaints soon unless something changes (better/faster patch, root
cause fix, etc.)".

Models (easy on 16GB M1 MacBook Air):
  - HistGradientBoostingClassifier (sklearn, modern boosted trees, no extra deps)
  - LogisticRegression (linear baseline, super fast, interpretable coeffs)

Split strategy for "generalize into the future":
  - Strict temporal: train on earlier period, validate middle, test on latest months.
  - No random shuffling, no future leakage into features.
  - We also compute a simple "earlier intervention" heuristic on the clusters:
    what fraction of follow-on reports arrived quickly after the first in a cell?
    This supports the claim that faster/better response on high-risk cases
    could have reduced total enquiry volume and associated deterioration.

Usage (after venv + prepare):
  source .venv/bin/activate
  python src/train_recurrence_risk.py \
      --data data/processed/potholes_2016_prepared.csv \
      --model_out models/recurrence_risk_hgb.joblib
"""
import argparse
import json
import os
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def temporal_split(df, train_end="2016-12-31", val_end="2017-02-28"):
    """Return train / val / test masks based on date_recorded. Purely temporal."""
    df = df.sort_values("date_recorded")
    train_mask = df["date_recorded"] <= pd.Timestamp(train_end)
    val_mask = (df["date_recorded"] > pd.Timestamp(train_end)) & (df["date_recorded"] <= pd.Timestamp(val_end))
    test_mask = df["date_recorded"] > pd.Timestamp(val_end)
    return train_mask, val_mask, test_mask


def build_feature_pipeline(cat_cols, num_cols):
    """
    ColumnTransformer + model pipeline.
    HistGradientBoosting handles categoricals natively if we pass them as such,
    but for simplicity + to show a full pipeline we one-hot the cats here
    (tiny cardinality, no problem). For very high-card you would use
    HGB's categorical feature support or target encoding.
    """
    preproc = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preproc


def add_simple_lag_features(df, grid_col="grid_cell"):
    """
    Very lightweight leakage-free lags: for each row, count of reports in
    same grid cell in the previous 7 and 14 days (computed from the sorted log).
    These are safe because we only use past data relative to each row.
    """
    df = df.sort_values("date_recorded").reset_index(drop=True)
    df["lag_7d_same_cell"] = 0
    df["lag_14d_same_cell"] = 0

    # Group by cell, then for each group walk forward
    for cell, g in df.groupby(grid_col, observed=True):
        idx = g.index.values
        times = g["date_recorded"].values
        for pos, i in enumerate(idx):
            t0 = times[pos]
            # count how many previous in this group within windows
            cnt7 = 0
            cnt14 = 0
            j = pos - 1
            while j >= 0:
                delta = (t0 - times[j]) / np.timedelta64(1, "D")
                if delta > 14:
                    break
                if delta <= 7:
                    cnt7 += 1
                cnt14 += 1
                j -= 1
            df.at[i, "lag_7d_same_cell"] = cnt7
            df.at[i, "lag_14d_same_cell"] = cnt14
    return df


def compute_earlier_patch_heuristic(df, radius_m=200, grid_for_cluster="grid_cell"):
    """
    Robust logic around 'if we had acted earlier on the first report in a cluster,
    could we have stopped subsequent reports/decay?'

    We look at all 200m (or grid) cells that eventually had 2+ reports.
    For each such cell, take the chronologically first report as t0.
    Then look at the arrival times of all later reports in that cell.

    Output: simple tables/stats you can quote:
      - % of multi-report cells that got their 2nd report within 1/3/7/14 days of first.
      - Overall, what fraction of all "repeat" reports (2nd+) arrived within 7 days of
        the previous one in the same cluster.

    Caveats (documented): no actual repair timestamps or quality, so this is a
    proxy heuristic only. Still useful for prioritization and for arguing value
    of fast response on predicted high-risk cases.
    """
    print("\n=== 'EARLIER PATCH' / DECAY PREVENTION HEURISTIC ===")
    # Use the 200m grid we already have (or could re-cluster, but grid is fine proxy)
    cell_groups = defaultdict(list)
    for _, row in df.iterrows():
        cell_groups[row[grid_for_cluster]].append(row["date_recorded"])

    multi_cells = {c: sorted(ts) for c, ts in cell_groups.items() if len(ts) >= 2}
    print(f"Cells with 2+ reports (using {grid_for_cluster}): {len(multi_cells)}")

    # For each multi cell, gaps from the *first* report
    first_to_second_days = []
    all_inter_report_gaps = []  # any consecutive in the cell's report list

    for ts in multi_cells.values():
        ts = sorted(ts)
        first_to_second_days.append((ts[1] - ts[0]) / np.timedelta64(1, "D"))
        for k in range(len(ts) - 1):
            gap = (ts[k + 1] - ts[k]) / np.timedelta64(1, "D")
            all_inter_report_gaps.append(gap)

    first_to_second = np.array(first_to_second_days)
    gaps = np.array(all_inter_report_gaps)

    def pct_under(x, thresh):
        return 100.0 * np.mean(x <= thresh)

    print("\nTime from FIRST report in cell to the SECOND report:")
    for d in [1, 3, 7, 14, 30]:
        print(f"  <= {d:2d} days: {pct_under(first_to_second, d):5.1f}% of multi-cells")

    print("\nAll consecutive inter-report gaps (any 2nd/3rd/... within a cell):")
    for d in [1, 3, 7, 14]:
        print(f"  <= {d:2d} days: {pct_under(gaps, d):5.1f}%")

    print(f"\nMedian gap between consecutive reports in multi-cells: {np.median(gaps):.2f} days")
    print(f"Mean gap: {np.mean(gaps):.2f} days")

    # A simple "avoidable" estimate for narrative
    within_7 = np.mean(gaps <= 7)
    print(f"\nRough narrative stat: ~{100*within_7:.0f}% of follow-on reports in these clusters")
    print("arrived within 7 days of the previous report in the same ~200m cell.")
    print("A system that (a) predicts high recurrence risk and (b) triggers a high-quality")
    print("repair within 48-72h on those cases could plausibly have eliminated a large")
    print("fraction of the repeat enquiries (and the vehicle damage / claims risk they represent).")
    print("This is a proxy analysis only -- we do not observe actual patch dates or quality.")

    return {
        "n_multi_cells": len(multi_cells),
        "pct_2nd_within_7d_of_first": float(pct_under(first_to_second, 7)),
        "pct_any_gap_within_7d": float(pct_under(gaps, 7)),
        "median_gap_days": float(np.median(gaps)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/potholes_2016_prepared.csv")
    parser.add_argument("--model_out", default="models/recurrence_risk_hgb.joblib")
    parser.add_argument("--train_end", default="2016-12-31")
    parser.add_argument("--val_end", default="2017-02-28")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    print(f"Loading {args.data}...")
    df = pd.read_csv(args.data, parse_dates=["date_recorded"])
    print(f"  {len(df)} rows, date range {df['date_recorded'].min()} - {df['date_recorded'].max()}")

    # Add cheap lag features (history only)
    print("Adding lag features (safe, past-only)...")
    df = add_simple_lag_features(df)

    # Define cats / nums for model (small set, easy to run)
    cat_cols = ["enquiry_category", "division", "client_office", "month", "dayofweek"]
    num_cols = ["lon", "lat", "hour", "is_weekend", "is_winter_spring",
                "lag_7d_same_cell", "lag_14d_same_cell", "days_since_apr2016"]

    # Ensure categoricals are treated as such for OHE
    for c in cat_cols:
        df[c] = df[c].astype(str).astype("category")

    # Temporal split
    train_m, val_m, test_m = temporal_split(df, args.train_end, args.val_end)
    print(f"\nTemporal split:")
    print(f"  train <= {args.train_end}: {train_m.sum()}")
    print(f"  val   <= {args.val_end}:   {val_m.sum()}")
    print(f"  test  >  {args.val_end}:   {test_m.sum()}")

    X = df[cat_cols + num_cols].copy().reset_index(drop=True)
    y = df["repeat_within_14d_200m"].values

    # Reset masks to positional after any prior reindexing
    train_idx = np.where(train_m.values)[0]
    val_idx = np.where(val_m.values)[0]
    test_idx = np.where(test_m.values)[0]

    preproc = build_feature_pipeline(cat_cols, num_cols)

    # Two models: the "fancy" one and a simple interpretable one
    models = {
        "hgb": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        ),
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    }

    results = {}
    for name, clf in models.items():
        print(f"\n=== Training {name} ===")
        pipe = Pipeline([("preproc", preproc), ("clf", clf)])

        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]

        pipe.fit(X_train, y_train)

        def eval_set(Xs, ys, tag):
            if len(ys) == 0 or ys.sum() == 0:
                print(f"  {tag}: no positive examples, skipping metrics")
                return {"roc_auc": None, "pr_auc": None}
            proba = pipe.predict_proba(Xs)[:, 1]
            roc = roc_auc_score(ys, proba)
            pr = average_precision_score(ys, proba)
            print(f"  {tag}: ROC-AUC={roc:.3f}  PR-AUC={pr:.3f}  (pos rate {ys.mean():.3f})")
            # Also quick thresholded report at 0.5 for intuition
            pred = (proba >= 0.5).astype(int)
            print(f"    classification_report @0.5 (val/test only shown for hgb):")
            if tag in ("val", "test") and name == "hgb":
                print(classification_report(ys, pred, zero_division=0))
            return {"roc_auc": float(roc), "pr_auc": float(pr), "n": int(len(ys)), "pos_rate": float(ys.mean())}

        res = {
            "train": eval_set(X_train, y_train, "train"),
            "val": eval_set(X_val, y_val, "val"),
            "test": eval_set(X_test, y_test, "test"),
        }
        results[name] = res

        if name == "hgb":
            # Save the best practical model
            joblib.dump(pipe, args.model_out)
            print(f"\nSaved HGB pipeline -> {args.model_out}")

    # The key "earlier patch prevents decay" analysis
    heuristic = compute_earlier_patch_heuristic(df)

    # Write a small results summary
    summary = {
        "temporal_split": {"train_end": args.train_end, "val_end": args.val_end},
        "metrics": results,
        "earlier_patch_heuristic": heuristic,
        "note": "All splits are strictly temporal. Features use only information available at or before the report time (lags are past-only).",
    }
    out_json = args.model_out.replace(".joblib", "_results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote results summary -> {out_json}")

    print("\nDone. Key takeaway for your use-case:")
    print("  The model ranks new pothole *reports* by predicted probability that the")
    print("  location will generate more reports soon. You can use the score to")
    print("  prioritize fast, high-quality intervention on the ones most likely to")
    print("  'decay' into repeat problems (the proxy we can actually observe).")


if __name__ == "__main__":
    main()
