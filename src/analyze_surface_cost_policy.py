#!/usr/bin/env python3
"""
Cost-sensitivity analysis for the surface-defect pothole recurrence model.

This script answers the operational question:

  At which risk threshold does fast/permanent intervention become worthwhile?

The raw DfI data does not contain repair costs, so costs are explicit scenario
assumptions. The model supplies precision/recall/workload on a future holdout;
the scenarios translate those operating points into expected net value.
"""
import argparse
import csv
import json
import os

import joblib
import numpy as np

from train_surface_recurrence_model import (
    add_section_lag_features,
    add_spatial_temporal_features_and_label,
    add_time_and_grid_features,
    load_surface_potholes,
    temporal_masks,
)


DEFAULT_INPUTS = [
    "data/raw/surface_defects/surface_defects_2016.csv",
    "data/raw/surface_defects/surface_defects_2017.csv",
]


SCENARIOS = [
    {
        "name": "direct_repeat_only_low_increment",
        "incremental_fast_repair_cost": 30.86,
        "avoided_repeat_cost": 87.80,
        "effectiveness": 0.50,
        "description": (
            "Very conservative: extra intervention cost is only the ALARM reactive-vs-planned "
            "fill delta; benefit is avoiding one average reactive fill; intervention prevents "
            "only half of would-be recurrences."
        ),
    },
    {
        "name": "direct_repeat_only_high_effect",
        "incremental_fast_repair_cost": 30.86,
        "avoided_repeat_cost": 87.80,
        "effectiveness": 0.75,
        "description": (
            "Conservative but assumes a good-quality intervention prevents 75% of would-be "
            "nearby repeat defects."
        ),
    },
    {
        "name": "ops_admin_low_claim",
        "incremental_fast_repair_cost": 75.00,
        "avoided_repeat_cost": 155.30,
        "effectiveness": 0.60,
        "description": (
            "Adds modest repeat inspection/admin cost and a low probability of driver damage "
            "externality to one avoided reactive fill."
        ),
    },
    {
        "name": "moderate_deterioration",
        "incremental_fast_repair_cost": 125.00,
        "avoided_repeat_cost": 272.80,
        "effectiveness": 0.60,
        "description": (
            "Assumes a better first repair is materially more expensive but avoids repeat work, "
            "some deterioration escalation, and low-probability vehicle damage."
        ),
    },
    {
        "name": "large_patch_premium",
        "incremental_fast_repair_cost": 300.00,
        "avoided_repeat_cost": 579.80,
        "effectiveness": 0.60,
        "description": (
            "Large quality/traffic-management premium, justified only if repeats carry larger "
            "deterioration, disruption, and damage costs."
        ),
    },
    {
        "name": "medium_patch_full_cost_not_incremental",
        "incremental_fast_repair_cost": 1328.00,
        "avoided_repeat_cost": 579.80,
        "effectiveness": 0.60,
        "description": (
            "Stress test: treats a medium patch cost as if it were entirely incremental. This "
            "is intentionally harsh and usually not economically viable from recurrence alone."
        ),
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/surface_recurrence_best.joblib")
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--out_json", default="models/surface_cost_policy_results.json")
    parser.add_argument("--out_csv", default="models/surface_cost_policy_thresholds.csv")
    parser.add_argument("--radius_m", type=float, default=200.0)
    parser.add_argument("--horizon_days", type=int, default=14)
    parser.add_argument("--grid_res_m", type=int, default=200)
    parser.add_argument("--train_end", default="2017-03-31")
    parser.add_argument("--val_end", default="2017-09-30")
    parser.add_argument("--threshold_min", type=float, default=0.20)
    parser.add_argument("--threshold_max", type=float, default=0.90)
    parser.add_argument("--threshold_step", type=float, default=0.025)
    return parser.parse_args()


def threshold_rows(y, proba, thresholds, scenarios):
    rows = []
    n = len(y)
    positives = int(y.sum())
    for threshold in thresholds:
        flagged = proba >= threshold
        tp = int(np.sum(flagged & (y == 1)))
        fp = int(np.sum(flagged & (y == 0)))
        fn = int(np.sum((~flagged) & (y == 1)))
        flagged_n = int(flagged.sum())
        precision = tp / flagged_n if flagged_n else 0.0
        recall = tp / positives if positives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        for scenario in scenarios:
            incremental = scenario["incremental_fast_repair_cost"]
            avoided = scenario["avoided_repeat_cost"]
            effectiveness = scenario["effectiveness"]
            gross_benefit = tp * effectiveness * avoided
            intervention_cost = flagged_n * incremental
            net_value = gross_benefit - intervention_cost
            rows.append(
                {
                    "scenario": scenario["name"],
                    "threshold": round(float(threshold), 3),
                    "n": n,
                    "actual_positives": positives,
                    "flagged_n": flagged_n,
                    "flagged_rate": flagged_n / n,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "incremental_fast_repair_cost": incremental,
                    "avoided_repeat_cost": avoided,
                    "effectiveness": effectiveness,
                    "gross_benefit": gross_benefit,
                    "intervention_cost": intervention_cost,
                    "net_value": net_value,
                    "net_value_per_flagged": net_value / flagged_n if flagged_n else 0.0,
                    "break_even_avoided_cost": (
                        incremental / (precision * effectiveness)
                        if precision > 0 and effectiveness > 0
                        else None
                    ),
                    "break_even_avoided_cost_multiple": (
                        1.0 / (precision * effectiveness)
                        if precision > 0 and effectiveness > 0
                        else None
                    ),
                }
            )
    return rows


def best_rows(rows):
    by_scenario = {}
    for row in rows:
        current = by_scenario.get(row["scenario"])
        if current is None or row["net_value"] > current["net_value"]:
            by_scenario[row["scenario"]] = row
    return by_scenario


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    artifact = joblib.load(args.model)
    pipe = artifact["pipeline"]
    features = artifact["features"]
    target = artifact["target"]

    print("Rebuilding holdout features from raw surface-defect data...")
    df = load_surface_potholes(args.inputs)
    df = add_time_and_grid_features(df, args.grid_res_m)
    df = add_spatial_temporal_features_and_label(df, args.radius_m, args.horizon_days)
    df = add_section_lag_features(df)

    _, _, test_m, censor_cutoff = temporal_masks(
        df, args.train_end, args.val_end, args.horizon_days
    )
    y = df.loc[test_m, target].astype(int).values
    proba = pipe.predict_proba(df.loc[test_m, features])[:, 1]

    thresholds = np.round(
        np.arange(
            args.threshold_min,
            args.threshold_max + args.threshold_step / 2,
            args.threshold_step,
        ),
        3,
    )
    rows = threshold_rows(y, proba, thresholds, SCENARIOS)
    best = best_rows(rows)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": args.model,
        "target": target,
        "holdout": {
            "date_rule": f"> {args.val_end} and <= {censor_cutoff}",
            "n": int(len(y)),
            "positive_rate": float(np.mean(y)),
        },
        "scenarios": SCENARIOS,
        "best_by_net_value": best,
        "notes": [
            "Net value = true_positives * effectiveness * avoided_repeat_cost - flagged_n * incremental_fast_repair_cost.",
            "This estimates value from avoided 14-day nearby recurrence only; it does not include wider safety, political, journey-delay, or long-run asset-condition benefits unless included in scenario avoided_repeat_cost.",
            "The medium_patch_full_cost_not_incremental scenario is a stress test, not the recommended interpretation: in practice only the premium above the default repair should be treated as incremental.",
        ],
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_csv}")
    print("\nBest thresholds by scenario:")
    for scenario, row in best.items():
        print(
            f"  {scenario}: threshold={row['threshold']:.3f}, "
            f"net=£{row['net_value']:,.0f}, flagged={row['flagged_rate']:.1%}, "
            f"precision={row['precision']:.3f}, recall={row['recall']:.3f}"
        )


if __name__ == "__main__":
    main()
