#!/usr/bin/env python3
"""
Counterfactual repair-policy simulation for the surface-defect model.

The classifier says which current defects are likely to be followed by nearby
future defects. This script asks a different question:

  If high-risk defects receive a faster/better intervention, what percentage
  cost saving is plausible?

The open data does not contain actual repair dates, repair types, unit costs,
or randomized interventions. Therefore this is not causal proof. It is an
auditable simulation driven by:

  - model scores on a true future holdout
  - observed future defect locations/times
  - explicit cost and intervention-effect assumptions

Baseline:
  every observed defect occurs and costs `event_cost`.

Policy:
  an occurring defect still costs `event_cost`; if its model score is above the
  threshold, it also costs `intervention_premium` and suppresses nearby future
  defects with probability `effectiveness` within `suppression_horizon_days`.
"""
import argparse
import csv
import json
import os

import joblib
import numpy as np
from scipy.spatial import cKDTree

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
        "name": "repair_only_14d_conservative",
        "event_cost": 87.80,
        "intervention_premium": 30.86,
        "effectiveness": 0.50,
        "suppression_horizon_days": 14,
        "description": (
            "Authority repair budget only. Each prevented future defect avoids one average "
            "reactive fill. Intervention premium is the ALARM reactive-minus-planned cost delta."
        ),
    },
    {
        "name": "repair_only_14d_good_fix",
        "event_cost": 87.80,
        "intervention_premium": 30.86,
        "effectiveness": 0.75,
        "suppression_horizon_days": 14,
        "description": "Same costs as above, but assumes good repair quality prevents 75% of short-window repeats.",
    },
    {
        "name": "authority_plus_admin_14d",
        "event_cost": 155.30,
        "intervention_premium": 75.00,
        "effectiveness": 0.60,
        "suppression_horizon_days": 14,
        "description": (
            "Authority-centric case including reactive repair plus modest inspection/admin/claim handling burden."
        ),
    },
    {
        "name": "moderate_asset_30d",
        "event_cost": 272.80,
        "intervention_premium": 125.00,
        "effectiveness": 0.60,
        "suppression_horizon_days": 30,
        "description": (
            "Moderate asset-management case: prevents nearby repeats over 30 days and includes "
            "some deterioration/road-user externality in avoided event cost."
        ),
    },
    {
        "name": "strong_asset_90d",
        "event_cost": 579.80,
        "intervention_premium": 300.00,
        "effectiveness": 0.60,
        "suppression_horizon_days": 90,
        "description": (
            "Stronger asset-effect case: expensive intervention but assumed to suppress local "
            "defect recurrence for a quarter."
        ),
    },
    {
        "name": "medium_patch_premium_90d_stress",
        "event_cost": 579.80,
        "intervention_premium": 1240.20,
        "effectiveness": 0.60,
        "suppression_horizon_days": 90,
        "description": (
            "Stress test: medium patch premium approximated as £1,328 minus one average reactive fill. "
            "Only viable if broader benefits are large."
        ),
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/surface_recurrence_best.joblib")
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--out_json", default="models/surface_repair_policy_simulation.json")
    parser.add_argument("--out_csv", default="models/surface_repair_policy_simulation.csv")
    parser.add_argument("--radius_m", type=float, default=200.0)
    parser.add_argument("--horizon_days", type=int, default=14)
    parser.add_argument("--grid_res_m", type=int, default=200)
    parser.add_argument("--train_end", default="2017-03-31")
    parser.add_argument("--val_end", default="2017-09-30")
    parser.add_argument("--threshold_min", type=float, default=0.20)
    parser.add_argument("--threshold_max", type=float, default=0.90)
    parser.add_argument("--threshold_step", type=float, default=0.025)
    return parser.parse_args()


def future_neighbor_lists(coords, times_ns, radius_m, horizon_days):
    tree = cKDTree(coords)
    horizon_ns = int(horizon_days * 24 * 60 * 60 * 1_000_000_000)
    lists = []
    for i in range(len(coords)):
        candidates = tree.query_ball_point(coords[i], radius_m)
        candidates = np.asarray(candidates, dtype=np.int64)
        candidates = candidates[candidates > i]
        if len(candidates) == 0:
            lists.append(candidates)
            continue
        dt = times_ns[candidates] - times_ns[i]
        lists.append(candidates[(dt >= 0) & (dt <= horizon_ns)])
    return lists


def simulate_threshold(proba, neighbor_lists, threshold, scenario):
    event_cost = scenario["event_cost"]
    premium = scenario["intervention_premium"]
    effectiveness = scenario["effectiveness"]
    flagged = proba >= threshold

    occurrence_prob = np.ones(len(proba), dtype=float)
    policy_cost = 0.0
    expected_interventions = 0.0
    expected_prevented = 0.0

    for i in range(len(proba)):
        p_occurs = occurrence_prob[i]
        if p_occurs <= 1e-12:
            continue

        policy_cost += p_occurs * event_cost

        if not flagged[i]:
            continue

        expected_interventions += p_occurs
        policy_cost += p_occurs * premium

        # If the current event occurs, the intervention exists. Expected future
        # suppression probability is p_occurs * effectiveness.
        suppression = p_occurs * effectiveness
        if suppression <= 0:
            continue

        for j in neighbor_lists[i]:
            old = occurrence_prob[j]
            new = old * (1.0 - suppression)
            occurrence_prob[j] = new
            expected_prevented += old - new

    baseline_cost = len(proba) * event_cost
    expected_events = float(np.sum(occurrence_prob))
    return {
        "threshold": round(float(threshold), 3),
        "baseline_cost": baseline_cost,
        "policy_cost": policy_cost,
        "cost_saving": baseline_cost - policy_cost,
        "cost_saving_pct": (baseline_cost - policy_cost) / baseline_cost,
        "expected_events": expected_events,
        "expected_prevented_events": expected_prevented,
        "event_reduction_pct": (len(proba) - expected_events) / len(proba),
        "expected_interventions": expected_interventions,
        "intervention_rate": expected_interventions / len(proba),
        "nominal_flagged_rate_before_suppression": float(np.mean(flagged)),
    }


def best_by_scenario(rows):
    best = {}
    for row in rows:
        current = best.get(row["scenario"])
        if current is None or row["cost_saving_pct"] > current["cost_saving_pct"]:
            best[row["scenario"]] = row
    return best


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
    holdout = df.loc[test_m].copy().reset_index(drop=True)
    proba = pipe.predict_proba(holdout[features])[:, 1]

    coords = holdout[["easting", "northing"]].values.astype(float)
    times_ns = holdout["date_recorded"].values.astype("datetime64[ns]").astype("int64")

    thresholds = np.round(
        np.arange(
            args.threshold_min,
            args.threshold_max + args.threshold_step / 2,
            args.threshold_step,
        ),
        3,
    )

    unique_horizons = sorted({s["suppression_horizon_days"] for s in SCENARIOS})
    neighbors_by_horizon = {}
    for horizon in unique_horizons:
        print(f"Building future-neighbor lists for {horizon}d suppression...")
        neighbors_by_horizon[horizon] = future_neighbor_lists(
            coords, times_ns, args.radius_m, horizon
        )

    rows = []
    for scenario in SCENARIOS:
        neighbor_lists = neighbors_by_horizon[scenario["suppression_horizon_days"]]
        for threshold in thresholds:
            row = simulate_threshold(proba, neighbor_lists, threshold, scenario)
            row = {
                "scenario": scenario["name"],
                "event_cost": scenario["event_cost"],
                "intervention_premium": scenario["intervention_premium"],
                "effectiveness": scenario["effectiveness"],
                "suppression_horizon_days": scenario["suppression_horizon_days"],
                **row,
            }
            rows.append(row)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = best_by_scenario(rows)
    summary = {
        "model": args.model,
        "target": target,
        "holdout": {
            "date_rule": f"> {args.val_end} and <= {censor_cutoff}",
            "n": int(len(holdout)),
            "observed_target_positive_rate": float(holdout[target].mean()),
        },
        "simulation_definition": {
            "baseline": "Every observed holdout defect occurs and costs event_cost.",
            "policy": (
                "If an occurring defect has model risk >= threshold, pay intervention_premium "
                "and probabilistically suppress future defects within radius and suppression horizon."
            ),
            "radius_m": args.radius_m,
            "caveat": (
                "This is a counterfactual simulation, not causal proof, because the open data "
                "does not contain actual intervention timings, repair types, or repair outcomes."
            ),
        },
        "scenarios": SCENARIOS,
        "best_by_cost_saving_pct": best,
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_csv}")
    print("\nBest simulated percentage savings:")
    for scenario, row in best.items():
        print(
            f"  {scenario}: threshold={row['threshold']:.3f}, "
            f"saving={row['cost_saving_pct']:.1%}, "
            f"event_reduction={row['event_reduction_pct']:.1%}, "
            f"interventions={row['intervention_rate']:.1%}"
        )


if __name__ == "__main__":
    main()
