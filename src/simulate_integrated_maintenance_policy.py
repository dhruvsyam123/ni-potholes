#!/usr/bin/env python3
"""
Integrated maintenance policy simulation.

Architecture:
  1. Before winter, use the section-burden model to select sections for planned
     bundled/heavy treatment.
  2. During winter, use the defect-level recurrence model to triage remaining
     observed defects for faster/better spot intervention.

This is closer to a real maintenance strategy than either model alone.
It estimates expected percentage saving versus a reactive baseline while avoiding
double counting: planned section treatment reduces the probability of future
events first; spot interventions then operate on the remaining expected events.
"""
import argparse
import csv
import json
import os

import joblib
import numpy as np
import pandas as pd
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
        "name": "integrated_authority_conservative",
        "event_cost": 155.30,
        "section_effectiveness": 0.45,
        "section_fixed_cost": 250.00,
        "section_per_recent_cell_cost": 60.00,
        "section_min_cost": 350.00,
        "spot_premium": 75.00,
        "spot_effectiveness": 0.50,
        "spot_horizon_days": 14,
        "description": "Conservative authority-cost case with modest section and spot effects.",
    },
    {
        "name": "integrated_authority_good_ops",
        "event_cost": 155.30,
        "section_effectiveness": 0.60,
        "section_fixed_cost": 250.00,
        "section_per_recent_cell_cost": 60.00,
        "section_min_cost": 350.00,
        "spot_premium": 75.00,
        "spot_effectiveness": 0.60,
        "spot_horizon_days": 14,
        "description": "Good operational execution: bundled section patching plus spot triage.",
    },
    {
        "name": "integrated_moderate_asset",
        "event_cost": 272.80,
        "section_effectiveness": 0.65,
        "section_fixed_cost": 1000.00,
        "section_per_recent_cell_cost": 150.00,
        "section_min_cost": 1500.00,
        "spot_premium": 125.00,
        "spot_effectiveness": 0.60,
        "spot_horizon_days": 30,
        "description": "Moderate asset case with heavier section patching and 30-day spot suppression.",
    },
    {
        "name": "integrated_strong_asset",
        "event_cost": 579.80,
        "section_effectiveness": 0.75,
        "section_fixed_cost": 2500.00,
        "section_per_recent_cell_cost": 300.00,
        "section_min_cost": 4000.00,
        "spot_premium": 300.00,
        "spot_effectiveness": 0.60,
        "spot_horizon_days": 90,
        "description": "Stronger asset-effect scenario with low-cost resurfacing-style section treatment.",
    },
    {
        "name": "integrated_full_resurfacing_stress",
        "event_cost": 579.80,
        "section_effectiveness": 0.85,
        "section_cost_per_km": 150000.00,
        "section_min_length_km": 0.20,
        "section_max_length_km": 2.00,
        "spot_premium": 300.00,
        "spot_effectiveness": 0.60,
        "spot_horizon_days": 90,
        "description": "Stress test: full resurfacing costs, plus spot triage for non-treated sections.",
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--defect_model", default="models/surface_recurrence_best.joblib")
    parser.add_argument("--section_forecast", default="models/segment_winter_forecast.csv")
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--out_json", default="models/integrated_maintenance_policy_results.json")
    parser.add_argument("--out_csv", default="models/integrated_maintenance_policy_thresholds.csv")
    parser.add_argument("--radius_m", type=float, default=200.0)
    parser.add_argument("--horizon_days", type=int, default=14)
    parser.add_argument("--grid_res_m", type=int, default=200)
    parser.add_argument("--train_end", default="2017-03-31")
    parser.add_argument("--val_end", default="2017-09-30")
    parser.add_argument("--threshold_min", type=float, default=0.20)
    parser.add_argument("--threshold_max", type=float, default=0.90)
    parser.add_argument("--threshold_step", type=float, default=0.025)
    return parser.parse_args()


def section_treatment_cost(row, scenario):
    if "section_cost_per_km" in scenario:
        length = min(
            max(float(row["segment_length_km"]), scenario["section_min_length_km"]),
            scenario["section_max_length_km"],
        )
        return scenario["section_cost_per_km"] * length
    return max(
        scenario["section_min_cost"],
        scenario["section_fixed_cost"]
        + scenario["section_per_recent_cell_cost"] * max(float(row["recent_active_cells_180d"]), 1.0),
    )


def selected_sections(section_df, scenario):
    df = section_df.copy()
    df["treatment_cost"] = df.apply(lambda r: section_treatment_cost(r, scenario), axis=1)
    expected_benefit = (
        df["predicted_future_count"] * scenario["section_effectiveness"] * scenario["event_cost"]
    )
    df["selected"] = expected_benefit > df["treatment_cost"]
    return df[df["selected"]].copy()


def future_neighbor_lists(coords, times_ns, radius_m, horizon_days):
    tree = cKDTree(coords)
    horizon_ns = int(horizon_days * 24 * 60 * 60 * 1_000_000_000)
    lists = []
    for i in range(len(coords)):
        candidates = np.asarray(tree.query_ball_point(coords[i], radius_m), dtype=np.int64)
        candidates = candidates[candidates > i]
        if len(candidates) == 0:
            lists.append(candidates)
            continue
        dt = times_ns[candidates] - times_ns[i]
        lists.append(candidates[(dt >= 0) & (dt <= horizon_ns)])
    return lists


def simulate(proba, section_codes, neighbor_lists, selected_section_codes, section_cost, threshold, scenario):
    event_cost = scenario["event_cost"]
    baseline_cost = len(proba) * event_cost
    occurrence_prob = np.ones(len(proba), dtype=float)

    in_selected_section = np.isin(section_codes, list(selected_section_codes))
    occurrence_prob[in_selected_section] *= 1.0 - scenario["section_effectiveness"]

    policy_cost = float(section_cost)
    expected_spot_interventions = 0.0
    expected_prevented_by_spot = 0.0
    flagged = proba >= threshold

    for i in range(len(proba)):
        p_occurs = occurrence_prob[i]
        if p_occurs <= 1e-12:
            continue
        policy_cost += p_occurs * event_cost

        # If a section has already been treated, do not also pay for individual
        # enhanced spot treatment on the same expected event.
        if in_selected_section[i] or not flagged[i]:
            continue

        expected_spot_interventions += p_occurs
        policy_cost += p_occurs * scenario["spot_premium"]
        suppression = p_occurs * scenario["spot_effectiveness"]
        for j in neighbor_lists[i]:
            if in_selected_section[j]:
                continue
            old = occurrence_prob[j]
            new = old * (1.0 - suppression)
            occurrence_prob[j] = new
            expected_prevented_by_spot += old - new

    expected_events = float(np.sum(occurrence_prob))
    expected_prevented_total = len(proba) - expected_events
    return {
        "threshold": round(float(threshold), 3),
        "baseline_cost": baseline_cost,
        "policy_cost": policy_cost,
        "cost_saving": baseline_cost - policy_cost,
        "cost_saving_pct": (baseline_cost - policy_cost) / baseline_cost,
        "expected_events": expected_events,
        "expected_prevented_events_total": expected_prevented_total,
        "event_reduction_pct": expected_prevented_total / len(proba),
        "expected_spot_interventions": expected_spot_interventions,
        "spot_intervention_rate": expected_spot_interventions / len(proba),
        "expected_prevented_by_spot": expected_prevented_by_spot,
        "events_initially_in_selected_sections": int(np.sum(in_selected_section)),
        "selected_section_event_share": float(np.mean(in_selected_section)),
        "nominal_flagged_rate_before_prevention": float(np.mean(flagged)),
    }


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    artifact = joblib.load(args.defect_model)
    pipe = artifact["pipeline"]
    features = artifact["features"]
    target = artifact["target"]

    print("Rebuilding holdout features from raw surface-defect data...")
    df = load_surface_potholes(args.inputs)
    df = add_time_and_grid_features(df, args.grid_res_m)
    df = add_spatial_temporal_features_and_label(df, args.radius_m, args.horizon_days)
    df = add_section_lag_features(df)
    _, _, test_m, censor_cutoff = temporal_masks(df, args.train_end, args.val_end, args.horizon_days)
    holdout = df.loc[test_m].copy().reset_index(drop=True)
    holdout["section_code"] = holdout["section_name"].astype(str).str.split().str[0]
    proba = pipe.predict_proba(holdout[features])[:, 1]

    section_df = pd.read_csv(args.section_forecast)
    section_df["section_code"] = section_df["section_code"].astype(str)

    coords = holdout[["easting", "northing"]].values.astype(float)
    times_ns = holdout["date_recorded"].values.astype("datetime64[ns]").astype("int64")

    unique_horizons = sorted({s["spot_horizon_days"] for s in SCENARIOS})
    neighbors_by_horizon = {}
    for horizon in unique_horizons:
        print(f"Building future-neighbor lists for {horizon}d spot suppression...")
        neighbors_by_horizon[horizon] = future_neighbor_lists(coords, times_ns, args.radius_m, horizon)

    thresholds = np.round(
        np.arange(
            args.threshold_min,
            args.threshold_max + args.threshold_step / 2,
            args.threshold_step,
        ),
        3,
    )
    thresholds = np.append(thresholds, 1.001)  # explicit section-only / no spot-triage option

    rows = []
    best = {}
    for scenario in SCENARIOS:
        selected = selected_sections(section_df, scenario)
        selected_codes = set(selected["section_code"])
        section_cost = float(selected["treatment_cost"].sum())
        neighbor_lists = neighbors_by_horizon[scenario["spot_horizon_days"]]
        for threshold in thresholds:
            row = simulate(
                proba,
                holdout["section_code"].values.astype(str),
                neighbor_lists,
                selected_codes,
                section_cost,
                threshold,
                scenario,
            )
            row = {
                "scenario": scenario["name"],
                "event_cost": scenario["event_cost"],
                "section_effectiveness": scenario["section_effectiveness"],
                "selected_sections": int(len(selected)),
                "section_treatment_cost": section_cost,
                "section_predicted_burden": float(selected["predicted_future_count"].sum()),
                "section_actual_holdout_events": int(
                    holdout["section_code"].isin(selected_codes).sum()
                ),
                "spot_premium": scenario["spot_premium"],
                "spot_effectiveness": scenario["spot_effectiveness"],
                "spot_horizon_days": scenario["spot_horizon_days"],
                **row,
            }
            rows.append(row)
            if scenario["name"] not in best or row["cost_saving_pct"] > best[scenario["name"]]["cost_saving_pct"]:
                best[scenario["name"]] = row

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "defect_model": args.defect_model,
        "section_forecast": args.section_forecast,
        "target": target,
        "holdout": {
            "date_rule": f"> {args.val_end} and <= {censor_cutoff}",
            "n_events": int(len(holdout)),
            "observed_recurrence_positive_rate": float(holdout[target].mean()),
        },
        "simulation_definition": {
            "baseline": "Every observed holdout defect occurs and costs event_cost.",
            "policy": (
                "Pre-winter section treatments are selected from section predicted burden. "
                "They reduce expected events on selected sections. Then defect-level spot "
                "triage is applied chronologically to remaining expected events."
            ),
            "caveat": "Treatment effects are scenario assumptions, not learned causal effects.",
        },
        "scenarios": SCENARIOS,
        "best_by_cost_saving_pct": best,
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_csv}")
    print("\nBest integrated savings:")
    for scenario, row in best.items():
        print(
            f"  {scenario}: threshold={row['threshold']:.3f}, "
            f"saving={row['cost_saving_pct']:.1%}, "
            f"events_reduced={row['event_reduction_pct']:.1%}, "
            f"sections={row['selected_sections']}, "
            f"spot_interventions={row['spot_intervention_rate']:.1%}"
        )


if __name__ == "__main__":
    main()
