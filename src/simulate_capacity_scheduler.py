#!/usr/bin/env python3
"""
Capacity-constrained repair scheduling simulation.

This adds the operational layer that threshold simulations miss:

  - repairs are made on a schedule: daily, weekly, fortnightly, monthly
  - crews can only do a limited number of enhanced interventions
  - candidates can be prioritised by model risk or FIFO
  - optional pre-winter section treatment reduces the event stream first

The result is an expected cost-saving estimate under realistic frequency and
capacity constraints.
"""
import argparse
import csv
import json
import math
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
        "name": "authority_low_cost_spot",
        "event_cost": 155.30,
        "spot_premium": 30.86,
        "spot_effectiveness": 0.60,
        "spot_horizon_days": 14,
        "section_effectiveness": 0.60,
        "section_fixed_cost": 250.00,
        "section_per_recent_cell_cost": 60.00,
        "section_min_cost": 350.00,
    },
    {
        "name": "authority_good_ops",
        "event_cost": 155.30,
        "spot_premium": 75.00,
        "spot_effectiveness": 0.60,
        "spot_horizon_days": 14,
        "section_effectiveness": 0.60,
        "section_fixed_cost": 250.00,
        "section_per_recent_cell_cost": 60.00,
        "section_min_cost": 350.00,
    },
    {
        "name": "moderate_asset",
        "event_cost": 272.80,
        "spot_premium": 125.00,
        "spot_effectiveness": 0.60,
        "spot_horizon_days": 30,
        "section_effectiveness": 0.65,
        "section_fixed_cost": 1000.00,
        "section_per_recent_cell_cost": 150.00,
        "section_min_cost": 1500.00,
    },
    {
        "name": "strong_asset",
        "event_cost": 579.80,
        "spot_premium": 300.00,
        "spot_effectiveness": 0.60,
        "spot_horizon_days": 90,
        "section_effectiveness": 0.75,
        "section_fixed_cost": 2500.00,
        "section_per_recent_cell_cost": 300.00,
        "section_min_cost": 4000.00,
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--defect_model", default="models/surface_recurrence_best.joblib")
    parser.add_argument("--section_forecast", default="models/segment_winter_forecast.csv")
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--out_json", default="models/capacity_scheduler_fast_results.json")
    parser.add_argument("--out_csv", default="models/capacity_scheduler_fast_results.csv")
    parser.add_argument("--label_radius_m", type=float, default=200.0)
    parser.add_argument("--grid_res_m", type=int, default=200)
    parser.add_argument("--train_end", default="2017-03-31")
    parser.add_argument("--val_end", default="2017-09-30")
    parser.add_argument("--fast", action="store_true", help="Run a focused operational grid.")
    return parser.parse_args()


def treatment_cost(row, scenario):
    return max(
        scenario["section_min_cost"],
        scenario["section_fixed_cost"]
        + scenario["section_per_recent_cell_cost"] * max(float(row["recent_active_cells_180d"]), 1.0),
    )


def selected_sections(section_df, scenario):
    df = section_df.copy()
    df["treatment_cost"] = df.apply(lambda r: treatment_cost(r, scenario), axis=1)
    benefit = df["predicted_future_count"] * scenario["section_effectiveness"] * scenario["event_cost"]
    df["selected"] = benefit > df["treatment_cost"]
    return df[df["selected"]].copy()


def service_dates(times, frequency_days):
    start = pd.Timestamp(times.min()).floor("D")
    end = pd.Timestamp(times.max()).ceil("D")
    dates = list(pd.date_range(start=start, end=end, freq=f"{frequency_days}D"))
    if dates[-1] < end:
        dates.append(end)
    return np.asarray(dates, dtype="datetime64[ns]").astype("int64")


def simulate_scheduler(
    times_ns,
    coords,
    section_codes,
    proba,
    scenario,
    policy,
    frequency_days,
    capacity_rate,
    spot_radius_m,
    use_sections,
    section_df,
):
    n = len(times_ns)
    baseline_cost = n * scenario["event_cost"]
    occurrence_prob = np.ones(n, dtype=float)
    policy_cost = 0.0

    treated_section_codes = set()
    if use_sections:
        selected = selected_sections(section_df, scenario)
        treated_section_codes = set(selected["section_code"].astype(str))
        in_section = np.isin(section_codes, list(treated_section_codes))
        occurrence_prob[in_section] *= 1.0 - scenario["section_effectiveness"]
        policy_cost += float(selected["treatment_cost"].sum())
    else:
        selected = pd.DataFrame()
        in_section = np.zeros(n, dtype=bool)

    tree = cKDTree(coords)
    dates = service_dates(pd.to_datetime(times_ns), frequency_days)
    horizon_ns = int(scenario["spot_horizon_days"] * 24 * 60 * 60 * 1_000_000_000)
    total_capacity = int(round(capacity_rate * n))
    per_service_capacity = max(1, int(math.ceil(total_capacity / max(len(dates), 1))))

    serviced = np.zeros(n, dtype=bool)
    pointer = 0
    pool = []
    expected_spot_interventions = 0.0
    expected_prevented_by_spot = 0.0
    used_capacity = 0

    order = np.argsort(times_ns)
    for service_time in dates:
        while pointer < n and times_ns[order[pointer]] <= service_time:
            idx = order[pointer]
            if not in_section[idx]:
                pool.append(idx)
            pointer += 1

        candidates = [
            i for i in pool
            if (not serviced[i]) and occurrence_prob[i] > 1e-6 and times_ns[i] <= service_time
        ]
        if not candidates:
            continue

        if policy == "risk":
            candidates.sort(key=lambda i: proba[i], reverse=True)
        elif policy == "fifo":
            candidates.sort(key=lambda i: times_ns[i])
        elif policy == "risk_per_cost":
            candidates.sort(key=lambda i: proba[i] * occurrence_prob[i], reverse=True)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        slots = min(per_service_capacity, total_capacity - used_capacity, len(candidates))
        if slots <= 0:
            break

        for i in candidates[:slots]:
            if serviced[i] or occurrence_prob[i] <= 1e-6:
                continue
            serviced[i] = True
            used_capacity += 1
            p_occurs = occurrence_prob[i]
            expected_spot_interventions += p_occurs
            policy_cost += p_occurs * scenario["spot_premium"]

            candidates_near = np.asarray(tree.query_ball_point(coords[i], spot_radius_m), dtype=np.int64)
            if len(candidates_near) == 0:
                continue
            dt = times_ns[candidates_near] - service_time
            future = candidates_near[(dt > 0) & (dt <= horizon_ns)]
            suppression = p_occurs * scenario["spot_effectiveness"]
            for j in future:
                if in_section[j]:
                    continue
                old = occurrence_prob[j]
                new = old * (1.0 - suppression)
                occurrence_prob[j] = new
                expected_prevented_by_spot += old - new

    policy_cost += float(np.sum(occurrence_prob) * scenario["event_cost"])
    expected_events = float(np.sum(occurrence_prob))
    return {
        "scenario": scenario["name"],
        "policy": policy,
        "frequency_days": int(frequency_days),
        "capacity_rate": float(capacity_rate),
        "spot_radius_m": float(spot_radius_m),
        "use_sections": bool(use_sections),
        "selected_sections": int(len(selected)),
        "section_treatment_cost": float(selected["treatment_cost"].sum()) if len(selected) else 0.0,
        "baseline_cost": baseline_cost,
        "policy_cost": policy_cost,
        "cost_saving": baseline_cost - policy_cost,
        "cost_saving_pct": (baseline_cost - policy_cost) / baseline_cost,
        "expected_events": expected_events,
        "event_reduction_pct": (n - expected_events) / n,
        "expected_spot_interventions": expected_spot_interventions,
        "used_capacity": int(used_capacity),
        "expected_spot_intervention_rate": expected_spot_interventions / n,
        "expected_prevented_by_spot": expected_prevented_by_spot,
    }


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    artifact = joblib.load(args.defect_model)
    pipe = artifact["pipeline"]
    features = artifact["features"]

    print("Rebuilding holdout features...")
    df = load_surface_potholes(args.inputs)
    df = add_time_and_grid_features(df, args.grid_res_m)
    df = add_spatial_temporal_features_and_label(df, args.label_radius_m, 14)
    df = add_section_lag_features(df)
    _, _, test_m, censor_cutoff = temporal_masks(df, args.train_end, args.val_end, 14)
    holdout = df.loc[test_m].copy().reset_index(drop=True)
    holdout["section_code"] = holdout["section_name"].astype(str).str.split().str[0]
    proba = pipe.predict_proba(holdout[features])[:, 1]
    times_ns = holdout["date_recorded"].values.astype("datetime64[ns]").astype("int64")
    coords = holdout[["easting", "northing"]].values.astype(float)
    section_codes = holdout["section_code"].values.astype(str)
    section_df = pd.read_csv(args.section_forecast)
    section_df["section_code"] = section_df["section_code"].astype(str)

    rows = []
    if args.fast:
        radii = [100.0, 200.0]
        frequencies = [1, 7]
        capacities = [0.00, 0.05, 0.10]
        policies = ["risk"]
    else:
        radii = [100.0, 200.0]
        frequencies = [1, 7, 14, 28]
        capacities = [0.00, 0.05, 0.10, 0.20, 0.35]
        policies = ["fifo", "risk"]

    for scenario in SCENARIOS:
        for spot_radius_m in radii:
            for frequency_days in frequencies:
                for capacity_rate in capacities:
                    for policy in policies:
                        rows.append(
                            simulate_scheduler(
                                times_ns,
                                coords,
                                section_codes,
                                proba,
                                scenario,
                                policy,
                                frequency_days,
                                capacity_rate,
                                spot_radius_m,
                                False,
                                section_df,
                            )
                        )
                    rows.append(
                        simulate_scheduler(
                            times_ns,
                            coords,
                            section_codes,
                            proba,
                            scenario,
                            "risk",
                            frequency_days,
                            capacity_rate,
                            spot_radius_m,
                            True,
                            section_df,
                        )
                    )

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_by_scenario = {}
    for row in rows:
        key = (row["scenario"], row["spot_radius_m"])
        if key not in best_by_scenario or row["cost_saving_pct"] > best_by_scenario[key]["cost_saving_pct"]:
            best_by_scenario[key] = row

    summary = {
        "holdout": {
            "date_rule": f"> {args.val_end} and <= {censor_cutoff}",
            "n_events": int(len(holdout)),
        },
        "definition": {
            "baseline": "Every observed holdout defect occurs and costs event_cost.",
            "scheduler": "Candidates are repaired only on service days, with limited enhanced-intervention capacity.",
            "policies": {
                "fifo": "oldest waiting defect first",
                "risk": "highest model recurrence-risk score first",
                "use_sections": "pre-winter section treatment first, then risk scheduling for remaining events",
            },
        },
        "best_by_scenario_and_radius": {f"{k[0]}_{int(k[1])}m": v for k, v in best_by_scenario.items()},
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_csv}")
    print("\nBest capacity-constrained policies:")
    for key, row in best_by_scenario.items():
        print(
            f"  {key[0]} radius={int(key[1])}m: saving={row['cost_saving_pct']:.1%}, "
            f"events_reduced={row['event_reduction_pct']:.1%}, "
            f"policy={row['policy']}, sections={row['use_sections']}, "
            f"freq={row['frequency_days']}d, capacity={row['capacity_rate']:.0%}"
        )


if __name__ == "__main__":
    main()
