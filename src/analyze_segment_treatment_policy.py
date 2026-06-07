#!/usr/bin/env python3
"""
Section-level forecast and treatment policy.

This handles the higher-level decision:

  repeated spot patching vs treating a whole road section / geographic segment.

The open data has no true road geometry or resurfacing invoices, so SECTION_NAME
is used as the segment proxy. The model forecasts future pothole-defect burden
per section, then policy scenarios decide whether the predicted burden is high
enough to justify bundled patching or resurfacing-style treatment.
"""
import argparse
import json
import os
import re
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import pyproj
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DAY_NS = 24 * 60 * 60 * 1_000_000_000


SCENARIOS = [
    {
        "name": "bundled_permanent_patching",
        "event_cost": 155.30,
        "effectiveness": 0.60,
        "fixed_cost": 250.00,
        "per_recent_cell_cost": 60.00,
        "min_cost": 350.00,
        "description": (
            "Crew treats the active cluster/section in one visit with better materials/process. "
            "This is closer to bundled patching than resurfacing."
        ),
    },
    {
        "name": "heavy_section_patching",
        "event_cost": 272.80,
        "effectiveness": 0.70,
        "fixed_cost": 1000.00,
        "per_recent_cell_cost": 150.00,
        "min_cost": 1500.00,
        "description": (
            "More substantial section patching. Requires a higher predicted defect burden."
        ),
    },
    {
        "name": "micro_resurfacing_low_cost_proxy",
        "event_cost": 579.80,
        "effectiveness": 0.80,
        "fixed_cost": 2500.00,
        "per_recent_cell_cost": 300.00,
        "min_cost": 4000.00,
        "description": (
            "Low-cost resurfacing-style proxy for short segments/clusters. This is not full "
            "carriageway resurfacing; it tests when larger area treatment starts to make sense."
        ),
    },
    {
        "name": "full_resurfacing_stress_150k_per_km",
        "event_cost": 579.80,
        "effectiveness": 0.90,
        "cost_per_km": 150000.00,
        "min_length_km": 0.20,
        "max_length_km": 2.00,
        "description": (
            "Stress test using a high resurfacing cost. The section length is approximated from "
            "observed defect-coordinate spread, so this should be treated as directional only."
        ),
    },
]


def route_code(section_name):
    if not isinstance(section_name, str) or not section_name.strip():
        return "UNKNOWN"
    return section_name.strip().split()[0].upper()


def road_class(route):
    match = re.search(r"([ABCUM])\d", route)
    return match.group(1) if match else "UNKNOWN"


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
    parser.add_argument("--model_out", default="models/segment_burden_hgb.joblib")
    parser.add_argument("--results_out", default="models/segment_treatment_policy_results.json")
    parser.add_argument("--panel_out", default="models/segment_winter_forecast.csv")
    parser.add_argument("--road_geojson", default="data/raw/dfi_road_network.geojson")
    parser.add_argument("--highway_csv", default="data/raw/highway_network.csv")
    parser.add_argument("--horizon_days", type=int, default=180)
    parser.add_argument("--decision_date", default="2017-10-01")
    parser.add_argument("--train_end", default="2017-03-01")
    parser.add_argument("--val_end", default="2017-09-01")
    parser.add_argument("--grid_res_m", type=int, default=200)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def mode_or_unknown(values):
    clean = [str(v).strip() for v in values if str(v).strip()]
    if not clean:
        return "UNKNOWN"
    return Counter(clean).most_common(1)[0][0]


def load_potholes(paths, grid_res_m):
    frames = []
    for path in paths:
        df = pd.read_csv(path, low_memory=False)
        df["source_file"] = os.path.basename(path)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["date_recorded"] = pd.to_datetime(
        df["RECORDED_DATE"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    df = df.dropna(subset=["date_recorded", "EASTING", "NORTHING", "SECTION_NAME", "DEFECT_DETAIL"])
    df = df[df["DEFECT_DETAIL"].str.contains("POTHOLE", case=False, na=False)].copy()
    df = df[(df["EASTING"] > 100) & (df["NORTHING"] > 100)].copy()
    df["easting"] = df["EASTING"].astype(float)
    df["northing"] = df["NORTHING"].astype(float)
    transformer = pyproj.Transformer.from_crs("EPSG:29902", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(df["easting"].values, df["northing"].values)
    df["lon"] = lon
    df["lat"] = lat
    df["section_name"] = df["SECTION_NAME"].astype(str).str.strip()
    df["division"] = df["DIVISION"].astype(str).str.strip()
    df["section_office"] = df["SECTION_OFFICE"].astype(str).str.strip()
    df["route_code"] = df["section_name"].map(route_code)
    df["road_class"] = df["route_code"].map(road_class)
    df["surface_material"] = df["DEFECT_DETAIL"].astype(str).str.split().str[0].fillna("UNKNOWN")
    df["grid_x"] = (df["easting"] // grid_res_m).astype(int)
    df["grid_y"] = (df["northing"] // grid_res_m).astype(int)
    df["grid_cell"] = df["grid_x"].astype(str) + "_" + df["grid_y"].astype(str)
    return df.sort_values("date_recorded").reset_index(drop=True)


def load_road_network(geojson_path=None, csv_path=None):
    rows = []
    if geojson_path and os.path.exists(geojson_path):
        with open(geojson_path) as f:
            data = json.load(f)
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            rows.append(
                {
                    "section_code": props.get("Section_Code"),
                    "road_section_name": props.get("SECTION_NA"),
                    "road_division": props.get("DIVISION_N"),
                    "road_section_office": props.get("SECTION_OF"),
                    "road_class_name": props.get("CLASS_NAME"),
                    "adoption_status": props.get("ADOPTION_S"),
                    "section_type": props.get("SECTION_TY"),
                    "official_length_m": props.get("Shape__Length"),
                    "geometry_source": "DFI_Road_Network_FeatureServer",
                }
            )
    elif csv_path and os.path.exists(csv_path):
        h = pd.read_csv(csv_path, low_memory=False)
        h["section_code"] = h["SECTION_CODE"].astype(str)
        rows = [
            {
                "section_code": r["section_code"],
                "road_section_name": r.get("SECTION_NAME"),
                "road_division": r.get("DIVISION_NAME"),
                "road_section_office": r.get("SECTION_OFFICE_NAME"),
                "road_class_name": r.get("CLASS"),
                "adoption_status": r.get("ADOPTION_STATUS_NAME"),
                "section_type": r.get("SECTION_TYPE_NAME"),
                "official_length_m": r.get("DIGITAL_LENGTH"),
                "geometry_source": "HIGHWAY_NETWORK_CSV",
            }
            for _, r in h.iterrows()
        ]

    if not rows:
        return pd.DataFrame()

    road = pd.DataFrame(rows)
    road = road.dropna(subset=["section_code"]).copy()
    road["section_code"] = road["section_code"].astype(str).str.strip()
    road["official_length_m"] = pd.to_numeric(road["official_length_m"], errors="coerce")
    road = road.sort_values("official_length_m", ascending=False)
    road = road.drop_duplicates("section_code", keep="first")
    return road


def section_static(df, road_network):
    rows = []
    for section, g in df.groupby("section_name", sort=False):
        section_code = route_code(section)
        rows.append(
            {
                "section_name": section,
                "section_code": section_code,
                "division": mode_or_unknown(g["division"]),
                "section_office": mode_or_unknown(g["section_office"]),
                "route_code": mode_or_unknown(g["route_code"]),
                "road_class": mode_or_unknown(g["road_class"]),
                "surface_material": mode_or_unknown(g["surface_material"]),
                "section_total_observed": int(len(g)),
            }
        )
    static = pd.DataFrame(rows)
    if road_network is not None and len(road_network):
        static = static.merge(road_network, on="section_code", how="left")
    else:
        static["official_length_m"] = np.nan
        static["road_class_name"] = np.nan
        static["adoption_status"] = np.nan
        static["section_type"] = np.nan
        static["geometry_source"] = "none"
    static["has_official_geometry"] = static["official_length_m"].notna().astype(int)
    return static


def build_panel(df, horizon_days, road_network):
    start = df["date_recorded"].min().to_period("M").to_timestamp()
    end = (df["date_recorded"].max() - pd.Timedelta(days=horizon_days)).to_period("M").to_timestamp()
    anchors = pd.date_range(start=start + pd.offsets.MonthBegin(1), end=end, freq="MS")
    anchors_ns = anchors.values.astype("datetime64[ns]").astype("int64")
    horizon_ns = horizon_days * DAY_NS
    rows = []
    static = section_static(df, road_network).set_index("section_name")

    for section, g in df.groupby("section_name", sort=False):
        g = g.sort_values("date_recorded")
        t = g["date_recorded"].values.astype("datetime64[ns]").astype("int64")
        east = g["easting"].values
        north = g["northing"].values
        grid = g["grid_cell"].values
        s = static.loc[section]

        for anchor, anchor_ns in zip(anchors, anchors_ns):
            past_hi = np.searchsorted(t, anchor_ns, side="left")
            future_hi = np.searchsorted(t, anchor_ns + horizon_ns, side="left")
            future_count = future_hi - past_hi

            past_365_lo = np.searchsorted(t, anchor_ns - 365 * DAY_NS, side="left")
            past_365_count = past_hi - past_365_lo
            if past_365_count == 0 and future_count == 0:
                continue

            def past_count(days):
                lo = np.searchsorted(t, anchor_ns - days * DAY_NS, side="left")
                return past_hi - lo, lo

            past_30, past_30_lo = past_count(30)
            past_90, past_90_lo = past_count(90)
            past_180, past_180_lo = past_count(180)

            if past_365_count > 1:
                pe = east[past_365_lo:past_hi]
                pn = north[past_365_lo:past_hi]
                span_m = float(((pe.max() - pe.min()) ** 2 + (pn.max() - pn.min()) ** 2) ** 0.5)
            else:
                span_m = 0.0
            official_length_m = s.get("official_length_m")
            if pd.isna(official_length_m) or official_length_m <= 0:
                length_m = max(50.0, span_m)
                length_source = "defect_coordinate_span"
            else:
                length_m = float(official_length_m)
                length_source = "official_dfi_geometry"

            recent_cells = len(set(grid[past_180_lo:past_hi])) if past_180 > 0 else 0
            rows.append(
                {
                    "decision_date": anchor,
                    "section_name": section,
                    "section_code": s["section_code"],
                    "future_count": int(future_count),
                    "past_30d": int(past_30),
                    "past_90d": int(past_90),
                    "past_180d": int(past_180),
                    "past_365d": int(past_365_count),
                    "recent_active_cells_180d": int(recent_cells),
                    "defects_per_recent_cell_180d": float(past_180 / max(recent_cells, 1)),
                    "observed_span_m_365d": span_m,
                    "official_length_m": float(official_length_m) if not pd.isna(official_length_m) else np.nan,
                    "segment_length_km": length_m / 1000.0,
                    "length_source": length_source,
                    "has_official_geometry": int(s["has_official_geometry"]),
                    "month": str(anchor.month),
                    "quarter": str(anchor.quarter),
                    "days_since_start": float((anchor - start).days),
                    "division": s["division"],
                    "section_office": s["section_office"],
                    "route_code": s["route_code"],
                    "road_class": s["road_class"],
                    "road_class_name": s.get("road_class_name") if not pd.isna(s.get("road_class_name")) else s["road_class"],
                    "adoption_status": s.get("adoption_status") if not pd.isna(s.get("adoption_status")) else "UNKNOWN",
                    "section_type": s.get("section_type") if not pd.isna(s.get("section_type")) else "UNKNOWN",
                    "geometry_source": s.get("geometry_source") if not pd.isna(s.get("geometry_source")) else "none",
                    "surface_material": s["surface_material"],
                    "section_total_observed": int(s["section_total_observed"]),
                }
            )
    return pd.DataFrame(rows)


def add_treatment_cost(row, scenario):
    if "cost_per_km" in scenario:
        length = min(
            max(float(row["segment_length_km"]), scenario["min_length_km"]),
            scenario["max_length_km"],
        )
        return scenario["cost_per_km"] * length
    return max(
        scenario["min_cost"],
        scenario["fixed_cost"] + scenario["per_recent_cell_cost"] * max(row["recent_active_cells_180d"], 1),
    )


def evaluate_policy(decision_df, pred_col, scenario):
    df = decision_df.copy()
    df["treatment_cost"] = df.apply(lambda r: add_treatment_cost(r, scenario), axis=1)
    expected_benefit = df[pred_col] * scenario["effectiveness"] * scenario["event_cost"]
    df["selected"] = expected_benefit > df["treatment_cost"]
    selected = df[df["selected"]]
    total_future = float(df["future_count"].sum())
    baseline_cost = total_future * scenario["event_cost"]
    gross_avoided = float((selected["future_count"] * scenario["effectiveness"] * scenario["event_cost"]).sum())
    treatment_cost = float(selected["treatment_cost"].sum())
    saving = gross_avoided - treatment_cost
    selected_future = float(selected["future_count"].sum())
    return {
        "scenario": scenario["name"],
        "selected_sections": int(len(selected)),
        "eligible_sections": int(len(df)),
        "selected_rate": float(len(selected) / len(df)) if len(df) else 0.0,
        "actual_future_defects_total": total_future,
        "actual_future_defects_in_selected": selected_future,
        "future_defect_coverage": float(selected_future / total_future) if total_future else 0.0,
        "baseline_cost": baseline_cost,
        "treatment_cost": treatment_cost,
        "gross_avoided_cost": gross_avoided,
        "net_saving": saving,
        "net_saving_pct": float(saving / baseline_cost) if baseline_cost else 0.0,
        "event_reduction_pct": float((selected_future * scenario["effectiveness"]) / total_future)
        if total_future
        else 0.0,
        "mean_predicted_count_selected": float(selected[pred_col].mean()) if len(selected) else 0.0,
        "mean_actual_count_selected": float(selected["future_count"].mean()) if len(selected) else 0.0,
    }


def oracle_policy(decision_df, scenario):
    df = decision_df.copy()
    df["treatment_cost"] = df.apply(lambda r: add_treatment_cost(r, scenario), axis=1)
    expected_benefit = df["future_count"] * scenario["effectiveness"] * scenario["event_cost"]
    df["oracle_score"] = df["future_count"]
    df["selected"] = expected_benefit > df["treatment_cost"]
    return evaluate_policy(df.rename(columns={"oracle_score": "oracle_pred"}), "oracle_pred", scenario)


def regression_metrics(y, pred):
    pred = np.clip(pred, 0, None)
    return {
        "n": int(len(y)),
        "mean_actual": float(np.mean(y)),
        "mean_predicted": float(np.mean(pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "poisson_deviance": float(mean_poisson_deviance(y, np.clip(pred, 1e-6, None))),
    }


def top_capture(y, pred, fractions=(0.01, 0.02, 0.05, 0.10)):
    order = np.argsort(-pred)
    total = y.sum()
    rows = {}
    for frac in fractions:
        k = max(1, int(len(y) * frac))
        rows[f"top_{int(frac * 100)}pct_capture"] = float(y[order[:k]].sum() / total) if total else 0.0
    return rows


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.results_out), exist_ok=True)

    print("Loading official pothole defects...")
    df = load_potholes(args.inputs, args.grid_res_m)
    print(f"  rows={len(df):,}, sections={df['section_name'].nunique():,}")
    road_network = load_road_network(args.road_geojson, args.highway_csv)
    if len(road_network):
        match_rate = df["route_code"].isin(set(road_network["section_code"])).mean()
        print(f"  road-network features={len(road_network):,}, defect-row join rate={match_rate:.1%}")
    else:
        print("  no road-network geometry loaded; falling back to defect-coordinate span")

    print("Building section-month panel...")
    panel = build_panel(df, args.horizon_days, road_network)
    panel = panel.sort_values(["decision_date", "section_name"]).reset_index(drop=True)
    print(f"  panel rows={len(panel):,}")

    cat_cols = [
        "division",
        "section_office",
        "road_class",
        "road_class_name",
        "adoption_status",
        "section_type",
        "surface_material",
        "month",
        "quarter",
    ]
    num_cols = [
        "past_30d",
        "past_90d",
        "past_180d",
        "past_365d",
        "recent_active_cells_180d",
        "defects_per_recent_cell_180d",
        "observed_span_m_365d",
        "official_length_m",
        "segment_length_km",
        "has_official_geometry",
        "days_since_start",
        "section_total_observed",
    ]
    for col in cat_cols:
        panel[col] = panel[col].astype(str)

    train_m = panel["decision_date"] <= pd.Timestamp(args.train_end)
    val_m = (panel["decision_date"] > pd.Timestamp(args.train_end)) & (
        panel["decision_date"] <= pd.Timestamp(args.val_end)
    )
    decision_date = pd.Timestamp(args.decision_date)
    decision_m = panel["decision_date"] == decision_date

    X = panel[cat_cols + num_cols]
    y = panel["future_count"].astype(float).values

    hgb = Pipeline(
        [
            (
                "preproc",
                ColumnTransformer(
                    [
                        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                        ("num", StandardScaler(), num_cols),
                    ],
                    verbose_feature_names_out=False,
                ),
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="poisson",
                    max_iter=300,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    l2_regularization=0.05,
                    min_samples_leaf=40,
                    random_state=args.random_state,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
            ),
        ]
    )
    dummy = DummyRegressor(strategy="mean")

    print("Training section burden model...")
    hgb.fit(X.loc[train_m], y[train_m.values])
    dummy.fit(X.loc[train_m], y[train_m.values])

    metrics = {}
    for name, model in [("hgb_poisson", hgb), ("dummy_mean", dummy)]:
        metrics[name] = {}
        for split_name, mask in [("train", train_m), ("val", val_m), ("decision_winter", decision_m)]:
            pred = model.predict(X.loc[mask])
            yy = y[mask.values]
            block = regression_metrics(yy, pred)
            block.update(top_capture(yy, pred))
            metrics[name][split_name] = block
            print(
                f"  {name} {split_name}: MAE={block['mae']:.3f}, "
                f"RMSE={block['rmse']:.3f}, mean={block['mean_actual']:.3f}, "
                f"top5_capture={block['top_5pct_capture']:.1%}"
            )

    # Refit on train+val before making the seasonal policy decision.
    train_val_m = train_m | val_m
    hgb.fit(X.loc[train_val_m], y[train_val_m.values])
    joblib.dump({"pipeline": hgb, "features": cat_cols + num_cols, "horizon_days": args.horizon_days}, args.model_out)

    decision_df = panel.loc[decision_m].copy()
    decision_df["predicted_future_count"] = np.clip(hgb.predict(decision_df[cat_cols + num_cols]), 0, None)
    decision_df = decision_df.sort_values("predicted_future_count", ascending=False)
    decision_df.to_csv(args.panel_out, index=False)

    policies = {}
    for scenario in SCENARIOS:
        model_policy = evaluate_policy(decision_df, "predicted_future_count", scenario)
        oracle = oracle_policy(decision_df, scenario)
        policies[scenario["name"]] = {
            "model_policy": model_policy,
            "oracle_upper_bound": oracle,
            "scenario": scenario,
        }
        print(
            f"Policy {scenario['name']}: selected={model_policy['selected_sections']}, "
            f"saving={model_policy['net_saving_pct']:.1%}, "
            f"coverage={model_policy['future_defect_coverage']:.1%}, "
            f"oracle={oracle['net_saving_pct']:.1%}"
        )

    top_sections = decision_df.head(25)[
        [
            "section_name",
            "predicted_future_count",
            "future_count",
            "past_90d",
            "past_180d",
            "recent_active_cells_180d",
            "observed_span_m_365d",
            "official_length_m",
            "segment_length_km",
            "length_source",
            "section_office",
            "road_class_name",
            "adoption_status",
            "section_type",
        ]
    ].to_dict(orient="records")

    summary = {
        "data": {
            "inputs": args.inputs,
            "rows": int(len(df)),
            "sections": int(df["section_name"].nunique()),
            "horizon_days": args.horizon_days,
            "decision_date": args.decision_date,
            "panel_rows": int(len(panel)),
            "road_network_features": int(len(road_network)) if road_network is not None else 0,
            "panel_official_geometry_rate": float(panel["has_official_geometry"].mean()),
        },
        "split": {
            "train_end": args.train_end,
            "val_end": args.val_end,
            "decision_date": args.decision_date,
            "train_rows": int(train_m.sum()),
            "val_rows": int(val_m.sum()),
            "decision_rows": int(decision_m.sum()),
        },
        "metrics": metrics,
        "policies": policies,
        "top_predicted_sections": top_sections,
        "outputs": {
            "model": args.model_out,
            "decision_panel": args.panel_out,
        },
        "caveats": [
            "SECTION_NAME/SECTION_CODE is used as the road segment key.",
            "DfI FeatureServer geometry supplies official road-section lengths for almost all matched pothole defects.",
            "Full resurfacing cost still needs actual scheme design; this model only screens candidate sections.",
            "Treatment effects are scenario assumptions, not learned causal effects, because repair type/date/outcome is absent.",
        ],
    }
    with open(args.results_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved model -> {args.model_out}")
    print(f"Saved results -> {args.results_out}")
    print(f"Saved winter forecast panel -> {args.panel_out}")


if __name__ == "__main__":
    main()
