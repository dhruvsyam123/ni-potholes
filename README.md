# NI Potholes — Cost-Saving Maintenance Model

This repo contains the final retained workflow for turning historic pothole
defect data into a maintenance policy that aims to reduce repair cost.

The project is built on official DfI surface-defect pothole records and
official DfI road-network geometry. It does not try to model asphalt physics
directly. It learns operational recurrence patterns from historic defect data,
then uses those predictions inside a repair-planning simulation.

The strongest final conclusion is simple: the main saving comes from identifying
high-burden road sections before winter and planning bundled permanent patching
on those sections. The best defensible headline from the final scheduler is
roughly `10-20%` expected cost saving, with the strongest practical case around
`19.7%` saving and `35.4%` fewer expected defect events when bundled section
patching is executed well.

## What The Model Actually Is

This is not one single model. It is a three-layer system.

1. A **defect-level recurrence model** predicts whether an official pothole
defect is likely to be followed by another nearby defect soon. Technically this
is a `HistGradientBoostingClassifier` from scikit-learn, trained on official
surface-defect pothole records.

2. A **section-level burden model** predicts how many pothole defects a road
section is likely to generate over the next 180 days. Technically this is a
`HistGradientBoostingRegressor` with Poisson loss, again using scikit-learn.

3. A **capacity-constrained maintenance scheduler** is layered on top of those
predictions. This is not another ML model. It is a simulation that decides
whether it is cheaper to do section treatment, selective spot upgrades, or just
continue with reactive repair.

The repo therefore answers a policy question, not just a prediction question:
given limited crews and intervention costs, what should be repaired first, at
what scale, and what saving is plausible?

## Final Workflow

Run the retained pipeline in this order:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/download_dfi_road_network_geometry.py
python src/train_surface_recurrence_model.py
python src/analyze_segment_treatment_policy.py
python src/simulate_capacity_scheduler.py --fast \
  --out_json models/capacity_scheduler_fast_results.json \
  --out_csv models/capacity_scheduler_fast_results.csv
```

The pipeline is intentionally narrow:

- download and join official DfI road geometry
- train the defect recurrence model
- forecast winter burden by road section
- simulate the final maintenance policy under cost/capacity constraints

## Main Scripts

- `src/download_dfi_road_network_geometry.py`
  Downloads official DfI road-section geometry from the public ArcGIS
  FeatureServer.

- `src/train_surface_recurrence_model.py`
  Trains the main defect-level recurrence model on official pothole-coded
  surface defects.

- `src/analyze_segment_treatment_policy.py`
  Forecasts 180-day future pothole burden per DfI road section using official
  road geometry and section length.

- `src/simulate_capacity_scheduler.py`
  Runs the final operations simulation with crew frequency/capacity constraints
  and compares section treatment versus enhanced spot intervention.

## Technical Summary

The retained technical stack is:

- `HistGradientBoostingClassifier` for defect recurrence risk
- `HistGradientBoostingRegressor` with Poisson loss for section burden
- official DfI road-network geometry joined by `SECTION_CODE`
- a final scheduler/simulation layer for economic decisions

This is why the project should not be described as “an XGBoost model.” The
machine-learning part is gradient-boosted trees from scikit-learn, and the
final decision layer is a simulation.

## Main Outputs

- `models/surface_recurrence_best.joblib`
- `models/surface_recurrence_best_results.json`
- `models/segment_burden_hgb.joblib`
- `models/segment_treatment_policy_results.json`
- `models/segment_winter_forecast.csv`
- `models/capacity_scheduler_fast_results.json`
- `models/capacity_scheduler_fast_results.csv`

## Data Used

Kept raw inputs:

- `data/raw/surface_defects/surface_defects_2016.csv`
- `data/raw/surface_defects/surface_defects_2017.csv`
- `data/raw/highway_network.csv`
- `data/raw/dfi_road_network.geojson`

Reference-only data still kept:

- `data/raw/pothole_enquiries_2015.csv`
- `data/raw/pothole_enquiries_2016.csv`
- `data/raw/pothole_enquiries_2017.csv`
- `docs/data_summary_2016.md`

The enquiry dataset is no longer the main modelling path. It remains as context
and documentation only.

## Final Findings

Best results from the final capacity-constrained simulation:

| Scenario | Best policy | Sections treated | Enhanced spot capacity | Cost saving | Event reduction |
|---|---|---:|---:|---:|---:|
| Authority low-cost spot | Section treatment only | 2,971 | 0% | 19.7% | 35.4% |
| Authority good ops | Section treatment only | 2,971 | 0% | 19.7% | 35.4% |
| Moderate asset | Section treatment only | 837 | 0% | 10.0% | 20.1% |
| Strong asset | Section + limited spot | 755 | 10% | 13.3% | 28.2% |

What this means operationally:

- The strongest cost-saving lever is planned section treatment before winter.
- Broad high-volume enhanced spot repair is usually not the cheapest policy.
- Full resurfacing is not justified from pothole recurrence alone in this data.
  It needs separate pavement-condition and scheme-cost evidence.

The important practical point is that the model is more useful for **planning**
than for firing off lots of one-off “smart repairs.” Historic data is most
valuable when it shows which sections repeatedly fail and should be treated as a
bundle rather than patched over and over.

## Docs

- `docs/FINAL_POLICY_RESULTS.md`
  Final recommended numbers and policy summary.

- `docs/SEGMENT_TREATMENT_POLICY.md`
  Section-level model, road geometry join, and segment-treatment findings.

- `docs/data_summary_2016.md`
  Detailed documentation of the older public-enquiry dataset, kept as reference.

## Assumptions and Limits

The open data does not contain:

- repair timestamp
- repair type
- repair cost
- repair quality
- pavement condition survey data

So the “repair changes future defects” part is simulated with explicit cost and
effect assumptions. The model is therefore a robust decision-support system, not
causal proof of exact savings.

That means the strongest claims in this repo are:

- which sections are historically likely to generate heavy future pothole burden
- which policies look cost-effective under stated cost/effect assumptions

The repo does **not** prove a specific repair technique causes a specific future
reduction without additional intervention data.
