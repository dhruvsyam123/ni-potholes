# NI Potholes — Cost-Saving Maintenance Model

This repo focuses on the final retained workflow for predicting pothole
recurrence risk and turning that into a maintenance policy that aims to reduce
repair cost.

The final pipeline uses official DfI surface-defect pothole records, official
DfI road-network geometry and section lengths, a defect-level recurrence model,
a section-level winter burden forecast, and a capacity-constrained maintenance
scheduler.

The strongest final conclusion is:

```text
the main saving comes from identifying high-burden road sections before winter
and planning bundled permanent patching on those sections
```

The best defensible headline from the final scheduler is roughly:

```text
~10-20% expected cost saving
```

with the strongest practical case around:

```text
~19.7% saving and ~35.4% fewer expected defect events
```

when bundled section patching is executed well.

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
