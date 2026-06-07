# Integrated Maintenance Architecture

This is the strongest current architecture in the repo.

It combines:

1. Defect-level recurrence risk.
2. Section-level winter burden forecasting.
3. Official DfI road-network geometry and segment lengths.
4. Counterfactual maintenance-policy simulation.

The goal is not just to predict potholes. The goal is to choose the cheapest
intervention level:

```text
do nothing extra
vs faster/better spot repair
vs bundled section patching
vs heavy section patching / resurfacing candidate review
```

## Scripts

Run in this order:

```bash
source .venv/bin/activate

python src/download_dfi_road_network_geometry.py
python src/train_surface_recurrence_model.py
python src/analyze_segment_treatment_policy.py
python src/simulate_integrated_maintenance_policy.py
```

Key outputs:

- `models/surface_recurrence_best.joblib`
- `models/segment_burden_hgb.joblib`
- `models/segment_winter_forecast.csv`
- `models/integrated_maintenance_policy_results.json`
- `models/integrated_maintenance_policy_thresholds.csv`

## Policy Stack

Before winter:

```text
Use section model to forecast 180-day pothole burden per DfI road section.
Treat high-value sections with planned bundled/heavy patching.
```

During winter:

```text
For defects not already covered by a treated section, use the defect-level model
to decide whether to do faster/better spot intervention.
```

The simulation avoids double-counting: if a section is already treated, events
on that section are not also charged for enhanced spot treatment.

## Integrated Savings

Compared with reactive repair of every observed future defect:

| Scenario | Section-only saving | Integrated saving | Event reduction | Section treatments | Spot intervention rate |
|---|---:|---:|---:|---:|---:|
| Authority conservative | 11.0% | 13.3% | 27.2% | 1,840 | 8.3% |
| Authority good ops | 19.7% | 21.5% | 40.2% | 2,971 | 6.3% |
| Moderate asset | 10.0% | 17.9% | 37.7% | 837 | 21.1% |
| Strong asset | 11.2% | 22.0% | 44.2% | 755 | 21.8% |
| Full resurfacing stress | 0.0% | 23.4% | 42.1% | 0 | 36.2% |

The last row does not mean full resurfacing works. It means the optimiser
selects zero full-resurfacing sections and falls back to defect-level spot
triage.

## Best Current Savings Claim

The most defensible range is:

```text
~13-22% expected cost saving
```

The low end assumes conservative treatment effects and authority-centric costs.
The high end assumes stronger asset effects but still does not rely on full
resurfacing being justified by the pothole data alone.

An ambitious but still explainable claim is:

```text
Model-guided integrated maintenance could reduce expected defect events by
~27-44% in the simulated winter holdout, translating into ~13-22% cost saving
under the tested intervention-cost assumptions.
```

## Architecture Rationale

Defect-level models are good for immediate operational triage, but they miss the
planning value of repeated failures along the same road section.

Section-level models are good for planning, but they are too coarse for daily
crew decisions.

The integrated architecture uses both:

- Section model: where should we plan bundled treatment?
- Defect model: which remaining live defects deserve better/faster intervention?

This is the right structure for saving money because the cheapest intervention
depends on scale.

The section model also beats simple history ranking. In the final winter
forecast table, the top 5% of HGB-ranked sections capture `36.0%` of future
burden. Ranking by past 180-day defects captures `24.5%`; ranking by past
365-day defects captures `26.3%`.

## Limits

The model still does not learn true repair causality. Treatment effects are
scenario assumptions because the open data does not contain:

- repair timestamp
- repair type
- repair quality
- unit repair cost
- traffic management cost
- condition survey score
- whether a treated patch failed again

The next major improvement is to obtain repair/order data and train an uplift or
causal model:

```text
Which intervention type reduces future defects the most, conditional on road
section, defect history, and cost?
```

Without that data, this architecture is a robust decision-support simulator, not
a proof of causal savings.
