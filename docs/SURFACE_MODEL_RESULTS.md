# Surface-Defect Pothole Model Results

This is now the primary model for operational cost saving. It uses official DfI
surface-defect records filtered to pothole-coded defects, rather than citizen
enquiries.

## Target

For each official pothole defect, predict:

> Will another official pothole defect be recorded within 200 m in the next 14 days?

This remains a recurrence-risk proxy, not a physical deterioration simulation.
It is much closer to maintenance cost than the enquiry model because it uses
official defect/work records.

## Data Used

- Raw files: `surface_defects_2016.csv` and `surface_defects_2017.csv`
- Pothole-coded rows: 178,645
- Date range: 2016-04-01 08:32 to 2018-03-31 11:34
- Overall positive rate: 57.3%

The training script combines both fiscal years before computing labels. This is
important because a March 2017 defect can now correctly see an April 2017 repeat.
The final 14 days are excluded because future labels are censored.

## Split

| Split | Date rule | Rows |
|---|---:|---:|
| Train | <= 2017-03-31 | 94,490 |
| Validation | 2017-04-01 to 2017-09-30 | 27,925 |
| Test | 2017-10-01 to 2018-03-17 | 49,360 |

## Model Comparison

| Model | Test ROC-AUC | Test PR-AUC | Test Brier | Verdict |
|---|---:|---:|---:|---|
| Histogram Gradient Boosting | 0.708 | 0.749 | 0.215 | Best |
| Sparse logistic regression + route code | 0.666 | 0.728 | 0.236 | Good baseline |

The selected saved model is:

`models/surface_recurrence_best.joblib`

The full metrics JSON is:

`models/surface_recurrence_best_results.json`

## Operating Points

These are measured on the untouched future test period.

| Risk threshold | Precision | Recall | Flagged workload | Break-even avoided-cost multiple |
|---:|---:|---:|---:|---:|
| 0.40 | 0.637 | 0.872 | 77.9% | 1.57x |
| 0.50 | 0.697 | 0.713 | 58.2% | 1.44x |
| 0.60 | 0.749 | 0.527 | 40.0% | 1.34x |
| 0.70 | 0.802 | 0.317 | 22.5% | 1.25x |
| 0.80 | 0.853 | 0.127 | 8.5% | 1.17x |

Recommended pilot setting: start around `risk >= 0.60`.

At that point, about 40% of official pothole defects are fast-tracked. Roughly
75% of the fast-tracked group would otherwise have another official pothole
record nearby within 14 days.

## Cost Function

The data does not contain repair cost, material cost, crew time, claims, or
repair quality, so the monetary function has to be parameterized:

```text
fast/permanent repair is justified when:

p_repeat * intervention_effectiveness * avoided_repeat_cost
  > incremental_fast_repair_cost
```

Equivalent threshold:

```text
p_repeat
  > incremental_fast_repair_cost
    / (intervention_effectiveness * avoided_repeat_cost)
```

For threshold `0.60`, measured precision is `0.749`.

If the intervention fully prevents repeats, the avoided repeat/deterioration
cost only needs to be about `1.34x` the incremental cost of the faster/better
repair. If the intervention is only 50% effective, it needs to be about `2.67x`.

For a more rigorous sensitivity analysis using UK repair/damage cost anchors,
see `docs/COST_POLICY_ANALYSIS.md`. The short version is:

- `0.55-0.60` is suitable when the intervention is a low-cost upgrade over the
  default repair.
- `0.675-0.70` is more defensible when the intervention costs around
  `£75-£125` extra.
- `0.80+` is only suitable for expensive interventions where high precision is
  more important than recall.
- A full medium patch cost should not be treated as justified by this recurrence
  model alone unless the incremental premium is much lower than the full patch
  price or broader asset/safety benefits are included.

## Why This Is Better Than The Enquiry Model

The older enquiry model predicts repeat citizen complaints. It is useful for
public-report triage, but it has weak future holdout performance:

- Enquiry HGB: ROC-AUC 0.643, PR-AUC 0.304
- Enquiry logistic regression: ROC-AUC 0.667, PR-AUC 0.317

The surface-defect model predicts recurrence in official pothole records and
has stronger performance, more rows, and a more direct link to maintenance cost.

## Reproduce

```bash
source .venv/bin/activate
python src/train_surface_recurrence_model.py
```
