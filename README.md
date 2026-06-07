# NI Potholes — Predictive Models + "Solve Quicker" Analysis

## Current best model

The best operational model in this repo is now the official surface-defect
recurrence model:

- Script: `src/train_surface_recurrence_model.py`
- Saved model: `models/surface_recurrence_best.joblib`
- Results: `models/surface_recurrence_best_results.json`
- Explanation: `docs/SURFACE_MODEL_RESULTS.md`

It uses 178,645 official pothole-coded surface-defect records from 2016/17 and
2017/18. On a true future holdout period (2017-10-01 to 2018-03-17), the selected
Histogram Gradient Boosting model scores:

- ROC-AUC: 0.708
- PR-AUC: 0.749
- Brier score: 0.215

Recommended pilot policy: fast-track defects with predicted recurrence risk
`>= 0.60`. On the future holdout this flags 40.0% of official pothole defects,
with 74.9% precision and 52.7% recall.

Economic thresholding is scenario-dependent. See
`docs/COST_POLICY_ANALYSIS.md` and `src/analyze_surface_cost_policy.py` for the
cost sensitivity layer. In short: use roughly `0.55-0.60` for low-cost repair
upgrades, `0.675-0.70` for moderate extra-cost interventions, and `0.80+` only
for expensive interventions where precision matters more than recall.

For the dynamic "repair changes what happens next" question, see
`docs/REPAIR_POLICY_SIMULATION.md` and `src/simulate_repair_policy.py`. The
counterfactual simulation estimates roughly `14-15%` short-term authority-cost
savings under conservative assumptions, and `18-23%` savings if better repairs
have a 30-90 day local asset effect.

For the section-level "patch repeatedly or treat the segment" decision, see
`docs/SEGMENT_TREATMENT_POLICY.md` and
`src/analyze_segment_treatment_policy.py`. This uses official DfI road-network
geometry/lengths joined by section code. It supports bundled permanent patching
on high-burden sections, but not full resurfacing from pothole recurrence alone.

The strongest current architecture is documented in
`docs/INTEGRATED_MAINTENANCE_ARCHITECTURE.md`. It combines section-level winter
planning with defect-level triage and estimates roughly `13-22%` cost saving
under tested intervention assumptions.

The most conservative final policy summary is in `docs/FINAL_POLICY_RESULTS.md`.
After adding crew frequency and capacity constraints, the best defensible
headline is roughly `10-20%` expected saving, with the strongest practical policy
being pre-winter bundled section patching rather than broad enhanced spot
repairs.

The original citizen-enquiry model is still useful for public-report triage, but
it is secondary. The surface-defect model is more relevant for repair planning
and cost-saving decisions because it predicts recurrence in official defect
records rather than repeat complaints.

This repo pulls and analyzes the official Northern Ireland DfI Roads **Pothole Enquiries 2016/2017** open data, understands its (very limited) schema in depth, and builds a lightweight, easy-to-train predictive model focused on **recurrence risk** as a proxy for "potholes that are decaying or were not fixed well enough, and will keep generating complaints and deterioration unless you intervene faster/better".

Everything is designed to run easily on a 16 GB M1 MacBook Air (or similar). No heavy GIS, no massive data, no finicky native extensions required for the core path.

## What the 2016 data actually is (the most important part)

**File**: `Pothole_Enquiries_2016.csv` (5,929 rows, fiscal year 1 Apr 2016 – 31 Mar 2017).

**Exactly 8 columns** (no more, no depth/crack/repair info):

- `ENQUIRY_TYPE`: always "Pothole" (pre-filtered dataset).
- `ENQUIRY_CATEGORY`: Public Enquiry (~53%), Call Centre Public Enquiry (~45%), tiny Complaints / Correspondence.
- `DATE_RECORDED`: second-level timestamp of when the report was logged in the system.
- `DIVISION`: SOUTHERN / WESTERN / NORTHERN / EASTERN (plus 1 junk "ALL CLIENTS").
- `CLIENT_OFFICE_NAME`: 18 district-level client areas (Castlereagh/Lisburn highest volume, etc.).
- `EASTING` / `NORTHING`: Irish Grid (EPSG:29902) projected meters. ~48 junk near-0,0 rows dropped in processing.
- `APPROVAL_STATUS_NAME`: 96% "Completed Enquiries" at extract time. Not a repair log.

**Critical limitations (you must not claim more than this)**:
- No depth, width, crack type, severity, photos, or free-text description.
- No repair date, repair quality, or "was this fixed on day X?" timeline.
- No unique pothole/defect ID. Repeats only detectable by crude spatial + temporal proximity.
- Status field is not usable as "fixed on this date".

Full documented profile (distributions, temporal patterns, spatial sparsity, etc.) lives in:

- [docs/data_summary_2016.md](docs/data_summary_2016.md) — the "mads"/metadata write-up you asked for first. Use this as the reference for any further work.

Also downloaded for context/comparison (same schema):
- 2015 (~6.4k rows)
- 2017 (~18.3k rows)

## The modeling goal & the "solve them quicker" logic

Because there are no physical measurements, we cannot directly predict "depth will reach 50 mm by next week".

Instead we predict a clean, observable proxy:

**Target**: for a given report, will there be *another* report within ~200 m (Irish Grid) in the next 14 days?

- Positive rate in 2016 data: ~15.4%.
- This captures "clusters of complaints that keep arriving at the same small location" — exactly the places where either (a) the initial report wasn't actioned fast enough, or (b) whatever was done didn't arrest the deterioration, or (c) the underlying road condition is so bad that new potholes form right next to the old one quickly.

**"If I had patched it earlier, would I have stopped it decaying?" — robust proxy logic**

We compute this directly from the data (see the "EARLIER PATCH / DECAY PREVENTION HEURISTIC" section in the training run):

- 795 of the 200 m cells had 2+ reports over the year.
- 29.7% of those multi-report cells got their *second* report within 7 days of the very first report in that cell.
- 39.1% of *all consecutive inter-report gaps* inside multi-cells were ≤ 7 days.
- Median gap between consecutive reports in a multi-cell: ~16.6 days.

**Claim you can defensibly make** (with the caveats):

> "In the 2016/17 data, nearly 40% of the follow-on complaints in locations that generated multiple reports arrived within a week of the previous one in the same ~200 m area. A predictive system that (1) scores incoming reports for recurrence risk and (2) triggers a high-quality, permanent repair within 48–72 hours on the high-risk ones could plausibly have eliminated a substantial fraction of those repeat enquiries — and the continued road deterioration and vehicle damage risk they represent. This is a proxy analysis (we do not observe actual patch dates or quality), but it is grounded in the only outcome the open data gives us: citizen reports."

This is exactly the kind of evidence-based "we could have solved them quicker / stopped the decay" argument that is useful for operational justification or business case, while staying honest about the data.

## Model & performance (temporal generalization)

We use **strictly temporal splits** only (train on earlier data, validate middle, test on latest months) so the numbers reflect "how well would this have worked on future data?"

Split used (2016 focus):
- Train: up to 31 Dec 2016 (3,715 reports)
- Val: Jan–Feb 2017 (1,183)
- Test: Mar 2017 (983)

**Main model** (saved): `HistGradientBoostingClassifier` via sklearn (modern histogram gradient boosting — the practical equivalent of "XGBoost something" for this size and for zero-hassle M1 training). No libomp, no compilation, trains in <1 second.

Results (from the run):

- HGB: Train ROC-AUC 0.86 / PR-AUC ~0.60
- Val: 0.681 / 0.397
- Test: 0.643 / 0.304

(Positive rate rises a bit in the later winter months; ranking power drops from the overfit-y train number but stays well above random and usable for prioritization.)

A plain LogisticRegression baseline is surprisingly competitive on this data (test ROC-AUC 0.667), which tells you the signal is mostly in the time-of-year + location + recent local history, not complex interactions.

**Features** (all available at report time, no leakage):
- Enquiry category, Division, Client office (one-hot)
- lon/lat (transformed), hour, is_weekend, is_winter_spring, days since start of period
- Simple past-only lags: how many reports in the *same* 200 m cell in the prior 7d / 14d

The saved pipeline (`models/recurrence_risk_hgb.joblib`) can be loaded with joblib and used to `.predict_proba` on new rows that have the same columns (after the same light feature prep).

## How to run everything (M1-friendly)

```bash
# 1. One-time setup (takes a minute)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # or the packages listed below
# (optional, only if you want real xgboost/lightgbm later)
# brew install libomp
# pip install xgboost lightgbm

# 2. (Data is already downloaded in data/raw/)

# 3. Prepare features + recurrence labels (2016)
python src/prepare_data.py --inputs data/raw/pothole_enquiries_2016.csv \
    --out data/processed/potholes_2016_prepared.csv

# 4. Train + evaluate + compute the "earlier patch" stats
python src/train_recurrence_risk.py \
    --data data/processed/potholes_2016_prepared.csv \
    --model_out models/recurrence_risk_hgb.joblib

# You can also feed 2015+2016 together for a bit more volume:
# python src/prepare_data.py --inputs data/raw/pothole_enquiries_2015.csv data/raw/pothole_enquiries_2016.csv ...
```

Core requirements (already frozen in `requirements.txt` after setup):
pandas, numpy, scikit-learn, pyproj, joblib, matplotlib (optional for plots).

The whole pipeline on 2016 data is a few seconds on a laptop.

## Project layout

```
data/
  raw/                    # the three CSVs from data.gov.uk
  processed/              # cleaned + lat/lon + time feats + target + lags
docs/
  data_summary_2016.md    # the full "what the data has" write-up + limitations + suggested uses
src/
  prepare_data.py
  train_recurrence_risk.py
models/
  recurrence_risk_hgb.joblib
  ..._results.json
README.md
```

## Next steps / extensions you can do later

- Add 2015 + 2017 for bigger training set + true multi-year temporal CV / future holdout on 2017.
- Switch the booster to real xgboost/lightgbm once `brew install libomp` + reinstall is done on your native arm python (the model code already has the structure; just change the import + constructor).
- Aggregate to cell × week counts and do a count / Poisson-style model for area-level risk forecasting.
- Enrich with open weather (frost/rain days) or road class if you pull extra public datasets.
- Turn the scorer into a tiny FastAPI or Streamlit triage tool that ingests a new enquiry and outputs a priority score + "expected repeat risk" + suggested response SLA.
- Cluster analysis or survival modeling on the inter-report times for more sophisticated "time-to-next-complaint" predictions.

The data documentation in `docs/data_summary_2016.md` is deliberately written so you (or a future you / colleague) can pick this up and extend without re-doing all the schema archaeology.

## License / attribution

Data © DfI Roads / OpenDataNI, published under the Open Government Licence v3.0. See the original dataset page for full terms.

---

Built end-to-end in one session starting from an empty directory. The emphasis was on (1) ruthlessly understanding and documenting the actual columns and limitations first, then (2) building the simplest defensible predictive + "earlier action prevents repeats" logic that can actually run on modest hardware and generalize forward in time.
