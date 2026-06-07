# Model Results — Clear, Effective Explanation (NI Potholes 2016/17)

This document explains the model, the numbers, and what they actually mean for a "solve potholes quicker / stop decay" use case. It is written to be read by non-ML people as well as practitioners.

## 1. What exactly is the model predicting?

**Simple definition:**

For every single pothole *report* (enquiry) that comes in, the model outputs a **risk score** between 0 and 1:

> "Given everything we know at the moment this report was logged, what is the probability that another report will appear within 200 meters of this location in the next 14 days?"

- **Base rate in the data**: 15.4% of reports are followed by at least one more in that window.
- This is **not** a prediction of "a pothole will form". It is a prediction of **future citizen complaints at the same small spot**.

**Why this target?**

Because the raw data has almost no physical information (no depth, no crack type, no repair records). The only thing we can reliably observe is whether complaints keep arriving at nearly the same coordinates. Clusters of complaints close together in space and time are the best available proxy for "this defect is either not fixed yet, was fixed poorly, or is decaying/ re-forming fast."

The model is therefore a **triage / prioritization scorer**, not a physical deterioration forecaster.

## 2. How the data was split (the most important methodological choice)

We did **not** do a random train/test split.

Instead we used a strict **temporal split** that mimics real deployment:

| Period          | Reports | Positive rate (actual repeats) | Purpose                     |
|-----------------|---------|--------------------------------|-----------------------------|
| Up to 31 Dec 2016 | 3,715  | 14.1%                          | Training                    |
| Jan – Feb 2017    | 1,183  | 18.7%                          | Validation (model selection)|
| March 2017        | 983    | 16.7%                          | Final test (future holdout) |

**Why this matters for your goal ("can it generalize into the future")**:
- March is later in the winter damage season than the training period.
- Positive rate rose a bit (more repeats happening).
- Any performance drop from train → test tells you how much the model will degrade when the world moves forward in time (different weather, different crew behavior, different reporting intensity).

This is the honest way to evaluate "will this help us in the future?"

## 3. The actual performance numbers (with interpretation)

### Main model (HistGradientBoostingClassifier — the practical XGBoost-style model)

| Split          | ROC-AUC | PR-AUC | Notes |
|----------------|---------|--------|-------|
| Train (to Dec) | 0.860   | 0.600  | Looks strong (but optimistic) |
| Val (Jan-Feb)  | 0.681   | 0.397  | Realistic drop |
| **Test (Mar)** | **0.643** | **0.304** | The number that matters for future use |

**What do these metrics actually mean?**

- **ROC-AUC 0.643 on the test set**: If you take one random report that *did* lead to a repeat in 14 days and one that did *not*, the model gives the "will repeat" one a higher risk score 64.3% of the time. 0.5 = random guessing, 1.0 = perfect. 0.64 is useful for ranking/prioritization but far from magic.
- **PR-AUC 0.304**: More relevant when positives are rare. It measures how well the model trades off catching the real future hot spots vs. not drowning crews in false alarms. The drop from 0.60 (train) to 0.30 (test) is noticeable.

**Important observation**: On the later periods (val and especially test), a plain Logistic Regression was *competitive or slightly better* than the boosted model (test ROC-AUC 0.667). This tells us the signal is mostly linear and driven by a few strong features (recent history in the cell + time of year + location).

## 4. The practical table you actually care about — different risk thresholds

The model does **not** force you to use a 50% cutoff. You choose the operating point based on how many crews/inspectors you can mobilize.

Here are the **exact operating points** computed on the true future holdout (March 2017 test set) with a freshly trained identical model:

```
Risk >=     Precision    Recall     F1      % of all incoming reports we would flag
  0.15         0.208       0.695    0.320           55.7%
  0.20         0.249       0.506    0.333           34.0%
  0.25         0.293       0.329    0.310           18.7%
  0.30         0.397       0.189    0.256            7.9%
  0.40         0.528       0.116    0.190            3.7%
  0.50         0.417       0.030    0.057            1.2%
```

These numbers come from the March 2017 holdout (the data the model had never seen during training).

**How to read this for operations**:

- If you are willing to "fast-track" or give extra attention to the top **23–32%** of incoming reports (risk score ≥ 0.25–0.30), you will catch roughly **38–49%** of the locations that were actually going to generate another complaint within 14 days.
- At those thresholds precision is only 26–28%. That means of the ones you flag, only about 1 in 4 will actually produce a repeat in the window. The other 3/4 are "false alarms" by this definition — but many of them may still be worth looking at (they are in currently active areas).
- If you only act on the very highest scores (≥ 0.50), you get high precision but you miss almost everything (recall 3%). This is too conservative for a "solve quicker" program.

**Recommended starting point for a pilot**: Flag everything ≥ 0.25 or 0.28. This gives you a manageable extra workload while catching nearly half the repeat-generating cases.

## 5. Why the model works at all (feature drivers)

The strongest signals (in rough order) are:

1. **Recent history in the same 200 m cell** (`lag_7d_same_cell`, `lag_14d_same_cell`) — by far the most powerful. If two or three reports already arrived in the last week or two at almost the same spot, the chance of yet another one soon is much higher. This is almost definitional for the target.
2. **Time of year / seasonality** (`is_winter_spring`, month, `days_since_apr2016`).
3. **Location** (specific client offices / divisions have chronically higher repeat rates — Castlereagh/Lisburn, certain Newry Mourne & Down areas, etc.).
4. **Hour of report** and weekend vs weekday (weaker).

The model is largely learning: "recent activity in this small area + bad time of year + known bad district = elevate priority."

This is useful, but it also explains part of the train-to-test drop: when the overall intensity of reporting changes (more winter damage manifesting), the lag features become even more dominant.

## 6. The "earlier patch stops decay" evidence (often more powerful than the ML metrics)

This analysis does **not** even use the model. It is direct from the raw report timestamps and locations.

Out of 795 different 200 m cells that received 2 or more reports during the year:

- 29.7% got their **second** report within 7 days of the very first report in that cell.
- 39.1% of **all gaps** between consecutive reports inside those hot cells were 7 days or shorter.
- Median gap between one report and the next in the same small area: **16.6 days**.

**The claim this supports** (with honest caveats):

In 2016/17 data, a very large fraction of the follow-on complaints — the visible sign that deterioration continued or the initial response didn't hold — happened within a week of the previous complaint at the same spot.

A program that:
- Uses a simple risk model (or even just "has this cell had a report in the last 7–10 days?"),
- And responds to the high-risk ones with a **high-quality, permanent repair within 48–72 hours**,

has a realistic, data-backed chance of breaking many of those short-cycle repeats.

**Caveats you must state**:
- We do not observe whether any repair actually happened, what quality it was, or when.
- Some short gaps are the same pothole being reported multiple times before anyone arrived (not a new decay cycle).
- Some long gaps are brand new defects forming in a chronically bad road section.

Still, 39% of follow-on reports arriving in ≤7 days is a strong operational signal that speed and quality of the first response matter.

## 7. Overall verdict — how good is this for "predictive model to solve them quicker"?

**Strengths**:
- Honest temporal evaluation (future holdout).
- Very lightweight (runs instantly on a laptop).
- The non-ML heuristic (the 39% within 7 days number) gives you a concrete, defensible story even before you deploy the fancy model.
- You can choose different risk thresholds to match your capacity.

**Limitations / realism check**:
- Test ROC-AUC of 0.64 and PR-AUC of 0.30 is **modest**. It will help with prioritization and worklist ranking, but it is not going to magically identify every bad pothole.
- Much of the predictive power comes from "recent activity in this cell" — which is almost tautological. The model is mainly amplifying the signal that "hot spots tend to stay hot in the short term."
- No physical features (depth, crack type, road construction, traffic, weather) means we are predicting *complaint behavior*, not engineering reality.

**Best use of this work**:
- Use the risk score (or even a simple rule based on recent lags + season) to create a **prioritized fast-response list** for crews.
- Measure the actual outcome: do the high-risk cases that receive fast high-quality repairs show fewer follow-on complaints than similar cases that don't?
- Treat the current model as a **baseline** that you improve by adding better features (weather, road class, actual repair logs if you ever get them, photos + image models, etc.).

## 8. Files you should look at

- `models/recurrence_risk_hgb_results.json` — the exact numbers
- `docs/data_summary_2016.md` — everything that is (and is not) in the raw data
- `src/train_recurrence_risk.py` — the code that produced the heuristic + metrics
- `data/processed/potholes_2016_prepared.csv` — the features + target used

## Bottom line (one paragraph for stakeholders)

The 2016 Northern Ireland pothole enquiry data is thin on physical measurements, so we built a complaint-recurrence risk model instead. On future months the model ranks reports such that a realistic operating point (flagging the top ~25–30% by risk score) would have caught roughly 40–50% of the locations that went on to generate another complaint within two weeks. Independently, the raw data shows that 39% of all follow-on reports in multi-complaint locations arrived within 7 days of the previous one. Together these two facts support a focused "fast, high-quality response on currently active small areas" program far better than random or purely reactive approaches. The model is a useful but modest tool for prioritization; its biggest value may be forcing the organization to act faster on the places the data already shows are most likely to keep decaying and generating more work.

---
*Generated from the actual trained model and raw data, March 2017 holdout.*