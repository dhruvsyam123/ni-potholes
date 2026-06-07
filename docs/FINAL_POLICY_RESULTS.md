# Final Policy Results

This is the current best version of the modelling work.

## Architecture

The final architecture is a maintenance-policy simulator, not just a prediction
model.

It has three layers:

1. **Defect recurrence model**
   - Predicts whether an official pothole defect is likely to be followed by
     another nearby defect soon.
   - Useful for daily triage, but not always cost-effective once crew scheduling
     and intervention premiums are included.

2. **Section burden model**
   - Uses official DfI road-network geometry and `SECTION_CODE`.
   - Forecasts 180-day future pothole burden per road section.
   - This is the strongest cost-saving lever.

3. **Capacity-constrained scheduler**
   - Simulates repair frequency: daily or weekly.
   - Simulates enhanced-repair capacity: 0%, 5%, 10%.
   - Compares section treatment only vs section treatment plus risk-prioritised
     spot intervention.

## Final Numbers

The most realistic operational simulation is:

```text
pre-winter planned section treatment
+ optional risk-prioritised enhanced spot repairs
+ crew frequency/capacity constraints
```

Best results from `models/capacity_scheduler_fast_results.json`:

| Scenario | Best policy | Sections treated | Enhanced spot capacity | Cost saving | Event reduction |
|---|---|---:|---:|---:|---:|
| Authority low-cost spot | Section treatment only | 2,971 | 0% | 19.7% | 35.4% |
| Authority good ops | Section treatment only | 2,971 | 0% | 19.7% | 35.4% |
| Moderate asset | Section treatment only | 837 | 0% | 10.0% | 20.1% |
| Strong asset | Section + limited spot | 755 | 10% | 13.3% | 28.2% |

Interpretation:

```text
Best defensible saving range: ~10-20%.
Most likely operational saving: ~20% if bundled section patching is executed well.
```

The earlier `21-23%` figures are still useful as optimistic immediate-action
simulations, but the capacity-constrained scheduler is more realistic. The
headline should therefore be closer to `~20%`, not `~23%+`.

## What Actually Saves Money

The strongest result is not “repair every high-risk pothole individually.”

The strongest result is:

```text
Use historical defects + road geometry to identify high-burden sections before
winter, then plan bundled permanent patching on those sections.
```

Why:

- The section model concentrates future work well.
- Top model-ranked sections contain a large share of future winter defects.
- Treating a section once avoids repeated callouts and repeated small patches.
- Once the worst sections are treated, many remaining individual spot upgrades
  do not produce enough extra avoided defects to justify their premium.

## Recommended Policy

Use this policy first:

```text
1. Before winter, rank road sections by predicted 180-day pothole burden.
2. Treat the top cost-effective sections with bundled permanent patching.
3. Do not automatically upgrade lots of individual live defects.
4. Reserve enhanced spot repair for exceptional high-risk defects or where the
   incremental premium is very low.
```

Operationally:

```text
Primary intervention: section-level bundled patching.
Secondary intervention: selective spot upgrade.
Avoid: broad high-volume enhanced spot patching.
```

## Why Not Full Resurfacing?

Full resurfacing is not justified from pothole recurrence alone in the tested
cost scenarios. The model can create a resurfacing-candidate list, but final
resurfacing decisions need:

- pavement-condition surveys
- cracking/rutting/skid resistance
- drainage condition
- traffic volume
- scheme width/length
- actual resurfacing costs

## What Historic Data Is Used For

Historic data is used to learn two things:

```text
Which individual defect reports tend to be followed by more nearby defects?
Which road sections repeatedly build up high winter pothole burden?
```

The model is not learning physics of asphalt failure directly. It is learning
operational recurrence patterns from official defect history.

## Files

- `src/train_surface_recurrence_model.py`
- `src/analyze_segment_treatment_policy.py`
- `src/simulate_capacity_scheduler.py`
- `models/capacity_scheduler_fast_results.json`
- `models/capacity_scheduler_fast_results.csv`
- `models/segment_winter_forecast.csv`
