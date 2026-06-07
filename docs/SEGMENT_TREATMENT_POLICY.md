# Segment Treatment / Resurfacing Policy

This layer addresses the more complex decision:

> If a road section has repeated pothole failures, should we stop reactive spot
> patching and treat the segment?

The answer is different for bundled patching versus full resurfacing.

## Geometry / Map Data Added

Official DfI road-network data was pulled from the public Highway Network
dataset and ArcGIS FeatureServer:

- Dataset page: https://www.data.gov.uk/dataset/411c9fc1-1192-4efe-8456-44eafc362387/highway-network1
- CSV downloaded to: `data/raw/highway_network.csv`
- GeoJSON downloaded to: `data/raw/dfi_road_network.geojson`
- FeatureServer source:
  `https://services1.arcgis.com/i8LHQZrSk9zIffRU/arcgis/rest/services/DFI_Road_Network/FeatureServer/0`

The road geometry has:

- 71,596 road-section polyline features
- `Section_Code`
- `SECTION_NA`
- `CLASS_NAME`
- `ADOPTION_S`
- `SECTION_TY`
- `Shape__Length`

Join quality is strong:

- 99.7% of official pothole-defect rows match a DfI road-network section code.
- 99.6% of section-panel rows have official geometry/length.

## Forecast Model

Script:

```bash
source .venv/bin/activate
python src/analyze_segment_treatment_policy.py
```

Outputs:

- `models/segment_burden_hgb.joblib`
- `models/segment_treatment_policy_results.json`
- `models/segment_winter_forecast.csv`

The model predicts 180-day future pothole-defect count per road section. The
planning decision date is `2017-10-01`, forecasting the winter period through
late March 2018.

Decision-period model performance:

| Model | MAE | RMSE | Top 5% burden captured | Top 10% burden captured |
|---|---:|---:|---:|---:|
| Section HGB Poisson | 1.42 | 3.48 | 33.6% | 49.6% |
| Mean baseline | 2.92 | 5.41 | 7.1% | 10.3% |

The forecasting model is clearly useful for ranking sections, even though exact
future counts are noisy.

The final refit used for the deployment-style winter forecast is also stronger
than simple history rules:

| Ranking method | Top 5% burden captured | Top 10% burden captured |
|---|---:|---:|
| HGB predicted future count | 36.0% | 51.5% |
| Past 180-day count | 24.5% | 36.5% |
| Past 365-day count | 26.3% | 39.3% |

## Segment Treatment Results

These are policy simulations on the winter 2017/18 decision set:

| Policy | Sections selected | Future burden covered | Event reduction | Net saving | Oracle upper bound |
|---|---:|---:|---:|---:|---:|
| Bundled permanent patching | 2,971 | 59.4% | 35.7% | 21.9% | 25.5% |
| Heavy section patching | 955 | 33.6% | 23.5% | 13.4% | 17.6% |
| Low-cost micro-resurfacing proxy | 855 | 31.9% | 25.5% | 14.6% | 19.1% |
| Full resurfacing stress test, £150k/km | 0 | 0.0% | 0.0% | 0.0% | 0.0% |

## Interpretation

The model supports this:

```text
Use section-level forecasts to bundle repeated local failures into planned
permanent patching / heavy patching programmes.
```

It does not support this from pothole recurrence alone:

```text
Fully resurface whole road sections at £150k/km purely because the pothole model
predicts future potholes.
```

That is not because resurfacing is never right. It is because the available data
does not contain the evidence needed to justify that capital decision:

- pavement condition surveys
- skid resistance
- rutting
- cracking
- traffic volume
- drainage condition
- structural layer condition
- scheme design length and width
- actual resurfacing unit costs

## Best Operational Design

Use a three-tier policy:

```text
Tier 1: defect-level triage
  If individual recurrence risk is high, do faster/better spot intervention.

Tier 2: section-level bundled patching
  If predicted 180-day section burden is high, plan a bundled permanent patching
  visit across the section/cluster.

Tier 3: resurfacing candidate list
  Only escalate to resurfacing when the section model agrees with external
  asset-condition evidence.
```

## Practical Recommendation

For the current data, the best defensible saving claim is:

```text
~10-20% cost saving from model-guided repair planning,
with the strongest practical result coming from bundled section patching.
```

Avoid claiming:

```text
The model proves full resurfacing will save money.
```

Better wording:

```text
The model identifies sections where repeated failures make bundled patching
economically attractive, and produces a ranked resurfacing-candidate list that
should be cross-checked against pavement-condition surveys before capital works.
```

After adding crew frequency/capacity constraints in the final scheduler, the
most realistic headline is closer to `~20%` than the earlier optimistic
immediate-action simulations.

## Top Forecasted Sections

The full ranked table is in `models/segment_winter_forecast.csv`.

The top forecasted winter-risk sections include:

- `7035B0094_01 HILLHEAD RD: COLEMANS CORNER RBT TO MILL RD`
- `7050A1002_08 STRAND RD1: PENNYBURN RAB TO QUEEN'S QUAY RAB`
- `7020A0043_01 CUSHENDALL RD: BROUGHSHANE RD TO CUSHENDALL RD RBT`
- `7020A0036_12 SHANESHILL RD1: ENT.KILLYLANE RESERVOIR TO U4036 UPPER BALLYBOLEY RD`
- `7020C0554_01 SIXTOWNS RD1: DIV BDRY AT BLACKROCK RD TO BEALNAMALA BRIDGE`

These should be interpreted as planning candidates, not automatic resurfacing
orders.
