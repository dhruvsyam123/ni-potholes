# Pothole Enquiries 2016/2017 (Northern Ireland) — Data Summary & Metadata

**Source**: OpenDataNI / DfI Roads (Department for Infrastructure Roads), Northern Ireland.  
**Dataset page**: https://www.data.gov.uk/dataset/4551d515-2236-4891-874a-6bd986d961ad/pothole-enquiries  
**File used**: `Pothole_Enquiries_2016.csv` (direct: https://apps1.wdm.co.uk/odr/drdni/Pothole_Enquiries_2016.csv)  
**Period covered**: 1 April 2016 – 31 March 2017 (fiscal / "2016/2017" reporting year)  
**Rows**: 5,929  
**License**: Open Government Licence v3.0  

## Purpose (from publisher)
> "This dataset shows public enquiries and complaints of potholes recorded on public roads maintained by DfI Roads in Northern Ireland."

It is a **log of citizen / call-centre reports** (enquiries), **not** a log of physical defects with measurements or repairs.

## Schema (exactly 8 columns, no others in this file)

| Column                  | Type          | Description / Notes                                                                 | Cardinality / Values |
|-------------------------|---------------|-------------------------------------------------------------------------------------|----------------------|
| ENQUIRY_TYPE            | string (cat)  | Always "Pothole" in this extract. Dataset appears pre-filtered to pothole reports. | 1 unique |
| ENQUIRY_CATEGORY        | string (cat)  | How the report arrived: "Public Enquiry", "Call Centre Public Enquiry", "Complaints", "Correspondence" | 4 |
| DATE_RECORDED           | datetime      | Timestamp of logging (dd/mm/YYYY HH:MM:SS). Not necessarily the exact time the defect appeared. | 5,929 distinct (or near) |
| DIVISION                | string (cat)  | DfI Roads maintenance division: SOUTHERN, WESTERN, NORTHERN, EASTERN (one "ALL CLIENTS" row) | 5 |
| CLIENT_OFFICE_NAME      | string (cat)  | Local government district / client area (post-2015 11-council structure + splits like (EAST)/(WEST)). One "?" value (45 rows). | 18 |
| EASTING                 | float         | Projected easting (meters). Coordinate Reference System: **Irish Grid (EPSG:29902 / TM65 Irish Grid)**. | — |
| NORTHING                | float         | Projected northing (meters), same CRS. | — |
| APPROVAL_STATUS_NAME    | string (cat)  | Workflow status **at time of data extract** (not a repair log): "Completed Enquiries" (96.3%), "New Enquiries", "Enquiries In Progress". | 3 |

**No missing values** in any column.

## Key Distributions & Patterns (2016/2017)

### Temporal
- Almost daily coverage (350 of 365 days had ≥1 report).
- Median ~14 reports per day; max 109 (14 Apr 2016).
- Strong seasonality: April 2016 spike (1,087 – possibly post-winter backlog), summer lull (lowest ~270-350/mo), then sharp rise Jan 547 → Feb 667 → Mar 966.
- Reporting concentrated 08:00–16:00, peak 09:00–11:00. Some overnight and evening reports.

### Spatial / Administrative
- 4,907 unique ~100 m grid cells for 5,881 valid points (very sparse).
- Only 35 cells had 5+ reports over the whole year.
- Highest volume client offices (districts): Castlereagh/Lisburn (690), Ards & North Down (514), Newry Mourne & Down West (508), Antrim & Newtownabbey (508), etc.
- Southern Division highest (2,068).

### Coordinate notes
- ~48 rows have near-zero (junk) coordinates (E or N < 100). Filter them for spatial work.
- Precision to ~0.1 m but likely derived from map-click, GPS, or address match in the back-office system. No indication of snapping to road centerlines.
- To use in modern tools, transform to WGS84 (EPSG:4326) using pyproj (Irish Grid → lon/lat).

### Duplication & "Repeats"
- Exact (E,N,datetime) duplicates: only 3 cases (max multiplicity 3). Data is essentially unique per logged enquiry.
- **Proxy repeats via proximity**: Using 150 m cells, 229 cells had 3+ reports over the year. Some clusters show very short gaps (same-day multiple reports or same location logged twice), others have median gaps of weeks (possible re-formation or poor repair).

## What the Data Does NOT Contain (Critical Limitations)

- **No defect attributes**: depth, diameter/width, crack type (fatigue/crocodile, edge, linear, pothole-in-pothole, etc.), photos, or any severity/priority score.
- **No free-text**: no description of the defect, surrounding road condition, or precise verbal location.
- **No repair / intervention data**: no date patched, repair type (temporary/permanent), crew, materials, or post-repair inspection outcome.
- **No outcome for the citizen**: no linked claims, follow-up satisfaction, or "was it fixed?" flag with timestamp.
- **No unique defect identifier**: you cannot track the *same physical pothole* across time except by crude spatial-temporal proximity.
- **Status field is not a timeline**: "Completed Enquiries" reflects the state when the CSV was generated for open data, not that every report led to a repair on a certain date.

## "MADs" / Summary Statistics (quick reference)

- Rows: 5,929
- Valid spatial points: 5,881
- Unique 100 m cells: 4,907
- Cells with 5+ reports (hot): 35
- Cells with 3+ reports (150 m proxy): 229
- Peak daily volume: 109
- Busiest month: 2016-04 (1,087) and 2017-03 (966)
- % "Completed" status: 96.3%
- Junk coords: 48 (0.8%)

## Suggested Uses & Modeling Framing (for "predict + solve quicker")

Because there are no physical measurements, any predictive model is fundamentally a **model of report *generation*** (citizen complaint intensity), not of physical pothole formation/decay mechanics.

**Useful proxy targets** (observable in this data):
1. **Future report count** in a small spatial cell (e.g. 200 m grid) over next 7/14/30 days.
2. **Binary "will re-report"**: after a report arrives in a cell, probability of another report within 200 m in the next 14 days. This can be used to prioritize "fast response / quality check" on the ones most likely to generate follow-ups (i.e. the ones that would have "decayed" or not been arrested by a quick patch).
3. **Hotspot risk scoring** for proactive inspection: which cells that are currently quiet are likely to light up soon (using lags + seasonality).

**"If we had patched earlier, would we have stopped the decay / further reports?" pseudo-logic**:
- For every multi-report cluster (150–200 m), treat the first report as t0.
- Compute the empirical distribution of time-to-subsequent-reports.
- Statistic: "X% of all follow-on reports in clusters arrived within 7 days of the initial report in that cell."
- Interpretation (with caveats): locations showing short inter-report times are candidates where a same-day or next-day high-quality permanent repair might have eliminated the subsequent citizen reports (and associated vehicle damage risk, claims, and repeat crew visits). Locations with long gaps may be "new" defects forming repeatedly due to underlying road condition.
- This is **not causal identification** (no randomized patching, no actual repair timestamps). It is a data-driven heuristic for triage/prioritization and for arguing the value of faster response on high-recurrence-risk sites.
- Can be turned into a simple decision rule or feature in the ML model: "if this cell already has 1+ recent reports, boost priority".

## Files in this project (after processing)
- `data/raw/pothole_enquiries_2016.csv` (and 2015/2017 for comparison or more volume)
- `data/processed/` — cleaned + feature-engineered (lon/lat, time feats, grid ID, lag counts, etc.)
- `src/` or `notebooks/` — EDA, feature pipelines, model training (temporal CV only)
- `models/` — saved pipelines / xgb models
- `reports/` — this doc + model performance + "earlier patch" analysis figures/tables

## Recommendations for further use
- Always respect temporal order for splits/CV (no leakage from future into past).
- Aggregate to grid × time buckets for modeling (cell-day or cell-week) to reduce sparsity.
- Consider enriching with external open data: daily weather (rain/frost), road class (A/B/C/unclassified), traffic estimates, or historical repair logs if DfI ever releases them.
- For operational "solve quicker": the model output can feed a ranked worklist for inspectors/crews: "these 50 cells have elevated probability of generating complaints or re-complaints in the next fortnight — inspect/patch proactively."

**Last updated in this doc**: 2026-06 (analysis of 2016 file)
