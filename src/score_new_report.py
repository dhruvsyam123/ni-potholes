#!/usr/bin/env python3
"""
Tiny demo: load the trained pipeline and score a brand-new (fake) pothole enquiry.

This shows exactly how you would plug the model into an operational triage
workflow: when a new report lands, featurize it the same way, run predict_proba,
and use the "probability it will generate a repeat complaint inside 200m in
the next 14 days" as a priority / "act fast on this one" score.
"""
import joblib
import pandas as pd
from datetime import datetime

# Load the saved HGB pipeline (includes the ColumnTransformer)
pipe = joblib.load("models/recurrence_risk_hgb.joblib")

# Example new report that just came in (you would construct this from your intake form / system)
dt = pd.Timestamp("2017-03-15 09:22:00")
new_report = pd.DataFrame([{
    "date_recorded": dt,
    "lon": -5.95,
    "lat": 54.60,
    "enquiry_category": "Public Enquiry",
    "division": "EASTERN DIVISION",
    "client_office": "ARDS & NORTH DOWN",
    "month": str(dt.month),
    "dayofweek": str(dt.dayofweek),
    "hour": dt.hour,
    "is_weekend": int(dt.dayofweek >= 5),
    "is_winter_spring": int(dt.month in (1,2,3,4,11,12)),
    "lag_7d_same_cell": 2,          # you would compute this from recent history in the same cell
    "lag_14d_same_cell": 3,
    "days_since_apr2016": (dt - pd.Timestamp("2016-04-01")).days,
}])

# The pipeline expects the columns it was trained on (cats + the nums we used)
# We only need to supply the ones the preproc uses; extra cols are ignored by the pipeline.
proba = pipe.predict_proba(new_report)[:, 1][0]
print(f"Recurrence risk (P(repeat within 14d / 200m)): {proba:.1%}")
print("Interpretation: if this is high (e.g. >0.30-0.40), consider fast-tracking a quality repair")
print("or root-cause inspection for this location to try to break the repeat/decay cycle.")
