#!/usr/bin/env python3
"""
Download official DfI Road Network geometry from the public ArcGIS FeatureServer.

Data source discovered from the DfI Road Network Public Viewer:
https://dfi-ni.maps.arcgis.com/apps/webappviewer/index.html?id=f8a42fc35a3d48788e651a1d47865ce1

Layer:
https://services1.arcgis.com/i8LHQZrSk9zIffRU/arcgis/rest/services/DFI_Road_Network/FeatureServer/0

The layer supports GeoJSON export and pagination. This script writes one
FeatureCollection containing the road-section polylines and the join fields used
by the pothole surface-defect data.
"""
import argparse
import json
import os
import time
from urllib.parse import urlencode
from urllib.request import urlopen


SERVICE_URL = (
    "https://services1.arcgis.com/i8LHQZrSk9zIffRU/arcgis/rest/services/"
    "DFI_Road_Network/FeatureServer/0/query"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/dfi_road_network.geojson")
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--sleep", type=float, default=0.05)
    return parser.parse_args()


def fetch_json(params):
    url = SERVICE_URL + "?" + urlencode(params)
    with urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    base_params = {
        "where": "1=1",
        "outFields": ",".join(
            [
                "OBJECTID",
                "Section_Code",
                "SECTION_NA",
                "DIVISION_N",
                "SECTION_OF",
                "CLASS_NAME",
                "ADOPTION_S",
                "SECTION_TY",
                "Shape__Length",
            ]
        ),
        "returnGeometry": "true",
        "f": "geojson",
        "orderByFields": "OBJECTID",
    }

    count = fetch_json({**base_params, "returnCountOnly": "true", "f": "json"})["count"]
    print(f"Downloading {count:,} DfI road-network features...")

    features = []
    offset = 0
    while offset < count:
        params = {
            **base_params,
            "resultOffset": offset,
            "resultRecordCount": args.batch_size,
        }
        batch = fetch_json(params)
        batch_features = batch.get("features", [])
        features.extend(batch_features)
        offset += len(batch_features)
        print(f"  {offset:,}/{count:,}")
        if not batch_features:
            break
        time.sleep(args.sleep)

    collection = {
        "type": "FeatureCollection",
        "name": "DFI_Road_Network",
        "features": features,
        "source": "DfI Road Network FeatureServer",
        "source_url": SERVICE_URL,
    }
    with open(args.out, "w") as f:
        json.dump(collection, f)
    print(f"Wrote {args.out} ({len(features):,} features)")


if __name__ == "__main__":
    main()
