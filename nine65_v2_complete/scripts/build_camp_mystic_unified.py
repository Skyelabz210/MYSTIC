#!/usr/bin/env python3
"""
Build unified Camp Mystic dataset with provenance and quality flags.

Inputs:
- camp_mystic_2007_synthetic.csv (baseline timeline)
- camp_mystic_2007_usgs_daily.csv (optional daily USGS values)

Output:
- camp_mystic_2007_unified.csv (same schema + data_quality column)
"""

import csv
import os
from datetime import datetime

SYNTHETIC_FILE = "../data/camp_mystic_2007_synthetic.csv"
USGS_DAILY_FILE = "../data/camp_mystic_2007_usgs_daily.csv"
OUTPUT_FILE = "../data/camp_mystic_2007_unified.csv"


def load_synthetic(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        records = []
        for row in reader:
            row["stream_cm"] = float(row["stream_cm"])
            row["rain_mm_hr"] = float(row["rain_mm_hr"])
            row["soil_pct"] = float(row["soil_pct"])
            row["data_quality"] = row.get("data_quality", "synthetic")
            records.append(row)
        return records


def load_usgs_daily(path):
    if not os.path.exists(path):
        return {}

    daily = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row["date"]
            daily[date] = float(row["gage_height_cm"])
    return daily


def scale_stream_values(records, daily_map):
    if not daily_map:
        for row in records:
            row["data_quality"] = "synthetic"
        return records

    by_date = {}
    for row in records:
        date = row["timestamp"].split("T")[0]
        by_date.setdefault(date, []).append(row["stream_cm"])

    for row in records:
        date = row["timestamp"].split("T")[0]
        if date in daily_map and by_date.get(date):
            avg = sum(by_date[date]) / len(by_date[date])
            if avg > 0:
                scale = daily_map[date] / avg
                row["stream_cm"] = row["stream_cm"] * scale
                row["data_quality"] = "usgs_daily_scaled_synthetic"
            else:
                row["data_quality"] = "synthetic"
        else:
            row["data_quality"] = "synthetic"

    return records


def write_unified(records, output_path):
    fieldnames = [
        "timestamp",
        "station_id",
        "temp_c",
        "dewpoint_c",
        "pressure_hpa",
        "wind_mps",
        "rain_mm_hr",
        "soil_pct",
        "stream_cm",
        "event_type",
        "data_quality",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({
                "timestamp": row["timestamp"],
                "station_id": row["station_id"],
                "temp_c": row["temp_c"],
                "dewpoint_c": row["dewpoint_c"],
                "pressure_hpa": row["pressure_hpa"],
                "wind_mps": row["wind_mps"],
                "rain_mm_hr": row["rain_mm_hr"],
                "soil_pct": row["soil_pct"],
                "stream_cm": f"{row['stream_cm']:.2f}",
                "event_type": row["event_type"],
                "data_quality": row["data_quality"],
            })


def main():
    if not os.path.exists(SYNTHETIC_FILE):
        raise FileNotFoundError(f"Missing synthetic dataset: {SYNTHETIC_FILE}")

    records = load_synthetic(SYNTHETIC_FILE)
    daily_map = load_usgs_daily(USGS_DAILY_FILE)

    records = scale_stream_values(records, daily_map)
    write_unified(records, OUTPUT_FILE)

    print("Unified dataset created:")
    print(f"  {OUTPUT_FILE}")
    print(f"  Records: {len(records)}")
    print(f"  USGS daily values: {'yes' if daily_map else 'no'}")


if __name__ == "__main__":
    main()
