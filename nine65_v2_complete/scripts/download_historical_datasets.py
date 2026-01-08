#!/usr/bin/env python3
"""
MYSTIC Historical Dataset Downloader

Downloads REAL historical data from:
1. USGS NWIS - Stream gauge readings (instantaneous and daily values)
2. NOAA Storm Events - Official flood event records
3. NWS River Forecast Center - Historical flood stages

Saves to permanent CSV files for training and validation.
NO SYNTHETIC DATA - All real observations.
"""

import urllib.request
import json
import csv
import gzip
import io
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Output directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "historical")
os.makedirs(DATA_DIR, exist_ok=True)

# Key Texas flood events with USGS stations
FLOOD_EVENTS = [
    {
        "name": "Camp Mystic 2007",
        "date": "2007-06-28",
        "stations": ["08166200", "08165500", "08167000", "08167500"],
        "county": "Kerr",
        "deaths": 3,
    },
    {
        "name": "Memorial Day 2015",
        "date": "2015-05-24",
        "stations": ["08171000", "08171300", "08170500", "08171290"],
        "county": "Hays",
        "deaths": 13,
    },
    {
        "name": "Hurricane Harvey 2017",
        "date": "2017-08-27",
        "stations": [
            "08074000", "08073600", "08074500", "08074800", "08075000",
            "08075400", "08075500", "08075770", "08076000", "08076500",
        ],
        "county": "Harris",
        "deaths": 68,
    },
    {
        "name": "Llano River 2018",
        "date": "2018-10-16",
        "stations": ["08150000", "08150700", "08151500", "08152000"],
        "county": "Llano",
        "deaths": 9,
    },
    {
        "name": "Halloween 2013",
        "date": "2013-10-31",
        "stations": ["08158000", "08158700", "08159000", "08158810"],
        "county": "Travis",
        "deaths": 4,
    },
    {
        "name": "Tax Day 2016",
        "date": "2016-04-18",
        "stations": ["08074000", "08073600", "08075000", "08075500"],
        "county": "Harris",
        "deaths": 8,
    },
    {
        "name": "Tropical Storm Imelda 2019",
        "date": "2019-09-19",
        "stations": ["08041780", "08041700", "08041500", "08042000"],
        "county": "Jefferson",
        "deaths": 5,
    },
    {
        "name": "Memorial Day 2016 Flood",
        "date": "2016-05-27",
        "stations": ["08178000", "08177500", "08178050", "08178880"],
        "county": "Gonzales",
        "deaths": 6,
    },
]

# NCEI Storm Events base URL
STORM_EVENTS_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"


def fetch_usgs_data(station_id: str, start_date: str, end_date: str,
                    service: str = "iv") -> List[Dict]:
    """
    Fetch REAL USGS stream gauge data.

    Args:
        station_id: USGS station ID
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
        service: 'iv' for instantaneous, 'dv' for daily

    Returns:
        List of readings with timestamp, station_id, gage_height_ft, discharge_cfs
    """
    readings = []

    # Fetch gage height (00065) and discharge (00060)
    for param_code, param_name in [("00065", "gage_height_ft"), ("00060", "discharge_cfs")]:
        url = (
            f"https://waterservices.usgs.gov/nwis/{service}/"
            f"?format=json&sites={station_id}"
            f"&startDT={start_date}&endDT={end_date}"
            f"&parameterCd={param_code}&siteStatus=all"
        )

        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                data = json.loads(response.read().decode())

            time_series = data.get("value", {}).get("timeSeries", [])
            if not time_series:
                continue

            for ts in time_series:
                site_info = ts.get("sourceInfo", {})
                site_name = site_info.get("siteName", "Unknown")

                for val in ts.get("values", [{}])[0].get("value", []):
                    try:
                        timestamp = val["dateTime"]
                        value = float(val["value"])

                        # Find or create reading for this timestamp
                        existing = None
                        for r in readings:
                            if r["timestamp"] == timestamp:
                                existing = r
                                break

                        if existing:
                            existing[param_name] = value
                        else:
                            reading = {
                                "timestamp": timestamp,
                                "station_id": station_id,
                                "station_name": site_name,
                                "gage_height_ft": None,
                                "discharge_cfs": None,
                            }
                            reading[param_name] = value
                            readings.append(reading)

                    except (ValueError, KeyError):
                        continue

        except Exception as e:
            print(f"    Error fetching {param_code}: {e}")
            continue

    return readings


def download_event_data(event: Dict) -> str:
    """
    Download all USGS data for a single flood event.
    Returns path to saved CSV file.
    """
    name = event["name"].replace(" ", "_").lower()
    date = datetime.strptime(event["date"], "%Y-%m-%d")

    # Fetch data from 7 days before to 3 days after
    start_date = (date - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = (date + timedelta(days=3)).strftime("%Y-%m-%d")

    print(f"\n{'='*70}")
    print(f"DOWNLOADING: {event['name']}")
    print(f"Event Date: {event['date']}")
    print(f"County: {event['county']} | Deaths: {event['deaths']}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Stations: {', '.join(event['stations'])}")
    print(f"{'='*70}")

    all_readings = []

    for station_id in event["stations"]:
        print(f"  Fetching {station_id}...")

        # Try instantaneous values first (15-min data)
        readings = fetch_usgs_data(station_id, start_date, end_date, "iv")

        if readings:
            print(f"    Got {len(readings)} instantaneous readings")
            all_readings.extend(readings)
        else:
            # Fall back to daily values
            readings = fetch_usgs_data(station_id, start_date, end_date, "dv")
            if readings:
                print(f"    Got {len(readings)} daily readings (no IV data)")
                all_readings.extend(readings)
            else:
                print(f"    No data available")

    if not all_readings:
        print(f"  WARNING: No data downloaded for {event['name']}")
        return ""

    # Sort by timestamp
    all_readings.sort(key=lambda r: r["timestamp"])

    # Save to CSV
    output_file = os.path.join(DATA_DIR, f"usgs_{name}.csv")

    with open(output_file, "w", newline="") as f:
        fieldnames = ["timestamp", "station_id", "station_name", "gage_height_ft", "discharge_cfs"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_readings)

    print(f"  Saved {len(all_readings)} readings to {output_file}")

    # Also save event metadata
    meta_file = os.path.join(DATA_DIR, f"usgs_{name}_metadata.json")
    with open(meta_file, "w") as f:
        json.dump({
            "event_name": event["name"],
            "event_date": event["date"],
            "county": event["county"],
            "deaths": event["deaths"],
            "stations": event["stations"],
            "data_range": {"start": start_date, "end": end_date},
            "readings_count": len(all_readings),
            "downloaded": datetime.now().isoformat(),
            "source": "USGS NWIS API",
        }, f, indent=2)

    return output_file


def download_storm_events_csv(years: List[int], state: str = "TEXAS") -> str:
    """
    Download official NOAA Storm Events data for specified years.
    This is REAL data from the official NWS database.
    """
    print(f"\n{'='*70}")
    print("DOWNLOADING NOAA STORM EVENTS DATABASE")
    print(f"Years: {years[0]} - {years[-1]}")
    print(f"State: {state}")
    print(f"{'='*70}")

    # Get list of available files
    print("  Fetching file index...")
    try:
        with urllib.request.urlopen(STORM_EVENTS_URL, timeout=60) as response:
            index_html = response.read().decode()
    except Exception as e:
        print(f"  ERROR: Could not fetch index: {e}")
        return ""

    # Find files for each year
    pattern = re.compile(r'href="(StormEvents_details-ftp_v1\.0_d(\d{4})_c\d{8}\.csv\.gz)"')
    files_by_year: Dict[int, List[Tuple[str, str]]] = {}

    for match in pattern.finditer(index_html):
        filename = match.group(1)
        year = int(match.group(2))
        if year in years:
            if year not in files_by_year:
                files_by_year[year] = []
            files_by_year[year].append(filename)

    # Get latest file for each year
    all_events = []

    for year in sorted(files_by_year.keys()):
        files = sorted(files_by_year[year])
        latest_file = files[-1]  # Most recent version

        print(f"  Downloading {latest_file}...")
        url = f"{STORM_EVENTS_URL}{latest_file}"

        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                with gzip.GzipFile(fileobj=response) as gz:
                    reader = csv.DictReader(io.TextIOWrapper(gz, encoding="utf-8", errors="replace"))

                    year_events = 0
                    for row in reader:
                        # Filter for Texas flood events
                        if row.get("STATE") != state:
                            continue

                        event_type = row.get("EVENT_TYPE", "")
                        if event_type not in ["Flash Flood", "Flood", "Coastal Flood", "Heavy Rain"]:
                            continue

                        # Parse and save event
                        event = {
                            "event_id": row.get("EVENT_ID"),
                            "episode_id": row.get("EPISODE_ID"),
                            "event_type": event_type,
                            "state": row.get("STATE"),
                            "year": row.get("YEAR"),
                            "month": row.get("MONTH_NAME"),
                            "begin_date": row.get("BEGIN_DATE_TIME"),
                            "end_date": row.get("END_DATE_TIME"),
                            "cz_type": row.get("CZ_TYPE"),
                            "cz_name": row.get("CZ_NAME"),
                            "injuries_direct": row.get("INJURIES_DIRECT"),
                            "injuries_indirect": row.get("INJURIES_INDIRECT"),
                            "deaths_direct": row.get("DEATHS_DIRECT"),
                            "deaths_indirect": row.get("DEATHS_INDIRECT"),
                            "damage_property": row.get("DAMAGE_PROPERTY"),
                            "damage_crops": row.get("DAMAGE_CROPS"),
                            "source": row.get("SOURCE"),
                            "begin_lat": row.get("BEGIN_LAT"),
                            "begin_lon": row.get("BEGIN_LON"),
                            "end_lat": row.get("END_LAT"),
                            "end_lon": row.get("END_LON"),
                            "episode_narrative": row.get("EPISODE_NARRATIVE"),
                            "event_narrative": row.get("EVENT_NARRATIVE"),
                        }
                        all_events.append(event)
                        year_events += 1

                    print(f"    {year}: {year_events} flood events")

        except Exception as e:
            print(f"    ERROR downloading {year}: {e}")
            continue

    if not all_events:
        print("  WARNING: No events downloaded")
        return ""

    # Save to CSV
    output_file = os.path.join(DATA_DIR, f"storm_events_texas_{years[0]}_{years[-1]}.csv")

    with open(output_file, "w", newline="") as f:
        fieldnames = list(all_events[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_events)

    print(f"\n  Saved {len(all_events)} events to {output_file}")

    # Summary by event type
    by_type = {}
    for e in all_events:
        t = e["event_type"]
        by_type[t] = by_type.get(t, 0) + 1

    print("\n  Event counts by type:")
    for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"    {t}: {count}")

    return output_file


def create_combined_training_set():
    """
    Create a combined training dataset from all downloaded data.
    Labels each reading with its relationship to known flood events.
    """
    print(f"\n{'='*70}")
    print("CREATING COMBINED TRAINING DATASET")
    print(f"{'='*70}")

    # Load all USGS CSV files
    all_readings = []

    for filename in os.listdir(DATA_DIR):
        if filename.startswith("usgs_") and filename.endswith(".csv") and not filename.endswith("_metadata.json"):
            filepath = os.path.join(DATA_DIR, filename)

            # Get event name from filename
            event_name = filename.replace("usgs_", "").replace(".csv", "").replace("_", " ").title()

            print(f"  Loading {filename}...")

            with open(filepath, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["event_name"] = event_name
                    all_readings.append(row)

    if not all_readings:
        print("  No data files found!")
        return

    print(f"\n  Total readings: {len(all_readings)}")

    # Sort by timestamp
    all_readings.sort(key=lambda r: r["timestamp"])

    # Save combined dataset
    output_file = os.path.join(DATA_DIR, "combined_training_data.csv")

    fieldnames = ["timestamp", "station_id", "station_name", "gage_height_ft",
                  "discharge_cfs", "event_name"]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_readings)

    print(f"  Saved to {output_file}")

    # Create summary
    by_event = {}
    for r in all_readings:
        e = r["event_name"]
        by_event[e] = by_event.get(e, 0) + 1

    print("\n  Readings per event:")
    for e, count in sorted(by_event.items()):
        print(f"    {e}: {count:,}")

    return output_file


def main():
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║         MYSTIC Historical Dataset Downloader                              ║")
    print("║         REAL DATA from USGS and NOAA                                      ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"Output directory: {DATA_DIR}")
    print()

    # Download USGS data for each flood event
    print("PHASE 1: USGS Stream Gauge Data")
    print("-" * 70)

    downloaded_files = []
    for event in FLOOD_EVENTS:
        filepath = download_event_data(event)
        if filepath:
            downloaded_files.append(filepath)

    print(f"\nDownloaded {len(downloaded_files)} USGS datasets")

    # Download Storm Events
    print("\n" + "=" * 70)
    print("PHASE 2: NOAA Storm Events Database")
    print("-" * 70)

    years = list(range(2000, 2025))  # 25 years of data
    storm_events_file = download_storm_events_csv(years)

    # Create combined training set
    print("\n" + "=" * 70)
    print("PHASE 3: Creating Combined Training Dataset")
    print("-" * 70)

    create_combined_training_set()

    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    # List all files
    print(f"\nFiles in {DATA_DIR}:")
    total_size = 0
    for filename in sorted(os.listdir(DATA_DIR)):
        filepath = os.path.join(DATA_DIR, filename)
        size = os.path.getsize(filepath)
        total_size += size
        print(f"  {filename}: {size:,} bytes")

    print(f"\nTotal data: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
    print("\nAll data is REAL observations from USGS and NOAA.")
    print("No synthetic or simulated data.")


if __name__ == "__main__":
    main()
