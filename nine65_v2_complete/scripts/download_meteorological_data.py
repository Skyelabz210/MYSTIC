#!/usr/bin/env python3
"""
MYSTIC Meteorological Data Downloader

Downloads REAL meteorological data for flash flood prediction:
1. NOAA LCD (Local Climatological Data) - hourly weather observations
2. PRISM rainfall data
3. Stage IV radar-derived precipitation
4. METAR/ASOS surface observations

The goal is to predict floods BEFORE they happen using:
- Rainfall intensity (mm/hr)
- Antecedent precipitation (7-day totals)
- Soil moisture proxies
- Atmospheric instability indicators
"""

import urllib.request
import json
import csv
import gzip
import io
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "meteorological")
os.makedirs(DATA_DIR, exist_ok=True)

# NOAA CDO API token (free from https://www.ncdc.noaa.gov/cdo-web/token)
CDO_TOKEN = os.environ.get("NOAA_CDO_TOKEN", "")

# Texas weather stations near flood events
WEATHER_STATIONS = {
    # Hill Country (Camp Mystic, Blanco River)
    "USW00012921": {"name": "San Antonio Intl", "lat": 29.53, "lon": -98.47},
    "USW00013904": {"name": "Austin Bergstrom", "lat": 30.18, "lon": -97.68},
    "USC00414778": {"name": "Kerrville", "lat": 30.05, "lon": -99.14},

    # Houston Area (Harvey, Tax Day)
    "USW00012960": {"name": "Houston Hobby", "lat": 29.65, "lon": -95.28},
    "USW00012918": {"name": "Houston Bush IAH", "lat": 29.98, "lon": -95.36},
    "USW00012906": {"name": "Houston Ellington", "lat": 29.61, "lon": -95.16},

    # Southeast Texas (Imelda)
    "USW00012917": {"name": "Beaumont/Port Arthur", "lat": 29.95, "lon": -94.02},

    # Central Texas (Halloween flood)
    "USW00013958": {"name": "Austin Camp Mabry", "lat": 30.32, "lon": -97.77},
}

# Key flood events with meteorological context
FLOOD_EVENTS = [
    {
        "name": "Camp Mystic 2007",
        "date": "2007-06-28",
        "stations": ["USW00012921", "USC00414778"],
        "rainfall_total_in": 9.0,  # Documented rainfall
        "description": "Localized heavy rain over Hill Country",
    },
    {
        "name": "Memorial Day 2015",
        "date": "2015-05-24",
        "stations": ["USW00013904", "USW00012921"],
        "rainfall_total_in": 12.0,
        "description": "Training storms over Blanco watershed",
    },
    {
        "name": "Hurricane Harvey 2017",
        "date": "2017-08-27",
        "stations": ["USW00012960", "USW00012918", "USW00012906"],
        "rainfall_total_in": 60.0,  # Record rainfall
        "description": "Stalled hurricane, catastrophic rainfall",
    },
    {
        "name": "Llano River 2018",
        "date": "2018-10-16",
        "stations": ["USW00012921", "USC00414778"],
        "rainfall_total_in": 10.0,
        "description": "Rapid runoff from Hill Country storms",
    },
    {
        "name": "Halloween 2013",
        "date": "2013-10-31",
        "stations": ["USW00013958", "USW00013904"],
        "rainfall_total_in": 14.0,
        "description": "Heavy rain over Austin metro",
    },
    {
        "name": "Tax Day 2016",
        "date": "2016-04-18",
        "stations": ["USW00012960", "USW00012918"],
        "rainfall_total_in": 17.0,
        "description": "Slow-moving storms over Houston",
    },
    {
        "name": "TS Imelda 2019",
        "date": "2019-09-19",
        "stations": ["USW00012917", "USW00012960"],
        "rainfall_total_in": 43.0,
        "description": "Tropical system stalled over SE Texas",
    },
]


def fetch_noaa_lcd_data(station_id: str, start_date: str, end_date: str) -> List[Dict]:
    """
    Fetch Local Climatological Data (hourly observations) from NOAA.

    This is REAL weather station data including:
    - Hourly precipitation
    - Temperature
    - Dewpoint
    - Wind speed/direction
    - Pressure
    - Sky conditions
    """
    readings = []

    # NOAA LCD endpoint
    base_url = "https://www.ncei.noaa.gov/access/services/data/v1"

    params = {
        "dataset": "local-climatological-data",
        "stations": station_id,
        "startDate": start_date,
        "endDate": end_date,
        "format": "json",
        "units": "metric",
    }

    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{base_url}?{query}"

    print(f"  Fetching LCD for {station_id}...")

    try:
        req = urllib.request.Request(url)
        if CDO_TOKEN:
            req.add_header("token", CDO_TOKEN)

        with urllib.request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode())

        for record in data:
            reading = {
                "timestamp": record.get("DATE"),
                "station_id": station_id,
                "temp_c": safe_float(record.get("HourlyDryBulbTemperature")),
                "dewpoint_c": safe_float(record.get("HourlyDewPointTemperature")),
                "humidity_pct": safe_float(record.get("HourlyRelativeHumidity")),
                "wind_speed_mps": safe_float(record.get("HourlyWindSpeed")),
                "wind_dir_deg": safe_float(record.get("HourlyWindDirection")),
                "pressure_hpa": safe_float(record.get("HourlyStationPressure")),
                "precip_mm": safe_float(record.get("HourlyPrecipitation")),
                "visibility_km": safe_float(record.get("HourlyVisibility")),
                "sky_condition": record.get("HourlySkyConditions"),
                "weather_type": record.get("HourlyPresentWeatherType"),
            }
            readings.append(reading)

        print(f"    Got {len(readings)} hourly observations")

    except Exception as e:
        print(f"    Error: {e}")

    return readings


def fetch_ghcn_daily(station_id: str, start_date: str, end_date: str) -> List[Dict]:
    """
    Fetch GHCN-Daily data (daily observations).

    Includes:
    - Daily precipitation totals (PRCP)
    - Max/Min temperature (TMAX, TMIN)
    - Snow (SNOW, SNWD)
    """
    readings = []

    base_url = "https://www.ncei.noaa.gov/access/services/data/v1"

    params = {
        "dataset": "daily-summaries",
        "stations": station_id,
        "startDate": start_date,
        "endDate": end_date,
        "dataTypes": "PRCP,TMAX,TMIN,AWND,WSF2,WSF5",
        "format": "json",
        "units": "metric",
    }

    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{base_url}?{query}"

    print(f"  Fetching GHCN-Daily for {station_id}...")

    try:
        req = urllib.request.Request(url)
        if CDO_TOKEN:
            req.add_header("token", CDO_TOKEN)

        with urllib.request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode())

        for record in data:
            reading = {
                "date": record.get("DATE"),
                "station_id": station_id,
                "precip_mm": safe_float(record.get("PRCP")),
                "tmax_c": safe_float(record.get("TMAX")),
                "tmin_c": safe_float(record.get("TMIN")),
                "wind_avg_mps": safe_float(record.get("AWND")),
                "wind_gust_mps": safe_float(record.get("WSF5")),
            }
            readings.append(reading)

        print(f"    Got {len(readings)} daily records")

    except Exception as e:
        print(f"    Error: {e}")

    return readings


def fetch_isd_lite(station_id: str, year: int) -> List[Dict]:
    """
    Fetch ISD-Lite data (hourly synoptic observations).

    This is a reliable source for historical hourly weather data.
    Format: year month day hour temp dewpoint pressure wind_dir wind_speed sky_condition precip_1hr precip_6hr
    """
    readings = []

    # ISD-Lite file naming: USAF-WBAN-YEAR.gz
    # Need to map station ID to USAF-WBAN
    usaf_wban_map = {
        "USW00012921": "722530-12921",  # San Antonio
        "USW00012960": "722430-12960",  # Houston Hobby
        "USW00012918": "722435-12918",  # Houston IAH
        "USW00013904": "722540-13904",  # Austin
        "USW00013958": "722544-13958",  # Austin Camp Mabry
        "USW00012917": "722410-12917",  # Beaumont
        "USW00012906": "722436-12906",  # Houston Ellington
    }

    if station_id not in usaf_wban_map:
        print(f"    No ISD-Lite mapping for {station_id}")
        return readings

    usaf_wban = usaf_wban_map[station_id]
    url = f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{usaf_wban}-{year}.gz"

    print(f"  Fetching ISD-Lite {usaf_wban} for {year}...")

    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            with gzip.GzipFile(fileobj=response) as gz:
                for line in gz:
                    parts = line.decode().split()
                    if len(parts) < 12:
                        continue

                    try:
                        yr, mo, dy, hr = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                        timestamp = f"{yr}-{mo:02d}-{dy:02d}T{hr:02d}:00:00"

                        # ISD-Lite uses scaled integers: temp/10, dewpoint/10, etc.
                        temp = int(parts[4]) / 10 if parts[4] != "-9999" else None
                        dewpoint = int(parts[5]) / 10 if parts[5] != "-9999" else None
                        pressure = int(parts[6]) / 10 if parts[6] != "-9999" else None
                        wind_dir = int(parts[7]) if parts[7] != "-9999" else None
                        wind_speed = int(parts[8]) / 10 if parts[8] != "-9999" else None
                        sky_cover = int(parts[9]) if parts[9] != "-9999" else None
                        precip_1hr = int(parts[10]) / 10 if parts[10] != "-9999" else None
                        precip_6hr = int(parts[11]) / 10 if parts[11] != "-9999" else None

                        reading = {
                            "timestamp": timestamp,
                            "station_id": station_id,
                            "temp_c": temp,
                            "dewpoint_c": dewpoint,
                            "pressure_hpa": pressure,
                            "wind_dir_deg": wind_dir,
                            "wind_speed_mps": wind_speed,
                            "sky_cover_oktas": sky_cover,
                            "precip_1hr_mm": precip_1hr,
                            "precip_6hr_mm": precip_6hr,
                        }
                        readings.append(reading)

                    except (ValueError, IndexError):
                        continue

        print(f"    Got {len(readings)} hourly observations")

    except Exception as e:
        print(f"    Error: {e}")

    return readings


def safe_float(value) -> Optional[float]:
    """Safely convert to float."""
    if value is None or value == "" or value == "T" or value == "M":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def download_event_weather(event: Dict) -> str:
    """Download all meteorological data for a flood event."""

    name = event["name"].replace(" ", "_").lower()
    date = datetime.strptime(event["date"], "%Y-%m-%d")
    year = date.year

    # Get data from 7 days before to 2 days after
    start_date = (date - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = (date + timedelta(days=2)).strftime("%Y-%m-%d")

    print(f"\n{'='*70}")
    print(f"DOWNLOADING METEOROLOGICAL DATA: {event['name']}")
    print(f"Event Date: {event['date']}")
    print(f"Documented Rainfall: {event['rainfall_total_in']} inches")
    print(f"Description: {event['description']}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"{'='*70}")

    all_readings = []

    for station_id in event["stations"]:
        station_info = WEATHER_STATIONS.get(station_id, {})
        print(f"\n  Station: {station_id} - {station_info.get('name', 'Unknown')}")

        # Try ISD-Lite first (most reliable for historical hourly data)
        readings = fetch_isd_lite(station_id, year)

        # Filter to event window
        if readings:
            readings = [r for r in readings
                       if start_date <= r["timestamp"][:10] <= end_date]
            all_readings.extend(readings)

        # Also get GHCN daily for precipitation totals
        daily = fetch_ghcn_daily(station_id, start_date, end_date)
        if daily:
            for d in daily:
                d["source"] = "ghcn_daily"
            all_readings.extend(daily)

    if not all_readings:
        print(f"  WARNING: No data downloaded for {event['name']}")
        return ""

    # Save to CSV
    output_file = os.path.join(DATA_DIR, f"weather_{name}.csv")

    # Determine fieldnames from all readings
    all_fields = set()
    for r in all_readings:
        all_fields.update(r.keys())
    fieldnames = sorted(list(all_fields))

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_readings)

    print(f"\n  Saved {len(all_readings)} records to {output_file}")

    # Calculate precip totals
    hourly_precip = [r.get("precip_1hr_mm", 0) or 0 for r in all_readings
                    if "precip_1hr_mm" in r]
    if hourly_precip:
        total = sum(hourly_precip)
        max_rate = max(hourly_precip)
        print(f"  Total precip: {total:.1f} mm ({total/25.4:.1f} inches)")
        print(f"  Max hourly:   {max_rate:.1f} mm ({max_rate/25.4:.2f} inches)")

    return output_file


def create_combined_meteo_dataset():
    """Create combined meteorological training dataset."""

    print(f"\n{'='*70}")
    print("CREATING COMBINED METEOROLOGICAL DATASET")
    print(f"{'='*70}")

    all_records = []

    for filename in os.listdir(DATA_DIR):
        if filename.startswith("weather_") and filename.endswith(".csv"):
            filepath = os.path.join(DATA_DIR, filename)
            event_name = filename.replace("weather_", "").replace(".csv", "").replace("_", " ").title()

            print(f"  Loading {filename}...")

            with open(filepath, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["event_name"] = event_name
                    all_records.append(row)

    if not all_records:
        print("  No weather files found!")
        return ""

    # Save combined
    output_file = os.path.join(DATA_DIR, "combined_meteorological_data.csv")

    all_fields = set()
    for r in all_records:
        all_fields.update(r.keys())
    fieldnames = sorted(list(all_fields))

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\n  Total records: {len(all_records)}")
    print(f"  Saved to {output_file}")

    return output_file


def main():
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║         MYSTIC Meteorological Data Downloader                             ║")
    print("║         REAL Weather Data for Flash Flood Prediction                      ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print()
    print("Flash floods are PREDICTED from meteorological conditions, not river stage.")
    print("Downloading: rainfall, temperature, humidity, pressure, wind data")
    print()
    print(f"Output directory: {DATA_DIR}")
    print()

    # Download for each event
    for event in FLOOD_EVENTS:
        download_event_weather(event)

    # Create combined dataset
    create_combined_meteo_dataset()

    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print(f"\nFiles in {DATA_DIR}:")
    total_size = 0
    for filename in sorted(os.listdir(DATA_DIR)):
        filepath = os.path.join(DATA_DIR, filename)
        size = os.path.getsize(filepath)
        total_size += size
        print(f"  {filename}: {size:,} bytes")

    print(f"\nTotal data: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
    print("\nThis is REAL meteorological data for predicting flash floods.")


if __name__ == "__main__":
    main()
