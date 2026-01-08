#!/usr/bin/env python3
"""
MYSTIC Data Fetcher - USGS Stream Gauge Data

Downloads historical stream gauge and weather data from USGS NWIS
for Texas stream gauge stations (default: statewide).

Data is converted to MYSTIC training format for flood attractor learning.
"""

import urllib.request
import json
import csv
import gzip
import io
import math
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

DEFAULT_STATE = os.environ.get("MYSTIC_STATE", "TX")

# Seed stations near Camp Mystic; can be extended with statewide loading.
DEFAULT_STATIONS = {
    "08166200": "Guadalupe River at Kerrville, TX",
    "08165500": "Guadalupe River at Spring Branch, TX",
    "08167000": "Guadalupe River near Comfort, TX",
}

DEFAULT_LAT = float(os.environ.get("MYSTIC_LAT", "29.4"))
DEFAULT_LON = float(os.environ.get("MYSTIC_LON", "-98.5"))
DEFAULT_RADIUS_KM = float(os.environ.get("MYSTIC_RADIUS_KM", "150"))
DEFAULT_PRECURSOR_HOURS = float(os.environ.get("MYSTIC_PRECURSOR_HOURS", "6"))
DEFAULT_EVENT_TYPES = [
    value.strip()
    for value in os.environ.get("MYSTIC_STORMEVENT_TYPES", "Flash Flood,Flood,Heavy Rain").split(",")
    if value.strip()
]
DEFAULT_START_YEAR = int(os.environ.get("MYSTIC_STORMEVENT_START_YEAR", "1950"))
USGS_STATEWIDE = os.environ.get("MYSTIC_USGS_STATEWIDE", "1") != "0"
USGS_STATION_LIMIT = int(os.environ.get("MYSTIC_USGS_STATION_LIMIT", "0"))
STORMEVENT_STATEWIDE = os.environ.get("MYSTIC_STORMEVENT_STATEWIDE", "1") != "0"
INCLUDE_EVENT_WINDOWS = os.environ.get("MYSTIC_INCLUDE_EVENT_WINDOWS", "1") != "0"
EVENT_WINDOW_DAYS_BEFORE = int(os.environ.get("MYSTIC_EVENT_WINDOW_DAYS_BEFORE", "3"))
EVENT_WINDOW_DAYS_AFTER = int(os.environ.get("MYSTIC_EVENT_WINDOW_DAYS_AFTER", "1"))
EVENT_WINDOW_LIMIT = int(os.environ.get("MYSTIC_EVENT_WINDOW_LIMIT", "20"))

# NWIS parameter codes
PARAMS = {
    "00060": "stream_discharge_cfs",  # Discharge (cubic feet per second)
    "00065": "stream_height_ft",      # Gage height (feet)
    "00045": "precip_total_in",       # Precipitation total
    "00021": "temp_water_c",          # Water temperature
}

EVENT_TYPE_MAP = {
    "Flash Flood": "flash_flood",
    "Flood": "major_flood",
    "Heavy Rain": "watch",
}

STORMEVENT_INDEX = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"


def safe_float(value: Optional[str]) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_event_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%d-%b-%y %H:%M:%S", "%d-%b-%Y %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def list_storm_event_files() -> List[str]:
    with urllib.request.urlopen(STORMEVENT_INDEX, timeout=30) as response:
        text = response.read().decode()
    pattern = re.compile(r'href="(StormEvents_details-ftp_v1.0_d\\d{4}_c\\d{8}\\.csv\\.gz)"')
    return pattern.findall(text)


def select_storm_event_files(years: List[int]) -> List[str]:
    files = list_storm_event_files()
    best_by_year: Dict[int, Tuple[str, str]] = {}
    pattern = re.compile(r"StormEvents_details-ftp_v1.0_d(\\d{4})_c(\\d{8})\\.csv\\.gz")
    for filename in files:
        match = pattern.match(filename)
        if not match:
            continue
        year = int(match.group(1))
        cdate = match.group(2)
        if year not in years:
            continue
        if year not in best_by_year or cdate > best_by_year[year][1]:
            best_by_year[year] = (filename, cdate)
    return [best_by_year[year][0] for year in sorted(best_by_year.keys())]


def normalize_event_type(event_type: str) -> str:
    return EVENT_TYPE_MAP.get(event_type, event_type.lower().replace(" ", "_"))


def load_usgs_state_stations(state_code: str, param_code: str = "00065") -> Dict[str, str]:
    url = (
        "https://waterservices.usgs.gov/nwis/site/"
        f"?format=rdb&stateCd={state_code}&parameterCd={param_code}&siteStatus=active"
    )
    stations: Dict[str, str] = {}
    with urllib.request.urlopen(url, timeout=60) as response:
        text = response.read().decode()
    lines = [line for line in text.splitlines() if line.strip()]
    header = None
    skip_types = False
    for line in lines:
        if line.startswith("#"):
            continue
        if header is None:
            header = line.split("\t")
            skip_types = True
            continue
        if skip_types:
            skip_types = False
            continue
        parts = line.split("\t")
        if header and len(parts) >= len(header):
            row = dict(zip(header, parts))
            site_no = row.get("site_no")
            station_nm = row.get("station_nm")
            if site_no and station_nm:
                stations[site_no] = station_nm
    return stations


def load_storm_events_from_ftp(state: str,
                               years: List[int],
                               event_types: List[str],
                               lat: float,
                               lon: float,
                               radius_km: float,
                               statewide: bool) -> List[Dict]:
    files = select_storm_event_files(years)
    events: List[Dict] = []
    for filename in files:
        url = f"{STORMEVENT_INDEX}{filename}"
        print(f"  • Downloading {filename}")
        with urllib.request.urlopen(url, timeout=60) as response:
            with gzip.GzipFile(fileobj=response) as gz:
                text_stream = io.TextIOWrapper(gz, encoding="utf-8", errors="replace")
                reader = csv.DictReader(text_stream)
                for row in reader:
                    if row.get("STATE") != state:
                        continue
                    event_type = row.get("EVENT_TYPE")
                    if event_type not in event_types:
                        continue

                    begin_lat = safe_float(row.get("BEGIN_LAT"))
                    begin_lon = safe_float(row.get("BEGIN_LON"))
                    distance = None
                    if not statewide and begin_lat is not None and begin_lon is not None:
                        distance = haversine_km(lat, lon, begin_lat, begin_lon)
                        if distance > radius_km:
                            continue

                    begin_dt = parse_event_datetime(row.get("BEGIN_DATE_TIME"))
                    end_dt = parse_event_datetime(row.get("END_DATE_TIME"))

                    if begin_dt is None:
                        begin_date = row.get("BEGIN_DATE")
                        begin_time = row.get("BEGIN_TIME")
                        if begin_date and begin_time:
                            begin_time = begin_time.zfill(4)
                            begin_dt = datetime.strptime(f"{begin_date} {begin_time}", "%m/%d/%Y %H%M")

                    if end_dt is None:
                        end_date = row.get("END_DATE")
                        end_time = row.get("END_TIME")
                        if end_date and end_time:
                            end_time = end_time.zfill(4)
                            end_dt = datetime.strptime(f"{end_date} {end_time}", "%m/%d/%Y %H%M")

                    if begin_dt is None:
                        continue
                    if end_dt is None:
                        end_dt = begin_dt + timedelta(hours=6)

                    events.append({
                        "event_id": row.get("EVENT_ID"),
                        "event_type": event_type,
                        "normalized_type": normalize_event_type(event_type),
                        "state": row.get("STATE"),
                        "cz_name": row.get("CZ_NAME"),
                        "begin_time": begin_dt.isoformat(),
                        "end_time": end_dt.isoformat(),
                        "begin_lat": begin_lat,
                        "begin_lon": begin_lon,
                        "distance_km": distance,
                        "source": filename,
                    })
    return events


def fetch_event_window_readings(station_id: str, start_dt: datetime, end_dt: datetime) -> List[Dict]:
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    data = fetch_usgs_data(station_id, start_str, end_str, "00065", service="iv")
    readings = parse_usgs_timeseries(data) if data else []

    if not readings:
        data = fetch_usgs_data(station_id, start_str, end_str, "00065", service="dv")
        readings = parse_usgs_timeseries(data) if data else []

    return readings


def fetch_usgs_data(station_id: str,
                    start_date: str,
                    end_date: str,
                    param_code: str = "00065",
                    service: str = "iv") -> Optional[Dict]:
    """
    Fetch data from USGS NWIS for a given station and date range.

    Args:
        station_id: USGS station ID (e.g., "08166200")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        param_code: USGS parameter code (default: 00065 = gage height)

    Returns:
        Dictionary with time series data or None if request fails
    """
    base_url = f"https://waterservices.usgs.gov/nwis/{service}/"

    # Build query parameters
    params = {
        "format": "json",
        "sites": station_id,
        "startDT": start_date,
        "endDT": end_date,
        "parameterCd": param_code,
        "siteStatus": "all"
    }

    # Construct URL
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    url = f"{base_url}?{query_string}"

    print(f"Fetching {station_id} - {PARAMS.get(param_code, param_code)} ({service})")
    print(f"  URL: {url[:80]}...")

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode())
            return data
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def parse_usgs_timeseries(data: Dict) -> List[Dict]:
    """
    Parse USGS JSON response into flat list of readings.

    Args:
        data: USGS JSON response

    Returns:
        List of {timestamp, value, station_id, parameter} dictionaries
    """
    readings = []

    try:
        time_series = data["value"]["timeSeries"]

        for ts in time_series:
            station_id = ts["sourceInfo"]["siteCode"][0]["value"]
            param_code = ts["variable"]["variableCode"][0]["value"]
            param_name = PARAMS.get(param_code, param_code)

            for value_entry in ts["values"][0]["value"]:
                timestamp = value_entry["dateTime"]
                value = float(value_entry["value"])

                readings.append({
                    "timestamp": timestamp,
                    "station_id": station_id,
                    "parameter": param_name,
                    "value": value,
                    "param_code": param_code
                })

        print(f"  Parsed {len(readings)} readings")
        return readings

    except (KeyError, IndexError, ValueError) as e:
        print(f"  ERROR parsing: {e}")
        return []


def fetch_nws_flood_events(state: str = DEFAULT_STATE,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           start_year: int = DEFAULT_START_YEAR) -> List[Dict]:
    """
    Fetch historical flash flood events from NOAA Storm Events Database.

    Uses NCEI Storm Events CSV (free, no token) and filters for Texas Hill Country.
    Falls back to known local events if download fails.
    """
    known_events = [
        {
            "event": "Camp Mystic Flash Flood",
            "location": "Kerr County, TX",
            "begin_time": "2007-06-28T14:00:00",
            "end_time": "2007-06-28T20:00:00",
            "event_type": "Flash Flood",
            "deaths": 3,
            "description": "Camp on Guadalupe River",
            "source": "manual"
        },
        {
            "event": "Memorial Day Flood",
            "location": "Wimberley, TX",
            "begin_time": "2015-05-23T01:00:00",
            "end_time": "2015-05-24T01:00:00",
            "event_type": "Flash Flood",
            "deaths": 13,
            "description": "Blanco River flash flood",
            "source": "manual"
        },
        {
            "event": "Halloween Flood",
            "location": "San Antonio, TX",
            "begin_time": "2013-10-30T06:00:00",
            "end_time": "2013-10-30T20:00:00",
            "event_type": "Flash Flood",
            "deaths": 4,
            "description": "Salado Creek flash flood",
            "source": "manual"
        },
        {
            "event": "Llano River Flash Flood",
            "location": "Llano County, TX",
            "begin_time": "2018-10-16T02:00:00",
            "end_time": "2018-10-16T18:00:00",
            "event_type": "Flash Flood",
            "deaths": 9,
            "description": "Record river rise",
            "source": "manual"
        }
    ]

    years: List[int] = []
    years_env = os.environ.get("MYSTIC_STORMEVENT_YEARS")
    if years_env:
        years = [int(y.strip()) for y in years_env.split(",") if y.strip().isdigit()]
    elif start_date and end_date:
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            years = list(range(start_dt.year, end_dt.year + 1))
        except ValueError:
            years = []

    if not years:
        current_year = datetime.now().year
        years = list(range(start_year, current_year + 1))

    events: List[Dict] = []
    try:
        print("Fetching NOAA Storm Events data (NCEI CSV)...")
        events = load_storm_events_from_ftp(
            state=state,
            years=years,
            event_types=DEFAULT_EVENT_TYPES,
            lat=DEFAULT_LAT,
            lon=DEFAULT_LON,
            radius_km=DEFAULT_RADIUS_KM,
            statewide=STORMEVENT_STATEWIDE,
        )
    except Exception as e:
        print(f"⚠ Storm Events download failed: {e}")

    # Always include known historical events as a baseline
    for event in known_events:
        event = event.copy()
        event["normalized_type"] = normalize_event_type(event["event_type"])
        events.append(event)

    # Normalize begin/end times to datetime objects for downstream labeling
    for event in events:
        begin_dt = parse_event_datetime(event.get("begin_time"))
        if begin_dt is None and event.get("begin_time"):
            try:
                begin_dt = datetime.fromisoformat(event["begin_time"])
            except ValueError:
                begin_dt = None
        end_dt = parse_event_datetime(event.get("end_time"))
        if end_dt is None and event.get("end_time"):
            try:
                end_dt = datetime.fromisoformat(event["end_time"])
            except ValueError:
                end_dt = None
        if begin_dt and not end_dt:
            end_dt = begin_dt + timedelta(hours=6)
        event["begin_dt"] = begin_dt
        event["end_dt"] = end_dt

    print(f"Texas Hill Country flood events: {len(events)}")
    for event in events[:10]:
        begin = event.get("begin_time", "unknown")
        print(f"  {begin}: {event.get('event', event.get('event_type'))}")

    return events


def classify_event_label(timestamp: str, flood_events: List[Dict], precursor_hours: float) -> str:
    try:
        ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        ts = ts.replace(tzinfo=None)
    except ValueError:
        return "normal"

    for event in flood_events:
        begin_dt = event.get("begin_dt")
        end_dt = event.get("end_dt")
        if begin_dt is None:
            begin_dt = parse_event_datetime(event.get("begin_time"))
        if end_dt is None:
            end_dt = parse_event_datetime(event.get("end_time"))
        if begin_dt is None:
            continue
        if end_dt is None:
            end_dt = begin_dt + timedelta(hours=6)

        if begin_dt - timedelta(hours=precursor_hours) <= ts < begin_dt:
            return "precursor"
        if begin_dt <= ts <= end_dt:
            return event.get("normalized_type", "flash_flood")

    return "normal"


def convert_to_mystic_format(usgs_readings: List[Dict], output_file: str, flood_events: List[Dict]):
    """
    Convert USGS data to MYSTIC training CSV format.

    MYSTIC CSV format:
    timestamp,station_id,temp_c,dewpoint_c,pressure_hpa,wind_mps,rain_mm_hr,soil_pct,stream_cm,event_type
    """
    # Group readings by timestamp
    by_timestamp = {}
    for reading in usgs_readings:
        ts = reading["timestamp"]
        station = reading["station_id"]
        key = (ts, station)

        if key not in by_timestamp:
            by_timestamp[key] = {
                "timestamp": ts,
                "station_id": station,
                "temp_c": 0.0,
                "dewpoint_c": 0.0,
                "pressure_hpa": 1013.0,
                "wind_mps": 0.0,
                "rain_mm_hr": 0.0,
                "soil_pct": 0.0,
                "stream_cm": 0.0,
                "event_type": "normal"
            }

        # Map USGS parameter to MYSTIC field
        param = reading["parameter"]
        value = reading["value"]

        if param == "stream_height_ft":
            # Convert feet to cm
            by_timestamp[key]["stream_cm"] = value * 30.48
        elif param == "precip_total_in":
            # Convert inches to mm/hr (approximation)
            by_timestamp[key]["rain_mm_hr"] = value * 25.4
        elif param == "temp_water_c":
            # Use water temp as proxy for air temp if not available
            by_timestamp[key]["temp_c"] = value

    if flood_events:
        for record in by_timestamp.values():
            record["event_type"] = classify_event_label(
                record["timestamp"], flood_events, DEFAULT_PRECURSOR_HOURS
            )

    # Write to CSV
    fieldnames = [
        "timestamp", "station_id", "temp_c", "dewpoint_c", "pressure_hpa",
        "wind_mps", "rain_mm_hr", "soil_pct", "stream_cm", "event_type"
    ]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in sorted(by_timestamp.values(), key=lambda x: x["timestamp"]):
            writer.writerow(row)

    print(f"Wrote {len(by_timestamp)} records to {output_file}")


def main():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║          MYSTIC Data Fetcher - USGS Stream Gauges         ║")
    print("║       Texas Flash Flood Training Data                     ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

    # Fetch recent data (last 30 days) for all configured stations
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    print(f"Date range: {start_str} to {end_str}")
    print()

    all_readings = []

    # Load statewide stations if enabled
    stations: Dict[str, str] = dict(DEFAULT_STATIONS)
    if USGS_STATEWIDE:
        print("Loading statewide USGS stations (gage height, active)...")
        statewide = load_usgs_state_stations(DEFAULT_STATE)
        if USGS_STATION_LIMIT > 0:
            statewide = dict(list(statewide.items())[:USGS_STATION_LIMIT])
        stations.update(statewide)
        print(f"Total stations loaded: {len(stations)}")
        print()
    else:
        print("USGS statewide loading disabled; using default Hill Country stations.")
        print()

    # Fetch gage height data for each station
    for station_id, name in stations.items():
        data = fetch_usgs_data(station_id, start_str, end_str, "00065")
        if data:
            readings = parse_usgs_timeseries(data)
            all_readings.extend(readings)
        print()

    # Fetch known flood events
    print("─" * 60)
    flood_events = fetch_nws_flood_events(start_date=start_str, end_date=end_str)
    print()

    if INCLUDE_EVENT_WINDOWS and flood_events:
        print("─" * 60)
        print("Fetching historical event windows for training...")
        for idx, event in enumerate(flood_events):
            if EVENT_WINDOW_LIMIT > 0 and idx >= EVENT_WINDOW_LIMIT:
                print(f"  Reached event window limit ({EVENT_WINDOW_LIMIT}); skipping remaining events")
                break
            begin_dt = event.get("begin_dt")
            end_dt = event.get("end_dt") or begin_dt
            if begin_dt is None:
                continue
            window_start = begin_dt - timedelta(days=EVENT_WINDOW_DAYS_BEFORE)
            window_end = end_dt + timedelta(days=EVENT_WINDOW_DAYS_AFTER)
            print(f"  Event window: {event.get('event', event.get('event_type'))}")
            print(f"    {window_start.date()} to {window_end.date()}")
            for station_id in STATIONS:
                readings = fetch_event_window_readings(station_id, window_start, window_end)
                all_readings.extend(readings)
        print()

    # Convert to MYSTIC format
    output_file = "../data/texas_hill_country_usgs.csv"
    print("─" * 60)
    print("Converting to MYSTIC training format...")
    convert_to_mystic_format(all_readings, output_file, flood_events)
    print()

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║                    FETCH COMPLETE                          ║")
    print("╠═══════════════════════════════════════════════════════════╣")
    print(f"║  Total readings: {len(all_readings):>6}                                  ║")
    print(f"║  Flood events:   {len(flood_events):>6}                                  ║")
    print(f"║  Output file:    {output_file:<40} ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()
    print("Next steps:")
    print("  1. Review the CSV file for data quality")
    print("  2. Verify event labeling in event_type column")
    print("  3. Run training script to learn attractor signatures")
    print("  4. Test with MYSTIC demo")


if __name__ == "__main__":
    main()
