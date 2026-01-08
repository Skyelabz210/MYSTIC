#!/usr/bin/env python3
"""
MYSTIC Omniscient Data Integrator

Fetches ALL available data sources from terrestrial to celestial scales:
- Weather: NOAA, NWS, NEXRAD radar, satellites
- Seismic: USGS earthquakes, tremors
- Atmospheric: Pressure, temperature, humidity
- Ocean: Sea surface temp, currents, tides
- Space Weather: Solar flares, geomagnetic storms, cosmic rays
- Planetary: Moon phase, planetary positions
- Gravitational: Tidal forces

Goal: Multi-scale pattern detection for disaster prediction
"""

import urllib.request
import urllib.error
import urllib.parse
import json
import os
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

print("╔═══════════════════════════════════════════════════════════════════╗")
print("║         MYSTIC OMNISCIENT DATA INTEGRATOR                          ║")
print("║    From Seismic Tremors to Solar Flares - All Scales Unified      ║")
print("╚═══════════════════════════════════════════════════════════════════╝")
print()

# ============================================================================
# CONFIG
# ============================================================================

DEFAULT_LAT = float(os.environ.get("MYSTIC_LAT", "29.4"))
DEFAULT_LON = float(os.environ.get("MYSTIC_LON", "-98.5"))
DEFAULT_RADIUS_KM = float(os.environ.get("MYSTIC_RADIUS_KM", "150"))
DEFAULT_NEXRAD_SITES = [
    site.strip().upper()
    for site in os.environ.get("MYSTIC_NEXRAD_SITES", "KEWX,KDFX,KCRP").split(",")
    if site.strip()
]
DEFAULT_BUOY_ID = os.environ.get("MYSTIC_BUOY_ID", "42019")
DEFAULT_TIDE_STATION = os.environ.get("MYSTIC_TIDE_STATION", "8771450")
DEFAULT_CURRENT_STATION = os.environ.get("MYSTIC_CURRENT_STATION", "g09010")
NWS_USER_AGENT = os.environ.get("NWS_USER_AGENT", "MYSTIC/1.0 (contact: local)")
OFFLINE = os.environ.get("MYSTIC_OFFLINE") == "1"

# ============================================================================
# HELPERS
# ============================================================================

def http_get_json(url: str, headers: Optional[Dict[str, str]] = None) -> Dict:
    if OFFLINE:
        raise RuntimeError("offline mode")
    req = urllib.request.Request(url)
    if headers:
        for key, value in headers.items():
            req.add_header(key, value)
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode())


def http_get_text(url: str) -> str:
    if OFFLINE:
        raise RuntimeError("offline mode")
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode()


def bbox_from_radius(lat: float, lon: float, radius_km: float) -> Tuple[float, float, float, float]:
    delta = radius_km / 111.0
    return (lat - delta, lon - delta, lat + delta, lon + delta)


def safe_float(value: Optional[str]) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_wind_speed_mps(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    cleaned = text.replace("mph", " ").replace("to", " ")
    for token in cleaned.split():
        try:
            return float(token) * 0.44704
        except ValueError:
            continue
    return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_horizons_vector(result_text: str) -> Optional[float]:
    if "$$SOE" not in result_text or "$$EOE" not in result_text:
        return None
    block = result_text.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    for line in block.splitlines():
        if "X =" in line and "Y =" in line and "Z =" in line:
            match = re.search(r"X\s*=\s*([-+0-9.Ee]+)\s+Y\s*=\s*([-+0-9.Ee]+)\s+Z\s*=\s*([-+0-9.Ee]+)", line)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                z = float(match.group(3))
                return math.sqrt(x * x + y * y + z * z)
    return None

# ============================================================================
# SCALE 1: TERRESTRIAL - WEATHER & SEISMIC
# ============================================================================

def fetch_noaa_weather_stations(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON, radius: float = DEFAULT_RADIUS_KM):
    """
    Fetch all NOAA weather stations within radius (km) of location.
    Default: San Antonio / Hill Country region
    """
    print("─" * 70)
    print("SCALE 1A: NOAA WEATHER STATIONS")
    print("─" * 70)

    token = os.environ.get("NOAA_CDO_TOKEN")
    if OFFLINE:
        print("⚠ Offline mode - skipping station fetch")
        return {"status": "offline", "count": 0, "stations": []}
    if not token:
        print("⚠ Missing NOAA_CDO_TOKEN - skipping station fetch")
        return {"status": "missing_token", "count": 0, "stations": []}

    print(f"Target: {lat}°N, {lon}°W (radius: {radius} km)")
    print("API: https://www.ncdc.noaa.gov/cdo-web/api/v2/")
    print()

    lat_min, lon_min, lat_max, lon_max = bbox_from_radius(lat, lon, radius)
    extent = f"{lat_min:.3f},{lon_min:.3f},{lat_max:.3f},{lon_max:.3f}"
    url = (
        "https://www.ncdc.noaa.gov/cdo-web/api/v2/stations"
        f"?extent={extent}&datasetid=GHCND&limit=1000"
    )

    try:
        data = http_get_json(url, headers={"token": token})
        stations = data.get("results", [])
        print(f"✓ Retrieved {len(stations)} NOAA stations")
        print()
        return {
            "status": "ok",
            "count": len(stations),
            "stations": [
                {
                    "id": s.get("id"),
                    "name": s.get("name"),
                    "lat": s.get("latitude"),
                    "lon": s.get("longitude"),
                }
                for s in stations[:10]
            ],
        }
    except Exception as e:
        print(f"✗ Error fetching NOAA stations: {e}")
        return {"status": f"error: {e}", "count": 0, "stations": []}


def fetch_noaa_daily_observations(station_id: str, start_date: str, end_date: str):
    """
    Fetch NOAA daily observations (GHCND) for a station.
    """
    token = os.environ.get("NOAA_CDO_TOKEN")
    if OFFLINE:
        return {"status": "offline", "data": []}
    if not token:
        return {"status": "missing_token", "data": []}

    url = (
        "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
        f"?datasetid=GHCND&stationid={station_id}"
        f"&startdate={start_date}&enddate={end_date}&units=metric&limit=1000"
    )

    try:
        data = http_get_json(url, headers={"token": token})
        return {"status": "ok", "data": data.get("results", [])}
    except Exception as e:
        return {"status": f"error: {e}", "data": []}


def fetch_nws_alerts(lat: float, lon: float):
    """
    Fetch active NWS alerts for a point.
    """
    url = f"https://api.weather.gov/alerts/active?point={lat},{lon}"
    headers = {"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"}
    try:
        data = http_get_json(url, headers=headers)
        features = data.get("features", [])
        return {"status": "ok", "count": len(features), "alerts": features[:5]}
    except Exception as e:
        return {"status": f"error: {e}", "count": 0, "alerts": []}


def fetch_nws_forecast(lat: float, lon: float):
    """
    Fetch NWS hourly forecast for a point.
    """
    headers = {"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"}
    try:
        point = http_get_json(f"https://api.weather.gov/points/{lat},{lon}", headers=headers)
        props = point.get("properties", {})
        hourly_url = props.get("forecastHourly")
        if not hourly_url:
            return {"status": "error", "message": "no forecastHourly in point data"}
        hourly = http_get_json(hourly_url, headers=headers)
        periods = hourly.get("properties", {}).get("periods", [])
        if not periods:
            return {"status": "empty", "period": {}}
        period = periods[0]
        temp = period.get("temperature")
        temp_unit = period.get("temperatureUnit", "F")
        temp_c = None
        if temp is not None:
            if temp_unit.upper() == "F":
                temp_c = (temp - 32) * 5.0 / 9.0
            else:
                temp_c = float(temp)
        dewpoint = period.get("dewpoint", {})
        return {
            "status": "ok",
            "period": {
                "start_time": period.get("startTime"),
                "temp_c": temp_c,
                "dewpoint_c": dewpoint.get("value"),
                "wind_mps": parse_wind_speed_mps(period.get("windSpeed")),
                "wind_dir": period.get("windDirection"),
                "precip_prob": period.get("probabilityOfPrecipitation", {}).get("value"),
                "short_forecast": period.get("shortForecast"),
            },
            "source": hourly_url,
        }
    except Exception as e:
        return {"status": f"error: {e}", "period": {}}


def fetch_nexrad_latest(site_id: str, date: datetime):
    """
    Fetch latest NEXRAD Level II object key from NOAA S3 (public).
    """
    if OFFLINE:
        return {"status": "offline", "latest_key": None}
    prefix = f"{date:%Y/%m/%d}/{site_id}/"
    url = f"https://noaa-nexrad-level2.s3.amazonaws.com/?list-type=2&prefix={prefix}"
    try:
        xml_text = http_get_text(url)
        root = ET.fromstring(xml_text)
        keys = [elem.text for elem in root.iter() if elem.tag.endswith("Key")]
        if not keys:
            return {"status": "empty", "latest_key": None}
        latest_key = sorted(keys)[-1]
        return {"status": "ok", "latest_key": latest_key}
    except Exception as e:
        return {"status": f"error: {e}", "latest_key": None}


def fetch_usgs_earthquakes(start_date: str, end_date: str, min_magnitude: float = 2.5,
                           lat: Optional[float] = None, lon: Optional[float] = None):
    """
    Fetch earthquake data from USGS.
    """
    print("─" * 70)
    print("SCALE 1B: USGS SEISMIC DATA")
    print("─" * 70)

    if OFFLINE:
        print("⚠ Offline mode - skipping USGS earthquake fetch")
        return {"status": "offline", "count": 0, "events": [], "nearest": None}

    base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    params = {
        "format": "geojson",
        "starttime": start_date,
        "endtime": end_date,
        "minmagnitude": min_magnitude,
        "orderby": "time"
    }

    query = "&".join([f"{k}={v}" for k, v in params.items()])
    url = f"{base_url}?{query}"

    print(f"Fetching earthquakes: {start_date} to {end_date}")
    print(f"Minimum magnitude: {min_magnitude}")
    print()

    try:
        data = http_get_json(url)
        earthquakes = data.get("features", [])
        print(f"✓ Retrieved {len(earthquakes)} earthquakes")
        print()

        events = []
        nearest = None
        for eq in earthquakes:
            props = eq.get("properties", {})
            coords = eq.get("geometry", {}).get("coordinates", [None, None, None])
            if coords[0] is None or coords[1] is None:
                continue
            event = {
                "mag": props.get("mag"),
                "place": props.get("place"),
                "time": datetime.fromtimestamp(props.get("time", 0) / 1000).isoformat(),
                "lon": coords[0],
                "lat": coords[1],
                "depth_km": coords[2],
            }
            if lat is not None and lon is not None:
                event["distance_km"] = haversine_km(lat, lon, coords[1], coords[0])
                if nearest is None or event["distance_km"] < nearest["distance_km"]:
                    nearest = event
            events.append(event)

        if events:
            print("Recent events:")
            for event in events[:5]:
                print(f"  M{event['mag']} - {event['place']}")
                print(f"    Time: {event['time']}")
                if "distance_km" in event:
                    print(f"    Distance: {event['distance_km']:.1f} km")
            print()

        return {
            "status": "ok",
            "count": len(events),
            "events": events[:10],
            "nearest": nearest,
        }

    except Exception as e:
        print(f"✗ Error: {e}")
        return {"status": f"error: {e}", "count": 0, "events": [], "nearest": None}


# ============================================================================
# SCALE 2: ATMOSPHERIC - GLOBAL WEATHER
# ============================================================================

def fetch_noaa_gfs_forecast():
    """
    NOAA Global Forecast System (GFS) - worldwide weather model
    """
    print("─" * 70)
    print("SCALE 2: NOAA GLOBAL FORECAST SYSTEM (GFS)")
    print("─" * 70)

    print("GFS provides global weather forecasts up to 16 days")
    print("Resolution: 0.25° (~28 km)")
    print("Data: Temperature, pressure, wind, humidity, precipitation")
    print()
    print("Access via NOAA NOMADS:")
    print("  https://nomads.ncep.noaa.gov/")
    print("  GRIB2 format, requires parsing tools")
    print()
    return {"status": "catalog", "source": "https://nomads.ncep.noaa.gov/"}


def fetch_satellite_data():
    """
    NOAA GOES satellite imagery
    """
    print("─" * 70)
    print("SCALE 2B: NOAA GOES SATELLITE DATA")
    print("─" * 70)

    print("GOES-16/17: Geostationary weather satellites")
    print("Coverage: Americas, Pacific")
    print("Imagery:")
    print("  - Visible (0.64 µm)")
    print("  - Near-IR (0.86-2.2 µm)")
    print("  - IR (3.9-13.3 µm)")
    print("  - Water vapor")
    print()
    print("Access: https://www.goes.noaa.gov/")
    print("Real-time viewer: https://www.star.nesdis.noaa.gov/GOES/")
    print()
    return {"status": "catalog", "source": "https://www.goes.noaa.gov/"}


# ============================================================================
# SCALE 3: OCEANIC - SEA STATE
# ============================================================================

def parse_ndbc_latest(lines: List[str]) -> Optional[Dict[str, Optional[float]]]:
    if len(lines) < 3:
        return None
    headers = lines[0].split()
    values = lines[2].split()
    if len(values) < len(headers):
        return None
    record = dict(zip(headers, values))

    def val(key: str) -> Optional[float]:
        raw = record.get(key)
        if raw in (None, "MM"):
            return None
        return safe_float(raw)

    year = record.get("#YY") or record.get("YY")
    timestamp = None
    if year and record.get("MM") and record.get("DD") and record.get("hh"):
        year_str = year if len(year) == 4 else f"20{year}"
        timestamp = f"{year_str}-{record.get('MM')}-{record.get('DD')}T{record.get('hh')}:{record.get('mm','00')}Z"

    return {
        "timestamp": timestamp,
        "wind_direction_deg": val("WDIR"),
        "wind_speed_ms": val("WSPD"),
        "gust_ms": val("GST"),
        "wave_height_m": val("WVHT"),
        "water_temp_c": val("WTMP"),
        "air_temp_c": val("ATMP"),
        "pressure_hpa": val("PRES"),
    }


def fetch_noaa_buoy_data(buoy_id: str = DEFAULT_BUOY_ID):
    """
    NOAA NDBC - National Data Buoy Center
    """
    print("─" * 70)
    print("SCALE 3: NOAA BUOY NETWORK (NDBC)")
    print("─" * 70)

    url = f"https://www.ndbc.noaa.gov/data/realtime2/{buoy_id}.txt"

    print(f"Buoy {buoy_id}: Gulf of Mexico")
    print(f"URL: {url}")
    print()

    if OFFLINE:
        print("⚠ Offline mode - skipping buoy fetch")
        return {"status": "offline", "data": {}}

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            lines = response.read().decode().split('\n')
        data = parse_ndbc_latest(lines)
        if data:
            print("Latest observation:")
            if data.get("timestamp"):
                print(f"  Date/Time: {data['timestamp']}")
            print()
            return {"status": "ok", "data": data}

    except Exception as e:
        print(f"⚠ Could not fetch buoy data: {e}")
        print()
        return {"status": f"error: {e}", "data": {}}


def fetch_noaa_tides(station_id: str = DEFAULT_TIDE_STATION):
    """
    NOAA CO-OPS tides/water levels
    """
    now = datetime.now(timezone.utc)
    begin_date = (now - timedelta(days=1)).strftime("%Y%m%d")
    end_date = now.strftime("%Y%m%d")

    if OFFLINE:
        return {"status": "offline", "station_id": station_id}

    water_url = (
        "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
        f"?product=water_level&application=MYSTIC&begin_date={begin_date}&end_date={end_date}"
        f"&datum=MLLW&station={station_id}&time_zone=gmt&units=metric&format=json"
    )
    pred_url = (
        "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
        f"?product=predictions&application=MYSTIC&begin_date={begin_date}&end_date={end_date}"
        f"&datum=MLLW&station={station_id}&time_zone=gmt&units=metric&interval=h&format=json"
    )

    try:
        water_data = http_get_json(water_url)
        water_rows = water_data.get("data", [])
        latest = water_rows[-1] if water_rows else {}
        water_level_m = safe_float(latest.get("v"))
        pred_data = http_get_json(pred_url)
        pred_rows = pred_data.get("predictions", [])
        pred_latest = pred_rows[-1] if pred_rows else {}
        pred_level_m = safe_float(pred_latest.get("v"))

        return {
            "status": "ok",
            "station_id": station_id,
            "water_level_m": water_level_m,
            "water_level_cm": water_level_m * 100 if water_level_m is not None else None,
            "water_time": latest.get("t"),
            "prediction_m": pred_level_m,
            "prediction_time": pred_latest.get("t"),
        }
    except Exception as e:
        return {"status": f"error: {e}", "station_id": station_id}


def fetch_ocean_currents(station_id: str = DEFAULT_CURRENT_STATION):
    """
    NOAA CO-OPS currents (PORTS stations)
    """
    print("─" * 70)
    print("SCALE 3B: OCEAN CURRENTS & TIDES")
    print("─" * 70)

    if OFFLINE:
        print("⚠ Offline mode - skipping currents fetch")
        return {"status": "offline", "station_id": station_id}

    now = datetime.now(timezone.utc)
    begin_date = (now - timedelta(days=1)).strftime("%Y%m%d")
    end_date = now.strftime("%Y%m%d")
    url = (
        "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
        f"?product=currents&application=MYSTIC&begin_date={begin_date}&end_date={end_date}"
        f"&station={station_id}&time_zone=gmt&units=metric&format=json"
    )

    try:
        data = http_get_json(url)
        rows = data.get("data", [])
        latest = rows[-1] if rows else {}
        speed_cm_s = safe_float(latest.get("s"))
        speed_mps = speed_cm_s / 100.0 if speed_cm_s is not None else None
        return {
            "status": "ok",
            "station_id": station_id,
            "speed_mps": speed_mps,
            "direction_deg": safe_float(latest.get("d")),
            "bin": latest.get("b"),
            "time": latest.get("t"),
        }
    except Exception as e:
        print(f"⚠ Could not fetch currents: {e}")
        return {"status": f"error: {e}", "station_id": station_id}


# ============================================================================
# SCALE 4: SPACE WEATHER - SOLAR & GEOMAGNETIC
# ============================================================================

def fetch_noaa_space_weather():
    """
    NOAA Space Weather Prediction Center
    """
    print("─" * 70)
    print("SCALE 4: NOAA SPACE WEATHER")
    print("─" * 70)

    url = "https://services.swpc.noaa.gov/products/alerts.json"

    print("Fetching space weather alerts...")
    print()

    if OFFLINE:
        print("⚠ Offline mode - skipping space weather alerts")
        return {"status": "offline", "count": 0, "alerts": []}

    try:
        alerts = http_get_json(url)

        print(f"✓ Retrieved {len(alerts)} space weather alerts/observations")
        print()

        if alerts:
            print("Recent alerts:")
            for alert in alerts[:5]:
                print(f"  [{alert.get('issue_datetime')}] {alert.get('message', '')[:80]}...")
            print()

        return {"status": "ok", "count": len(alerts), "alerts": alerts[:5]}

    except Exception as e:
        print(f"✗ Error: {e}")
        print()
        return {"status": f"error: {e}", "count": 0, "alerts": []}


def fetch_solar_data():
    """
    NOAA SWPC solar data (X-ray flux, solar wind)
    """
    print("─" * 70)
    print("SCALE 4B: SOLAR ACTIVITY")
    print("─" * 70)

    if OFFLINE:
        print("⚠ Offline mode - skipping solar data")
        return {"status": "offline"}

    xray_url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"
    plasma_url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-5-minute.json"
    mag_url = "https://services.swpc.noaa.gov/products/solar-wind/mag-5-minute.json"

    xray_flux = None
    xray_time = None
    solar_wind = {}
    magnetic = {}

    try:
        xray_data = http_get_json(xray_url)
        for entry in reversed(xray_data):
            if entry.get("energy") == "0.1-0.8nm":
                xray_flux = safe_float(entry.get("flux"))
                xray_time = entry.get("time_tag")
                break
    except Exception as e:
        xray_flux = None
        xray_time = None
        magnetic["xray_error"] = str(e)

    try:
        plasma = http_get_json(plasma_url)
        if len(plasma) > 1:
            row = plasma[-1]
            solar_wind = {
                "time": row[0],
                "density_cm3": safe_float(row[1]),
                "speed_km_s": safe_float(row[2]),
                "temperature_k": safe_float(row[3]),
            }
    except Exception as e:
        solar_wind["error"] = str(e)

    try:
        mag = http_get_json(mag_url)
        if len(mag) > 1:
            row = mag[-1]
            magnetic.update({
                "time": row[0],
                "bx_gsm": safe_float(row[1]),
                "by_gsm": safe_float(row[2]),
                "bz_gsm": safe_float(row[3]),
                "bt": safe_float(row[6]) if len(row) > 6 else None,
            })
    except Exception as e:
        magnetic["error"] = str(e)

    return {
        "status": "ok",
        "xray_flux": xray_flux,
        "xray_time": xray_time,
        "solar_wind": solar_wind,
        "magnetic_field": magnetic,
    }


def fetch_geomagnetic_data():
    """
    Geomagnetic field data (Kp index, Dst, etc.)
    """
    print("─" * 70)
    print("SCALE 4C: GEOMAGNETIC FIELD")
    print("─" * 70)

    url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"

    print("Fetching Kp index (geomagnetic activity)...")
    print()

    if OFFLINE:
        print("⚠ Offline mode - skipping Kp index")
        return {"status": "offline"}

    try:
        data = http_get_json(url)
        kp_data = data[1:]

        print(f"✓ Retrieved {len(kp_data)} Kp measurements")
        print()

        if kp_data:
            latest = kp_data[-1]
            kp_val = safe_float(latest[1])
            if kp_val is None:
                status = "Unknown"
            elif kp_val < 4:
                status = "Quiet to unsettled"
            elif kp_val < 6:
                status = "Active geomagnetic conditions"
            elif kp_val < 8:
                status = "Geomagnetic storm"
            else:
                status = "Severe geomagnetic storm"

            print("Latest Kp index:")
            print(f"  Time: {latest[0]}")
            print(f"  Kp: {latest[1]} (0=quiet, 9=extreme storm)")
            print(f"  Status: {status}")
            print()

            return {
                "status": "ok",
                "time": latest[0],
                "kp": kp_val,
                "a_running": latest[2],
                "station_count": latest[3],
                "status_text": status,
            }

        return {"status": "empty"}

    except Exception as e:
        print(f"✗ Error: {e}")
        print()
        return {"status": f"error: {e}"}


# ============================================================================
# SCALE 5: PLANETARY - CELESTIAL MECHANICS
# ============================================================================

def fetch_nasa_planetary_data():
    """
    NASA Horizons System - planetary ephemerides
    """
    print("─" * 70)
    print("SCALE 5: NASA PLANETARY EPHEMERIDES")
    print("─" * 70)

    if OFFLINE:
        print("⚠ Offline mode - skipping Horizons fetch")
        return {"status": "offline"}

    start = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stop = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

    params = {
        "format": "json",
        "COMMAND": "'301'",
        "CENTER": "'500@399'",
        "EPHEM_TYPE": "'V'",
        "VEC_TABLE": "1",
        "START_TIME": f"'{start}'",
        "STOP_TIME": f"'{stop}'",
        "STEP_SIZE": "'1 d'",
    }

    url = "https://ssd.jpl.nasa.gov/api/horizons.api?" + urllib.parse.urlencode(params)

    try:
        data = http_get_json(url)
        result_text = data.get("result", "")
        moon_distance_km = parse_horizons_vector(result_text)
        return {"status": "ok", "moon_distance_km": moon_distance_km}
    except Exception as e:
        return {"status": f"error: {e}"}


def calculate_moon_phase():
    """
    Calculate current moon phase (affects tides, potentially seismic activity)
    """
    print("─" * 70)
    print("SCALE 5B: LUNAR PHASE & TIDAL FORCES")
    print("─" * 70)

    now = datetime.now(timezone.utc)

    # Known new moon: Jan 11, 2024
    known_new_moon = datetime(2024, 1, 11, 11, 57, tzinfo=timezone.utc)
    lunar_month = 29.53059

    days_since = (now - known_new_moon).total_seconds() / 86400
    phase = (days_since % lunar_month) / lunar_month

    if phase < 0.03 or phase > 0.97:
        phase_name = "New Moon"
        tidal = "Spring tide (high tidal forces)"
    elif 0.22 < phase < 0.28:
        phase_name = "First Quarter"
        tidal = "Neap tide (low tidal forces)"
    elif 0.47 < phase < 0.53:
        phase_name = "Full Moon"
        tidal = "Spring tide (high tidal forces)"
    elif 0.72 < phase < 0.78:
        phase_name = "Last Quarter"
        tidal = "Neap tide (low tidal forces)"
    else:
        phase_name = "Waxing/Waning"
        tidal = "Moderate tidal forces"

    tidal_force = abs(math.sin(phase * 2 * math.pi))

    print(f"Current date: {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"Lunar phase: {phase:.3f} (0=new, 0.5=full)")
    print(f"Phase name: {phase_name}")
    print(f"Tidal effect: {tidal}")
    print()

    return {
        "phase": phase,
        "phase_name": phase_name,
        "tidal_effect": tidal,
        "tidal_force_index": tidal_force,
    }


# ============================================================================
# SCALE 6: COSMIC - GALACTIC EVENTS
# ============================================================================

def fetch_cosmic_ray_data():
    """
    Cosmic ray proxy: GOES integral proton flux
    """
    print("─" * 70)
    print("SCALE 6: COSMIC RAY FLUX (PROTON PROXY)")
    print("─" * 70)

    if OFFLINE:
        print("⚠ Offline mode - skipping cosmic ray proxy")
        return {"status": "offline"}

    url = "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-1-day.json"
    try:
        data = http_get_json(url)
        latest = data[-1] if data else {}
        return {
            "status": "ok",
            "flux": safe_float(latest.get("flux")),
            "energy": latest.get("energy"),
            "time": latest.get("time_tag"),
        }
    except Exception as e:
        print(f"✗ Error: {e}")
        return {"status": f"error: {e}"}


def fetch_gamma_ray_bursts():
    """
    NASA Fermi GRB detections (catalog only)
    """
    print("─" * 70)
    print("SCALE 6B: GAMMA RAY BURSTS")
    print("─" * 70)

    print("NASA Fermi Gamma-ray Space Telescope:")
    print("  Detects gamma-ray bursts from distant galaxies")
    print("  Most energetic events in universe")
    print()
    print("Data: https://fermi.gsfc.nasa.gov/ssc/observations/types/grbs/")
    print()
    print("Note: GRBs are cosmological - no Earth impact")
    print("      Included for completeness of multi-scale monitoring")
    print()
    return {"status": "catalog", "source": "https://fermi.gsfc.nasa.gov/ssc/observations/types/grbs/"}


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    output_file = "../data/omniscient_data_summary.json"

    location = {
        "lat": DEFAULT_LAT,
        "lon": DEFAULT_LON,
        "radius_km": DEFAULT_RADIUS_KM,
        "nexrad_sites": DEFAULT_NEXRAD_SITES,
    }

    all_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "location": location,
        "scale_1_terrestrial": {},
        "scale_2_atmospheric": {},
        "scale_3_oceanic": {},
        "scale_4_space_weather": {},
        "scale_5_planetary": {},
        "scale_6_cosmic": {},
    }

    # SCALE 1: Terrestrial
    print()
    stations = fetch_noaa_weather_stations(DEFAULT_LAT, DEFAULT_LON, DEFAULT_RADIUS_KM)
    nws_alerts = fetch_nws_alerts(DEFAULT_LAT, DEFAULT_LON)
    nws_forecast = fetch_nws_forecast(DEFAULT_LAT, DEFAULT_LON)
    nexrad_latest = {
        site: fetch_nexrad_latest(site, datetime.now(timezone.utc)) for site in DEFAULT_NEXRAD_SITES
    }
    earthquakes = fetch_usgs_earthquakes(
        (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d"),
        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        min_magnitude=2.5,
        lat=DEFAULT_LAT,
        lon=DEFAULT_LON,
    )

    all_data["scale_1_terrestrial"] = {
        "weather_stations": stations,
        "nws_alerts": nws_alerts,
        "nws_forecast": nws_forecast,
        "nexrad_latest": nexrad_latest,
        "earthquakes": earthquakes,
    }

    # SCALE 2: Atmospheric
    gfs = fetch_noaa_gfs_forecast()
    goes = fetch_satellite_data()
    all_data["scale_2_atmospheric"] = {
        "gfs": gfs,
        "goes": goes,
    }

    # SCALE 3: Oceanic
    buoy = fetch_noaa_buoy_data(DEFAULT_BUOY_ID)
    tides = fetch_noaa_tides(DEFAULT_TIDE_STATION)
    currents = fetch_ocean_currents(DEFAULT_CURRENT_STATION)
    all_data["scale_3_oceanic"] = {
        "buoy": buoy,
        "tides": tides,
        "currents": currents,
    }

    # SCALE 4: Space Weather
    space_alerts = fetch_noaa_space_weather()
    solar = fetch_solar_data()
    geomagnetic = fetch_geomagnetic_data()
    all_data["scale_4_space_weather"] = {
        "alerts": space_alerts,
        "solar": solar,
        "geomagnetic_kp": geomagnetic,
    }

    # SCALE 5: Planetary
    planetary = fetch_nasa_planetary_data()
    moon_phase = calculate_moon_phase()
    all_data["scale_5_planetary"] = {
        "planetary": planetary,
        "moon_phase": moon_phase,
    }

    # SCALE 6: Cosmic
    cosmic = fetch_cosmic_ray_data()
    grb = fetch_gamma_ray_bursts()
    all_data["scale_6_cosmic"] = {
        "proton_flux": cosmic,
        "gamma_bursts": grb,
    }

    # Save summary
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)

    print("═" * 70)
    print("INTEGRATION SUMMARY")
    print("═" * 70)
    print()
    print("Data sources catalogued:")
    print(f"  ✓ SCALE 1: Terrestrial (NOAA stations: {stations.get('count')}, earthquakes: {earthquakes.get('count')})")
    print(f"  ✓ SCALE 2: Atmospheric (GFS, GOES catalogues)")
    print(f"  ✓ SCALE 3: Oceanic (buoy: {buoy.get('status')}, tides: {tides.get('status')}, currents: {currents.get('status')})")
    print(f"  ✓ SCALE 4: Space Weather (Kp: {geomagnetic.get('kp')}, X-ray: {solar.get('xray_flux')})")
    print(f"  ✓ SCALE 5: Planetary (moon phase: {moon_phase.get('phase'):.3f})")
    print(f"  ✓ SCALE 6: Cosmic (proton flux: {cosmic.get('flux')})")
    print()
    print(f"Summary saved to: {output_file}")
    print()
    print("Next step: Create unified ingestion pipeline")
    print("  python3 create_unified_pipeline.py")
    print()


if __name__ == "__main__":
    main()
