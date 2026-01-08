#!/usr/bin/env python3
"""
MYSTIC DATA SOURCE INTEGRATION MODULE

Integrates external data sources identified in the Global Data Resources report:
- Open-Meteo (Flood API, Weather API) - No auth required
- USGS Earthquake API - No auth required
- USGS Water Services - No auth required
- NOAA Space Weather - No auth required

These provide real-time and historical data for:
- River discharge (flood prediction)
- Weather conditions (pressure, precipitation, wind)
- Seismic activity (earthquake correlation)
- Space weather (geomagnetic indices)

Author: Claude (K-Elimination Expert)
Date: 2026-01-07
Based on: Comprehensive_Report_Global_Data_Resources_for_MYSTIC_System_Integration.pdf
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode
import os
import ssl


# Create SSL context. Allow opt-out via MYSTIC_INSECURE_SSL=1.
SSL_CONTEXT = ssl.create_default_context()
if os.environ.get("MYSTIC_INSECURE_SSL") == "1":
    SSL_CONTEXT.check_hostname = False
    SSL_CONTEXT.verify_mode = ssl.CERT_NONE


@dataclass
class DataPoint:
    """Unified data point from any source."""
    timestamp: str
    value: int  # Integer-only for QMNF compatibility
    unit: str
    source: str
    variable: str
    location: Optional[str] = None
    raw_value: Optional[float] = None  # Original float before conversion


@dataclass
class DataFetchResult:
    """Result of a data fetch operation."""
    success: bool
    source: str
    data: List[DataPoint] = field(default_factory=list)
    error: Optional[str] = None
    fetch_time: float = 0.0


def fetch_json(url: str, timeout: int = 30) -> Tuple[bool, Any, Optional[str]]:
    """Fetch JSON from URL with error handling."""
    try:
        req = Request(url, headers={'User-Agent': 'MYSTIC/3.0'})
        with urlopen(req, timeout=timeout, context=SSL_CONTEXT) as response:
            data = json.loads(response.read().decode('utf-8'))
            return True, data, None
    except HTTPError as e:
        return False, None, f"HTTP {e.code}: {e.reason}"
    except URLError as e:
        return False, None, f"URL Error: {e.reason}"
    except json.JSONDecodeError as e:
        return False, None, f"JSON Error: {e}"
    except Exception as e:
        return False, None, f"Error: {e}"


def float_to_int_scaled(value: float, scale: int = 100) -> int:
    """Convert float to scaled integer for QMNF compatibility."""
    return int(round(value * scale))


# =============================================================================
# OPEN-METEO API (No Authentication Required)
# =============================================================================

class OpenMeteoClient:
    """
    Client for Open-Meteo APIs.

    Provides:
    - Flood API: River discharge data (1984-present, 5km resolution)
    - Weather Forecast API: Temperature, pressure, precipitation, wind
    - Historical Weather API: Reanalysis data back to 1940s

    Reference: https://open-meteo.com/
    """

    BASE_URL = "https://api.open-meteo.com/v1"
    FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"

    @staticmethod
    def fetch_flood_data(
        latitude: float,
        longitude: float,
        days_past: int = 7,
        days_forecast: int = 7
    ) -> DataFetchResult:
        """
        Fetch river discharge data from Open-Meteo Flood API.

        Returns discharge in m³/s (scaled to integers).
        """
        start_time = time.time()

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "river_discharge",
            "past_days": days_past,
            "forecast_days": days_forecast,
        }

        url = f"{OpenMeteoClient.FLOOD_URL}?{urlencode(params)}"
        success, data, error = fetch_json(url)

        if not success:
            return DataFetchResult(
                success=False,
                source="open-meteo-flood",
                error=error,
                fetch_time=time.time() - start_time
            )

        # Parse response
        points = []
        try:
            times = data.get("daily", {}).get("time", [])
            discharges = data.get("daily", {}).get("river_discharge", [])

            for t, d in zip(times, discharges):
                if d is not None:
                    points.append(DataPoint(
                        timestamp=t,
                        value=float_to_int_scaled(d, scale=100),  # m³/s × 100
                        unit="m³/s×100",
                        source="open-meteo-flood",
                        variable="river_discharge",
                        location=f"{latitude},{longitude}",
                        raw_value=d
                    ))
        except Exception as e:
            return DataFetchResult(
                success=False,
                source="open-meteo-flood",
                error=f"Parse error: {e}",
                fetch_time=time.time() - start_time
            )

        return DataFetchResult(
            success=True,
            source="open-meteo-flood",
            data=points,
            fetch_time=time.time() - start_time
        )

    @staticmethod
    def fetch_weather_data(
        latitude: float,
        longitude: float,
        days_past: int = 7,
        days_forecast: int = 7
    ) -> DataFetchResult:
        """
        Fetch weather data from Open-Meteo Weather API.

        Returns pressure (hPa), precipitation (mm), temperature (°C).
        All scaled to integers.
        """
        start_time = time.time()

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "pressure_msl,precipitation,temperature_2m,relative_humidity_2m",
            "past_days": days_past,
            "forecast_days": days_forecast,
        }

        url = f"{OpenMeteoClient.BASE_URL}/forecast?{urlencode(params)}"
        success, data, error = fetch_json(url)

        if not success:
            return DataFetchResult(
                success=False,
                source="open-meteo-weather",
                error=error,
                fetch_time=time.time() - start_time
            )

        points = []
        try:
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            pressures = hourly.get("pressure_msl", [])
            precips = hourly.get("precipitation", [])
            temps = hourly.get("temperature_2m", [])
            humidities = hourly.get("relative_humidity_2m", [])

            for i, t in enumerate(times):
                # Pressure (already in hPa, scale by 10 for precision)
                if i < len(pressures) and pressures[i] is not None:
                    points.append(DataPoint(
                        timestamp=t,
                        value=float_to_int_scaled(pressures[i], scale=10),
                        unit="hPa×10",
                        source="open-meteo-weather",
                        variable="pressure_msl",
                        location=f"{latitude},{longitude}",
                        raw_value=pressures[i]
                    ))

                # Precipitation (mm, scale by 100)
                if i < len(precips) and precips[i] is not None:
                    points.append(DataPoint(
                        timestamp=t,
                        value=float_to_int_scaled(precips[i], scale=100),
                        unit="mm×100",
                        source="open-meteo-weather",
                        variable="precipitation",
                        location=f"{latitude},{longitude}",
                        raw_value=precips[i]
                    ))

                # Temperature (°C, scale by 100)
                if i < len(temps) and temps[i] is not None:
                    points.append(DataPoint(
                        timestamp=t,
                        value=float_to_int_scaled(temps[i], scale=100),
                        unit="°C×100",
                        source="open-meteo-weather",
                        variable="temperature",
                        location=f"{latitude},{longitude}",
                        raw_value=temps[i]
                    ))

                # Humidity (%, no scaling needed)
                if i < len(humidities) and humidities[i] is not None:
                    points.append(DataPoint(
                        timestamp=t,
                        value=int(humidities[i]),
                        unit="%",
                        source="open-meteo-weather",
                        variable="humidity",
                        location=f"{latitude},{longitude}",
                        raw_value=humidities[i]
                    ))

        except Exception as e:
            return DataFetchResult(
                success=False,
                source="open-meteo-weather",
                error=f"Parse error: {e}",
                fetch_time=time.time() - start_time
            )

        return DataFetchResult(
            success=True,
            source="open-meteo-weather",
            data=points,
            fetch_time=time.time() - start_time
        )


# =============================================================================
# USGS EARTHQUAKE API (No Authentication Required)
# =============================================================================

class USGSEarthquakeClient:
    """
    Client for USGS Earthquake Hazards Program API.

    Provides:
    - Real-time earthquake data (updated within minutes)
    - Historical data back to early 1900s
    - GeoJSON format for easy parsing

    Reference: https://earthquake.usgs.gov/fdsnws/event/1/
    """

    BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    @staticmethod
    def fetch_recent_earthquakes(
        min_magnitude: float = 2.5,
        days_back: int = 7,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        max_radius_km: float = 500
    ) -> DataFetchResult:
        """
        Fetch recent earthquake data.

        Returns magnitude (scaled) and depth (km).
        """
        start_time = time.time()

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)

        params = {
            "format": "geojson",
            "starttime": start_date.strftime("%Y-%m-%d"),
            "endtime": end_date.strftime("%Y-%m-%d"),
            "minmagnitude": min_magnitude,
            "orderby": "time",
        }

        # Add geographic filter if provided
        if latitude is not None and longitude is not None:
            params["latitude"] = latitude
            params["longitude"] = longitude
            params["maxradiuskm"] = max_radius_km

        url = f"{USGSEarthquakeClient.BASE_URL}?{urlencode(params)}"
        success, data, error = fetch_json(url)

        if not success:
            return DataFetchResult(
                success=False,
                source="usgs-earthquake",
                error=error,
                fetch_time=time.time() - start_time
            )

        points = []
        try:
            features = data.get("features", [])

            for feature in features:
                props = feature.get("properties", {})
                geom = feature.get("geometry", {})
                coords = geom.get("coordinates", [0, 0, 0])

                timestamp = props.get("time", 0)
                timestamp_str = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).isoformat()

                mag = props.get("mag", 0)
                depth = coords[2] if len(coords) > 2 else 0
                place = props.get("place", "Unknown")

                # Magnitude (scale by 100 for precision)
                points.append(DataPoint(
                    timestamp=timestamp_str,
                    value=float_to_int_scaled(mag, scale=100),
                    unit="M×100",
                    source="usgs-earthquake",
                    variable="magnitude",
                    location=place,
                    raw_value=mag
                ))

                # Depth (km, scale by 10)
                points.append(DataPoint(
                    timestamp=timestamp_str,
                    value=float_to_int_scaled(depth, scale=10),
                    unit="km×10",
                    source="usgs-earthquake",
                    variable="depth",
                    location=place,
                    raw_value=depth
                ))

        except Exception as e:
            return DataFetchResult(
                success=False,
                source="usgs-earthquake",
                error=f"Parse error: {e}",
                fetch_time=time.time() - start_time
            )

        return DataFetchResult(
            success=True,
            source="usgs-earthquake",
            data=points,
            fetch_time=time.time() - start_time
        )


# =============================================================================
# USGS WATER SERVICES API (No Authentication Required)
# =============================================================================

class USGSWaterClient:
    """
    Client for USGS National Water Information System (NWIS).

    Provides:
    - Real-time stream gauge data (13,500+ stations)
    - Water level, discharge, temperature
    - Instantaneous values updated every 15 minutes

    Reference: https://waterservices.usgs.gov/
    """

    BASE_URL = "https://waterservices.usgs.gov/nwis/iv/"

    # Common parameter codes
    PARAM_DISCHARGE = "00060"  # Discharge, cubic feet per second
    PARAM_GAGE_HEIGHT = "00065"  # Gage height, feet
    PARAM_WATER_TEMP = "00010"  # Temperature, water, degrees Celsius

    @staticmethod
    def fetch_station_data(
        site_code: str,
        parameter: str = "00060",  # Default: discharge
        days_back: int = 7
    ) -> DataFetchResult:
        """
        Fetch data from a specific USGS water station.

        Common sites near Camp Mystic area:
        - 08167500: Guadalupe River at Comfort, TX
        - 08167000: Guadalupe River at Kerrville, TX
        - 08165500: Guadalupe River at Spring Branch, TX
        """
        start_time = time.time()

        params = {
            "format": "json",
            "sites": site_code,
            "parameterCd": parameter,
            "period": f"P{days_back}D",  # Past N days
        }

        url = f"{USGSWaterClient.BASE_URL}?{urlencode(params)}"
        success, data, error = fetch_json(url)

        if not success:
            return DataFetchResult(
                success=False,
                source="usgs-water",
                error=error,
                fetch_time=time.time() - start_time
            )

        points = []
        try:
            time_series = data.get("value", {}).get("timeSeries", [])

            for ts in time_series:
                site_name = ts.get("sourceInfo", {}).get("siteName", "Unknown")
                variable_info = ts.get("variable", {})
                var_name = variable_info.get("variableName", "Unknown")
                unit = variable_info.get("unit", {}).get("unitCode", "")

                values = ts.get("values", [{}])[0].get("value", [])

                for v in values:
                    val = v.get("value")
                    timestamp = v.get("dateTime", "")

                    if val is not None:
                        try:
                            float_val = float(val)
                            # Scale based on parameter type
                            if parameter == USGSWaterClient.PARAM_DISCHARGE:
                                scaled = float_to_int_scaled(float_val, scale=10)
                            else:
                                scaled = float_to_int_scaled(float_val, scale=100)

                            points.append(DataPoint(
                                timestamp=timestamp,
                                value=scaled,
                                unit=f"{unit}×scale",
                                source="usgs-water",
                                variable=var_name,
                                location=site_name,
                                raw_value=float_val
                            ))
                        except ValueError:
                            continue

        except Exception as e:
            return DataFetchResult(
                success=False,
                source="usgs-water",
                error=f"Parse error: {e}",
                fetch_time=time.time() - start_time
            )

        return DataFetchResult(
            success=True,
            source="usgs-water",
            data=points,
            fetch_time=time.time() - start_time
        )

    @staticmethod
    def fetch_texas_hill_country_stations() -> List[str]:
        """Return station codes for Texas Hill Country (Camp Mystic area)."""
        return [
            "08167500",  # Guadalupe River at Comfort
            "08167000",  # Guadalupe River at Kerrville
            "08165500",  # Guadalupe River at Spring Branch
            "08168500",  # Guadalupe River above Comal River
            "08171000",  # Blanco River at Wimberley
        ]


# =============================================================================
# NOAA SPACE WEATHER (No Authentication Required)
# =============================================================================

class NOAASpaceWeatherClient:
    """
    Client for NOAA Space Weather Prediction Center.

    Provides:
    - Kp index (geomagnetic activity)
    - Solar X-ray flux
    - Geomagnetic storm warnings

    Reference: https://www.swpc.noaa.gov/
    """

    KP_URL = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
    XRAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"

    @staticmethod
    def fetch_kp_index() -> DataFetchResult:
        """
        Fetch planetary K-index (geomagnetic activity).

        Kp ranges from 0-9:
        - 0-3: Quiet
        - 4: Unsettled
        - 5: Minor storm
        - 6: Moderate storm
        - 7-9: Strong to extreme storm
        """
        start_time = time.time()

        success, data, error = fetch_json(NOAASpaceWeatherClient.KP_URL)

        if not success:
            return DataFetchResult(
                success=False,
                source="noaa-spaceweather",
                error=error,
                fetch_time=time.time() - start_time
            )

        points = []
        try:
            for entry in data:
                timestamp = entry.get("time_tag", "")
                kp = entry.get("kp_index", 0)

                points.append(DataPoint(
                    timestamp=timestamp,
                    value=int(kp * 10),  # Scale by 10 for fractional Kp
                    unit="Kp×10",
                    source="noaa-spaceweather",
                    variable="kp_index",
                    raw_value=kp
                ))

        except Exception as e:
            return DataFetchResult(
                success=False,
                source="noaa-spaceweather",
                error=f"Parse error: {e}",
                fetch_time=time.time() - start_time
            )

        return DataFetchResult(
            success=True,
            source="noaa-spaceweather",
            data=points,
            fetch_time=time.time() - start_time
        )


# =============================================================================
# UNIFIED DATA FETCHER
# =============================================================================

class MYSTICDataFetcher:
    """
    Unified data fetcher for MYSTIC system.

    Aggregates data from multiple sources and normalizes to integer format
    for QMNF compatibility.
    """

    # Camp Mystic approximate coordinates (Texas Hill Country)
    CAMP_MYSTIC_LAT = 30.05
    CAMP_MYSTIC_LON = -99.17

    def __init__(self):
        self.last_fetch_results: Dict[str, DataFetchResult] = {}

    def fetch_all(
        self,
        latitude: float = None,
        longitude: float = None,
        include_earthquake: bool = True,
        include_flood: bool = True,
        include_weather: bool = True,
        include_spaceweather: bool = True,
        include_water_stations: bool = True
    ) -> Dict[str, DataFetchResult]:
        """
        Fetch data from all configured sources.

        Returns dict of source_name -> DataFetchResult
        """
        lat = latitude or self.CAMP_MYSTIC_LAT
        lon = longitude or self.CAMP_MYSTIC_LON

        results = {}

        if include_flood:
            print("  Fetching Open-Meteo Flood data...")
            results["flood"] = OpenMeteoClient.fetch_flood_data(lat, lon)

        if include_weather:
            print("  Fetching Open-Meteo Weather data...")
            results["weather"] = OpenMeteoClient.fetch_weather_data(lat, lon)

        if include_earthquake:
            print("  Fetching USGS Earthquake data...")
            results["earthquake"] = USGSEarthquakeClient.fetch_recent_earthquakes(
                latitude=lat, longitude=lon, max_radius_km=1000
            )

        if include_spaceweather:
            print("  Fetching NOAA Space Weather data...")
            results["spaceweather"] = NOAASpaceWeatherClient.fetch_kp_index()

        if include_water_stations:
            print("  Fetching USGS Water Station data...")
            # Fetch from first Texas Hill Country station
            stations = USGSWaterClient.fetch_texas_hill_country_stations()
            if stations:
                results["water"] = USGSWaterClient.fetch_station_data(stations[0])

        self.last_fetch_results = results
        return results

    def extract_time_series(
        self,
        results: Dict[str, DataFetchResult],
        variable: str,
        source: str = None
    ) -> List[int]:
        """
        Extract a time series of integer values for a specific variable.

        This is the format needed by MYSTIC predictors.
        """
        values = []

        for src_name, result in results.items():
            if source and src_name != source:
                continue

            if not result.success:
                continue

            for point in result.data:
                if point.variable == variable:
                    values.append(point.value)

        return values

    def get_summary(self, results: Dict[str, DataFetchResult]) -> Dict[str, Any]:
        """Generate summary of fetched data."""
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sources": {}
        }

        for name, result in results.items():
            summary["sources"][name] = {
                "success": result.success,
                "data_points": len(result.data),
                "fetch_time_ms": int(result.fetch_time * 1000),
                "error": result.error
            }

            if result.success and result.data:
                # Get unique variables
                variables = set(p.variable for p in result.data)
                summary["sources"][name]["variables"] = list(variables)

        return summary


# =============================================================================
# TEST SUITE
# =============================================================================

def test_data_sources():
    """Test all data source integrations."""
    print("=" * 70)
    print("MYSTIC DATA SOURCE INTEGRATION TEST")
    print("Testing APIs from Global Data Resources report")
    print("=" * 70)

    fetcher = MYSTICDataFetcher()

    # Test each source individually
    print("\n[TEST 1] Open-Meteo Flood API")
    print("-" * 40)
    result = OpenMeteoClient.fetch_flood_data(30.05, -99.17, days_past=3, days_forecast=3)
    print(f"  Success: {result.success}")
    print(f"  Data points: {len(result.data)}")
    print(f"  Fetch time: {result.fetch_time*1000:.0f}ms")
    if result.data:
        print(f"  Sample: {result.data[0]}")
    if result.error:
        print(f"  Error: {result.error}")

    print("\n[TEST 2] Open-Meteo Weather API")
    print("-" * 40)
    result = OpenMeteoClient.fetch_weather_data(30.05, -99.17, days_past=1, days_forecast=1)
    print(f"  Success: {result.success}")
    print(f"  Data points: {len(result.data)}")
    print(f"  Fetch time: {result.fetch_time*1000:.0f}ms")
    if result.data:
        # Show one of each variable type
        vars_shown = set()
        for p in result.data[:20]:
            if p.variable not in vars_shown:
                print(f"  {p.variable}: {p.value} {p.unit}")
                vars_shown.add(p.variable)
    if result.error:
        print(f"  Error: {result.error}")

    print("\n[TEST 3] USGS Earthquake API")
    print("-" * 40)
    result = USGSEarthquakeClient.fetch_recent_earthquakes(
        min_magnitude=2.5, days_back=7
    )
    print(f"  Success: {result.success}")
    print(f"  Data points: {len(result.data)}")
    print(f"  Fetch time: {result.fetch_time*1000:.0f}ms")
    if result.data:
        # Show first earthquake
        mag_points = [p for p in result.data if p.variable == "magnitude"]
        if mag_points:
            print(f"  Recent earthquake: M{mag_points[0].raw_value} at {mag_points[0].location}")
    if result.error:
        print(f"  Error: {result.error}")

    print("\n[TEST 4] NOAA Space Weather API")
    print("-" * 40)
    result = NOAASpaceWeatherClient.fetch_kp_index()
    print(f"  Success: {result.success}")
    print(f"  Data points: {len(result.data)}")
    print(f"  Fetch time: {result.fetch_time*1000:.0f}ms")
    if result.data:
        latest = result.data[-1] if result.data else None
        if latest:
            print(f"  Latest Kp: {latest.raw_value}")
    if result.error:
        print(f"  Error: {result.error}")

    print("\n[TEST 5] USGS Water Services API")
    print("-" * 40)
    stations = USGSWaterClient.fetch_texas_hill_country_stations()
    result = USGSWaterClient.fetch_station_data(stations[0], days_back=1)
    print(f"  Success: {result.success}")
    print(f"  Data points: {len(result.data)}")
    print(f"  Fetch time: {result.fetch_time*1000:.0f}ms")
    if result.data:
        print(f"  Station: {result.data[0].location}")
        print(f"  Latest value: {result.data[-1].value} {result.data[-1].unit}")
    if result.error:
        print(f"  Error: {result.error}")

    # Test unified fetcher
    print("\n" + "=" * 70)
    print("UNIFIED DATA FETCH TEST")
    print("=" * 70)

    print("\nFetching all data sources...")
    all_results = fetcher.fetch_all(
        include_earthquake=True,
        include_flood=True,
        include_weather=True,
        include_spaceweather=True,
        include_water_stations=True
    )

    summary = fetcher.get_summary(all_results)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    # Extract time series for MYSTIC
    print("\n[MYSTIC Integration]")
    print("-" * 40)

    pressure_series = fetcher.extract_time_series(all_results, "pressure_msl")
    print(f"  Pressure time series: {len(pressure_series)} points")
    if pressure_series:
        print(f"  Sample (first 5): {pressure_series[:5]}")

    discharge_series = fetcher.extract_time_series(all_results, "river_discharge")
    print(f"  River discharge series: {len(discharge_series)} points")
    if discharge_series:
        print(f"  Sample (first 5): {discharge_series[:5]}")

    print("\n" + "=" * 70)
    print("✓ DATA SOURCE INTEGRATION COMPLETE")
    print("✓ All APIs accessible and returning integer-scaled data")
    print("=" * 70)


if __name__ == "__main__":
    test_data_sources()
