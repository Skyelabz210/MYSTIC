#!/usr/bin/env python3
"""
HISTORICAL DATA LOADER - Real-World Event Data for MYSTIC

Replaces synthetic test patterns with actual historical data from:
- USGS Water Services (streamflow, gage height)
- Open-Meteo Archive (pressure, precipitation, temperature, humidity)
- NOAA CO-OPS (water levels, tides)

Events with verified historical data:
1. Hurricane Harvey (Aug 2017) - Houston flooding
2. Blanco River Flash Flood (May 2015) - Wimberley
3. Camp Fire (Nov 2018) - Paradise, CA fire weather
4. Memorial Day Floods Texas (May 2015)
5. Joplin Tornado region (May 2011)

Author: Claude (K-Elimination Expert)
Date: 2026-01-07
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from data_sources_extended import (
    USGSWaterServices, OpenMeteoWeather, NOAATidesCurrents
)


@dataclass
class HistoricalEvent:
    """Represents a historical weather event with real data."""
    name: str
    description: str
    event_date: str
    location: Dict[str, float]  # lat, lon
    usgs_sites: List[str]
    data: Dict[str, List[int]]  # Variable -> integer-scaled values
    expected_risk: str
    expected_min_score: int
    source: str
    data_quality: str  # VERIFIED, PARTIAL, RECONSTRUCTED


class HistoricalDataLoader:
    """
    Loads real historical weather event data from APIs.

    All data is integer-scaled for QMNF compatibility.
    """

    def __init__(self):
        self.usgs = USGSWaterServices()
        self.weather = OpenMeteoWeather()
        self.coops = NOAATidesCurrents()

        # Define known historical events
        self.events = {
            "harvey_2017": {
                "name": "Hurricane Harvey (2017)",
                "description": "Category 4 hurricane, catastrophic Houston flooding",
                "start_date": "2017-08-25",
                "end_date": "2017-08-31",
                "lat": 29.76,
                "lon": -95.37,
                "usgs_sites": ["08074000", "08075000"],  # Buffalo Bayou, Brays Bayou
                "hazard_type": "HURRICANE",
                "expected_risk": "CRITICAL",
                "expected_min_score": 70,
                "lead_window_ratio": 0.5,
                "lead_min_score": 60,
            },
            "blanco_2015": {
                "name": "Blanco River Flash Flood (2015)",
                "description": "Memorial Day flash flood, Wimberley TX",
                "start_date": "2015-05-23",
                "end_date": "2015-05-26",
                "lat": 29.99,
                "lon": -98.10,
                "usgs_sites": ["08171000"],  # Blanco River at Wimberley
                "hazard_type": "FLASH_FLOOD",
                "expected_risk": "CRITICAL",
                "expected_min_score": 70,
                "lead_window_ratio": 0.6,
                "lead_min_score": 50,
            },
            "camp_mystic_2007": {
                "name": "Camp Mystic Flood (2007)",
                "description": "Guadalupe River flash flood near Hunt, TX",
                "start_date": "2007-06-21",
                "end_date": "2007-07-01",
                "lat": 30.05,
                "lon": -99.33,
                "usgs_sites": ["08165500"],  # Guadalupe River at Hunt, TX
                "hazard_type": "FLASH_FLOOD",
                "expected_risk": "CRITICAL",
                "expected_min_score": 70,
                "lead_window_ratio": 0.6,
                "lead_min_score": 50,
            },
            "camp_fire_2018": {
                "name": "Camp Fire (2018)",
                "description": "Deadliest California wildfire, Paradise CA",
                "start_date": "2018-11-08",
                "end_date": "2018-11-12",
                "lat": 39.76,
                "lon": -121.62,
                "usgs_sites": [],
                "hazard_type": "FIRE_WEATHER",
                "expected_risk": "HIGH",
                "expected_min_score": 50,
                "lead_window_ratio": 0.6,
                "lead_min_score": 35,
            },
            "halloween_2013": {
                "name": "Halloween Floods (2013)",
                "description": "Central Texas flooding across Austin metro",
                "start_date": "2013-10-24",
                "end_date": "2013-11-03",
                "lat": 30.27,
                "lon": -97.74,
                "usgs_sites": ["08159000"],  # Onion Creek at US Hwy 183, Austin
                "hazard_type": "FLASH_FLOOD",
                "expected_risk": "HIGH",
                "expected_min_score": 50,
                "lead_window_ratio": 0.6,
                "lead_min_score": 40,
            },
            "joplin_2011": {
                "name": "Joplin Tornado Region (2011)",
                "description": "EF5 tornado conditions, Joplin MO",
                "start_date": "2011-05-22",
                "end_date": "2011-05-23",
                "lat": 37.08,
                "lon": -94.51,
                "usgs_sites": [],
                "hazard_type": "TORNADO",
                "expected_risk": "CRITICAL",
                "expected_min_score": 70,
                "lead_window_ratio": 0.5,
                "lead_min_score": 40,
            },
            "llano_2018": {
                "name": "Llano River Flood (2018)",
                "description": "Llano River flash flood near Llano, TX",
                "start_date": "2018-10-09",
                "end_date": "2018-10-19",
                "lat": 30.75,
                "lon": -98.67,
                "usgs_sites": ["08151500"],  # Llano River at Llano, TX
                "hazard_type": "FLASH_FLOOD",
                "expected_risk": "HIGH",
                "expected_min_score": 50,
                "lead_window_ratio": 0.6,
                "lead_min_score": 40,
            },
            "memorial_day_2015": {
                "name": "Memorial Day Floods (2015)",
                "description": "Central Texas Memorial Day flood sequence",
                "start_date": "2015-05-17",
                "end_date": "2015-05-27",
                "lat": 29.99,
                "lon": -98.10,
                "usgs_sites": ["08171000"],  # Blanco River at Wimberley
                "hazard_type": "FLASH_FLOOD",
                "expected_risk": "CRITICAL",
                "expected_min_score": 70,
                "lead_window_ratio": 0.6,
                "lead_min_score": 50,
            },
            "memorial_day_2016": {
                "name": "Memorial Day Floods (2016)",
                "description": "San Antonio area flood event",
                "start_date": "2016-05-20",
                "end_date": "2016-05-30",
                "lat": 29.42,
                "lon": -98.49,
                "usgs_sites": ["08178050"],  # San Antonio River at Mitchell St
                "hazard_type": "FLASH_FLOOD",
                "expected_risk": "HIGH",
                "expected_min_score": 50,
                "lead_window_ratio": 0.6,
                "lead_min_score": 40,
            },
            "tax_day_2016": {
                "name": "Tax Day Flood (2016)",
                "description": "Houston metro flood event",
                "start_date": "2016-04-11",
                "end_date": "2016-04-21",
                "lat": 29.76,
                "lon": -95.37,
                "usgs_sites": ["08074000"],  # Buffalo Bayou at Houston
                "hazard_type": "FLASH_FLOOD",
                "expected_risk": "HIGH",
                "expected_min_score": 50,
                "lead_window_ratio": 0.6,
                "lead_min_score": 40,
            },
            "imelda_2019": {
                "name": "Tropical Storm Imelda (2019)",
                "description": "Extreme rainfall and flooding in SE Texas",
                "start_date": "2019-09-12",
                "end_date": "2019-09-22",
                "lat": 30.09,
                "lon": -94.10,
                "usgs_sites": ["08041780"],  # Neches River at Beaumont
                "hazard_type": "HURRICANE",
                "expected_risk": "CRITICAL",
                "expected_min_score": 70,
                "lead_window_ratio": 0.5,
                "lead_min_score": 60,
            },
            "derecho_2012": {
                "name": "June 2012 Derecho",
                "description": "Long-lived derecho from Midwest to Mid-Atlantic",
                "start_date": "2012-06-29",
                "end_date": "2012-06-30",
                "lat": 39.95,
                "lon": -83.00,  # Columbus, OH area
                "usgs_sites": [],
                "hazard_type": "SEVERE_STORM",
                "expected_risk": "HIGH",
                "expected_min_score": 50,
                "lead_window_ratio": 0.5,
                "lead_min_score": 40,
            },
            "stable_reference": {
                "name": "Stable Weather Reference",
                "description": "Clear weather period for baseline comparison",
                "start_date": "2020-07-15",
                "end_date": "2020-07-17",
                "lat": 30.05,
                "lon": -99.17,
                "usgs_sites": ["08171000"],
                "hazard_type": "STABLE",
                "expected_risk": "LOW",
                "expected_min_score": 0,
                "lead_window_ratio": 0.0,
                "lead_min_score": 0,
            },
        }

    def fetch_event_data(self, event_key: str) -> Optional[HistoricalEvent]:
        """
        Fetch all available data for a historical event.

        Returns HistoricalEvent with integer-scaled data.
        """
        if event_key not in self.events:
            return None

        event = self.events[event_key]
        data = {}
        quality_issues = []

        # Fetch weather data (pressure, precipitation)
        print(f"  Fetching weather data...")
        weather_data = self.weather.get_historical(
            lat=event["lat"],
            lon=event["lon"],
            start_date=event["start_date"],
            end_date=event["end_date"],
            hourly=["pressure_msl", "precipitation", "temperature_2m",
                   "relative_humidity_2m", "wind_speed_10m"]
        )

        if "error" not in weather_data:
            hourly = weather_data.get("hourly", {})

            # Scale to integers
            if hourly.get("pressure_msl"):
                data["pressure"] = [int(p * 10) for p in hourly["pressure_msl"]]
            if hourly.get("precipitation"):
                data["precipitation"] = [int(p * 100) for p in hourly["precipitation"]]
            if hourly.get("temperature_2m"):
                data["temperature"] = [int(t * 100) for t in hourly["temperature_2m"]]
            if hourly.get("relative_humidity_2m"):
                data["humidity"] = [int(h) for h in hourly["relative_humidity_2m"]]
            if hourly.get("wind_speed_10m"):
                data["wind_speed"] = [int(w * 10) for w in hourly["wind_speed_10m"]]
        else:
            quality_issues.append("Weather data unavailable")

        # Fetch USGS streamflow if sites specified
        usgs_source = "USGS NONE"
        if event["usgs_sites"]:
            print(f"  Fetching USGS streamflow...")

            def _extract_usgs_series(response: Dict[str, Any]) -> Dict[str, List[int]]:
                extracted = {}
                ts = response.get("value", {}).get("timeSeries", [])
                for series in ts:
                    var_name = series.get("variable", {}).get("variableName", "unknown")
                    values = series.get("values", [{}])[0].get("value", [])
                    parsed = [
                        int(float(v.get("value", 0)) * 100)
                        for v in values if v.get("value")
                    ]
                    if not parsed:
                        continue
                    if "Streamflow" in var_name:
                        extracted["streamflow"] = parsed
                    elif "Gage height" in var_name:
                        extracted["gage_height"] = parsed
                return extracted

            usgs_data = self.usgs.fetch_instantaneous_range(
                sites=event["usgs_sites"],
                parameters=["00060", "00065"],
                start_date=event["start_date"],
                end_date=event["end_date"]
            )
            usgs_parsed = {} if "error" in usgs_data else _extract_usgs_series(usgs_data)

            if usgs_parsed:
                data.update(usgs_parsed)
                usgs_source = "USGS IV"
            else:
                usgs_data = self.usgs.fetch_daily_values(
                    sites=event["usgs_sites"],
                    parameters=["00060", "00065"],
                    start_date=event["start_date"],
                    end_date=event["end_date"]
                )
                usgs_parsed = {} if "error" in usgs_data else _extract_usgs_series(usgs_data)

                if usgs_parsed:
                    data.update(usgs_parsed)
                    usgs_source = "USGS DV"
                    quality_issues.append("USGS IV unavailable; used daily values")
                else:
                    quality_issues.append("USGS data unavailable")

        # Determine data quality
        if not quality_issues and len(data) >= 3:
            quality = "VERIFIED"
        elif len(data) >= 1:
            quality = "PARTIAL"
        else:
            quality = "UNAVAILABLE"

        return HistoricalEvent(
            name=event["name"],
            description=event["description"],
            event_date=event["start_date"],
            location={"lat": event["lat"], "lon": event["lon"]},
            usgs_sites=event["usgs_sites"],
            data=data,
            expected_risk=event["expected_risk"],
            expected_min_score=event["expected_min_score"],
            source=f"{usgs_source}+Open-Meteo ({quality_issues if quality_issues else 'complete'})",
            data_quality=quality
        )

    def fetch_all_events(self) -> List[HistoricalEvent]:
        """Fetch data for all defined historical events."""
        events = []
        for key in self.events:
            print(f"\nLoading: {self.events[key]['name']}")
            event = self.fetch_event_data(key)
            if event and event.data:
                events.append(event)
        return events

    def get_pressure_series(self, event: HistoricalEvent) -> List[int]:
        """Get pressure time series for MYSTIC analysis."""
        return event.data.get("pressure", [])

    def get_streamflow_series(self, event: HistoricalEvent) -> List[int]:
        """Get streamflow time series for MYSTIC analysis."""
        return event.data.get("streamflow", [])


def run_historical_data_test():
    """Test historical data loading."""
    print("=" * 70)
    print("HISTORICAL DATA LOADER TEST")
    print("Loading real-world event data from APIs")
    print("=" * 70)

    loader = HistoricalDataLoader()

    # Test specific events
    test_events = ["harvey_2017", "blanco_2015", "camp_fire_2018", "stable_reference"]

    loaded_events = []

    for event_key in test_events:
        print(f"\n{'─' * 70}")
        print(f"Loading: {loader.events[event_key]['name']}")
        print(f"{'─' * 70}")

        event = loader.fetch_event_data(event_key)

        if event:
            print(f"  Quality: {event.data_quality}")
            print(f"  Available data:")
            for var, values in event.data.items():
                if values:
                    print(f"    {var}: {len(values)} points, range [{min(values)} - {max(values)}]")
            loaded_events.append(event)
        else:
            print("  ✗ Failed to load")

    # Summary
    print(f"\n{'=' * 70}")
    print("HISTORICAL DATA SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Events loaded: {len(loaded_events)}/{len(test_events)}")

    for event in loaded_events:
        total_points = sum(len(v) for v in event.data.values())
        print(f"  • {event.name}: {total_points} total data points ({event.data_quality})")

    return len(loaded_events) == len(test_events)


def create_mystic_validation_events() -> List[Dict[str, Any]]:
    """
    Create validation events using real historical data.

    Returns list compatible with historical_validation.py format.
    """
    print("Creating MYSTIC validation events from real data...")

    loader = HistoricalDataLoader()
    validation_events = []

    for event_key, event_config in loader.events.items():
        print(f"  Loading {event_config['name']}...")
        event = loader.fetch_event_data(event_key)

        if event and event.data:
            # Prefer pressure for weather events, streamflow for floods
            if "streamflow" in event.data and event.data["streamflow"]:
                primary_data = event.data["streamflow"]
                data_type = "streamflow"
            elif "pressure" in event.data and event.data["pressure"]:
                primary_data = event.data["pressure"]
                data_type = "pressure"
            elif "humidity" in event.data and event.data["humidity"]:
                primary_data = event.data["humidity"]
                data_type = "humidity"
            else:
                continue

            validation_events.append({
                "name": event.name,
                "description": event.description,
                "data": primary_data,
                "data_type": data_type,
                "expected_risk": event.expected_risk,
                "expected_min_score": event.expected_min_score,
                "source": event.source,
                "quality": event.data_quality,
            })

    return validation_events


if __name__ == "__main__":
    success = run_historical_data_test()

    print("\n" + "=" * 70)
    print("VALIDATION EVENT GENERATION")
    print("=" * 70)

    events = create_mystic_validation_events()
    print(f"\nGenerated {len(events)} validation events from real data:")
    for e in events:
        print(f"  • {e['name']}: {len(e['data'])} {e['data_type']} points ({e['quality']})")

    exit(0 if success else 1)
