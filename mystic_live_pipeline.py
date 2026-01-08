#!/usr/bin/env python3
"""
MYSTIC Live Pipeline

Fetches live data via MYSTICDataHub and runs MYSTIC V3 integrated prediction.
"""

from typing import Dict, List
import argparse

from data_sources_extended import MYSTICDataHub
from mystic_v3_integrated import MYSTICPredictorV3


def build_payload(results: Dict) -> Dict[str, List[int]]:
    """Convert MYSTICDataHub results into MultiVariableAnalyzer payload."""
    data: Dict[str, List[int]] = {}
    sources = results.get("sources", {})

    weather = sources.get("weather", {})
    if weather.get("pressure_series"):
        data["pressure"] = weather["pressure_series"]
    if weather.get("precipitation_series"):
        data["precipitation"] = weather["precipitation_series"]
    if weather.get("temperature_series"):
        data["temperature"] = weather["temperature_series"]
    if weather.get("humidity_series"):
        data["humidity"] = weather["humidity_series"]
    if weather.get("wind_speed_series"):
        data["wind_speed"] = weather["wind_speed_series"]

    usgs = sources.get("usgs", {})
    streamflow = usgs.get("streamflow_series", [])
    gage_height = usgs.get("gage_height_series", [])
    if streamflow:
        data["streamflow"] = streamflow
    elif gage_height:
        data["streamflow"] = gage_height

    return data


def select_primary_series(data: Dict[str, List[int]]) -> Dict[str, List[int]]:
    """Pick a primary series for the V3 predictor."""
    priority = [
        "pressure",
        "streamflow",
        "precipitation",
        "humidity",
        "wind_speed",
        "temperature",
    ]
    for key in priority:
        series = data.get(key, [])
        if series:
            return {"name": key, "series": series}
    return {"name": "UNKNOWN", "series": []}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MYSTIC live multi-variable analysis")
    parser.add_argument("--lat", type=float, default=None, help="Latitude")
    parser.add_argument("--lon", type=float, default=None, help="Longitude")
    parser.add_argument("--usgs-sites", type=str, default=None, help="Comma-separated USGS site IDs")
    parser.add_argument("--cache-ttl", type=int, default=300, help="Cache TTL seconds")
    args = parser.parse_args()

    sites = None
    if args.usgs_sites:
        sites = [s.strip() for s in args.usgs_sites.split(",") if s.strip()]

    hub = MYSTICDataHub(cache_ttl_seconds=args.cache_ttl)
    results = hub.fetch_comprehensive(lat=args.lat, lon=args.lon, usgs_sites=sites)
    payload = build_payload(results)

    if not payload:
        print("No usable data returned. Check data sources or network access.")
        return 1

    primary = select_primary_series(payload)
    if not primary["series"]:
        print("No primary series available for prediction.")
        return 1

    predictor = MYSTICPredictorV3()
    analysis = predictor.predict(
        primary["series"],
        location="LIVE",
        hazard_type="LIVE",
        multi_variable_data=payload
    )

    print("MYSTIC LIVE ANALYSIS (INTEGRATED)")
    print(f"  Primary: {primary['name']} ({len(primary['series'])} points)")
    print(f"  Risk: {analysis.risk_level} ({analysis.risk_score})")
    print(f"  Confidence: {analysis.confidence}%")
    print(f"  Attractor: {analysis.attractor_classification}")
    print(f"  Lyapunov: {analysis.lyapunov.stability}")

    summary = analysis.multi_variable_summary or {}
    if summary:
        print(f"  Multi-Variable Hazard: {summary.get('hazard_type', 'UNKNOWN')}")
        print(f"  Multi-Variable Risk: {summary.get('composite_risk', 'UNKNOWN')} ({summary.get('composite_score', 0)})")
        signals = summary.get("signals", [])
        if signals:
            print(f"  Signals: {', '.join(signals)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
