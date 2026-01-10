#!/usr/bin/env python3
"""
HISTORICAL WEATHER EVENT VALIDATION SUITE

Tests MYSTIC against archived, real-world historical CSV data
defined in predictive_gauntlet_events.json (offline, deterministic).

Author: Claude (K-Elimination Expert)
Date: 2026-01-07
Updated: 2026-01-08 - Switched validation to archived historical CSVs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from mystic_v3_production import MYSTICPredictorV3Production


@dataclass
class HistoricalEvent:
    """Represents a historical weather event for validation."""
    name: str
    description: str
    data: List[int]
    expected_risk: str
    expected_min_score: int
    source: str
    primary_series: str = "UNKNOWN"


DEFAULT_EVENTS_PATH = str(Path(__file__).with_name("predictive_gauntlet_events.json"))


def parse_float(value: str) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def relative_humidity(temp_c: float, dewpoint_c: float) -> float:
    """Approximate relative humidity from temp/dewpoint."""
    es = math.exp((17.625 * temp_c) / (243.04 + temp_c))
    esd = math.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c))
    return max(0.0, min(100.0, 100.0 * esd / es))


def read_usgs_csv(path: str) -> Dict[str, List[int]]:
    """Parse USGS CSV and return streamflow series (scaled)."""
    discharge_by_ts: Dict[str, float] = {}
    gage_by_ts: Dict[str, float] = {}

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row = {k.strip(): v for k, v in row.items() if k}
            timestamp = row.get("timestamp")
            if not timestamp:
                continue
            discharge = parse_float(row.get("discharge_cfs"))
            gage_height = parse_float(row.get("gage_height_ft"))
            if discharge is not None:
                prev = discharge_by_ts.get(timestamp)
                discharge_by_ts[timestamp] = discharge if prev is None else max(prev, discharge)
            if gage_height is not None:
                prev = gage_by_ts.get(timestamp)
                gage_by_ts[timestamp] = gage_height if prev is None else max(prev, gage_height)

    series: Dict[str, List[int]] = {}
    if discharge_by_ts:
        timestamps = sorted(discharge_by_ts.keys())
        series["streamflow"] = [int(round(discharge_by_ts[t] * 100)) for t in timestamps]
    elif gage_by_ts:
        timestamps = sorted(gage_by_ts.keys())
        series["streamflow"] = [int(round(gage_by_ts[t] * 100)) for t in timestamps]

    return series


def read_weather_csv(path: str) -> Dict[str, List[int]]:
    """Parse meteorological CSV and return scaled series."""
    pressure: List[int] = []
    precip: List[int] = []
    wind: List[int] = []
    temperature: List[int] = []
    humidity: List[int] = []

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row = {k.strip(): v for k, v in row.items() if k}

            pressure_hpa = parse_float(row.get("pressure_hpa"))
            precip_1hr = parse_float(row.get("precip_1hr_mm"))
            precip_total = parse_float(row.get("precip_mm"))
            wind_mps = parse_float(row.get("wind_speed_mps"))
            temp_c = parse_float(row.get("temp_c"))
            dewpoint_c = parse_float(row.get("dewpoint_c"))

            if pressure_hpa is not None:
                pressure.append(int(round(pressure_hpa * 10)))
            if precip_1hr is not None:
                precip.append(int(round(precip_1hr * 100)))
            elif precip_total is not None:
                precip.append(int(round(precip_total * 100)))
            if wind_mps is not None:
                wind.append(int(round(wind_mps * 3.6 * 10)))
            if temp_c is not None:
                temperature.append(int(round(temp_c * 100)))
            if temp_c is not None and dewpoint_c is not None:
                humidity.append(int(round(relative_humidity(temp_c, dewpoint_c))))

    series: Dict[str, List[int]] = {}
    if pressure:
        series["pressure"] = pressure
    if precip:
        series["precipitation"] = precip
    if wind:
        series["wind_speed"] = wind
    if temperature:
        series["temperature"] = temperature
    if humidity:
        series["humidity"] = humidity

    return series


def merge_payload(*sources: Dict[str, List[int]]) -> Dict[str, List[int]]:
    payload: Dict[str, List[int]] = {}
    for src in sources:
        for key, series in src.items():
            if series:
                payload[key] = series
    return payload


def choose_primary(payload: Dict[str, List[int]], hazard_type: str) -> Tuple[str, List[int]]:
    hazard_type = (hazard_type or "").upper()

    if hazard_type in ["FLASH_FLOOD", "HURRICANE"]:
        ordered = ["streamflow", "precipitation", "pressure"]
    elif hazard_type == "FIRE_WEATHER":
        ordered = ["humidity", "wind_speed", "temperature", "pressure"]
    elif hazard_type in ["TORNADO", "SEVERE_STORM"]:
        ordered = ["pressure", "wind_speed", "precipitation"]
    elif hazard_type == "STABLE":
        ordered = ["pressure", "streamflow", "temperature"]
    else:
        ordered = ["streamflow", "pressure", "precipitation", "humidity", "wind_speed", "temperature"]

    for key in ordered:
        series = payload.get(key, [])
        if series:
            return key, series

    return "UNKNOWN", []


def build_historical_events(events_path: str) -> Tuple[List[HistoricalEvent], Optional[str]]:
    try:
        with open(events_path, "r") as handle:
            events = json.load(handle)
    except FileNotFoundError:
        return [], f"Missing event config: {events_path}"
    except json.JSONDecodeError as exc:
        return [], f"Invalid JSON in {events_path}: {exc}"

    historical_events: List[HistoricalEvent] = []

    for event in events:
        payload: Dict[str, List[int]] = {}
        usgs_csv = event.get("usgs_csv")
        met_csv = event.get("met_csv")

        if usgs_csv:
            payload = merge_payload(payload, read_usgs_csv(usgs_csv))
        if met_csv:
            payload = merge_payload(payload, read_weather_csv(met_csv))

        if not payload:
            continue

        hazard_type = event.get("hazard_type", "HISTORICAL")
        primary_name, primary_series = choose_primary(payload, hazard_type)
        if not primary_series:
            continue

        source_parts = []
        if usgs_csv:
            source_parts.append(usgs_csv)
        if met_csv:
            source_parts.append(met_csv)
        record_reference = event.get("record_reference")
        source = ", ".join(source_parts) if source_parts else "archived CSVs"
        if record_reference:
            source = f"{record_reference} | CSV: {source}"
        else:
            source = f"CSV: {source}"

        historical_events.append(HistoricalEvent(
            name=event.get("name", event.get("id", "UNKNOWN")),
            description=event.get("name", "Historical event"),
            data=primary_series,
            expected_risk=event.get("expected_risk", "UNKNOWN"),
            expected_min_score=int(event.get("expected_min_score", 0)),
            source=source,
            primary_series=primary_name.upper()
        ))

    return historical_events, None


HISTORICAL_EVENTS, HISTORICAL_EVENTS_ERROR = build_historical_events(DEFAULT_EVENTS_PATH)


def run_historical_validation():
    """Run MYSTIC against historical weather event patterns."""
    print("=" * 75)
    print("HISTORICAL WEATHER EVENT VALIDATION SUITE")
    print("Testing MYSTIC against archived historical CSV data")
    print("=" * 75)

    if HISTORICAL_EVENTS_ERROR:
        print(f"\nMissing historical events: {HISTORICAL_EVENTS_ERROR}")
        return False
    if not HISTORICAL_EVENTS:
        print("\nNo historical events loaded.")
        return False

    predictor = MYSTICPredictorV3Production()

    results = []
    correct = 0
    total = len(HISTORICAL_EVENTS)

    for event in HISTORICAL_EVENTS:
        print(f"\n{'─' * 75}")
        print(f"EVENT: {event.name}")
        print(f"Description: {event.description}")
        print(f"Data points: {len(event.data)}")
        print(f"Source: {event.source}")
        print(f"Primary: {event.primary_series}")
        print(f"{'─' * 75}")

        result = predictor.predict(event.data, location="HISTORICAL", hazard_type="VALIDATION")

        # Check if prediction matches expected risk level
        risk_match = False
        if event.expected_risk == "LOW":
            risk_match = result.risk_level in ["LOW"]
        elif event.expected_risk == "MODERATE":
            risk_match = result.risk_level in ["LOW", "MODERATE"]
        elif event.expected_risk == "HIGH":
            risk_match = result.risk_level in ["HIGH", "CRITICAL"]
        elif event.expected_risk == "CRITICAL":
            risk_match = result.risk_level in ["HIGH", "CRITICAL"]

        # Check minimum score
        score_match = result.risk_score >= event.expected_min_score

        overall_match = risk_match and score_match
        if overall_match:
            correct += 1

        mark = "✓" if overall_match else "✗"

        print(f"  PREDICTION:")
        print(f"    Risk Level: {result.risk_level} (expected: {event.expected_risk}) {'✓' if risk_match else '✗'}")
        print(f"    Risk Score: {result.risk_score} (min expected: {event.expected_min_score}) {'✓' if score_match else '✗'}")
        print(f"    Trend: {result.trend_direction}")
        print(f"    Attractor: {result.attractor_classification}")
        print(f"    Lyapunov: {result.lyapunov.exponent_float:.4f} ({result.lyapunov.stability})")
        print(f"    Confidence: {result.confidence}%")
        print(f"  RESULT: {mark}")

        results.append({
            "event": event.name,
            "expected": event.expected_risk,
            "predicted": result.risk_level,
            "score": result.risk_score,
            "match": overall_match
        })

    # Summary
    accuracy = correct / total * 100

    print(f"\n{'=' * 75}")
    print("HISTORICAL VALIDATION SUMMARY")
    print(f"{'=' * 75}")
    print(f"  Events tested: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print()

    # Breakdown by risk level
    print("  By expected risk level:")
    for level in ["LOW", "MODERATE", "HIGH", "CRITICAL"]:
        level_results = [r for r in results if r["expected"] == level]
        if level_results:
            level_correct = sum(1 for r in level_results if r["match"])
            print(f"    {level}: {level_correct}/{len(level_results)}")

    print()
    if accuracy >= 70:
        print("✓ HISTORICAL VALIDATION PASSED (70%+ threshold)")
    else:
        print("○ Below 70% threshold - model needs calibration for real-world events")

    # Detailed failure analysis
    failures = [r for r in results if not r["match"]]
    if failures:
        print(f"\n  Failed cases ({len(failures)}):")
        for f in failures:
            print(f"    - {f['event']}: expected {f['expected']}, got {f['predicted']} (score: {f['score']})")

    return accuracy >= 70


def print_event_data_samples():
    """Print sample data from each event for inspection."""
    print("\n" + "=" * 75)
    print("SAMPLE DATA FROM HISTORICAL EVENTS")
    print("=" * 75)

    if HISTORICAL_EVENTS_ERROR:
        print(f"\nMissing historical events: {HISTORICAL_EVENTS_ERROR}")
        return
    if not HISTORICAL_EVENTS:
        print("\nNo historical events loaded.")
        return

    for event in HISTORICAL_EVENTS:
        print(f"\n{event.name}:")
        print(f"  Primary series: {event.primary_series}")
        print(f"  First 10 values: {event.data[:10]}")
        print(f"  Last 10 values: {event.data[-10:]}")
        print(f"  Range: {min(event.data)} - {max(event.data)}")
        print(f"  Mean: {sum(event.data) // len(event.data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run historical validation suite")
    parser.add_argument("--events", default=DEFAULT_EVENTS_PATH, help="Path to gauntlet events JSON")
    args = parser.parse_args()

    if args.events != DEFAULT_EVENTS_PATH:
        HISTORICAL_EVENTS, HISTORICAL_EVENTS_ERROR = build_historical_events(args.events)

    print_event_data_samples()
    print()
    success = run_historical_validation()
    raise SystemExit(0 if success else 1)
