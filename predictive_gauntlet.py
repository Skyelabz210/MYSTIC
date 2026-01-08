#!/usr/bin/env python3
"""
MYSTIC PREDICTIVE GAUNTLET

Runs MYSTIC V3 against archived historical events and evaluates predictions
against expected outcomes from historical records.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Any, Tuple, Optional

from mystic_v3_integrated import MYSTICPredictorV3


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


def choose_primary(payload: Dict[str, List[int]], event: Dict[str, Any]) -> Tuple[str, List[int]]:
    preferred = event.get("primary_series")
    if preferred and payload.get(preferred):
        return preferred, payload[preferred]

    for key in ["pressure", "streamflow", "precipitation", "humidity", "wind_speed", "temperature"]:
        series = payload.get(key, [])
        if series:
            return key, series

    return "UNKNOWN", []


def slice_payload(payload: Dict[str, List[int]], ratio: float) -> Dict[str, List[int]]:
    sliced: Dict[str, List[int]] = {}
    for key, series in payload.items():
        if not series:
            continue
        end = max(3, int(len(series) * ratio))
        sliced[key] = series[:end]
    return sliced


def risk_match(expected: str, predicted: str) -> bool:
    expected = expected.upper()
    predicted = predicted.upper()
    if expected == "LOW":
        return predicted in ["LOW", "MODERATE"]
    if expected == "MODERATE":
        return predicted in ["MODERATE", "HIGH"]
    if expected == "HIGH":
        return predicted in ["HIGH", "CRITICAL"]
    if expected == "CRITICAL":
        return predicted in ["HIGH", "CRITICAL"]
    return False


def run_gauntlet(events_path: str, output_path: Optional[str] = None) -> int:
    with open(events_path, "r") as handle:
        events = json.load(handle)

    predictor = MYSTICPredictorV3()
    results: List[Dict[str, Any]] = []

    risk_hits = 0
    score_hits = 0
    hazard_hits = 0
    lead_hits = 0

    for event in events:
        payload = {}
        usgs_csv = event.get("usgs_csv")
        met_csv = event.get("met_csv")

        if usgs_csv:
            payload = merge_payload(payload, read_usgs_csv(usgs_csv))
        if met_csv:
            payload = merge_payload(payload, read_weather_csv(met_csv))

        primary_name, primary_series = choose_primary(payload, event)
        if not primary_series:
            print(f"[SKIP] {event['id']}: no usable series")
            continue

        prediction = predictor.predict(
            primary_series,
            location=event.get("id", "HISTORICAL"),
            hazard_type=event.get("hazard_type", "HISTORICAL"),
            multi_variable_data=payload
        )

        expected_risk = event.get("expected_risk", "UNKNOWN")
        expected_min_score = int(event.get("expected_min_score", 0))

        risk_ok = risk_match(expected_risk, prediction.risk_level)
        score_ok = prediction.risk_score >= expected_min_score

        hazard_expected = event.get("hazard_type")
        hazard_predicted = None
        if prediction.multi_variable_summary:
            hazard_predicted = prediction.multi_variable_summary.get("hazard_type")
        hazard_ok = bool(hazard_expected and hazard_predicted and hazard_expected == hazard_predicted)

        lead_ratio = float(event.get("lead_window_ratio", 0.0))
        lead_ok = False
        lead_score = None
        lead_level = None
        if lead_ratio > 0:
            sliced_payload = slice_payload(payload, lead_ratio)
            lead_primary = slice_payload({"primary": primary_series}, lead_ratio)["primary"]
            lead_prediction = predictor.predict(
                lead_primary,
                location=event.get("id", "HISTORICAL"),
                hazard_type=event.get("hazard_type", "HISTORICAL"),
                multi_variable_data=sliced_payload
            )
            lead_score = lead_prediction.risk_score
            lead_level = lead_prediction.risk_level
            lead_min_score = int(event.get("lead_min_score", max(0, expected_min_score * 7 // 10)))
            lead_ok = lead_score >= lead_min_score

        if risk_ok:
            risk_hits += 1
        if score_ok:
            score_hits += 1
        if hazard_ok:
            hazard_hits += 1
        if lead_ratio > 0 and lead_ok:
            lead_hits += 1

        results.append({
            "id": event.get("id"),
            "name": event.get("name"),
            "primary_series": primary_name,
            "expected_risk": expected_risk,
            "predicted_risk": prediction.risk_level,
            "risk_score": prediction.risk_score,
            "risk_ok": risk_ok,
            "score_ok": score_ok,
            "hazard_expected": hazard_expected,
            "hazard_predicted": hazard_predicted,
            "hazard_ok": hazard_ok,
            "lead_ok": lead_ok,
            "lead_score": lead_score,
            "lead_level": lead_level,
            "record_reference": event.get("record_reference")
        })

        print(f"EVENT {event['id']}: risk={prediction.risk_level} score={prediction.risk_score} lead_ok={lead_ok}")

    total = len(results)
    hazard_total = sum(1 for r in results if r.get("hazard_expected"))
    lead_total = sum(1 for r in results if r.get("lead_score") is not None)

    summary = {
        "events_tested": total,
        "risk_accuracy": round((risk_hits / total) * 100, 2) if total else 0,
        "score_accuracy": round((score_hits / total) * 100, 2) if total else 0,
        "hazard_accuracy": round((hazard_hits / hazard_total) * 100, 2) if hazard_total else 0,
        "lead_success": round((lead_hits / lead_total) * 100, 2) if lead_total else 0,
        "results": results,
    }

    print("\nGAUNTLET SUMMARY")
    print(f"  Events tested: {total}")
    print(f"  Risk accuracy: {summary['risk_accuracy']}%")
    print(f"  Score accuracy: {summary['score_accuracy']}%")
    if hazard_total:
        print(f"  Hazard accuracy: {summary['hazard_accuracy']}%")
    if lead_total:
        print(f"  Lead success: {summary['lead_success']}%")

    if output_path:
        with open(output_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"\nReport written to {output_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MYSTIC predictive gauntlet")
    parser.add_argument("--events", default="predictive_gauntlet_events.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    return run_gauntlet(args.events, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
