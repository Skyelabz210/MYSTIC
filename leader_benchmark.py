#!/usr/bin/env python3
"""
MYSTIC Leader Benchmark

Compares MYSTIC lead time vs a simple baseline against NOAA Storm Events
begin times (Texas) using archived historical CSV data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from mystic_v3_integrated import MYSTICPredictorV3


DEFAULT_EVENTS_PATH = Path(__file__).with_name("predictive_gauntlet_events.json")
DEFAULT_NOAA_PATH = Path("nine65_v2_complete/data/historical/storm_events_texas_2000_2024.csv")


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


def parse_iso_timestamp(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.replace(tzinfo=None)
    except ValueError:
        try:
            dt = datetime.fromisoformat(value.split(".")[0])
            return dt.replace(tzinfo=None)
        except ValueError:
            return None


def parse_noaa_timestamp(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%d-%b-%y %H:%M:%S")
    except ValueError:
        try:
            return datetime.strptime(value.strip(), "%d-%b-%Y %H:%M:%S")
        except ValueError:
            return None


def relative_humidity(temp_c: float, dewpoint_c: float) -> float:
    es = math.exp((17.625 * temp_c) / (243.04 + temp_c))
    esd = math.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c))
    return max(0.0, min(100.0, 100.0 * esd / es))


def read_usgs_csv_with_ts(path: str) -> Tuple[Dict[str, List[int]], Dict[str, List[datetime]]]:
    discharge_by_ts: Dict[str, float] = {}
    gage_by_ts: Dict[str, float] = {}
    ts_lookup: Dict[str, datetime] = {}

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row = {k.strip(): v for k, v in row.items() if k}
            timestamp = row.get("timestamp")
            if not timestamp:
                continue
            dt = parse_iso_timestamp(timestamp)
            if not dt:
                continue
            ts_lookup[timestamp] = dt
            discharge = parse_float(row.get("discharge_cfs"))
            gage_height = parse_float(row.get("gage_height_ft"))
            if discharge is not None:
                prev = discharge_by_ts.get(timestamp)
                discharge_by_ts[timestamp] = discharge if prev is None else max(prev, discharge)
            if gage_height is not None:
                prev = gage_by_ts.get(timestamp)
                gage_by_ts[timestamp] = gage_height if prev is None else max(prev, gage_height)

    series: Dict[str, List[int]] = {}
    timestamps: Dict[str, List[datetime]] = {}
    if discharge_by_ts:
        ordered = sorted(discharge_by_ts.items(), key=lambda item: ts_lookup[item[0]])
        series["streamflow"] = [int(round(value * 100)) for _, value in ordered]
        timestamps["streamflow"] = [ts_lookup[key] for key, _ in ordered]
    elif gage_by_ts:
        ordered = sorted(gage_by_ts.items(), key=lambda item: ts_lookup[item[0]])
        series["streamflow"] = [int(round(value * 100)) for _, value in ordered]
        timestamps["streamflow"] = [ts_lookup[key] for key, _ in ordered]

    return series, timestamps


def read_weather_csv_with_ts(path: str) -> Tuple[Dict[str, List[int]], Dict[str, List[datetime]]]:
    series: Dict[str, List[int]] = {
        "pressure": [],
        "precipitation": [],
        "wind_speed": [],
        "temperature": [],
        "humidity": [],
    }
    timestamps: Dict[str, List[datetime]] = {key: [] for key in series}

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row = {k.strip(): v for k, v in row.items() if k}
            timestamp = row.get("timestamp")
            dt = parse_iso_timestamp(timestamp) if timestamp else None
            if not dt:
                continue

            pressure_hpa = parse_float(row.get("pressure_hpa"))
            precip_1hr = parse_float(row.get("precip_1hr_mm"))
            precip_total = parse_float(row.get("precip_mm"))
            wind_mps = parse_float(row.get("wind_speed_mps"))
            temp_c = parse_float(row.get("temp_c"))
            dewpoint_c = parse_float(row.get("dewpoint_c"))

            if pressure_hpa is not None:
                series["pressure"].append(int(round(pressure_hpa * 10)))
                timestamps["pressure"].append(dt)
            if precip_1hr is not None:
                series["precipitation"].append(int(round(precip_1hr * 100)))
                timestamps["precipitation"].append(dt)
            elif precip_total is not None:
                series["precipitation"].append(int(round(precip_total * 100)))
                timestamps["precipitation"].append(dt)
            if wind_mps is not None:
                series["wind_speed"].append(int(round(wind_mps * 3.6 * 10)))
                timestamps["wind_speed"].append(dt)
            if temp_c is not None:
                series["temperature"].append(int(round(temp_c * 100)))
                timestamps["temperature"].append(dt)
            if temp_c is not None and dewpoint_c is not None:
                series["humidity"].append(int(round(relative_humidity(temp_c, dewpoint_c))))
                timestamps["humidity"].append(dt)

    series = {k: v for k, v in series.items() if v}
    timestamps = {k: v for k, v in timestamps.items() if v}
    return series, timestamps


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


def slice_payload(payload: Dict[str, List[int]], ratio: float) -> Dict[str, List[int]]:
    sliced: Dict[str, List[int]] = {}
    for key, series in payload.items():
        if not series:
            continue
        end = max(3, int(len(series) * ratio))
        sliced[key] = series[:end]
    return sliced


def baseline_risk_score(payload: Dict[str, List[int]]) -> int:
    score = 0

    streamflow = payload.get("streamflow", [])
    if streamflow:
        min_flow = min(streamflow)
        max_flow = max(streamflow)
        ratio = max_flow / max(1, min_flow)
        if ratio >= 10:
            score = max(score, 90)
        elif ratio >= 5:
            score = max(score, 75)
        elif ratio >= 3:
            score = max(score, 60)
        elif ratio >= 2:
            score = max(score, 40)

    precip = payload.get("precipitation", [])
    if precip:
        max_precip = max(precip)
        if max_precip >= 5000:
            score = max(score, 80)
        elif max_precip >= 2500:
            score = max(score, 60)

    wind = payload.get("wind_speed", [])
    if wind:
        max_wind = max(wind)
        if max_wind >= 500:
            score = max(score, 80)
        elif max_wind >= 300:
            score = max(score, 60)

    pressure = payload.get("pressure", [])
    if pressure:
        min_pressure = min(pressure)
        if min_pressure <= 9800:
            score = max(score, 80)
        elif min_pressure <= 10000:
            score = max(score, 60)
        if len(pressure) > 1:
            max_drop = min(pressure[i + 1] - pressure[i] for i in range(len(pressure) - 1))
            if max_drop <= -30:
                score = max(score, 70)
            elif max_drop <= -15:
                score = max(score, 50)

    humidity = payload.get("humidity", [])
    if humidity:
        min_hum = min(humidity)
        if min_hum <= 15:
            score = max(score, 70)
        elif min_hum <= 25:
            score = max(score, 50)

    temperature = payload.get("temperature", [])
    if temperature:
        max_temp = max(temperature)
        if max_temp >= 4000:
            score = max(score, 60)
        elif max_temp >= 3500:
            score = max(score, 40)

    return score


def find_detection_index(
    series: List[int],
    payload: Dict[str, List[int]],
    lead_min_score: int,
    predictor: Optional[MYSTICPredictorV3] = None,
    hazard_type: str = "HISTORICAL"
) -> Optional[int]:
    if not series:
        return None

    n = len(series)
    step = max(1, n // 200)
    start_index = None

    for idx in range(step, n + 1, step):
        ratio = idx / n
        sliced_payload = slice_payload(payload, ratio)
        if predictor:
            prediction = predictor.predict(
                series[:idx],
                location="BENCHMARK",
                hazard_type=hazard_type,
                multi_variable_data=sliced_payload
            )
            score = prediction.risk_score
        else:
            score = baseline_risk_score(sliced_payload)
        if score >= lead_min_score:
            start_index = idx
            break

    if start_index is None:
        return None

    refine_start = max(1, start_index - step)
    for idx in range(refine_start, start_index + 1):
        ratio = idx / n
        sliced_payload = slice_payload(payload, ratio)
        if predictor:
            prediction = predictor.predict(
                series[:idx],
                location="BENCHMARK",
                hazard_type=hazard_type,
                multi_variable_data=sliced_payload
            )
            score = prediction.risk_score
        else:
            score = baseline_risk_score(sliced_payload)
        if score >= lead_min_score:
            return idx

    return start_index


def load_noaa_events(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("state") != "TEXAS":
                continue
            begin_date = parse_noaa_timestamp(row.get("begin_date", ""))
            if not begin_date:
                continue
            events.append({
                "event_type": row.get("event_type", ""),
                "begin_date": begin_date,
                "source": row.get("source", ""),
                "cz_name": row.get("cz_name", ""),
                "event_id": row.get("event_id", ""),
            })
    return events


def map_noaa_event(
    events: List[Dict[str, Any]],
    hazard_type: str,
    window_start: datetime,
    window_end: datetime
) -> Optional[Dict[str, Any]]:
    hazard_type = (hazard_type or "").upper()
    if hazard_type == "FLASH_FLOOD":
        types = {"Flash Flood", "Flood", "Heavy Rain"}
    elif hazard_type == "HURRICANE":
        types = {"Hurricane", "Tropical Storm"}
    elif hazard_type == "FIRE_WEATHER":
        types = {"Wildfire", "Fire"}
    elif hazard_type == "TORNADO":
        types = {"Tornado"}
    elif hazard_type == "SEVERE_STORM":
        types = {"Thunderstorm Wind", "Severe Thunderstorm"}
    else:
        types = {hazard_type.title()}

    candidates = [
        event for event in events
        if event["event_type"] in types
        and window_start <= event["begin_date"] <= window_end
    ]

    if not candidates:
        return None

    candidates.sort(key=lambda event: event["begin_date"])
    return candidates[0]


def compute_lead_hours(event_time: Optional[datetime], detection_time: Optional[datetime]) -> Optional[float]:
    if not event_time or not detection_time:
        return None
    delta = event_time - detection_time
    return round(delta.total_seconds() / 3600.0, 2)


def run_benchmark(events_path: Path, noaa_path: Path, output_path: Path) -> int:
    with events_path.open("r") as handle:
        events_config = json.load(handle)

    predictor = MYSTICPredictorV3()
    noaa_events = load_noaa_events(noaa_path)

    results = []
    lead_hours_mystic = []
    lead_hours_baseline = []
    lead_hours_mystic_noaa = []
    lead_hours_baseline_noaa = []

    for event in events_config:
        payload = {}
        timestamp_map: Dict[str, List[datetime]] = {}
        usgs_csv = event.get("usgs_csv")
        met_csv = event.get("met_csv")

        if usgs_csv:
            usgs_series, usgs_ts = read_usgs_csv_with_ts(usgs_csv)
            payload = merge_payload(payload, usgs_series)
            timestamp_map.update(usgs_ts)
        if met_csv:
            wx_series, wx_ts = read_weather_csv_with_ts(met_csv)
            payload = merge_payload(payload, wx_series)
            timestamp_map.update(wx_ts)

        hazard_type = event.get("hazard_type", "HISTORICAL")
        primary_name, primary_series = choose_primary(payload, hazard_type)
        if not primary_series:
            continue

        primary_timestamps = timestamp_map.get(primary_name, [])
        if not primary_timestamps or len(primary_timestamps) != len(primary_series):
            primary_timestamps = [None] * len(primary_series)  # type: ignore

        lead_min_score = int(event.get("lead_min_score", event.get("expected_min_score", 0)))

        mystic_index = find_detection_index(
            primary_series,
            payload,
            lead_min_score,
            predictor=predictor,
            hazard_type=hazard_type
        )
        baseline_index = find_detection_index(
            primary_series,
            payload,
            lead_min_score,
            predictor=None,
            hazard_type=hazard_type
        )

        mystic_time = primary_timestamps[mystic_index - 1] if mystic_index else None
        baseline_time = primary_timestamps[baseline_index - 1] if baseline_index else None
        window_start = primary_timestamps[0] if primary_timestamps else None
        window_end = primary_timestamps[-1] if primary_timestamps else None

        noaa_match = None
        if window_start and window_end and noaa_events:
            noaa_match = map_noaa_event(noaa_events, hazard_type, window_start, window_end)

        event_start = noaa_match["begin_date"] if noaa_match else window_start
        event_source = "NOAA Storm Events" if noaa_match else "SERIES_START"

        mystic_lead = compute_lead_hours(event_start, mystic_time)
        baseline_lead = compute_lead_hours(event_start, baseline_time)

        if mystic_lead is not None:
            lead_hours_mystic.append(mystic_lead)
        if baseline_lead is not None:
            lead_hours_baseline.append(baseline_lead)
        if event_source == "NOAA Storm Events":
            if mystic_lead is not None:
                lead_hours_mystic_noaa.append(mystic_lead)
            if baseline_lead is not None:
                lead_hours_baseline_noaa.append(baseline_lead)

        results.append({
            "id": event.get("id"),
            "name": event.get("name"),
            "hazard_type": hazard_type,
            "expected_risk": event.get("expected_risk"),
            "primary_series": primary_name,
            "event_start": event_start.isoformat() if event_start else None,
            "event_source": event_source,
            "noaa_event_type": noaa_match["event_type"] if noaa_match else None,
            "noaa_event_id": noaa_match["event_id"] if noaa_match else None,
            "window_start": window_start.isoformat() if window_start else None,
            "window_end": window_end.isoformat() if window_end else None,
            "mystic_detection": mystic_time.isoformat() if mystic_time else None,
            "baseline_detection": baseline_time.isoformat() if baseline_time else None,
            "mystic_lead_hours": mystic_lead,
            "baseline_lead_hours": baseline_lead,
        })

        print(
            f"{event.get('id')}: "
            f"mystic_lead={mystic_lead}h baseline_lead={baseline_lead}h source={event_source}"
        )

    def mean(values: List[float]) -> Optional[float]:
        return round(sum(values) / len(values), 2) if values else None

    def median(values: List[float]) -> Optional[float]:
        if not values:
            return None
        values = sorted(values)
        mid = len(values) // 2
        if len(values) % 2 == 0:
            return round((values[mid - 1] + values[mid]) / 2, 2)
        return round(values[mid], 2)

    summary = {
        "events_tested": len(results),
        "noaa_reference_count": sum(1 for r in results if r["event_source"] == "NOAA Storm Events"),
        "mystic_mean_lead_hours": mean(lead_hours_mystic),
        "baseline_mean_lead_hours": mean(lead_hours_baseline),
        "mystic_median_lead_hours": median(lead_hours_mystic),
        "baseline_median_lead_hours": median(lead_hours_baseline),
        "lead_delta_mean_hours": (
            round(mean(lead_hours_mystic) - mean(lead_hours_baseline), 2)
            if lead_hours_mystic and lead_hours_baseline else None
        ),
        "mystic_mean_lead_hours_noaa": mean(lead_hours_mystic_noaa),
        "baseline_mean_lead_hours_noaa": mean(lead_hours_baseline_noaa),
        "mystic_median_lead_hours_noaa": median(lead_hours_mystic_noaa),
        "baseline_median_lead_hours_noaa": median(lead_hours_baseline_noaa),
        "lead_delta_mean_hours_noaa": (
            round(mean(lead_hours_mystic_noaa) - mean(lead_hours_baseline_noaa), 2)
            if lead_hours_mystic_noaa and lead_hours_baseline_noaa else None
        ),
        "events": results,
    }

    with output_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nBenchmark report written to {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark MYSTIC vs NOAA Storm Events")
    parser.add_argument("--events", default=str(DEFAULT_EVENTS_PATH))
    parser.add_argument("--noaa", default=str(DEFAULT_NOAA_PATH))
    parser.add_argument("--output", default="leader_benchmark_report.json")
    args = parser.parse_args()

    return run_benchmark(Path(args.events), Path(args.noaa), Path(args.output))


if __name__ == "__main__":
    raise SystemExit(main())
