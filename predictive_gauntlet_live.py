#!/usr/bin/env python3
"""
MYSTIC PREDICTIVE GAUNTLET - LIVE (Phase B)

Fetches real historical event windows via APIs and evaluates predictions.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Any, Optional, Tuple

from historical_data_loader import HistoricalDataLoader
from mystic_v3_integrated import MYSTICPredictorV3


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


QUALITY_GATES = {
    "risk_accuracy": 85.0,
    "score_accuracy": 85.0,
    "hazard_accuracy": 70.0,
    "lead_success": 70.0,
}


def evaluate_quality_gates(summary: Dict[str, Any]) -> bool:
    gates: Dict[str, Dict[str, Any]] = {}
    passed = True
    for metric, threshold in QUALITY_GATES.items():
        value = float(summary.get(metric, 0))
        gate_ok = value >= threshold
        gates[metric] = {
            "value": value,
            "threshold": threshold,
            "pass": gate_ok
        }
        if not gate_ok:
            passed = False
    summary["quality_gates"] = {
        "passed": passed,
        "gates": gates
    }
    return passed


def choose_primary(
    payload: Dict[str, List[int]],
    hazard_type: str
) -> Tuple[str, List[int]]:
    hazard_type = (hazard_type or "").upper()

    if hazard_type in ["FLASH_FLOOD", "HURRICANE"]:
        for key in ["streamflow", "gage_height", "precipitation", "pressure"]:
            if payload.get(key):
                return key, payload[key]
    elif hazard_type == "FIRE_WEATHER":
        for key in ["humidity", "wind_speed", "temperature", "pressure"]:
            if payload.get(key):
                return key, payload[key]
    elif hazard_type in ["TORNADO", "SEVERE_STORM"]:
        for key in ["pressure", "wind_speed", "precipitation"]:
            if payload.get(key):
                return key, payload[key]
    elif hazard_type == "STABLE":
        for key in ["pressure", "streamflow", "temperature"]:
            if payload.get(key):
                return key, payload[key]

    for key in [
        "streamflow",
        "pressure",
        "precipitation",
        "humidity",
        "wind_speed",
        "temperature",
        "gage_height",
    ]:
        if payload.get(key):
            return key, payload[key]

    return "UNKNOWN", []


def slice_payload(payload: Dict[str, List[int]], ratio: float) -> Dict[str, List[int]]:
    sliced: Dict[str, List[int]] = {}
    for key, series in payload.items():
        if not series:
            continue
        end = max(3, int(len(series) * ratio))
        sliced[key] = series[:end]
    return sliced


def run_live_gauntlet(
    output_path: Optional[str] = None,
    enforce_gates: bool = False
) -> int:
    loader = HistoricalDataLoader()
    predictor = MYSTICPredictorV3()

    risk_hits = 0
    score_hits = 0
    hazard_hits = 0
    lead_hits = 0
    results: List[Dict[str, Any]] = []

    for event_key, event_config in loader.events.items():
        print(f"\nLoading: {event_config['name']}")
        event = loader.fetch_event_data(event_key)
        if not event or not event.data:
            print(f"[SKIP] {event_key}: no data available")
            continue

        payload = dict(event.data)
        if "streamflow" not in payload and payload.get("gage_height"):
            payload["streamflow"] = payload["gage_height"]

        hazard_type = event_config.get("hazard_type", "HISTORICAL")
        primary_name, primary_series = choose_primary(payload, hazard_type)
        if not primary_series:
            print(f"[SKIP] {event_key}: no primary series")
            continue

        prediction = predictor.predict(
            primary_series,
            location=event_key,
            hazard_type=hazard_type,
            multi_variable_data=payload
        )

        expected_risk = event_config.get("expected_risk", "UNKNOWN")
        expected_min_score = int(event_config.get("expected_min_score", 0))

        risk_ok = risk_match(expected_risk, prediction.risk_level)
        score_ok = prediction.risk_score >= expected_min_score

        hazard_expected = event_config.get("hazard_type")
        hazard_predicted = None
        if prediction.multi_variable_summary:
            hazard_predicted = prediction.multi_variable_summary.get("hazard_type")
        hazard_ok = bool(hazard_expected and hazard_predicted and hazard_expected == hazard_predicted)

        lead_ratio = float(event_config.get("lead_window_ratio", 0.0))
        lead_ok = False
        lead_score = None
        lead_level = None
        if lead_ratio > 0:
            sliced_payload = slice_payload(payload, lead_ratio)
            lead_primary_name, lead_primary = choose_primary(sliced_payload, hazard_type)
            if lead_primary:
                lead_prediction = predictor.predict(
                    lead_primary,
                    location=event_key,
                    hazard_type=hazard_type,
                    multi_variable_data=sliced_payload
                )
                lead_score = lead_prediction.risk_score
                lead_level = lead_prediction.risk_level
                lead_min_score = int(
                    event_config.get("lead_min_score", max(0, expected_min_score * 7 // 10))
                )
                lead_ok = lead_score >= lead_min_score
                primary_name = primary_name or lead_primary_name

        if risk_ok:
            risk_hits += 1
        if score_ok:
            score_hits += 1
        if hazard_ok:
            hazard_hits += 1
        if lead_ratio > 0 and lead_ok:
            lead_hits += 1

        results.append({
            "id": event_key,
            "name": event.name,
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
            "data_quality": event.data_quality,
            "source": event.source,
        })

        print(
            f"EVENT {event_key}: risk={prediction.risk_level} "
            f"score={prediction.risk_score} lead_ok={lead_ok}"
        )

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

    print("\nGAUNTLET SUMMARY (LIVE)")
    print(f"  Events tested: {total}")
    print(f"  Risk accuracy: {summary['risk_accuracy']}%")
    print(f"  Score accuracy: {summary['score_accuracy']}%")
    if hazard_total:
        print(f"  Hazard accuracy: {summary['hazard_accuracy']}%")
    if lead_total:
        print(f"  Lead success: {summary['lead_success']}%")

    gates_passed = evaluate_quality_gates(summary)
    print("\nQUALITY GATES")
    for metric, gate in summary["quality_gates"]["gates"].items():
        status = "PASS" if gate["pass"] else "FAIL"
        print(f"  {metric}: {gate['value']}% (min {gate['threshold']}%) {status}")
    print(f"  Overall: {'PASS' if gates_passed else 'FAIL'}")

    if output_path:
        with open(output_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"\nReport written to {output_path}")

    if enforce_gates and not gates_passed:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live MYSTIC predictive gauntlet")
    parser.add_argument("--output", default="predictive_gauntlet_live_report.json")
    parser.add_argument("--enforce", action="store_true", help="Exit nonzero on gate failure")
    args = parser.parse_args()

    return run_live_gauntlet(args.output, args.enforce)


if __name__ == "__main__":
    raise SystemExit(main())
