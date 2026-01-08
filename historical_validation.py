#!/usr/bin/env python3
"""
HISTORICAL WEATHER EVENT VALIDATION SUITE

Tests MYSTIC against realistic data patterns based on documented
meteorological events. These patterns are derived from:
- NOAA storm reports
- NWS historical data
- Peer-reviewed meteorological literature

Note: These are RECONSTRUCTED patterns based on reported characteristics,
not raw sensor data (which would require proper data licensing).

Events modeled:
1. Hurricane Harvey (2017) - Texas flooding
2. Camp Fire precursors (2018) - California fire weather
3. Joplin Tornado (2011) - EF5 tornado conditions
4. Texas Hill Country Flash Flood pattern
5. Stable high-pressure system
6. Cold front passage
7. Derecho conditions (2012 North American)

Author: Claude (K-Elimination Expert)
Date: 2026-01-07
"""

from typing import List, Dict, Any
import random
from dataclasses import dataclass

# Import the production predictor (calibrated for real-world events)
from mystic_v3_production import MYSTICPredictorV3Production, PredictionResult


@dataclass
class HistoricalEvent:
    """Represents a historical weather event for validation."""
    name: str
    description: str
    data: List[int]
    expected_risk: str
    expected_min_score: int
    source: str


def add_realistic_noise(base: List[int], noise_scale: int = 5, seed: int = 42) -> List[int]:
    """Add realistic measurement noise to synthetic data."""
    random.seed(seed)
    return [v + random.randint(-noise_scale, noise_scale) for v in base]


def generate_hurricane_harvey_pattern() -> List[int]:
    """
    Hurricane Harvey (August 2017) - Houston, TX

    Characteristics:
    - Pressure dropped from ~1010 mb to ~940 mb over 48 hours
    - Extremely rapid intensification
    - Sustained flooding for days

    Pattern: Pressure readings (mb) over 48 1-hour intervals
    """
    # Start at normal pressure, drop rapidly as hurricane approaches
    base = []
    for i in range(48):
        if i < 10:
            # Normal conditions
            pressure = 1012 - i * 2
        elif i < 25:
            # Rapid intensification phase
            pressure = 992 - (i - 10) * 3
        else:
            # Near landfall, stabilizing at low pressure
            pressure = 947 - (i - 25) * 1

        base.append(max(940, pressure))

    return add_realistic_noise(base, noise_scale=3, seed=2017)


def generate_camp_fire_pattern() -> List[int]:
    """
    Camp Fire precursor conditions (November 2018) - Paradise, CA

    Fire weather characteristics:
    - Very low humidity (relative humidity proxy: 10-20%)
    - Rapid humidity drop
    - Diablo wind pattern

    Pattern: Relative humidity (%) over 24 1-hour intervals
    Note: Low humidity = high fire danger
    """
    base = []
    for i in range(36):
        if i < 8:
            # Morning, moderate humidity
            rh = 45 - i * 3
        elif i < 16:
            # Afternoon drying
            rh = 21 - (i - 8) * 1
        elif i < 24:
            # Diablo winds, extreme drying
            rh = 13 - (i - 16) * 0.5
        else:
            # Sustained dangerous conditions
            rh = 10 + random.randint(-2, 2)

        base.append(max(8, int(rh)))

    return add_realistic_noise(base, noise_scale=2, seed=2018)


def generate_joplin_tornado_pattern() -> List[int]:
    """
    Joplin Tornado (May 22, 2011) - Joplin, MO

    EF5 tornado characteristics:
    - Extreme pressure drop in tornado core
    - Rapid pressure oscillations
    - High atmospheric instability

    Pattern: Pressure (mb) with simulated tornado-scale variations
    """
    base = []
    for i in range(30):
        if i < 10:
            # Pre-storm, unstable conditions
            pressure = 1005 - i * 1 + ((-1) ** i) * 5
        elif i < 15:
            # Storm approach
            pressure = 995 - (i - 10) * 4 + random.randint(-10, 10)
        elif i < 20:
            # Tornado passage (extreme oscillations)
            pressure = 975 - (i - 15) * 8 + random.randint(-20, 20)
        else:
            # Post-passage, recovering
            pressure = 935 + (i - 20) * 5

        base.append(max(920, min(1010, pressure)))

    return add_realistic_noise(base, noise_scale=8, seed=2011)


def generate_flash_flood_hill_country() -> List[int]:
    """
    Texas Hill Country Flash Flood Pattern

    Characteristics:
    - Rapid water level rise (inches per hour)
    - Exponential accumulation in narrow canyons
    - Based on Blanco River 2015 event patterns

    Pattern: Stream gauge height (inches above baseline)
    """
    base = []
    for i in range(40):
        if i < 10:
            # Pre-storm baseline
            height = 10 + i * 0.5
        elif i < 20:
            # Initial rainfall accumulation
            height = 15 + (i - 10) * 5
        elif i < 30:
            # Rapid rise (exponential-like)
            height = 65 + int(40 * (1.15 ** (i - 20)))
        else:
            # Peak and initial recession
            height = max(200, 400 - (i - 30) * 20)

        base.append(int(height))

    return add_realistic_noise(base, noise_scale=5, seed=2015)


def generate_stable_high_pressure() -> List[int]:
    """
    Stable high-pressure system (typical fair weather)

    Characteristics:
    - Pressure slowly varying around 1020-1025 mb
    - Diurnal variation of ~2-3 mb
    - Very predictable

    Pattern: Pressure (mb) over 48 hours
    """
    base = []
    import math
    for i in range(48):
        # Base pressure with small diurnal cycle
        diurnal = 2 * math.sin(2 * math.pi * i / 24)
        pressure = 1022 + diurnal

        base.append(int(pressure))

    return add_realistic_noise(base, noise_scale=1, seed=1234)


def generate_cold_front_passage() -> List[int]:
    """
    Cold front passage pattern

    Characteristics:
    - Gradual pressure drop before front
    - Sharp pressure rise as front passes
    - Temperature drop (using pressure as proxy for simplicity)

    Pattern: Pressure (mb) over 24 hours
    """
    base = []
    for i in range(36):
        if i < 12:
            # Pre-frontal: gradual drop
            pressure = 1015 - i * 1.5
        elif i < 16:
            # Frontal passage: rapid changes
            pressure = 997 + (i - 12) * 5
        else:
            # Post-frontal: high pressure building
            pressure = 1017 + (i - 16) * 0.3

        base.append(int(pressure))

    return add_realistic_noise(base, noise_scale=2, seed=5678)


def generate_derecho_pattern() -> List[int]:
    """
    Derecho (June 29, 2012) - North American pattern

    Characteristics:
    - Long-lived bow echo
    - Sustained damaging winds
    - Pressure oscillations along squall line

    Pattern: Pressure (mb) during derecho passage
    """
    base = []
    for i in range(30):
        if i < 8:
            # Pre-derecho, unstable atmosphere
            pressure = 1008 - i * 0.5
        elif i < 15:
            # Gust front arrival, pressure spike then drop
            if i < 10:
                pressure = 1004 + (i - 8) * 3  # Pressure spike
            else:
                pressure = 1010 - (i - 10) * 4  # Rapid drop behind gust front
        elif i < 22:
            # Main convective region
            pressure = 990 - (i - 15) * 1 + ((-1) ** i) * 3
        else:
            # Recovery
            pressure = 983 + (i - 22) * 2

        base.append(int(pressure))

    return add_realistic_noise(base, noise_scale=4, seed=2012)


# Define all historical test cases
HISTORICAL_EVENTS = [
    HistoricalEvent(
        name="Hurricane Harvey (2017)",
        description="Category 4 hurricane, catastrophic flooding in Houston",
        data=generate_hurricane_harvey_pattern(),
        expected_risk="CRITICAL",
        expected_min_score=70,
        source="NOAA NHC reports, Houston Chronicle archives"
    ),
    HistoricalEvent(
        name="Camp Fire Precursors (2018)",
        description="Extreme fire weather conditions, Paradise CA",
        data=generate_camp_fire_pattern(),
        expected_risk="HIGH",
        expected_min_score=50,
        source="Cal Fire incident reports, NWS Red Flag warnings"
    ),
    HistoricalEvent(
        name="Joplin Tornado (2011)",
        description="EF5 tornado, extreme atmospheric instability",
        data=generate_joplin_tornado_pattern(),
        expected_risk="CRITICAL",
        expected_min_score=70,
        source="NWS Storm Prediction Center, peer-reviewed literature"
    ),
    HistoricalEvent(
        name="Hill Country Flash Flood",
        description="Texas Hill Country rapid flooding pattern",
        data=generate_flash_flood_hill_country(),
        expected_risk="CRITICAL",
        expected_min_score=70,
        source="USGS stream gauge data patterns, NWS flood reports"
    ),
    HistoricalEvent(
        name="Stable High Pressure",
        description="Fair weather, no hazards expected",
        data=generate_stable_high_pressure(),
        expected_risk="LOW",
        expected_min_score=0,
        source="Typical meteorological patterns"
    ),
    HistoricalEvent(
        name="Cold Front Passage",
        description="Routine cold front, minor weather changes",
        data=generate_cold_front_passage(),
        expected_risk="MODERATE",
        expected_min_score=15,
        source="Standard synoptic meteorology"
    ),
    HistoricalEvent(
        name="2012 Derecho",
        description="Long-lived damaging wind event",
        data=generate_derecho_pattern(),
        expected_risk="HIGH",
        expected_min_score=50,
        source="NWS Storm Prediction Center, NOAA post-storm reports"
    ),
]


def run_historical_validation():
    """Run MYSTIC against historical weather event patterns."""
    print("=" * 75)
    print("HISTORICAL WEATHER EVENT VALIDATION SUITE")
    print("Testing MYSTIC against realistic meteorological patterns")
    print("=" * 75)

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

    for event in HISTORICAL_EVENTS:
        print(f"\n{event.name}:")
        print(f"  First 10 values: {event.data[:10]}")
        print(f"  Last 10 values: {event.data[-10:]}")
        print(f"  Range: {min(event.data)} - {max(event.data)}")
        print(f"  Mean: {sum(event.data) // len(event.data)}")


if __name__ == "__main__":
    print_event_data_samples()
    print()
    success = run_historical_validation()
    exit(0 if success else 1)
