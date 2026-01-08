#!/usr/bin/env python3
"""
MYSTIC Flash Flood Predictor

Predicts flash floods from METEOROLOGICAL conditions BEFORE the river rises.

Key predictive factors:
1. Rainfall rate (mm/hr) - intensity threshold
2. Accumulated rainfall (last 6 hours) - total water input
3. Antecedent precipitation (7-day) - soil saturation proxy
4. Dewpoint depression - moisture availability
5. Pressure tendency - storm intensification

This is similar to storm trackers but uses NINE65's exact chaos mathematics
for attractor basin detection - identifying when atmospheric conditions
enter a "flash flood attractor basin".
"""

import csv
import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# QMNF: Import AttractorClassifier and PhiResonanceDetector for chaos-based detection
try:
    from mystic_advanced_math import AttractorClassifier, PhiResonanceDetector, SCALE
    QMNF_AVAILABLE = True
except ImportError:
    QMNF_AVAILABLE = False
    SCALE = 1_000_000

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "meteorological")


@dataclass
class MeteoReading:
    """A single meteorological observation."""
    timestamp: datetime
    station_id: str
    temp_c: Optional[float]
    dewpoint_c: Optional[float]
    pressure_hpa: Optional[float]
    wind_speed_mps: Optional[float]
    precip_1hr_mm: Optional[float]
    precip_6hr_mm: Optional[float]
    sky_cover: Optional[int]


@dataclass
class FloodPrediction:
    """Flash flood prediction result."""
    timestamp: str
    alert_level: str
    probability: float
    lead_time_hours: float
    factors: List[str]
    rain_rate_mm_hr: float
    rain_6hr_mm: float
    rain_24hr_mm: float
    dewpoint_depression_c: float
    pressure_change_hpa: float
    # QMNF: New chaos-based fields
    attractor_basin: str = "UNKNOWN"  # Basin classification from AttractorClassifier
    phi_resonance_detected: bool = False  # φ-resonance pattern in pressure history
    phi_resonance_confidence: int = 0  # Confidence (0-100)


# QMNF: Impact-weighted integer thresholds
# Scaling is proportional to variable's influence on flash flood risk:
#   - Rain rate: PRIMARY driver (70% of signal) → SCALE_RAIN = 1000 (mm/hr × 1000)
#   - Accumulation: PRIMARY driver (20% of signal) → SCALE_ACCUM = 100 (mm × 100)
#   - Dewpoint: SECONDARY modifier (adaptive) → starts at SCALE_DEW = 100
#   - Pressure: TERTIARY modifier (adaptive) → starts at SCALE_PRES = 100
SCALE_RAIN = 1000   # High precision for primary driver (fixed)
SCALE_ACCUM = 100   # Standard precision for accumulation (fixed)
SCALE_DEW = 100     # Lower precision for secondary factor (initial)
SCALE_PRES = 100    # Lower precision for modifier (initial)


# ============================================================================
# QMNF: Adaptive Weight System for Minor Variables
# ============================================================================
# Minor variables (dewpoint, pressure) self-tune based on prediction feedback.
# Uses exponential moving average of contribution-to-outcome correlation.

class AdaptiveWeight:
    """
    Self-tuning weight for minor prediction variables.

    Tracks how often a variable's signal correlates with actual outcomes,
    adjusting its influence weight via feedback loop.

    All arithmetic is integer-only (permille precision).
    """
    def __init__(self, name: str, initial_weight_permille: int = 100,
                 min_weight: int = 10, max_weight: int = 300):
        self.name = name
        self.weight_permille = initial_weight_permille  # Current weight (‰)
        self.min_weight = min_weight   # Floor (don't go below 1%)
        self.max_weight = max_weight   # Ceiling (don't exceed 30%)

        # Feedback tracking (all integers)
        self.total_signals = 0         # Times this variable signaled
        self.correct_signals = 0       # Times signal matched outcome
        self.ema_accuracy_permille = 500  # EMA of accuracy (start at 50%)
        self.ema_alpha = 100           # EMA smoothing (100 = 10% new, 900 = old)

    def record_outcome(self, signaled: bool, event_occurred: bool):
        """
        Record prediction outcome for feedback.

        signaled: Did this variable contribute to the prediction?
        event_occurred: Did the actual event happen?
        """
        if not signaled:
            return  # Only track when variable contributed

        self.total_signals += 1

        # Calculate this observation's accuracy (1000 = correct, 0 = wrong)
        if event_occurred:
            self.correct_signals += 1
            observation = 1000  # Correct positive
        else:
            observation = 0     # False positive

        # Update EMA: new_ema = alpha * observation + (1-alpha) * old_ema
        # Using integer math: ema = (alpha * obs + (1000-alpha) * ema) / 1000
        self.ema_accuracy_permille = (
            self.ema_alpha * observation +
            (1000 - self.ema_alpha) * self.ema_accuracy_permille
        ) // 1000

        # Adjust weight based on accuracy
        # If accuracy > 50%, increase weight; if < 50%, decrease
        self._adjust_weight()

    def _adjust_weight(self):
        """Adjust weight based on accumulated accuracy."""
        # Deviation from 50% accuracy determines adjustment
        deviation = self.ema_accuracy_permille - 500

        # Adjustment rate: 5% of deviation per update
        adjustment = deviation // 20

        # Apply adjustment with bounds
        new_weight = self.weight_permille + adjustment
        self.weight_permille = max(self.min_weight, min(self.max_weight, new_weight))

    def get_risk_contribution(self, base_risk_permille: int) -> int:
        """
        Calculate this variable's contribution to risk.

        Returns scaled risk contribution based on current adaptive weight.
        """
        # Scale base risk by weight (weight is in permille)
        return (base_risk_permille * self.weight_permille) // 1000

    def get_stats(self) -> dict:
        """Return current adaptive state for monitoring."""
        return {
            "name": self.name,
            "weight_permille": self.weight_permille,
            "total_signals": self.total_signals,
            "correct_signals": self.correct_signals,
            "ema_accuracy_permille": self.ema_accuracy_permille,
            "historical_accuracy": (
                (self.correct_signals * 1000) // self.total_signals
                if self.total_signals > 0 else 500
            ),
        }


# Global adaptive weights for minor variables
# These persist across predictions and learn from outcomes
ADAPTIVE_WEIGHTS = {
    "dewpoint": AdaptiveWeight("dewpoint", initial_weight_permille=100),  # 10% initial
    "pressure": AdaptiveWeight("pressure", initial_weight_permille=100),  # 10% initial
    "phi_resonance": AdaptiveWeight("phi_resonance", initial_weight_permille=150),  # 15% initial
    "attractor_basin": AdaptiveWeight("attractor_basin", initial_weight_permille=200),  # 20% initial
}

THRESHOLDS = {
    # Rainfall intensity (integer: mm/hr × SCALE_RAIN)
    "rain_light": 5000,        # 5.0 mm/hr ~0.2 in/hr
    "rain_moderate": 12500,    # 12.5 mm/hr ~0.5 in/hr
    "rain_heavy": 25000,       # 25.0 mm/hr ~1.0 in/hr
    "rain_intense": 50000,     # 50.0 mm/hr ~2.0 in/hr
    "rain_extreme": 75000,     # 75.0 mm/hr ~3.0 in/hr - rare, catastrophic

    # 6-hour accumulation (integer: mm × SCALE_ACCUM)
    "accum_6hr_moderate": 5000,   # 50 mm ~2 inches
    "accum_6hr_heavy": 10000,     # 100 mm ~4 inches
    "accum_6hr_extreme": 15000,   # 150 mm ~6 inches

    # 24-hour accumulation (integer: mm × SCALE_ACCUM)
    "accum_24hr_heavy": 15000,    # 150 mm ~6 inches
    "accum_24hr_extreme": 25000,  # 250 mm ~10 inches

    # Dewpoint depression (integer: °C × SCALE_DEW) - lower impact, smaller scale
    "dewpoint_saturated": 200,    # 2.0°C - Nearly saturated air
    "dewpoint_humid": 500,        # 5.0°C - High humidity

    # Pressure tendency (integer: hPa × SCALE_PRES) - modifier only
    "pressure_falling_fast": -300,  # -3.0 hPa per 3 hours
}


def load_meteorological_data(event_name: str) -> List[MeteoReading]:
    """Load meteorological data for an event."""
    filename = f"weather_{event_name.lower().replace(' ', '_')}.csv"
    filepath = os.path.join(DATA_DIR, filename)

    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return []

    readings = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse timestamp
            ts_str = row.get("timestamp") or row.get("date")
            if not ts_str:
                continue

            try:
                if "T" in ts_str:
                    timestamp = datetime.fromisoformat(ts_str.replace("Z", ""))
                else:
                    timestamp = datetime.strptime(ts_str, "%Y-%m-%d")
            except ValueError:
                continue

            reading = MeteoReading(
                timestamp=timestamp,
                station_id=row.get("station_id", ""),
                temp_c=safe_float(row.get("temp_c")),
                dewpoint_c=safe_float(row.get("dewpoint_c")),
                pressure_hpa=safe_float(row.get("pressure_hpa")),
                wind_speed_mps=safe_float(row.get("wind_speed_mps")),
                precip_1hr_mm=safe_float(row.get("precip_1hr_mm")),
                precip_6hr_mm=safe_float(row.get("precip_6hr_mm")),
                sky_cover=safe_int(row.get("sky_cover_oktas")),
            )
            readings.append(reading)

    return sorted(readings, key=lambda r: r.timestamp)


def safe_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def safe_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def calculate_accumulated_precip(readings: List[MeteoReading],
                                  target_time: datetime,
                                  hours: int) -> float:
    """Calculate total precipitation over the past N hours."""
    window_start = target_time - timedelta(hours=hours)

    total = 0.0
    for r in readings:
        if window_start <= r.timestamp <= target_time:
            if r.precip_1hr_mm is not None and r.precip_1hr_mm > 0:
                total += r.precip_1hr_mm

    return total


def calculate_pressure_tendency(readings: List[MeteoReading],
                                target_time: datetime,
                                hours: int = 3) -> float:
    """Calculate pressure change over past N hours (negative = falling)."""
    window_start = target_time - timedelta(hours=hours)

    start_pressure = None
    end_pressure = None

    for r in readings:
        if r.pressure_hpa is not None:
            if r.timestamp <= window_start and (start_pressure is None or r.timestamp > start_pressure[0]):
                start_pressure = (r.timestamp, r.pressure_hpa)
            if r.timestamp <= target_time and (end_pressure is None or r.timestamp > end_pressure[0]):
                end_pressure = (r.timestamp, r.pressure_hpa)

    if start_pressure and end_pressure:
        return end_pressure[1] - start_pressure[1]
    return 0.0


def predict_flash_flood(readings: List[MeteoReading],
                        target_time: datetime) -> FloodPrediction:
    """
    Predict flash flood risk from meteorological conditions.

    This is the core detection algorithm - identifying when conditions
    enter the "flash flood attractor basin" in chaos space.
    """
    factors = []
    risk = 0.0

    # Find nearest reading
    nearest = None
    min_diff = (1 << 63) - 1  # Integer max instead of float('inf')
    for r in readings:
        diff = abs((r.timestamp - target_time).total_seconds())
        if diff < min_diff:
            min_diff = diff
            nearest = r

    if nearest is None:
        return FloodPrediction(
            timestamp=target_time.isoformat(),
            alert_level="NO_DATA",
            probability=0.0,
            lead_time_hours=0,
            factors=[],
            rain_rate_mm_hr=0,
            rain_6hr_mm=0,
            rain_24hr_mm=0,
            dewpoint_depression_c=0,
            pressure_change_hpa=0,
        )

    # Current rainfall rate (convert to scaled integer for comparison)
    rain_rate_raw = nearest.precip_1hr_mm or 0
    rain_rate = int(rain_rate_raw * SCALE_RAIN)  # Scale to integer

    # Accumulated precipitation (scaled integers)
    rain_6hr_raw = calculate_accumulated_precip(readings, target_time, 6)
    rain_24hr_raw = calculate_accumulated_precip(readings, target_time, 24)
    rain_6hr = int(rain_6hr_raw * SCALE_ACCUM)
    rain_24hr = int(rain_24hr_raw * SCALE_ACCUM)

    # Dewpoint depression (temp - dewpoint), scaled
    dewpoint_depression_raw = 0
    if nearest.temp_c is not None and nearest.dewpoint_c is not None:
        dewpoint_depression_raw = nearest.temp_c - nearest.dewpoint_c
    dewpoint_depression = int(dewpoint_depression_raw * SCALE_DEW)

    # Pressure tendency, scaled
    pressure_change_raw = calculate_pressure_tendency(readings, target_time)
    pressure_change = int(pressure_change_raw * SCALE_PRES)

    # QMNF: Risk accumulator as integer (permille for precision)
    # Impact weights: rain=70%, accum=20%, dew=5%, pressure=5%
    risk_permille = 0

    # ========================================
    # FACTOR 1: Current Rainfall Rate (PRIMARY - 70% weight)
    # ========================================
    if rain_rate >= THRESHOLDS["rain_extreme"]:
        factors.append("rain_extreme")
        risk_permille += 400  # 40% as permille
    elif rain_rate >= THRESHOLDS["rain_intense"]:
        factors.append("rain_intense")
        risk_permille += 300
    elif rain_rate >= THRESHOLDS["rain_heavy"]:
        factors.append("rain_heavy")
        risk_permille += 200
    elif rain_rate >= THRESHOLDS["rain_moderate"]:
        factors.append("rain_moderate")
        risk_permille += 100

    # ========================================
    # FACTOR 2: 6-Hour Accumulation (PRIMARY - part of 20%)
    # ========================================
    if rain_6hr >= THRESHOLDS["accum_6hr_extreme"]:
        factors.append("6hr_extreme")
        risk_permille += 350
    elif rain_6hr >= THRESHOLDS["accum_6hr_heavy"]:
        factors.append("6hr_heavy")
        risk_permille += 250
    elif rain_6hr >= THRESHOLDS["accum_6hr_moderate"]:
        factors.append("6hr_moderate")
        risk_permille += 150

    # ========================================
    # FACTOR 3: 24-Hour Accumulation (soil saturation proxy)
    # ========================================
    if rain_24hr >= THRESHOLDS["accum_24hr_extreme"]:
        factors.append("24hr_extreme")
        risk_permille += 250
    elif rain_24hr >= THRESHOLDS["accum_24hr_heavy"]:
        factors.append("24hr_heavy")
        risk_permille += 150

    # ========================================
    # FACTOR 4: Atmospheric Moisture (ADAPTIVE weight)
    # ========================================
    dewpoint_signaled = False
    if dewpoint_depression <= THRESHOLDS["dewpoint_saturated"]:
        factors.append("saturated_air")
        dewpoint_signaled = True
        # Use adaptive weight instead of fixed 100
        base_contribution = 100  # Base risk if signal present
        risk_permille += ADAPTIVE_WEIGHTS["dewpoint"].get_risk_contribution(base_contribution)
    elif dewpoint_depression <= THRESHOLDS["dewpoint_humid"]:
        factors.append("humid_air")
        dewpoint_signaled = True
        base_contribution = 50
        risk_permille += ADAPTIVE_WEIGHTS["dewpoint"].get_risk_contribution(base_contribution)

    # ========================================
    # FACTOR 5: Storm Intensification (ADAPTIVE weight)
    # ========================================
    pressure_signaled = False
    if pressure_change <= THRESHOLDS["pressure_falling_fast"]:
        factors.append("pressure_falling")
        pressure_signaled = True
        # Use adaptive weight instead of fixed 100
        base_contribution = 100
        risk_permille += ADAPTIVE_WEIGHTS["pressure"].get_risk_contribution(base_contribution)

    # Convert permille to 0-1 scale for compatibility
    risk = risk_permille / 1000

    # ========================================
    # QMNF: Attractor Basin Classification
    # ========================================
    attractor_basin = "UNKNOWN"
    phi_resonance_detected = False
    phi_resonance_confidence = 0

    if QMNF_AVAILABLE:
        # rain_rate is already scaled by SCALE_RAIN, pass directly to classifier
        # pressure_change is already scaled by SCALE_PRES
        # humidity proxy from dewpoint depression: low depression = high humidity
        # dewpoint_depression is scaled by SCALE_DEW (100), so 200 = 2°C
        # Convert to 0-100 scale: 100% when dewpoint_depression = 0, 0% when >= 2000 (20°C)
        humidity_pct = max(0, min(100, 100 - dewpoint_depression // 20))

        # Classify attractor basin
        classifier = AttractorClassifier()
        basin_name, basin_sig = classifier.classify(
            rain_rate=rain_rate,  # Already scaled
            pressure_tendency=pressure_change,  # Already scaled
            humidity=humidity_pct
        )
        attractor_basin = basin_name

        # Add basin to factors if in dangerous basin (ADAPTIVE weight)
        basin_signaled = False
        if attractor_basin == "FLASH_FLOOD":
            factors.append("attractor_basin_ff")
            basin_signaled = True
            base_contribution = 200  # High base for flash flood basin
            risk_permille += ADAPTIVE_WEIGHTS["attractor_basin"].get_risk_contribution(base_contribution)
        elif attractor_basin == "STEADY_RAIN":
            factors.append("attractor_basin_rain")
            basin_signaled = True
            base_contribution = 75
            risk_permille += ADAPTIVE_WEIGHTS["attractor_basin"].get_risk_contribution(base_contribution)

        # Recalculate risk after QMNF boost
        risk = risk_permille / 1000

        # Check for φ-resonance in pressure history
        # Get recent pressure readings
        window_start = target_time - timedelta(hours=6)
        pressure_history = []
        for r in readings:
            if window_start <= r.timestamp <= target_time and r.pressure_hpa is not None:
                pressure_history.append(int(r.pressure_hpa * SCALE_PRES))  # Use consistent scale

        phi_signaled = False
        if len(pressure_history) >= 5:
            phi_detector = PhiResonanceDetector(tolerance_permille=30)
            phi_result = phi_detector.detect_resonance(pressure_history)
            phi_resonance_detected = phi_result["has_resonance"]
            phi_resonance_confidence = phi_result["confidence"]

            if phi_resonance_detected and phi_resonance_confidence >= 30:
                factors.append("phi_resonance")
                phi_signaled = True
                # φ-resonance contribution scales with confidence (ADAPTIVE)
                # Base 50 + confidence bonus, then apply adaptive weight
                base_contribution = 50 + (phi_resonance_confidence * 150) // 100
                risk_permille += ADAPTIVE_WEIGHTS["phi_resonance"].get_risk_contribution(base_contribution)
                risk = risk_permille / 1000

    # ========================================
    # Multi-Factor Requirement
    # ========================================
    # Flash floods need multiple factors - single heavy rain isn't enough

    risk = min(risk, 1.0)

    if len(factors) >= 4 and risk >= 0.70:
        alert_level = "FF_EMERGENCY"
    elif len(factors) >= 3 and risk >= 0.55:
        alert_level = "FF_WARNING"
    elif len(factors) >= 2 and risk >= 0.40:
        alert_level = "FF_ADVISORY"
    elif len(factors) >= 1 and risk >= 0.20:
        alert_level = "FF_WATCH"
    else:
        alert_level = "CLEAR"

    # Lead time is how far before flooding this would have been issued
    # (Will be set by the caller based on actual flood timing)

    # Return with raw (unscaled) values for human readability
    return FloodPrediction(
        timestamp=target_time.isoformat(),
        alert_level=alert_level,
        probability=risk,
        lead_time_hours=0,  # Set by caller
        factors=factors,
        rain_rate_mm_hr=rain_rate_raw,  # Original unscaled value
        rain_6hr_mm=rain_6hr_raw,
        rain_24hr_mm=rain_24hr_raw,
        dewpoint_depression_c=dewpoint_depression_raw,
        pressure_change_hpa=pressure_change_raw,
        attractor_basin=attractor_basin,
        phi_resonance_detected=phi_resonance_detected,
        phi_resonance_confidence=phi_resonance_confidence,
    )


def record_prediction_outcome(factors: List[str], event_occurred: bool):
    """
    Record prediction outcome for adaptive weight learning.

    Call this after determining whether a predicted flood actually occurred.
    This enables the adaptive weights to self-tune over time.

    Args:
        factors: List of factors from the prediction
        event_occurred: True if flood actually happened, False otherwise
    """
    # Update each adaptive weight based on whether it signaled
    ADAPTIVE_WEIGHTS["dewpoint"].record_outcome(
        signaled="saturated_air" in factors or "humid_air" in factors,
        event_occurred=event_occurred
    )
    ADAPTIVE_WEIGHTS["pressure"].record_outcome(
        signaled="pressure_falling" in factors,
        event_occurred=event_occurred
    )
    ADAPTIVE_WEIGHTS["phi_resonance"].record_outcome(
        signaled="phi_resonance" in factors,
        event_occurred=event_occurred
    )
    ADAPTIVE_WEIGHTS["attractor_basin"].record_outcome(
        signaled="attractor_basin_ff" in factors or "attractor_basin_rain" in factors,
        event_occurred=event_occurred
    )


def get_adaptive_weight_stats() -> dict:
    """Get current adaptive weight statistics for monitoring."""
    return {name: weight.get_stats() for name, weight in ADAPTIVE_WEIGHTS.items()}


def validate_event(event_name: str, flood_time: datetime,
                   documented_rainfall_in: float) -> List[FloodPrediction]:
    """
    Validate predictions for a historical flood event.

    Tests predictions at various times BEFORE the flood occurred
    to measure lead time and accuracy.
    """
    print(f"\n{'='*70}")
    print(f"VALIDATING: {event_name}")
    print(f"Flood Time: {flood_time}")
    print(f"Documented Rainfall: {documented_rainfall_in} inches")
    print(f"{'='*70}")

    readings = load_meteorological_data(event_name)
    if not readings:
        print("  No data available!")
        return []

    print(f"  Loaded {len(readings)} meteorological observations")

    # Test at various time offsets before the flood
    test_offsets = [-48, -36, -24, -18, -12, -6, -3, -2, -1, 0, 1, 2]
    predictions = []

    for offset in test_offsets:
        test_time = flood_time + timedelta(hours=offset)

        pred = predict_flash_flood(readings, test_time)
        pred.lead_time_hours = -offset if pred.alert_level not in ["CLEAR", "NO_DATA"] else 0

        predictions.append(pred)

        detected = pred.alert_level not in ["CLEAR", "NO_DATA"]
        status = "DETECTED" if detected else "missed"

        print(f"  T{offset:+3d}h: {pred.alert_level:12s} ({pred.probability:5.0%}) | "
              f"Rain: {pred.rain_rate_mm_hr:5.1f}mm/hr | "
              f"6hr: {pred.rain_6hr_mm:6.1f}mm | {status}")
        if pred.factors:
            print(f"         Factors: {', '.join(pred.factors)}")

    return predictions


def main():
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║         MYSTIC Flash Flood Predictor                                      ║")
    print("║         Meteorological-Based Detection Using Chaos Mathematics            ║")
    print("║         QMNF Enhanced: AttractorClassifier + φ-Resonance                  ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print()
    print("Predicting floods from RAINFALL and ATMOSPHERIC conditions")
    print("NOT from river stage (that's too late!)")
    if QMNF_AVAILABLE:
        print("✓ QMNF innovations loaded: AttractorClassifier, PhiResonanceDetector")
    else:
        print("⚠ QMNF innovations not available - using basic detection")
    print()

    # Historical flood events with their peak times
    events = [
        ("Camp Mystic 2007", datetime(2007, 6, 28, 16, 0), 9.0),
        ("Memorial Day 2015", datetime(2015, 5, 24, 2, 0), 12.0),
        ("Hurricane Harvey 2017", datetime(2017, 8, 27, 12, 0), 60.0),
        ("Llano River 2018", datetime(2018, 10, 16, 7, 0), 10.0),
        ("Halloween 2013", datetime(2013, 10, 31, 8, 0), 14.0),
        ("Tax Day 2016", datetime(2016, 4, 18, 6, 0), 17.0),
        ("TS Imelda 2019", datetime(2019, 9, 19, 12, 0), 43.0),
    ]

    all_predictions = {}
    all_results = []

    for event_name, flood_time, rainfall in events:
        predictions = validate_event(event_name, flood_time, rainfall)
        if predictions:
            all_predictions[event_name] = predictions
            all_results.extend(predictions)

    # Calculate metrics
    print("\n" + "=" * 70)
    print("VERIFICATION METRICS")
    print("=" * 70)

    # Count detections within critical window (-6 to +2 hours)
    hits = 0
    misses = 0
    for event_name, predictions in all_predictions.items():
        detected_in_window = False
        for p in predictions:
            # Parse timestamp to get offset
            # Within critical window should detect
            if p.alert_level not in ["CLEAR", "NO_DATA"]:
                if p.lead_time_hours > 0:  # Before flood
                    detected_in_window = True
                    break

        if detected_in_window:
            hits += 1
        else:
            misses += 1

    total_events = hits + misses
    pod = hits / total_events if total_events > 0 else 0

    print(f"\nEvents Detected:     {hits}/{total_events}")
    print(f"POD:                 {pod:.1%}")
    print(f"Events with lead time:")

    for event_name, predictions in all_predictions.items():
        max_lead = 0
        for p in predictions:
            if p.alert_level not in ["CLEAR", "NO_DATA"] and p.lead_time_hours > max_lead:
                max_lead = p.lead_time_hours

        if max_lead > 0:
            print(f"  {event_name}: {max_lead:.0f} hours lead time")
        else:
            print(f"  {event_name}: NOT DETECTED")

    # Save results
    output_file = os.path.join(DATA_DIR, "..", "flash_flood_predictions.json")
    results = {
        "generated": datetime.now().isoformat(),
        "method": "meteorological_prediction",
        "events_tested": len(all_predictions),
        "events_detected": hits,
        "POD": pod,
        "predictions": {}
    }

    for event_name, predictions in all_predictions.items():
        results["predictions"][event_name] = [
            {
                "timestamp": p.timestamp,
                "alert_level": p.alert_level,
                "probability": p.probability,
                "lead_time_hours": p.lead_time_hours,
                "factors": p.factors,
                "rain_rate_mm_hr": p.rain_rate_mm_hr,
                "rain_6hr_mm": p.rain_6hr_mm,
                "rain_24hr_mm": p.rain_24hr_mm,
                "attractor_basin": p.attractor_basin,
                "phi_resonance_detected": p.phi_resonance_detected,
                "phi_resonance_confidence": p.phi_resonance_confidence,
            }
            for p in predictions
        ]

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
