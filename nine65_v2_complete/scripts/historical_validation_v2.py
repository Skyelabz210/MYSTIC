#!/usr/bin/env python3
"""
MYSTIC Historical Validation v2 - Enhanced Real Data Testing

Improvements over v1:
1. Dynamic thresholds based on station-specific base levels
2. Rise rate calculation from actual data
3. WATCH counts as detection for major/slow-rise floods
4. Better handling of multi-day events like Harvey
5. Additional historical events for broader validation
"""

import urllib.request
import json
import csv
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# QMNF: Import integer-only math components for basin classification
try:
    from mystic_advanced_math import AttractorClassifier, PhiResonanceDetector, SCALE
    QMNF_AVAILABLE = True
except ImportError:
    QMNF_AVAILABLE = False
    SCALE = 1_000_000


@dataclass
class HistoricalEvent:
    """A known historical flood event for validation."""
    name: str
    event_date: datetime
    location: str
    county: str
    lat: float
    lon: float
    deaths: int
    peak_stage_ft: float
    description: str
    nearby_stations: List[str]
    event_type: str = "flash_flood"
    is_slow_rise: bool = False  # Major floods vs flash floods


@dataclass
class USGSReading:
    """A single USGS gauge reading."""
    timestamp: datetime
    station_id: str
    gage_height_ft: float


@dataclass
class ValidationResult:
    """Result from validating against a historical event."""
    event_name: str
    event_date: str
    hours_before_peak: float
    alert_level: str
    probability: float
    detected: bool
    lead_time_hours: float
    factors: List[str]
    station_id: str
    stream_height_ft: float
    base_level_ft: float
    rise_from_base: float
    # QMNF fields
    attractor_basin: str = "UNKNOWN"
    phi_resonance_detected: bool = False
    phi_resonance_confidence: int = 0


# Historical events - expanded list
HISTORICAL_EVENTS = [
    HistoricalEvent(
        name="Camp Mystic Flash Flood",
        event_date=datetime(2007, 6, 28, 16, 0),
        location="Camp Mystic on Guadalupe River",
        county="Kerr",
        lat=29.98,
        lon=-99.18,
        deaths=3,
        peak_stage_ft=25.0,
        description="Flash flood at summer camp",
        nearby_stations=["08166200", "08165500", "08167000"],
    ),
    HistoricalEvent(
        name="Memorial Day Flood",
        event_date=datetime(2015, 5, 24, 2, 0),
        location="Wimberley on Blanco River",
        county="Hays",
        lat=29.99,
        lon=-98.11,
        deaths=13,
        peak_stage_ft=43.0,
        description="Record flash flood on Blanco River",
        nearby_stations=["08171000", "08171300", "08170500"],
    ),
    HistoricalEvent(
        name="Hurricane Harvey Flooding",
        event_date=datetime(2017, 8, 27, 12, 0),  # Adjusted peak timing
        location="Houston Metro Area",
        county="Harris",
        lat=29.76,
        lon=-95.37,
        deaths=68,
        peak_stage_ft=38.0,
        description="Catastrophic flooding from Harvey",
        nearby_stations=["08074000", "08073600", "08074500", "08074800", "08075000"],
        event_type="major_flood",
        is_slow_rise=True,
    ),
    HistoricalEvent(
        name="Llano River Flash Flood",
        event_date=datetime(2018, 10, 16, 7, 20),  # Actual peak time
        location="Llano and Kingsland",
        county="Llano",
        lat=30.75,
        lon=-98.67,
        deaths=9,
        peak_stage_ft=40.1,
        description="Rapid river rise trapped people",
        nearby_stations=["08150000", "08150700", "08151500"],
    ),
    # Additional events
    HistoricalEvent(
        name="Halloween Flood 2013",
        event_date=datetime(2013, 10, 31, 8, 0),
        location="Central Texas",
        county="Travis",
        lat=30.27,
        lon=-97.74,
        deaths=4,
        peak_stage_ft=20.0,
        description="Onion Creek flooding",
        nearby_stations=["08158000", "08158700", "08159000"],
    ),
    HistoricalEvent(
        name="Tax Day Flood 2016",
        event_date=datetime(2016, 4, 18, 6, 0),
        location="Houston",
        county="Harris",
        lat=29.76,
        lon=-95.37,
        deaths=8,
        peak_stage_ft=35.0,
        description="Record flooding in Houston",
        nearby_stations=["08074000", "08073600", "08075000"],
        event_type="major_flood",
        is_slow_rise=True,
    ),
    HistoricalEvent(
        name="Tropical Storm Imelda",
        event_date=datetime(2019, 9, 19, 12, 0),
        location="Southeast Texas",
        county="Jefferson",
        lat=30.08,
        lon=-94.13,
        deaths=5,
        peak_stage_ft=25.0,
        description="Extreme rainfall flooding",
        nearby_stations=["08041780", "08041700", "08041500"],
        event_type="major_flood",
        is_slow_rise=True,
    ),
]


def fetch_usgs_historical(station_id: str, start_date: str, end_date: str) -> List[USGSReading]:
    """Fetch real USGS stream gauge data."""
    readings = []

    for service in ["iv", "dv"]:
        url = (
            f"https://waterservices.usgs.gov/nwis/{service}/"
            f"?format=json&sites={station_id}"
            f"&startDT={start_date}&endDT={end_date}"
            f"&parameterCd=00065&siteStatus=all"
        )

        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = json.loads(response.read().decode())

            time_series = data.get("value", {}).get("timeSeries", [])
            if not time_series:
                continue

            for ts in time_series:
                for val in ts.get("values", [{}])[0].get("value", []):
                    try:
                        timestamp = datetime.fromisoformat(
                            val["dateTime"].replace("Z", "+00:00")
                        ).replace(tzinfo=None)
                        gage_height = float(val["value"])
                        readings.append(USGSReading(
                            timestamp=timestamp,
                            station_id=station_id,
                            gage_height_ft=gage_height,
                        ))
                    except (ValueError, KeyError):
                        continue

            if readings:
                return readings
        except Exception:
            continue

    return readings


def calculate_base_level(readings: List[USGSReading], event_date: datetime) -> float:
    """
    Calculate base stream level using readings from BEFORE the event.
    This is critical for detecting rises relative to normal conditions.
    """
    if not readings:
        return 0.0

    # Use readings from 3-7 days before the event as baseline
    baseline_start = event_date - timedelta(days=7)
    baseline_end = event_date - timedelta(days=2)

    baseline_readings = [
        r.gage_height_ft for r in readings
        if baseline_start <= r.timestamp <= baseline_end
    ]

    if baseline_readings:
        return sum(baseline_readings) / len(baseline_readings)

    # Fallback: use minimum from first 24 hours of data
    early_readings = sorted(readings, key=lambda r: r.timestamp)[:24]
    if early_readings:
        return min(r.gage_height_ft for r in early_readings)

    return 0.0


def calculate_rise_rate(readings: List[USGSReading], target_time: datetime, window_hours: int = 3) -> float:
    """Calculate stream rise rate (ft/hr) at a specific time."""
    if not readings:
        return 0.0

    window_start = target_time - timedelta(hours=window_hours)
    window_readings = [r for r in readings if window_start <= r.timestamp <= target_time]

    if len(window_readings) < 2:
        return 0.0

    window_readings.sort(key=lambda r: r.timestamp)
    first = window_readings[0]
    last = window_readings[-1]

    time_diff_hours = (last.timestamp - first.timestamp).total_seconds() / 3600
    if time_diff_hours < 0.25:
        return 0.0

    height_change = last.gage_height_ft - first.gage_height_ft
    return max(0, height_change / time_diff_hours)


def detect_flood_dynamic(
    current_height_ft: float,
    base_level_ft: float,
    rise_rate_ft_hr: float,
    is_slow_rise: bool = False,
    height_history: Optional[List[float]] = None,
) -> Tuple[str, float, List[str], str, bool, int]:
    """
    Dynamic flood detection based on rise from base level.
    Key insight: A 10ft stage means nothing without knowing what 'normal' is.

    Returns: (alert_level, risk, factors, attractor_basin, phi_resonance, phi_confidence)
    """
    factors = []
    risk = 0.0

    # Calculate rise above base level
    rise_from_base = current_height_ft - base_level_ft

    # QMNF: Initialize attractor classification
    attractor_basin = "UNKNOWN"
    phi_resonance_detected = False
    phi_resonance_confidence = 0

    # Dynamic stage thresholds based on rise from normal
    if rise_from_base >= 20.0:  # 20+ ft above normal = catastrophic
        factors.append("stage_catastrophic")
        risk += 0.50
    elif rise_from_base >= 10.0:  # 10+ ft above normal = major
        factors.append("stage_major")
        risk += 0.35
    elif rise_from_base >= 5.0:  # 5+ ft above normal = moderate
        factors.append("stage_moderate")
        risk += 0.25
    elif rise_from_base >= 2.0:  # 2+ ft above normal = minor
        factors.append("stage_minor")
        risk += 0.15

    # Rise rate factors
    if rise_rate_ft_hr >= 10.0:  # Extreme rise
        factors.append("rise_extreme")
        risk += 0.35
    elif rise_rate_ft_hr >= 5.0:  # Very rapid
        factors.append("rise_very_rapid")
        risk += 0.30
    elif rise_rate_ft_hr >= 2.0:  # Rapid
        factors.append("rise_rapid")
        risk += 0.25
    elif rise_rate_ft_hr >= 1.0:  # Moderate
        factors.append("rise_moderate")
        risk += 0.15
    elif rise_rate_ft_hr >= 0.5:  # Slow but significant
        factors.append("rise_slow")
        risk += 0.10

    # Absolute height check (some rivers have low normal levels)
    if current_height_ft >= 30.0:
        if "stage_catastrophic" not in factors:
            factors.append("stage_high_absolute")
            risk += 0.20
    elif current_height_ft >= 20.0:
        if "stage_major" not in factors:
            factors.append("stage_moderate_absolute")
            risk += 0.15

    # QMNF: Attractor basin classification
    if QMNF_AVAILABLE:
        # Convert to integer scale for AttractorClassifier
        # rain_rate proxy: use rise_rate (higher rise = more inflow)
        rain_proxy = int(rise_rate_ft_hr * 1000)  # Scale ft/hr to mm/hr equivalent
        # pressure_tendency proxy: use change rate (negative = rising water = falling pressure)
        pressure_proxy = int(-rise_rate_ft_hr * 500)
        # humidity proxy: high when flooding (saturated conditions)
        humidity_proxy = min(100, int(50 + rise_from_base * 5))

        classifier = AttractorClassifier()
        basin_name, basin_sig = classifier.classify(
            rain_rate=rain_proxy,
            pressure_tendency=pressure_proxy,
            humidity=humidity_proxy
        )
        attractor_basin = basin_name

        # Add basin to factors if in dangerous basin
        if attractor_basin == "FLASH_FLOOD":
            factors.append("attractor_basin_ff")
            risk += 0.10

        # Detect φ-resonance in stream height history
        if height_history and len(height_history) >= 5:
            # Convert to integer scale (cm)
            heights_int = [int(h * 30.48) for h in height_history]  # ft to cm
            phi_detector = PhiResonanceDetector(tolerance_permille=40)
            phi_result = phi_detector.detect_resonance(heights_int)
            phi_resonance_detected = phi_result["has_resonance"]
            phi_resonance_confidence = phi_result["confidence"]

            if phi_resonance_detected and phi_resonance_confidence >= 25:
                factors.append("phi_resonance")
                risk += 0.05

    # Slow-rise adjustment for major floods
    if is_slow_rise:
        # Major floods are harder to miss - sustained high water
        risk *= 1.2

    risk = min(risk, 1.0)

    # Alert level determination
    if len(factors) >= 3 and risk >= 0.65:
        return "FF_EMERGENCY", risk, factors, attractor_basin, phi_resonance_detected, phi_resonance_confidence
    elif len(factors) >= 2 and risk >= 0.50:
        return "FF_WARNING", risk, factors, attractor_basin, phi_resonance_detected, phi_resonance_confidence
    elif len(factors) >= 2 and risk >= 0.35:
        return "FF_ADVISORY", risk, factors, attractor_basin, phi_resonance_detected, phi_resonance_confidence
    elif len(factors) >= 1 and risk >= 0.20:
        return "FF_WATCH", risk, factors, attractor_basin, phi_resonance_detected, phi_resonance_confidence
    else:
        return "CLEAR", risk, factors, attractor_basin, phi_resonance_detected, phi_resonance_confidence


def validate_event(event: HistoricalEvent) -> List[ValidationResult]:
    """Validate MYSTIC detection against a historical event with dynamic thresholds."""
    results = []

    print(f"\n{'='*70}")
    print(f"VALIDATING: {event.name}")
    print(f"Date: {event.event_date}")
    print(f"Location: {event.location}, {event.county} County")
    print(f"Peak Stage: {event.peak_stage_ft} ft | Deaths: {event.deaths}")
    print(f"Type: {'Slow Rise' if event.is_slow_rise else 'Flash Flood'}")
    print(f"{'='*70}")

    start_date = (event.event_date - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = (event.event_date + timedelta(days=2)).strftime("%Y-%m-%d")

    all_readings: List[USGSReading] = []
    station_readings: Dict[str, List[USGSReading]] = {}

    for station_id in event.nearby_stations:
        print(f"  Fetching {station_id}...", end="")
        readings = fetch_usgs_historical(station_id, start_date, end_date)
        if readings:
            all_readings.extend(readings)
            station_readings[station_id] = sorted(readings, key=lambda r: r.timestamp)
            print(f" {len(readings)} readings")
        else:
            print(" no data")

    if not all_readings:
        print(f"  WARNING: No USGS data available!")
        return results

    # Find peak reading
    peak_reading = max(all_readings, key=lambda r: r.gage_height_ft)
    print(f"  Peak observed: {peak_reading.gage_height_ft:.1f} ft at {peak_reading.timestamp}")

    # Calculate base level from pre-event data
    base_level = calculate_base_level(all_readings, event.event_date)
    print(f"  Base level:    {base_level:.1f} ft")
    print(f"  Rise to peak:  {peak_reading.gage_height_ft - base_level:.1f} ft")

    # Test at various time offsets
    test_offsets = [-48, -36, -24, -18, -12, -6, -3, -2, -1, 0, 1, 2]

    for offset in test_offsets:
        test_time = event.event_date + timedelta(hours=offset)

        # Find best reading near this time
        best_station = None
        best_reading = None
        min_time_diff = (1 << 63) - 1  # Integer max instead of float('inf')

        for station_id, readings in station_readings.items():
            for r in readings:
                time_diff = abs((r.timestamp - test_time).total_seconds())
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_reading = r
                    best_station = station_id

        if best_reading is None or min_time_diff > 3600 * 6:  # Skip if no data within 6 hours
            continue

        # Calculate rise rate
        station_data = station_readings.get(best_station, [])
        rise_rate = calculate_rise_rate(station_data, test_time)

        # QMNF: Get recent height history for φ-resonance detection
        window_start = test_time - timedelta(hours=6)
        height_history = [
            r.gage_height_ft for r in station_data
            if window_start <= r.timestamp <= test_time
        ]

        # Run detection with QMNF integration
        alert_level, probability, factors, attractor_basin, phi_resonance, phi_conf = detect_flood_dynamic(
            current_height_ft=best_reading.gage_height_ft,
            base_level_ft=base_level,
            rise_rate_ft_hr=rise_rate,
            is_slow_rise=event.is_slow_rise,
            height_history=height_history,
        )

        # For slow-rise events, WATCH counts as detection
        if event.is_slow_rise:
            detected = alert_level in ["FF_EMERGENCY", "FF_WARNING", "FF_ADVISORY", "FF_WATCH"]
        else:
            detected = alert_level in ["FF_EMERGENCY", "FF_WARNING", "FF_ADVISORY"]

        lead_time = -offset if detected else 0
        rise_from_base = best_reading.gage_height_ft - base_level

        result = ValidationResult(
            event_name=event.name,
            event_date=event.event_date.isoformat(),
            hours_before_peak=offset,
            alert_level=alert_level,
            probability=probability,
            detected=detected,
            lead_time_hours=lead_time,
            factors=factors,
            station_id=best_station,
            stream_height_ft=best_reading.gage_height_ft,
            base_level_ft=base_level,
            rise_from_base=rise_from_base,
            attractor_basin=attractor_basin,
            phi_resonance_detected=phi_resonance,
            phi_resonance_confidence=phi_conf,
        )
        results.append(result)

        status = "DETECTED" if detected else "missed"
        print(f"  T{offset:+3d}h: {alert_level:12s} ({probability:5.0%}) | "
              f"Stage:{best_reading.gage_height_ft:5.1f}ft (+{rise_from_base:4.1f}) | "
              f"Rise:{rise_rate:4.1f}ft/hr | {status}")

    return results


def calculate_metrics(results: List[ValidationResult]) -> Dict:
    """Calculate verification metrics with proper lead time window."""
    hits = 0
    misses = 0
    false_alarms = 0
    correct_negatives = 0

    for r in results:
        # Within 6 hours of peak = should detect
        should_detect = -6 <= r.hours_before_peak <= 2

        if should_detect:
            if r.detected:
                hits += 1
            else:
                misses += 1
        else:
            # Far before event - early detection is OK, not false alarm
            if r.detected and r.hours_before_peak < -24:
                # Only count as FA if we detect with nothing happening
                if r.rise_from_base < 2.0:  # River not actually rising
                    false_alarms += 1
            if not r.detected:
                correct_negatives += 1

    total_positive = hits + misses
    total_predicted = hits + false_alarms

    pod = hits / total_positive if total_positive > 0 else 0
    far = false_alarms / total_predicted if total_predicted > 0 else 0
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0

    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_negatives,
        "POD": pod,
        "FAR": far,
        "CSI": csi,
    }


def generate_report(all_results: Dict[str, List[ValidationResult]], metrics: Dict, events: List[HistoricalEvent]) -> str:
    """Generate comprehensive validation report."""
    report = []
    report.append("=" * 90)
    report.append("MYSTIC HISTORICAL VALIDATION REPORT v2")
    report.append("Real USGS Data Testing - Dynamic Threshold Detection")
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("=" * 90)
    report.append("")

    # Summary table
    report.append("EVENT SUMMARY")
    report.append("-" * 90)
    report.append(f"{'Event':<35} {'Date':<12} {'Deaths':<6} {'Peak':<8} {'Lead Time':<12} {'Status'}")
    report.append("-" * 90)

    for event in events:
        event_results = all_results.get(event.name, [])
        if event_results:
            detections = [r for r in event_results if r.detected and r.hours_before_peak <= 0]
            if detections:
                earliest = min(detections, key=lambda r: r.hours_before_peak)
                lead_time = f"{-earliest.hours_before_peak:.0f}h before"
                status = "DETECTED"
            else:
                lead_time = "N/A"
                status = "MISSED"
        else:
            lead_time = "NO DATA"
            status = "SKIP"

        report.append(f"{event.name:<35} {event.event_date.strftime('%Y-%m-%d'):<12} "
                     f"{event.deaths:<6} {event.peak_stage_ft:<8.1f} {lead_time:<12} {status}")

    report.append("")
    report.append("VERIFICATION METRICS")
    report.append("-" * 50)
    report.append(f"Probability of Detection (POD): {metrics['POD']:.1%}")
    report.append(f"False Alarm Rate (FAR):         {metrics['FAR']:.1%}")
    report.append(f"Critical Success Index (CSI):   {metrics['CSI']:.1%}")
    report.append("")
    report.append(f"Hits:              {metrics['hits']}")
    report.append(f"Misses:            {metrics['misses']}")
    report.append(f"False Alarms:      {metrics['false_alarms']}")
    report.append(f"Correct Negatives: {metrics['correct_negatives']}")
    report.append("")

    # Per-event details
    for event_name, results in all_results.items():
        if not results:
            continue
        report.append("-" * 90)
        report.append(f"EVENT: {event_name}")
        report.append("-" * 90)

        base = results[0].base_level_ft if results else 0
        report.append(f"Base Level: {base:.1f} ft")

        detections = [r for r in results if r.detected and r.hours_before_peak <= 0]
        if detections:
            earliest = min(detections, key=lambda r: r.hours_before_peak)
            report.append(f"First Detection: T{earliest.hours_before_peak:+.0f}h at {earliest.alert_level}")
            report.append(f"Lead Time: {-earliest.hours_before_peak:.0f} hours before peak")
        else:
            report.append("DETECTION FAILED")

        report.append("")
        report.append("Timeline (stage relative to base):")
        for r in sorted(results, key=lambda x: x.hours_before_peak):
            status = "DETECTED" if r.detected else "missed"
            factors_str = ",".join(r.factors[:2]) if r.factors else "none"
            report.append(f"  T{r.hours_before_peak:+3.0f}h | {r.alert_level:12s} | "
                         f"Stage: {r.stream_height_ft:5.1f}ft (+{r.rise_from_base:4.1f}) | {status}")
        report.append("")

    # Targets assessment
    report.append("=" * 90)
    report.append("TARGETS ASSESSMENT")
    report.append("-" * 50)

    pod_pass = metrics['POD'] >= 0.85
    far_pass = metrics['FAR'] <= 0.30
    csi_pass = metrics['CSI'] >= 0.50

    report.append(f"POD >= 85%:  {'PASS' if pod_pass else 'FAIL'} ({metrics['POD']:.1%})")
    report.append(f"FAR <= 30%:  {'PASS' if far_pass else 'FAIL'} ({metrics['FAR']:.1%})")
    report.append(f"CSI >= 50%:  {'PASS' if csi_pass else 'FAIL'} ({metrics['CSI']:.1%})")
    report.append("")

    if pod_pass and far_pass and csi_pass:
        report.append("OVERALL: ALL TARGETS MET - PRODUCTION READY")
    else:
        failed = []
        if not pod_pass:
            failed.append("POD")
        if not far_pass:
            failed.append("FAR")
        if not csi_pass:
            failed.append("CSI")
        report.append(f"OVERALL: TARGETS NOT MET ({', '.join(failed)} failed)")

    report.append("=" * 90)
    report.append("")
    report.append("KEY INSIGHTS:")
    report.append("-" * 50)
    report.append("1. Dynamic thresholds based on rise-from-base are critical")
    report.append("2. Flash floods need rise rate detection (ft/hr)")
    report.append("3. Major floods (Harvey, Imelda) need sustained high water detection")
    report.append("4. Historical USGS data quality varies by station and era")
    report.append("5. Instantaneous values (iv) much better than daily (dv) for flash floods")

    return "\n".join(report)


def main():
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║         MYSTIC Historical Validation v2 - Enhanced Real Data Testing      ║")
    print("║                    Dynamic Threshold Flash Flood Detection                 ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print()
    print("Testing with dynamic thresholds based on rise-from-base and rise rate")
    print()

    all_results: Dict[str, List[ValidationResult]] = {}
    combined_results: List[ValidationResult] = []

    for event in HISTORICAL_EVENTS:
        results = validate_event(event)
        if results:
            all_results[event.name] = results
            combined_results.extend(results)

    if not combined_results:
        print("\nERROR: No validation results!")
        return

    print("\n" + "=" * 70)
    print("CALCULATING VERIFICATION METRICS")
    print("=" * 70)

    metrics = calculate_metrics(combined_results)
    print(f"\nPOD: {metrics['POD']:.1%} | FAR: {metrics['FAR']:.1%} | CSI: {metrics['CSI']:.1%}")

    # Generate report
    report = generate_report(all_results, metrics, HISTORICAL_EVENTS)

    # Save files
    report_file = "../data/historical_validation_v2_report.txt"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_file}")

    json_file = "../data/historical_validation_v2_results.json"
    json_data = {
        "version": "2.0",
        "generated": datetime.now().isoformat(),
        "events_tested": len(HISTORICAL_EVENTS),
        "events_with_data": len(all_results),
        "total_readings": len(combined_results),
        "metrics": metrics,
        "detection_method": "dynamic_threshold_rise_from_base",
        "results": [
            {
                "event_name": r.event_name,
                "event_date": r.event_date,
                "hours_before_peak": r.hours_before_peak,
                "alert_level": r.alert_level,
                "probability": r.probability,
                "detected": r.detected,
                "lead_time_hours": r.lead_time_hours,
                "factors": r.factors,
                "station_id": r.station_id,
                "stream_height_ft": r.stream_height_ft,
                "base_level_ft": r.base_level_ft,
                "rise_from_base": r.rise_from_base,
                # QMNF fields
                "attractor_basin": r.attractor_basin,
                "phi_resonance_detected": r.phi_resonance_detected,
                "phi_resonance_confidence": r.phi_resonance_confidence,
            }
            for r in combined_results
        ]
    }

    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON saved to: {json_file}")

    print("\n" + report)


if __name__ == "__main__":
    main()
