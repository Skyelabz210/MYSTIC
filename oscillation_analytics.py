#!/usr/bin/env python3
"""
OSCILLATION ANALYTICS MODULE

Oscillations in weather data are NOT noise - they are diagnostic signals
that indicate specific meteorological phenomena.

Oscillation Pattern → Meteorological Meaning:
1. High-frequency large-amplitude → Mesocyclone/tornado precursor
2. Spike-then-drop pattern → Gust front (derecho, squall line)
3. Regular diurnal cycle → Normal fair weather
4. Irregular with downward trend → Frontal passage
5. Increasing amplitude over time → Atmospheric destabilization
6. Dampening oscillation → System stabilizing

This module classifies oscillation patterns and predicts what they precede.

Author: Claude (K-Elimination Expert)
Date: 2026-01-07
"""

from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

# K-Elimination for exact division
from k_elimination import KElimination, KEliminationContext

# Module-level K-Elimination instance for exact division operations
_KELIM = KElimination(KEliminationContext.for_weather())


def _divide_exact(dividend: int, divisor: int) -> int:
    """
    Division using K-Elimination for exact divides, fallback to // otherwise.

    This ensures maximum precision in oscillation pattern detection where
    exact amplitude and frequency calculations are critical.
    """
    if divisor == 0:
        return 0

    # Try K-Elimination for exact divides
    if dividend % divisor == 0:
        try:
            abs_result = _KELIM.exact_divide(abs(dividend), abs(divisor))
            # Handle sign
            if (dividend < 0) != (divisor < 0):  # XOR for sign
                return -abs_result
            return abs_result
        except (ValueError, OverflowError):
            # Fallback if K-Elimination fails
            pass

    # Fallback to standard integer division
    return dividend // divisor


class OscillationPattern(Enum):
    """Classification of oscillation patterns by meteorological meaning."""
    DIURNAL = "DIURNAL"                    # Normal day/night cycle
    GUST_FRONT = "GUST_FRONT"              # Pressure spike then drop
    MESOCYCLONE = "MESOCYCLONE"            # High-freq large-amplitude
    FRONTAL = "FRONTAL"                    # Irregular with trend
    DESTABILIZING = "DESTABILIZING"        # Increasing amplitude
    STABILIZING = "STABILIZING"            # Decreasing amplitude
    GRAVITY_WAVE = "GRAVITY_WAVE"          # Regular medium-frequency
    TURBULENT = "TURBULENT"                # Chaotic/random
    MINIMAL = "MINIMAL"                    # Very small oscillations
    UNKNOWN = "UNKNOWN"                    # Unclassified


@dataclass
class OscillationAnalysis:
    """Comprehensive oscillation analysis result."""
    pattern: OscillationPattern
    confidence: int  # 0-100

    # Characteristics
    frequency: int          # Oscillations per unit time (scaled ×100)
    amplitude_mean: int     # Mean oscillation amplitude
    amplitude_trend: str    # INCREASING, DECREASING, STABLE

    # Derived signals
    has_spike: bool         # Detected pressure spike
    spike_magnitude: int    # Size of largest spike
    has_precursor: bool     # Oscillation precedes known event
    precursor_type: str     # What event it might precede

    # Risk contribution
    risk_modifier: int      # Points to add/subtract from base risk
    risk_reason: str        # Explanation

    # Raw metrics
    sign_changes: int
    zero_crossings: int
    peak_count: int
    trough_count: int


def compute_oscillation_metrics(series: List[int]) -> Dict[str, Any]:
    """
    Compute detailed oscillation metrics from time series.

    Returns comprehensive metrics about oscillation behavior.
    """
    if len(series) < 5:
        return {"error": "Insufficient data"}

    # First differences
    diffs = [series[i + 1] - series[i] for i in range(len(series) - 1)]

    # Sign changes in differences (direction reversals)
    sign_changes = 0
    for i in range(len(diffs) - 1):
        if (diffs[i] > 0 and diffs[i + 1] < 0) or (diffs[i] < 0 and diffs[i + 1] > 0):
            sign_changes += 1

    # Calculate mean-centered oscillations
    mean_val = _divide_exact(sum(series), len(series))
    centered = [v - mean_val for v in series]

    # Zero crossings (around mean)
    zero_crossings = 0
    for i in range(len(centered) - 1):
        if (centered[i] >= 0 and centered[i + 1] < 0) or (centered[i] < 0 and centered[i + 1] >= 0):
            zero_crossings += 1

    # Find peaks and troughs
    peaks = []
    troughs = []
    for i in range(1, len(series) - 1):
        if series[i] > series[i - 1] and series[i] > series[i + 1]:
            peaks.append((i, series[i]))
        elif series[i] < series[i - 1] and series[i] < series[i + 1]:
            troughs.append((i, series[i]))

    # Oscillation amplitudes (peak-to-trough)
    amplitudes = []
    for i in range(min(len(peaks), len(troughs))):
        if peaks and troughs:
            # Find closest peak-trough pair
            amp = abs(peaks[i][1] - troughs[i][1]) if i < len(peaks) and i < len(troughs) else 0
            if amp > 0:
                amplitudes.append(amp)

    mean_amplitude = _divide_exact(sum(amplitudes), len(amplitudes)) if amplitudes else 0

    # Amplitude trend (is oscillation growing or shrinking?)
    if len(amplitudes) >= 3:
        first_third = _divide_exact(sum(amplitudes[:_divide_exact(len(amplitudes), 3)]), max(1, _divide_exact(len(amplitudes), 3)))
        last_third = _divide_exact(sum(amplitudes[-_divide_exact(len(amplitudes), 3):]), max(1, _divide_exact(len(amplitudes), 3)))
        if last_third > first_third * 1.3:
            amplitude_trend = "INCREASING"
        elif last_third < first_third * 0.7:
            amplitude_trend = "DECREASING"
        else:
            amplitude_trend = "STABLE"
    else:
        amplitude_trend = "STABLE"

    # Frequency estimate (oscillations per unit time)
    n = len(series)
    oscillation_count = min(len(peaks), len(troughs))
    frequency_scaled = _divide_exact(oscillation_count * 100, n) if n > 0 else 0

    # Detect spikes (sudden jumps compared to local mean)
    spikes = []
    data_range = max(series) - min(series)
    spike_threshold = max(4, _divide_exact(data_range, 12))
    for i in range(1, len(series) - 1):
        local_mean = _divide_exact(series[i - 1] + series[i + 1], 2)
        deviation = series[i] - local_mean
        if deviation >= spike_threshold:
            spikes.append((i, deviation, "up"))
        elif deviation <= -spike_threshold:
            spikes.append((i, abs(deviation), "down"))

    max_spike = max((s[1] for s in spikes), default=0)

    # Detect gust front pattern: stable baseline, spike, then sustained drop
    has_gust_front = False
    gust_front_magnitude = 0
    baseline_threshold = max(3, _divide_exact(data_range, 10))
    gust_spike_threshold = max(6, _divide_exact(data_range, 8))
    gust_drop_threshold = max(5, _divide_exact(data_range, 8))
    for i in range(3, len(series) - 5):
        baseline_window = series[i - 4:i - 1]
        if len(baseline_window) < 3:
            continue
        baseline = _divide_exact(sum(baseline_window), len(baseline_window))
        baseline_range = max(baseline_window) - min(baseline_window)
        if baseline_range > baseline_threshold:
            continue
        if not (series[i] >= series[i - 1] and series[i] >= series[i + 1]):
            continue
        spike_rise = series[i] - baseline
        if spike_rise < gust_spike_threshold:
            continue
        post_window = series[i + 1:i + 6]
        post_min = min(post_window)
        post_drop = series[i] - post_min
        below_baseline = sum(1 for v in post_window if v <= baseline - 2)
        if post_drop >= gust_drop_threshold and below_baseline >= 3:
            has_gust_front = True
            gust_front_magnitude = post_drop
            break

    # Check for diurnal pattern (regular ~24 hour cycle)
    # Look for consistent period in oscillations
    if len(peaks) >= 3:
        periods = [peaks[i + 1][0] - peaks[i][0] for i in range(len(peaks) - 1)]
        avg_period = _divide_exact(sum(periods), len(periods)) if periods else 0
        period_variance = _divide_exact(sum((p - avg_period) ** 2 for p in periods), len(periods)) if periods else 0
        is_regular = period_variance < 5 and avg_period > 8  # Regular, longer period
    else:
        is_regular = False
        avg_period = 0

    oscillation_ratio = _divide_exact(sign_changes * 100, max(1, len(diffs) - 1))

    return {
        "sign_changes": sign_changes,
        "zero_crossings": zero_crossings,
        "peaks": peaks,
        "troughs": troughs,
        "peak_count": len(peaks),
        "trough_count": len(troughs),
        "amplitudes": amplitudes,
        "mean_amplitude": mean_amplitude,
        "amplitude_trend": amplitude_trend,
        "frequency_scaled": frequency_scaled,
        "spikes": spikes,
        "max_spike": max_spike,
        "has_spike": len(spikes) > 0,
        "has_gust_front": has_gust_front,
        "gust_front_magnitude": gust_front_magnitude,
        "is_regular": is_regular,
        "avg_period": avg_period,
        "data_range": max(series) - min(series),
        "series_length": len(series),
        "oscillation_ratio": oscillation_ratio
    }


def classify_oscillation_pattern(metrics: Dict[str, Any]) -> Tuple[OscillationPattern, int]:
    """
    Classify oscillation pattern based on metrics.

    Returns (pattern_type, confidence)
    """
    mean_amp = metrics.get("mean_amplitude", 0)
    freq = metrics.get("frequency_scaled", 0)
    amp_trend = metrics.get("amplitude_trend", "STABLE")
    has_gust_front = metrics.get("has_gust_front", False)
    is_regular = metrics.get("is_regular", False)
    data_range = metrics.get("data_range", 0)
    sign_changes = metrics.get("sign_changes", 0)
    n = metrics.get("series_length", 1)

    # Calculate oscillation intensity
    osc_ratio = _divide_exact(sign_changes * 100, max(1, n - 2))

    # Minimal oscillation check - very small variations regardless of frequency
    if mean_amp <= 2 and data_range <= 3:
        return OscillationPattern.MINIMAL, 90

    # Gust front pattern (derecho/squall line signature)
    if has_gust_front:
        magnitude = metrics.get("gust_front_magnitude", 0)
        conf = min(95, 60 + magnitude)
        return OscillationPattern.GUST_FRONT, conf

    # Mesocyclone/tornado precursor (very high-freq, large-amp, increasing)
    if freq > 35 and mean_amp > 20 and amp_trend == "INCREASING":
        return OscillationPattern.MESOCYCLONE, 85

    # Destabilizing pattern (amplitude growing)
    if amp_trend == "INCREASING" and mean_amp > 5:
        return OscillationPattern.DESTABILIZING, 80

    # Stabilizing pattern (amplitude decreasing)
    if amp_trend == "DECREASING" and mean_amp > 5:
        return OscillationPattern.STABILIZING, 80

    # Diurnal pattern (regular, low-frequency)
    if is_regular and freq < 15:
        return OscillationPattern.DIURNAL, 85

    # Gravity wave (regular medium-frequency)
    if is_regular and 15 <= freq <= 30:
        return OscillationPattern.GRAVITY_WAVE, 75

    # Frontal passage (oscillations with downward trend in data)
    # Check for overall downward trend
    n = metrics.get("series_length", 1)
    if n > 5:
        # Look at peaks to see if they're trending down
        peaks = metrics.get("peaks", [])
        if len(peaks) >= 2:
            first_peak_val = peaks[0][1] if peaks else 0
            last_peak_val = peaks[-1][1] if peaks else 0
            if first_peak_val > last_peak_val + 5:  # Peaks are dropping
                return OscillationPattern.FRONTAL, 75

    # Frontal passage (irregular with oscillations)
    if osc_ratio > 40 and not is_regular:
        return OscillationPattern.FRONTAL, 70

    # Turbulent (high-freq, irregular)
    if freq > 30 and not is_regular:
        return OscillationPattern.TURBULENT, 75

    # Check for any oscillation with amplitude > 2 but not matching other patterns
    if mean_amp >= 2 and mean_amp < 10:
        # Mild oscillation, could be frontal or transitional
        if osc_ratio > 20:
            return OscillationPattern.FRONTAL, 60

    return OscillationPattern.UNKNOWN, 40


def determine_precursor_type(pattern: OscillationPattern, metrics: Dict) -> Tuple[bool, str]:
    """
    Determine if oscillation pattern is a precursor to a weather event.

    Returns (is_precursor, event_type)
    """
    precursor_map = {
        OscillationPattern.GUST_FRONT: (True, "DERECHO_SQUALL"),
        OscillationPattern.MESOCYCLONE: (True, "TORNADO"),
        OscillationPattern.DESTABILIZING: (True, "SEVERE_WEATHER"),
        OscillationPattern.FRONTAL: (True, "FRONTAL_PASSAGE"),
        OscillationPattern.DIURNAL: (False, "FAIR_WEATHER"),
        OscillationPattern.STABILIZING: (False, "IMPROVING"),
        OscillationPattern.GRAVITY_WAVE: (False, "NEUTRAL"),
        OscillationPattern.TURBULENT: (True, "TURBULENCE"),
        OscillationPattern.MINIMAL: (False, "STABLE"),
        OscillationPattern.UNKNOWN: (False, "UNKNOWN"),
    }
    return precursor_map.get(pattern, (False, "UNKNOWN"))


def calculate_risk_modifier(
    pattern: OscillationPattern,
    metrics: Dict[str, Any]
) -> Tuple[int, str]:
    """
    Calculate risk modifier based on oscillation pattern.

    Returns (risk_points, explanation)
    """
    risk_modifiers = {
        OscillationPattern.GUST_FRONT: (
            30,
            "Gust front oscillation detected - severe wind event likely"
        ),
        OscillationPattern.MESOCYCLONE: (
            40,
            "Mesocyclone oscillation signature - tornado conditions possible"
        ),
        OscillationPattern.DESTABILIZING: (
            25,
            "Increasing oscillation amplitude - atmosphere destabilizing"
        ),
        OscillationPattern.FRONTAL: (
            15,
            "Frontal oscillation pattern - weather change expected"
        ),
        OscillationPattern.TURBULENT: (
            20,
            "Turbulent oscillation pattern - unstable conditions"
        ),
        OscillationPattern.DIURNAL: (
            -10,
            "Normal diurnal oscillation - stable conditions"
        ),
        OscillationPattern.STABILIZING: (
            -15,
            "Decreasing oscillation amplitude - conditions improving"
        ),
        OscillationPattern.GRAVITY_WAVE: (
            0,
            "Gravity wave oscillation - neutral indicator"
        ),
        OscillationPattern.MINIMAL: (
            -5,
            "Minimal oscillation - very stable conditions"
        ),
        OscillationPattern.UNKNOWN: (
            0,
            "Oscillation pattern unclassified"
        ),
    }

    base_modifier, reason = risk_modifiers.get(pattern, (0, "Unknown"))

    # Amplify if amplitude is particularly large
    mean_amp = metrics.get("mean_amplitude", 0)
    if mean_amp > 20 and base_modifier > 0:
        base_modifier = int(base_modifier * 1.3)
        reason += f" (amplified by {mean_amp}mb amplitude)"

    return base_modifier, reason


def analyze_oscillations(series: List[int]) -> OscillationAnalysis:
    """
    Comprehensive oscillation analysis.

    Args:
        series: Time series data (integer values)

    Returns:
        OscillationAnalysis with pattern classification and risk assessment
    """
    metrics = compute_oscillation_metrics(series)

    if "error" in metrics:
        return OscillationAnalysis(
            pattern=OscillationPattern.UNKNOWN,
            confidence=0,
            frequency=0,
            amplitude_mean=0,
            amplitude_trend="UNKNOWN",
            has_spike=False,
            spike_magnitude=0,
            has_precursor=False,
            precursor_type="INSUFFICIENT_DATA",
            risk_modifier=0,
            risk_reason="Insufficient data for oscillation analysis",
            sign_changes=0,
            zero_crossings=0,
            peak_count=0,
            trough_count=0
        )

    pattern, confidence = classify_oscillation_pattern(metrics)
    is_precursor, precursor_type = determine_precursor_type(pattern, metrics)
    risk_modifier, risk_reason = calculate_risk_modifier(pattern, metrics)

    return OscillationAnalysis(
        pattern=pattern,
        confidence=confidence,
        frequency=metrics["frequency_scaled"],
        amplitude_mean=metrics["mean_amplitude"],
        amplitude_trend=metrics["amplitude_trend"],
        has_spike=len(metrics["spikes"]) > 0,
        spike_magnitude=metrics["max_spike"],
        has_precursor=is_precursor,
        precursor_type=precursor_type,
        risk_modifier=risk_modifier,
        risk_reason=risk_reason,
        sign_changes=metrics["sign_changes"],
        zero_crossings=metrics["zero_crossings"],
        peak_count=metrics["peak_count"],
        trough_count=metrics["trough_count"]
    )


def run_oscillation_tests():
    """Test oscillation analytics on known patterns."""
    import math

    print("=" * 70)
    print("OSCILLATION ANALYTICS TEST SUITE")
    print("Validating oscillation pattern recognition")
    print("=" * 70)

    # Generate realistic diurnal pattern (sinusoidal with 24-hour period)
    diurnal_data = [1020 + int(3 * math.sin(2 * math.pi * i / 24)) for i in range(72)]

    # Generate gust front: stable, then spike, then sharp drop
    gust_front_data = (
        [1008 + (i % 2) for i in range(8)] +  # Stable
        [1010, 1015, 1018, 1015] +              # Spike up
        [1010, 1000, 990, 985, 982, 980, 983, 985, 988, 990]  # Drop and partial recovery
    )

    # Destabilizing: oscillation amplitude grows over time
    destabilizing_data = [1000 + int((i // 4 + 1) * 2 * ((-1) ** i)) for i in range(30)]

    # Minimal: very small variations
    minimal_data = [1020, 1020, 1021, 1020, 1020, 1020, 1021, 1020, 1020, 1019,
                   1020, 1020, 1020, 1021, 1020, 1020, 1020, 1019, 1020, 1020]

    # Frontal: downward trend with oscillation
    frontal_data = [1015 - i + int(3 * math.sin(i)) for i in range(30)]

    test_cases = [
        {
            "name": "Stable High Pressure (Diurnal)",
            "data": diurnal_data,
            "expected_pattern": OscillationPattern.DIURNAL,
            "expected_precursor": False
        },
        {
            "name": "Derecho Gust Front",
            "data": gust_front_data,
            "expected_pattern": OscillationPattern.GUST_FRONT,
            "expected_precursor": True
        },
        {
            "name": "Destabilizing Atmosphere",
            "data": destabilizing_data,
            "expected_pattern": OscillationPattern.DESTABILIZING,
            "expected_precursor": True
        },
        {
            "name": "Minimal Oscillation (Stable)",
            "data": minimal_data,
            "expected_pattern": OscillationPattern.MINIMAL,
            "expected_precursor": False
        },
        {
            "name": "Frontal Passage",
            "data": frontal_data,
            "expected_pattern": OscillationPattern.FRONTAL,
            "expected_precursor": True
        },
    ]

    correct = 0
    total = len(test_cases)

    for case in test_cases:
        print(f"\n{'─' * 70}")
        print(f"TEST: {case['name']}")
        print(f"{'─' * 70}")

        result = analyze_oscillations(case["data"])

        # Pattern match is primary, precursor match is secondary
        pattern_match = result.pattern == case["expected_pattern"]
        precursor_match = result.has_precursor == case["expected_precursor"]

        # Count as correct only if both pattern and precursor match
        if pattern_match and precursor_match:
            correct += 1

        pattern_mark = "✓" if pattern_match else "○"
        precursor_mark = "✓" if precursor_match else "✗"

        print(f"  Pattern: {result.pattern.value} (expected: {case['expected_pattern'].value}) {pattern_mark}")
        print(f"  Precursor: {result.has_precursor} (expected: {case['expected_precursor']}) {precursor_mark}")
        print(f"  Precursor Type: {result.precursor_type}")
        print(f"  Confidence: {result.confidence}%")
        print(f"  Frequency: {result.frequency}, Amplitude: {result.amplitude_mean} ({result.amplitude_trend})")
        print(f"  Risk Modifier: {result.risk_modifier:+d}")
        print(f"  Reason: {result.risk_reason}")

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {correct}/{total} correct ({_divide_exact(100 * correct, total)}%)")
    print(f"{'=' * 70}")

    return correct == total


if __name__ == "__main__":
    success = run_oscillation_tests()
    exit(0 if success else 1)
