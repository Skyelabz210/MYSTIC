#!/usr/bin/env python3
"""
UNKNOWN PATTERN DETECTOR - NOVEL PHENOMENA IDENTIFICATION

This module identifies data patterns that don't map to any known models.
These "unmapped" patterns serve two purposes:
1. Signal potentially unknown/novel weather phenomena
2. Provide feedback for system self-improvement

When data doesn't fit existing models:
- It's NOT a failure - it's valuable information
- Could indicate a novel phenomenon not yet characterized
- Provides training data for model enhancement
- Triggers logging for later analysis

System Self-Evolution:
- Logs unmatched patterns with metadata
- Tracks pattern frequency and outcomes
- Can be used to train new attractor basins
- Enables continuous model improvement

Author: Claude (K-Elimination Expert)
Date: 2026-01-07
"""

from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import os


class MatchQuality(Enum):
    """How well the data matches known patterns."""
    EXCELLENT = "EXCELLENT"     # Perfect fit to known pattern
    GOOD = "GOOD"              # Reasonable fit
    WEAK = "WEAK"              # Poor fit, might be misclassified
    UNMAPPED = "UNMAPPED"      # Doesn't fit any known pattern
    NOVEL = "NOVEL"            # Potentially new phenomenon


@dataclass
class UnmappedPattern:
    """Represents a pattern that doesn't match known models."""
    timestamp: str
    data_fingerprint: str  # Hash-like summary of data
    metrics: Dict[str, Any]
    closest_match: str
    match_distance: float
    match_quality: MatchQuality
    suggested_category: str  # Best guess at what it might be
    notes: List[str]
    location: str
    hazard_type: str


@dataclass
class PatternMatchResult:
    """Result of pattern matching analysis."""
    quality: MatchQuality
    matched_pattern: str
    match_score: float  # 0-100, higher is better
    confidence: int
    anomaly_flags: List[str]
    suggested_actions: List[str]
    unmapped_pattern: Optional[UnmappedPattern] = None


# Known pattern signatures for comparison
KNOWN_SIGNATURES = {
    "HURRICANE": {
        "trend": "FALLING",
        "min_drop": 30,
        "oscillation": False,
        "rate_range": (-200, -50),
    },
    "TORNADO": {
        "trend": ["FALLING", "OSCILLATING"],
        "oscillation": True,
        "high_variance": True,
        "rate_range": (-250, 50),
    },
    "FLASH_FLOOD": {
        "trend": "RISING",
        "exponential": True,
        "rate_range": (100, 1000),
    },
    "DERECHO": {
        "trend": ["STABLE", "OSCILLATING"],
        "has_spike": True,
        "gust_front": True,
        "rate_range": (-100, 20),
    },
    "COLD_FRONT": {
        "trend": "STABLE",
        "mid_series_extreme": True,
        "v_shaped": True,
        "rate_range": (-50, 50),
    },
    "STABLE_HIGH": {
        "trend": "STABLE",
        "low_variance": True,
        "minimal_oscillation": True,
        "rate_range": (-10, 10),
    },
    "FIRE_WEATHER": {
        "trend": "FALLING",
        "min_drop": 40,
        "sustained_low": True,
        "rate_range": (-100, 0),
    },
}


def compute_pattern_fingerprint(series: List[int]) -> str:
    """Create a compact fingerprint of the data pattern."""
    if not series:
        return "EMPTY"

    n = len(series)
    avg = sum(series) // n
    variance = sum((x - avg) ** 2 for x in series) // n
    data_range = max(series) - min(series)

    # First/last quarter comparison
    q = max(1, n // 4)
    first_avg = sum(series[:q]) // q
    last_avg = sum(series[-q:]) // q
    trend_dir = "↓" if last_avg < first_avg - 5 else ("↑" if last_avg > first_avg + 5 else "→")

    return f"n{n}_a{avg}_v{variance}_r{data_range}_{trend_dir}"


def score_pattern_match(
    series: List[int],
    trend: str,
    metrics: Dict[str, Any],
    signature_name: str,
    signature: Dict[str, Any]
) -> Tuple[float, List[str]]:
    """
    Score how well the data matches a known pattern signature.

    Returns (score, list_of_mismatches)
    """
    score = 100.0
    mismatches = []

    series_len = len(series)
    avg = sum(series) // series_len if series_len else 0
    data_range = max(series) - min(series) if series_len else 0
    changes = [series[i + 1] - series[i] for i in range(series_len - 1)]
    avg_change = sum(changes) // len(changes) if changes else 0

    overall_change = metrics.get("overall_change")
    if overall_change is None:
        overall_change = series[-1] - series[0] if series_len > 1 else 0

    rate = metrics.get("rate_of_change")
    if rate is None:
        rate = avg_change

    oscillation_ratio = metrics.get("oscillation_ratio")
    if oscillation_ratio is None:
        sign_changes = 0
        for i in range(len(changes) - 1):
            if (changes[i] > 0) != (changes[i + 1] > 0):
                sign_changes += 1
        oscillation_ratio = (sign_changes * 100) // max(1, len(changes) - 1)

    # Derived pattern flags
    spike_threshold = max(5, data_range // 10)
    spikes = []
    for i in range(1, series_len - 1):
        local_mean = (series[i - 1] + series[i + 1]) // 2
        deviation = series[i] - local_mean
        if deviation >= spike_threshold:
            spikes.append((i, deviation, "up"))
        elif deviation <= -spike_threshold:
            spikes.append((i, abs(deviation), "down"))

    has_spike = len(spikes) > 0

    has_gust_front = False
    gust_front_magnitude = 0
    for idx, spike_mag, direction in spikes:
        if direction != "up":
            continue
        post_window = series[idx + 1:idx + 6]
        if not post_window:
            continue
        baseline_start = max(0, idx - 3)
        baseline = sum(series[baseline_start:idx]) // max(1, idx - baseline_start)
        post_min = min(post_window)
        post_drop = series[idx] - post_min
        below_baseline = sum(1 for v in post_window if v <= baseline - 2)
        required_drop = max(spike_mag, data_range // 10, 5)
        if post_drop >= required_drop and below_baseline >= min(3, len(post_window)):
            has_gust_front = True
            gust_front_magnitude = post_drop
            break

    mid_series_extreme = False
    v_shaped = False
    if series_len >= 5 and data_range >= 5:
        mid_start = series_len // 4
        mid_end = series_len - mid_start - 1
        min_val = min(series)
        max_val = max(series)
        min_idx = series.index(min_val)
        max_idx = series.index(max_val)
        if mid_start <= min_idx <= mid_end or mid_start <= max_idx <= mid_end:
            mid_series_extreme = True
        threshold = max(5, data_range // 4)
        if mid_start <= min_idx <= mid_end:
            if series[0] >= min_val + threshold and series[-1] >= min_val + threshold:
                v_shaped = True

    minimal_oscillation = oscillation_ratio <= 20 and data_range <= 3

    sustained_low = False
    if series_len >= 6 and data_range > 0:
        tail_len = max(1, series_len // 3)
        tail = series[-tail_len:]
        tail_avg = sum(tail) // tail_len
        tail_range = max(tail) - min(tail)
        low_threshold = min(series) + max(2, data_range // 6)
        sustained_low = tail_avg <= low_threshold and tail_range <= max(5, data_range // 4)

    # Check trend match
    sig_trend = signature.get("trend")
    if sig_trend:
        if isinstance(sig_trend, list):
            if trend not in sig_trend:
                score -= 25
                mismatches.append(f"trend={trend}, expected one of {sig_trend}")
        elif trend != sig_trend:
            score -= 25
            mismatches.append(f"trend={trend}, expected {sig_trend}")

    # Check overall direction vs signature trend
    direction_threshold = max(10, data_range // 4)
    if overall_change > direction_threshold:
        overall_direction = "RISING"
    elif overall_change < -direction_threshold:
        overall_direction = "FALLING"
    else:
        overall_direction = "STABLE"

    if sig_trend:
        allowed_trends = sig_trend if isinstance(sig_trend, list) else [sig_trend]
        if overall_direction in ["RISING", "FALLING"] and overall_direction not in allowed_trends:
            score -= 25
            mismatches.append(f"overall_direction={overall_direction}, expected one of {allowed_trends}")

    # Check rate of change
    rate_range = signature.get("rate_range")
    if rate_range:
        if not (rate_range[0] <= rate <= rate_range[1]):
            score -= 20
            mismatches.append(f"rate={rate}, expected {rate_range}")

    # Check drop magnitude
    min_drop = signature.get("min_drop")
    if min_drop:
        overall_change = metrics.get("overall_change", 0)
        if abs(overall_change) < min_drop:
            score -= 25
            mismatches.append(f"change={overall_change}, expected >={min_drop}")

    # Check oscillation requirement
    sig_osc = signature.get("oscillation")
    has_osc = oscillation_ratio > 30
    if sig_osc is not None:
        if sig_osc and not has_osc:
            score -= 25
            mismatches.append("expected oscillation, none found")
        elif not sig_osc and has_osc:
            score -= 10
            mismatches.append("unexpected oscillation detected")

    # Check variance expectations
    variance = sum((x - avg) ** 2 for x in series) // series_len if series_len else 0
    if signature.get("high_variance") and variance < 500:
        score -= 20
        mismatches.append(f"variance={variance}, expected high")
    if signature.get("low_variance") and variance > 100:
        score -= 15
        mismatches.append(f"variance={variance}, expected low")

    # Check exponential
    if signature.get("exponential") and not metrics.get("is_exponential", False):
        score -= 20
        mismatches.append("expected exponential growth, not detected")

    if signature.get("has_spike") and not has_spike:
        score -= 15
        mismatches.append("expected spike, none found")

    if signature.get("gust_front") and not has_gust_front:
        score -= 15
        mismatches.append("expected gust front drop, none found")

    if signature.get("mid_series_extreme") and not mid_series_extreme:
        score -= 15
        mismatches.append("expected mid-series extreme, none found")

    if signature.get("v_shaped") and not v_shaped:
        score -= 15
        mismatches.append("expected v-shaped signature, none found")

    if signature.get("minimal_oscillation") and not minimal_oscillation:
        score -= 10
        mismatches.append("expected minimal oscillation")

    if signature.get("sustained_low") and not sustained_low:
        score -= 10
        mismatches.append("expected sustained low values")

    if oscillation_ratio >= 60 and overall_change > 50:
        allowed_trends = sig_trend if isinstance(sig_trend, list) else [sig_trend] if sig_trend else []
        if "RISING" not in allowed_trends:
            score -= 40
            mismatches.append("strong oscillation with rising drift")

    if signature_name == "FIRE_WEATHER" and avg > 200:
        score -= 20
        mismatches.append(f"avg={avg} above fire-weather range")

    return max(0, score), mismatches


def classify_match_quality(score: float) -> MatchQuality:
    """Determine match quality from score."""
    if score >= 80:
        return MatchQuality.EXCELLENT
    elif score >= 60:
        return MatchQuality.GOOD
    elif score >= 40:
        return MatchQuality.WEAK
    else:
        return MatchQuality.UNMAPPED


def suggest_category_for_unmapped(
    series: List[int],
    trend: str,
    metrics: Dict[str, Any]
) -> Tuple[str, List[str]]:
    """
    Generate suggestions for what an unmapped pattern might be.

    Returns (suggested_category, notes)
    """
    notes = []

    rate = metrics.get("rate_of_change", 0)
    variance = sum((x - sum(series) // len(series)) ** 2 for x in series) // len(series)
    data_range = max(series) - min(series)
    overall_change = metrics.get("overall_change", 0)

    # Analyze characteristics
    if trend == "FALLING" and rate < -100:
        notes.append("Rapid decline detected - possible severe event precursor")
        if variance > 500:
            return "SEVERE_UNSTABLE", notes
        else:
            return "RAPID_DECLINE", notes

    if trend == "RISING" and rate > 200:
        notes.append("Rapid rise detected - possible flash event")
        return "RAPID_RISE", notes

    if variance > 1000:
        notes.append("Extremely high variance - highly unstable system")
        return "EXTREME_INSTABILITY", notes

    if metrics.get("oscillation_ratio", 0) > 60:
        if data_range > 30:
            notes.append("Strong oscillation with large amplitude")
            return "WAVE_EVENT", notes
        else:
            notes.append("Moderate oscillation pattern")
            return "CYCLIC_PATTERN", notes

    if abs(overall_change) < 5 and variance < 50:
        notes.append("Very stable pattern - might be instrument plateau")
        return "ULTRA_STABLE", notes

    notes.append("Pattern doesn't match known signatures - novel phenomenon possible")
    return "NOVEL_UNKNOWN", notes


def detect_anomaly_flags(
    series: List[int],
    metrics: Dict[str, Any]
) -> List[str]:
    """Detect specific anomalies in the data."""
    flags = []

    if len(series) < 10:
        flags.append("SHORT_SERIES")

    # Sudden jumps
    diffs = [abs(series[i + 1] - series[i]) for i in range(len(series) - 1)]
    max_jump = max(diffs) if diffs else 0
    avg_jump = sum(diffs) // len(diffs) if diffs else 0
    if max_jump > avg_jump * 5 and max_jump > 10:
        flags.append(f"SUDDEN_JUMP_{max_jump}")

    # Plateaus (identical values)
    plateaus = 0
    for i in range(len(series) - 1):
        if series[i] == series[i + 1]:
            plateaus += 1
    if plateaus > len(series) // 3:
        flags.append("POSSIBLE_SENSOR_PLATEAU")

    # Extreme values
    avg = sum(series) // len(series)
    extremes = sum(1 for x in series if abs(x - avg) > avg // 2)
    if extremes > len(series) // 4:
        flags.append("MULTIPLE_EXTREMES")

    return flags


def analyze_pattern_match(
    series: List[int],
    trend: str,
    metrics: Dict[str, Any],
    location: str = "UNKNOWN",
    hazard_type: str = "UNKNOWN"
) -> PatternMatchResult:
    """
    Comprehensive pattern matching analysis.

    Identifies if data matches known patterns or represents novel phenomena.
    """
    best_match = "UNKNOWN"
    best_score = 0.0
    best_mismatches = []

    # Score against all known patterns
    for sig_name, signature in KNOWN_SIGNATURES.items():
        score, mismatches = score_pattern_match(
            series, trend, metrics, sig_name, signature
        )
        if score > best_score:
            best_score = score
            best_match = sig_name
            best_mismatches = mismatches

    quality = classify_match_quality(best_score)
    anomaly_flags = detect_anomaly_flags(series, metrics)

    # Generate suggested actions
    actions = []
    if quality == MatchQuality.UNMAPPED:
        actions.append("Log pattern for later analysis")
        actions.append("Consider adding new attractor basin")
        actions.append("Review nearby sensor data for correlation")
    elif quality == MatchQuality.WEAK:
        actions.append("Pattern uncertain - monitor closely")
        actions.append("Cross-reference with other data sources")
    elif anomaly_flags:
        actions.append("Investigate flagged anomalies")

    # Create unmapped pattern record if needed
    unmapped = None
    if quality in [MatchQuality.UNMAPPED, MatchQuality.WEAK]:
        suggested_cat, notes = suggest_category_for_unmapped(series, trend, metrics)
        notes.extend([f"Closest known: {best_match} (score: {best_score:.1f})"])
        notes.extend([f"Mismatch: {m}" for m in best_mismatches])

        unmapped = UnmappedPattern(
            timestamp=datetime.now().isoformat(),
            data_fingerprint=compute_pattern_fingerprint(series),
            metrics={
                "trend": trend,
                "rate_of_change": metrics.get("rate_of_change", 0),
                "overall_change": metrics.get("overall_change", 0),
                "data_range": max(series) - min(series) if series else 0,
                "length": len(series),
            },
            closest_match=best_match,
            match_distance=100 - best_score,
            match_quality=quality,
            suggested_category=suggested_cat,
            notes=notes,
            location=location,
            hazard_type=hazard_type
        )

    return PatternMatchResult(
        quality=quality,
        matched_pattern=best_match,
        match_score=best_score,
        confidence=int(best_score),
        anomaly_flags=anomaly_flags,
        suggested_actions=actions,
        unmapped_pattern=unmapped
    )


def log_unmapped_pattern(
    pattern: UnmappedPattern,
    log_path: str = "/home/acid/Projects/MYSTIC/unmapped_patterns.jsonl"
) -> None:
    """
    Log unmapped pattern for later analysis and model improvement.

    Uses JSONL format for easy appending and parsing.
    """
    record = {
        "timestamp": pattern.timestamp,
        "fingerprint": pattern.data_fingerprint,
        "metrics": pattern.metrics,
        "closest_match": pattern.closest_match,
        "match_distance": pattern.match_distance,
        "quality": pattern.match_quality.value,
        "suggested_category": pattern.suggested_category,
        "notes": pattern.notes,
        "location": pattern.location,
        "hazard_type": pattern.hazard_type,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def get_evolution_suggestions(
    log_path: str = "/home/acid/Projects/MYSTIC/unmapped_patterns.jsonl"
) -> Dict[str, Any]:
    """
    Analyze logged unmapped patterns to suggest model improvements.

    Returns suggestions for new attractor basins or threshold adjustments.
    """
    if not os.path.exists(log_path):
        return {"status": "no_data", "suggestions": []}

    patterns = []
    with open(log_path, "r") as f:
        for line in f:
            if line.strip():
                patterns.append(json.loads(line))

    if not patterns:
        return {"status": "no_data", "suggestions": []}

    # Analyze patterns by suggested category
    category_counts = {}
    for p in patterns:
        cat = p.get("suggested_category", "UNKNOWN")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    suggestions = []

    # If a category appears frequently, suggest adding it as a new attractor
    for cat, count in category_counts.items():
        if count >= 3:
            suggestions.append({
                "type": "NEW_ATTRACTOR",
                "category": cat,
                "occurrences": count,
                "action": f"Consider adding '{cat}' as a new attractor basin"
            })

    # Check if specific known patterns have high miss rates
    miss_by_pattern = {}
    for p in patterns:
        closest = p.get("closest_match", "UNKNOWN")
        miss_by_pattern[closest] = miss_by_pattern.get(closest, 0) + 1

    for pattern, misses in miss_by_pattern.items():
        if misses >= 2:
            suggestions.append({
                "type": "THRESHOLD_ADJUSTMENT",
                "pattern": pattern,
                "misses": misses,
                "action": f"Review thresholds for '{pattern}' - {misses} near-misses logged"
            })

    return {
        "status": "analyzed",
        "pattern_count": len(patterns),
        "categories": category_counts,
        "suggestions": suggestions
    }


def run_unknown_pattern_tests():
    """Test unknown pattern detection."""
    print("=" * 70)
    print("UNKNOWN PATTERN DETECTOR TEST SUITE")
    print("=" * 70)

    # Test cases: known patterns and novel ones
    test_cases = [
        {
            "name": "Clear Hurricane Pattern",
            "data": [1010 - i * 2 for i in range(30)],
            "trend": "FALLING",
            "metrics": {"rate_of_change": -200, "overall_change": -58, "oscillation_ratio": 10},
            "expected_quality": MatchQuality.EXCELLENT,
        },
        {
            "name": "Novel: Oscillating Rise (no known signature)",
            "data": [500 + i * 5 + ((-1) ** i) * 30 for i in range(30)],
            "trend": "OSCILLATING",  # Rising with oscillation - doesn't match any
            "metrics": {"rate_of_change": 50, "overall_change": 150, "oscillation_ratio": 80, "is_exponential": False},
            "expected_quality": MatchQuality.UNMAPPED,
        },
        {
            "name": "Stable Pattern",
            "data": [1020 + (i % 2) for i in range(30)],
            "trend": "STABLE",
            "metrics": {"rate_of_change": 0, "overall_change": 0, "oscillation_ratio": 10},
            "expected_quality": MatchQuality.EXCELLENT,  # Will match STABLE_HIGH
        },
        {
            "name": "Novel: Moderate Fall No Oscillation",
            "data": [1000 - i for i in range(30)],
            "trend": "FALLING",
            "metrics": {"rate_of_change": -30, "overall_change": -29, "oscillation_ratio": 0},
            "expected_quality": MatchQuality.WEAK,  # Between cold front and hurricane
        },
    ]

    correct = 0
    for case in test_cases:
        print(f"\n{'─' * 70}")
        print(f"TEST: {case['name']}")

        result = analyze_pattern_match(
            case["data"],
            case["trend"],
            case["metrics"]
        )

        match = result.quality == case["expected_quality"]
        if match:
            correct += 1

        mark = "✓" if match else "○"
        print(f"  Quality: {result.quality.value} (expected: {case['expected_quality'].value}) {mark}")
        print(f"  Matched: {result.matched_pattern} (score: {result.match_score:.1f})")
        print(f"  Anomalies: {result.anomaly_flags}")

        if result.unmapped_pattern:
            print(f"  Suggested: {result.unmapped_pattern.suggested_category}")
            print(f"  Notes: {result.unmapped_pattern.notes[:2]}")

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {correct}/{len(test_cases)} correct")
    print(f"{'=' * 70}")

    return correct == len(test_cases)


if __name__ == "__main__":
    success = run_unknown_pattern_tests()

    # Also demonstrate evolution suggestions
    print("\n" + "=" * 70)
    print("EVOLUTION SUGGESTIONS")
    print("=" * 70)
    suggestions = get_evolution_suggestions()
    print(json.dumps(suggestions, indent=2))

    exit(0 if success else 1)
