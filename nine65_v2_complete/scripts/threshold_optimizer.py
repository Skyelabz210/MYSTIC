#!/usr/bin/env python3
"""
MYSTIC Threshold Optimizer

The verification showed:
- Tornado: POD=91.9%, FAR=24.6% - GOOD, all targets met
- Flash Flood: POD=84.6%, FAR=32.1% - Close, slight over-warning
- Hurricane RI: POD=98.8%, FAR=57.4% - High POD but way too many false alarms
- GIC: POD=100%, FAR=51.9% - Same problem

The issue is BIAS > 1.5 for RI and GIC means we're over-warning.
Need to raise thresholds to reduce false alarms while keeping POD > 85%.

This optimizer:
1. Sweeps threshold values
2. Calculates POD, FAR, CSI at each
3. Finds optimal threshold balancing detection vs false alarms
"""

import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple

# QMNF: random module removed - using ShadowEntropy via verification_metrics

# Import verification components
from verification_metrics import (
    generate_synthetic_events,
    ContingencyTable
)

print("╔═══════════════════════════════════════════════════════════════════╗")
print("║         MYSTIC THRESHOLD OPTIMIZER                               ║")
print("║      Finding Optimal POD/FAR Balance                              ║")
print("╚═══════════════════════════════════════════════════════════════════╝")
print()

# ============================================================================
# PARAMETERIZED DETECTORS
# ============================================================================

def detect_flash_flood(event: Dict, threshold: float) -> Tuple[bool, float]:
    """Flash flood detection with variable threshold."""
    rain = event["rain_mm_hr"]
    sat = event["soil_saturation"]
    rise = event["rise_rate_cm_hr"]

    effective_rain = rain * (1 + sat * 0.5)
    risk = 0.0

    if effective_rain >= 100:
        risk += 0.35
    elif effective_rain >= 65:
        risk += 0.25
    elif effective_rain >= 40:
        risk += 0.15
    elif rain >= 25 and rise >= 15:
        risk += 0.20

    if sat >= 0.8:
        risk += 0.20
    elif sat >= 0.6:
        risk += 0.10

    if rise >= 30:
        risk += 0.25
    elif rise >= 20:
        risk += 0.15

    risk = min(risk, 1.0)
    return risk >= threshold, risk

def detect_ri(event: Dict, threshold: float) -> Tuple[bool, float]:
    """RI detection with variable threshold."""
    sst = event["sst"]
    ohc = event["ohc"]
    shear = event["shear"]
    mld = event["mld"]

    risk = 0.0

    if sst >= 28.5:
        risk += 0.25
    elif sst >= 27:
        risk += 0.15
    elif sst >= 26 and ohc >= 60:
        risk += 0.10

    if ohc >= 80:
        risk += 0.15
    elif ohc >= 50:
        risk += 0.05

    if shear < 10:
        risk += 0.25
    elif shear < 15:
        risk += 0.15
    elif shear < 20:
        risk += 0.05
    else:
        risk -= 0.20

    if mld >= 50:
        risk += 0.10
    elif mld < 30:
        risk -= 0.10

    risk = max(0, min(risk, 1.0))
    return risk >= threshold, risk

def detect_gic(event: Dict, threshold: float) -> Tuple[bool, float]:
    """GIC detection with variable threshold."""
    kp = event["kp"]
    dbdt = event["dbdt"]
    bz = event["bz"]
    density = event["density"]

    risk = 0.0

    if kp >= 9:
        risk += 0.40
    elif kp >= 8:
        risk += 0.35
    elif kp >= 7:
        risk += 0.25
    elif kp >= 6:
        risk += 0.15
    elif kp >= 5:
        risk += 0.08
    elif kp >= 4 and dbdt >= 50:
        risk += 0.05

    if dbdt >= 500:
        risk += 0.35
    elif dbdt >= 300:
        risk += 0.25
    elif dbdt >= 100:
        risk += 0.15
    elif dbdt >= 50:
        risk += 0.05

    if bz <= -20:
        risk += 0.15
    elif bz <= -10:
        risk += 0.10

    if density >= 20:
        risk += 0.10

    risk = min(risk, 1.0)
    return risk >= threshold, risk

# ============================================================================
# THRESHOLD SWEEP
# ============================================================================

def sweep_threshold(event_type: str, detector, n_events: int = 1000):
    """
    Sweep threshold values and find optimal.

    Returns: List of (threshold, pod, far, csi, bias) tuples
    """
    events = generate_synthetic_events(n_events, event_type)

    results = []
    for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        ct = ContingencyTable()

        for event in events:
            warning, risk = detector(event, threshold)
            occurred = event["event_occurred"]

            if warning and occurred:
                ct.hits += 1
            elif warning and not occurred:
                ct.false_alarms += 1
            elif not warning and occurred:
                ct.misses += 1
            else:
                ct.correct_nulls += 1

        results.append({
            "threshold": threshold,
            "pod": ct.pod(),
            "far": ct.far(),
            "csi": ct.csi(),
            "bias": ct.bias(),
            "hss": ct.hss()
        })

    return results

def find_optimal_threshold(results: List[Dict], min_pod: float = 0.85, max_far: float = 0.30):
    """
    Find threshold that maximizes CSI while meeting POD/FAR constraints.
    """
    valid = [r for r in results if r["pod"] >= min_pod and r["far"] <= max_far]

    if valid:
        # Among valid, pick highest CSI
        return max(valid, key=lambda x: x["csi"])
    else:
        # No threshold meets both constraints
        # Prioritize POD, then minimize FAR
        pod_ok = [r for r in results if r["pod"] >= min_pod]
        if pod_ok:
            return min(pod_ok, key=lambda x: x["far"])

        # Fall back to highest CSI overall
        return max(results, key=lambda x: x["csi"])

# ============================================================================
# OPTIMIZE ALL MODULES
# ============================================================================

def optimize_all():
    """Run optimization for all modules."""

    modules = [
        ("flash_flood", detect_flash_flood),
        ("hurricane_ri", detect_ri),
        ("gic", detect_gic)
    ]

    all_results = {}

    for event_type, detector in modules:
        print(f"─" * 70)
        print(f"OPTIMIZING: {event_type.upper()}")
        print(f"─" * 70)
        print()

        results = sweep_threshold(event_type, detector, n_events=1000)

        print("Threshold Sweep Results:")
        print("  Thresh │ POD    │ FAR    │ CSI    │ Bias  │ HSS")
        print("  ───────┼────────┼────────┼────────┼───────┼──────")

        for r in results:
            pod_mark = "✓" if r["pod"] >= 0.85 else " "
            far_mark = "✓" if r["far"] <= 0.30 else " "
            print(f"  {r['threshold']:.2f}   │ {r['pod']:5.1%}{pod_mark} │ {r['far']:5.1%}{far_mark} │ {r['csi']:5.1%}  │ {r['bias']:5.2f} │ {r['hss']:+.3f}")

        print()

        # Find optimal
        optimal = find_optimal_threshold(results)

        print(f"OPTIMAL THRESHOLD: {optimal['threshold']:.2f}")
        print(f"  POD: {optimal['pod']:.1%}")
        print(f"  FAR: {optimal['far']:.1%}")
        print(f"  CSI: {optimal['csi']:.1%}")
        print()

        # Check if it meets targets
        if optimal['pod'] >= 0.85 and optimal['far'] <= 0.30:
            print("  ✓ Meets all targets")
        elif optimal['pod'] >= 0.85:
            print(f"  ⚠ POD OK but FAR {optimal['far']:.1%} above target")
            print(f"    Consider: Accept higher FAR or add more parameters")
        else:
            print(f"  ⚠ Cannot meet both targets simultaneously")
            print(f"    Trade-off required between POD and FAR")

        print()

        all_results[event_type] = {
            "sweep": results,
            "optimal": optimal
        }

    return all_results

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

def generate_recommendations(results: Dict):
    """Generate tuning recommendations based on optimization."""
    print("═" * 70)
    print("TUNING RECOMMENDATIONS")
    print("═" * 70)
    print()

    print("┌──────────────────┬──────────────┬──────────────┬─────────────────┐")
    print("│ Module           │ Current Th   │ Optimal Th   │ Status          │")
    print("├──────────────────┼──────────────┼──────────────┼─────────────────┤")

    current_thresholds = {
        "flash_flood": 0.40,
        "hurricane_ri": 0.30,
        "gic": 0.20
    }

    for module, data in results.items():
        current = current_thresholds.get(module, 0.30)
        optimal = data["optimal"]["threshold"]
        status = "✓ OK" if data["optimal"]["pod"] >= 0.85 and data["optimal"]["far"] <= 0.30 else "⚠ Trade-off"

        print(f"│ {module:16} │ {current:10.2f}   │ {optimal:10.2f}   │ {status:15} │")

    print("└──────────────────┴──────────────┴──────────────┴─────────────────┘")
    print()

    print("SPECIFIC RECOMMENDATIONS:")
    print()

    # Flash flood
    ff = results.get("flash_flood", {}).get("optimal", {})
    print("Flash Flood:")
    print(f"  • Current threshold: 0.40")
    print(f"  • Recommended: {ff.get('threshold', 0.45):.2f}")
    print(f"  • Expected: POD {ff.get('pod', 0)*100:.0f}%, FAR {ff.get('far', 0)*100:.0f}%")
    print()

    # Hurricane RI - needs higher threshold
    ri = results.get("hurricane_ri", {}).get("optimal", {})
    print("Hurricane RI:")
    print(f"  • Current threshold: 0.30 (FAR was 57%!)")
    print(f"  • Recommended: {ri.get('threshold', 0.50):.2f}")
    print(f"  • This is a known challenge - RI is rare but we warn often")
    print(f"  • Consider: Require MULTIPLE favorable factors before warning")
    print()

    # GIC
    gic = results.get("gic", {}).get("optimal", {})
    print("Space Weather GIC:")
    print(f"  • Current threshold: 0.20 (FAR was 52%!)")
    print(f"  • Recommended: {gic.get('threshold', 0.40):.2f}")
    print(f"  • Issue: Many minor storms trigger warnings")
    print(f"  • Consider: Require Kp ≥ 5 AND dB/dt ≥ 100 for WARNING level")
    print()

    print("ALGORITHMIC IMPROVEMENTS (beyond threshold tuning):")
    print()
    print("  1. Multi-factor requirements:")
    print("     Instead of: 'If ANY factor high, warn'")
    print("     Use:        'If 2+ factors high, warn'")
    print()
    print("  2. Confidence-weighted alerting:")
    print("     LOW confidence (0.30-0.50): ADVISORY only")
    print("     MEDIUM confidence (0.50-0.70): WATCH")
    print("     HIGH confidence (0.70+): WARNING")
    print()
    print("  3. Regional calibration:")
    print("     Train separate thresholds per region")
    print("     (e.g., Texas floods different from Maryland)")
    print()

# ============================================================================
# MAIN
# ============================================================================

def main():
    results = optimize_all()
    generate_recommendations(results)

    # Save results
    output = {
        "generated": datetime.now().isoformat(),
        "optimization_results": {}
    }

    for module, data in results.items():
        output["optimization_results"][module] = {
            "sweep": data["sweep"],
            "optimal_threshold": data["optimal"]["threshold"],
            "optimal_pod": round(data["optimal"]["pod"], 4),
            "optimal_far": round(data["optimal"]["far"], 4),
            "optimal_csi": round(data["optimal"]["csi"], 4)
        }

    with open('../data/threshold_optimization.json', 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print("✓ Results saved to: ../data/threshold_optimization.json")
    print()

if __name__ == "__main__":
    main()
