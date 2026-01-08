#!/usr/bin/env python3
"""
MYSTIC Verification Metrics Framework

Implements comprehensive skill scoring for disaster prediction:
1. POD (Probability of Detection) - Did we catch real events?
2. FAR (False Alarm Rate) - Did we cry wolf?
3. CSI (Critical Success Index) - Overall accuracy
4. Lead Time Distribution - How early do we warn?
5. Confidence Calibration - Are probability estimates accurate?

This is critical for operational credibility - a system that warns
100 times for 10 events will be ignored by emergency managers.

Target metrics (based on NWS verification):
  POD > 0.85 (catch 85% of events)
  FAR < 0.30 (less than 30% false alarms)
  CSI > 0.50 (threat score)
"""

import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# QMNF: Import ShadowEntropy for deterministic random (replaces random module)
try:
    from mystic_advanced_math import ShadowEntropy
except ImportError:
    class ShadowEntropy:
        def __init__(self, modulus=2147483647, seed=42):
            self.modulus = modulus
            self.state = seed % modulus
        def next_int(self, max_value=2**32):
            r = (3 * self.modulus) // 4
            self.state = ((r * self.state) % self.modulus * ((self.modulus - self.state) % self.modulus)) % self.modulus
            return self.state % max_value
        def next_uniform(self, low=0.0, high=1.0, scale=10000):
            return low + (self.next_int(scale) * (high - low)) / scale

# Global ShadowEntropy instance
_rng = ShadowEntropy(modulus=2147483647, seed=42)

print("╔═══════════════════════════════════════════════════════════════════╗")
print("║         MYSTIC VERIFICATION METRICS FRAMEWORK                    ║")
print("║      POD, FAR, CSI, and Skill Score Analysis                     ║")
print("║      QMNF Compliant: Deterministic Random                        ║")
print("╚═══════════════════════════════════════════════════════════════════╝")
print()

# ============================================================================
# CONTINGENCY TABLE
# ============================================================================

@dataclass
class ContingencyTable:
    """
    2x2 contingency table for verification.

                    EVENT OCCURRED
                    YES         NO
    WARNING   YES   Hit (a)     False Alarm (b)
    ISSUED    NO    Miss (c)    Correct Null (d)

    POD = a / (a + c)  - Probability of Detection
    FAR = b / (a + b)  - False Alarm Ratio
    CSI = a / (a + b + c)  - Critical Success Index
    HSS = (a + d - expected) / (n - expected)  - Heidke Skill Score
    """
    hits: int = 0           # a: Warned AND event occurred
    false_alarms: int = 0   # b: Warned but no event
    misses: int = 0         # c: No warning but event occurred
    correct_nulls: int = 0  # d: No warning AND no event

    def total(self) -> int:
        return self.hits + self.false_alarms + self.misses + self.correct_nulls

    def pod(self) -> float:
        """Probability of Detection (Hit Rate)."""
        if self.hits + self.misses == 0:
            return 0.0
        return self.hits / (self.hits + self.misses)

    def far(self) -> float:
        """False Alarm Ratio."""
        if self.hits + self.false_alarms == 0:
            return 0.0
        return self.false_alarms / (self.hits + self.false_alarms)

    def pofd(self) -> float:
        """Probability of False Detection (False Alarm Rate)."""
        if self.false_alarms + self.correct_nulls == 0:
            return 0.0
        return self.false_alarms / (self.false_alarms + self.correct_nulls)

    def csi(self) -> float:
        """Critical Success Index (Threat Score)."""
        denom = self.hits + self.false_alarms + self.misses
        if denom == 0:
            return 0.0
        return self.hits / denom

    def bias(self) -> float:
        """Frequency Bias (>1 = over-warning, <1 = under-warning)."""
        if self.hits + self.misses == 0:
            return 0.0
        return (self.hits + self.false_alarms) / (self.hits + self.misses)

    def hss(self) -> float:
        """Heidke Skill Score (-1 to 1, 0 = no skill vs random)."""
        n = self.total()
        if n == 0:
            return 0.0

        a, b, c, d = self.hits, self.false_alarms, self.misses, self.correct_nulls
        expected_correct = ((a + c) * (a + b) + (b + d) * (c + d)) / n
        actual_correct = a + d

        if n - expected_correct == 0:
            return 0.0
        return (actual_correct - expected_correct) / (n - expected_correct)

    def ets(self) -> float:
        """Equitable Threat Score (Gilbert Skill Score)."""
        n = self.total()
        if n == 0:
            return 0.0

        a, b, c = self.hits, self.false_alarms, self.misses
        expected_hits = (a + b) * (a + c) / n
        denom = a + b + c - expected_hits

        if denom == 0:
            return 0.0
        return (a - expected_hits) / denom

    def summary(self) -> str:
        """Generate text summary."""
        return f"""
Contingency Table:
                EVENT
              YES    NO
WARNING  YES  {self.hits:4}  {self.false_alarms:4}  (Hit/FA)
         NO   {self.misses:4}  {self.correct_nulls:4}  (Miss/CN)

Metrics:
  POD (Prob of Detection): {self.pod():.1%}
  FAR (False Alarm Ratio): {self.far():.1%}
  CSI (Threat Score):      {self.csi():.1%}
  Bias:                    {self.bias():.2f}
  HSS (Heidke Skill):      {self.hss():.3f}
  ETS (Equitable Threat):  {self.ets():.3f}
"""

# ============================================================================
# LEAD TIME ANALYSIS
# ============================================================================

@dataclass
class LeadTimeDistribution:
    """Track distribution of warning lead times."""
    lead_times_hours: List[float] = field(default_factory=list)

    def add(self, lead_time_hours: float):
        self.lead_times_hours.append(lead_time_hours)

    def mean(self) -> float:
        if not self.lead_times_hours:
            return 0.0
        return sum(self.lead_times_hours) / len(self.lead_times_hours)

    def median(self) -> float:
        if not self.lead_times_hours:
            return 0.0
        sorted_times = sorted(self.lead_times_hours)
        n = len(sorted_times)
        if n % 2 == 0:
            return (sorted_times[n//2 - 1] + sorted_times[n//2]) / 2
        return sorted_times[n//2]

    def percentile(self, p: float) -> float:
        """Get p-th percentile (0-100)."""
        if not self.lead_times_hours:
            return 0.0
        sorted_times = sorted(self.lead_times_hours)
        idx = int(len(sorted_times) * p / 100)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    def summary(self) -> str:
        if not self.lead_times_hours:
            return "No lead times recorded"

        return f"""
Lead Time Distribution (n={len(self.lead_times_hours)}):
  Mean:   {self.mean():.1f} hours
  Median: {self.median():.1f} hours
  10th %: {self.percentile(10):.1f} hours
  25th %: {self.percentile(25):.1f} hours
  75th %: {self.percentile(75):.1f} hours
  90th %: {self.percentile(90):.1f} hours
"""

# ============================================================================
# RELIABILITY DIAGRAM (Calibration)
# ============================================================================

@dataclass
class ReliabilityBin:
    """Track forecast reliability in probability bins."""
    forecast_sum: float = 0.0
    observed_sum: float = 0.0  # 1 if event, 0 if not
    count: int = 0

    def add(self, forecast_prob: float, observed: bool):
        self.forecast_sum += forecast_prob
        self.observed_sum += 1.0 if observed else 0.0
        self.count += 1

    def mean_forecast(self) -> float:
        return self.forecast_sum / self.count if self.count > 0 else 0.0

    def observed_frequency(self) -> float:
        return self.observed_sum / self.count if self.count > 0 else 0.0

class ReliabilityDiagram:
    """Track probability calibration across bins."""

    def __init__(self, n_bins: int = 10):
        self.bins = [ReliabilityBin() for _ in range(n_bins)]
        self.n_bins = n_bins

    def add(self, forecast_prob: float, observed: bool):
        """Add a forecast-observation pair."""
        bin_idx = min(int(forecast_prob * self.n_bins), self.n_bins - 1)
        self.bins[bin_idx].add(forecast_prob, observed)

    def brier_score(self) -> float:
        """Brier Score (0 = perfect, 1 = worst)."""
        total_bs = 0.0
        total_n = 0
        for bin in self.bins:
            if bin.count > 0:
                # BS = mean((forecast - observed)^2)
                # For each bin, approximate
                obs_freq = bin.observed_frequency()
                mean_fc = bin.mean_forecast()
                total_bs += bin.count * (mean_fc - obs_freq) ** 2
                total_n += bin.count
        return total_bs / total_n if total_n > 0 else 0.0

    def summary(self) -> str:
        lines = ["Reliability Diagram:"]
        lines.append("  Bin      | Forecast | Observed | Count")
        lines.append("  ---------|----------|----------|------")
        for i, bin in enumerate(self.bins):
            low = i / self.n_bins
            high = (i + 1) / self.n_bins
            if bin.count > 0:
                lines.append(f"  {low:.1f}-{high:.1f}  | {bin.mean_forecast():6.1%}   | {bin.observed_frequency():6.1%}   | {bin.count:5}")
        lines.append(f"\n  Brier Score: {self.brier_score():.4f} (0=perfect, 1=worst)")
        return "\n".join(lines)

# ============================================================================
# SYNTHETIC EVENT GENERATION FOR TESTING
# ============================================================================

def generate_synthetic_events(n_events: int, event_type: str) -> List[Dict]:
    """
    Generate synthetic events with known ground truth for testing.

    Returns list of events with:
    - conditions (input parameters)
    - event_occurred (ground truth)
    - actual_lead_time (if event occurred)
    """
    events = []
    _rng.reset(42)  # Reset for reproducibility

    for i in range(n_events):
        if event_type == "flash_flood":
            # Generate realistic flash flood scenarios
            rain = _rng.next_uniform(20, 150)
            saturation = _rng.next_uniform(0.2, 0.95)
            stream_ratio = _rng.next_uniform(0.3, 1.2)
            rise_rate = _rng.next_uniform(5, 40)

            # Ground truth: event occurs based on physics
            event_prob = 0.0
            if rain >= 75 and saturation >= 0.7:
                event_prob = 0.9
            elif rain >= 50 and saturation >= 0.8:
                event_prob = 0.7
            elif rain >= 100:
                event_prob = 0.8
            elif saturation >= 0.9 and rain >= 30:
                event_prob = 0.6
            elif rise_rate >= 30:
                event_prob = 0.5

            event_occurred = _rng.next_uniform() < event_prob
            lead_time = _rng.next_uniform(1, 8) if event_occurred else 0

            events.append({
                "id": f"FF_{i:03d}",
                "type": "flash_flood",
                "rain_mm_hr": rain,
                "soil_saturation": saturation,
                "stream_ratio": stream_ratio,
                "rise_rate_cm_hr": rise_rate,
                "event_occurred": event_occurred,
                "actual_lead_time_hours": lead_time
            })

        elif event_type == "tornado":
            cape = _rng.next_uniform(500, 4500)
            srh = _rng.next_uniform(50, 500)
            shear = _rng.next_uniform(15, 60)
            cin = _rng.next_uniform(10, 300)

            # Ground truth
            stp = (cape/1500) * (srh/150) * (shear/20) * 0.8  # Simplified
            event_prob = min(0.95, stp * 0.15) if stp > 1 else stp * 0.05

            event_occurred = _rng.next_uniform() < event_prob
            lead_time = _rng.next_uniform(0.5, 4) if event_occurred else 0

            events.append({
                "id": f"TOR_{i:03d}",
                "type": "tornado",
                "cape": cape,
                "srh": srh,
                "shear": shear,
                "cin": cin,
                "event_occurred": event_occurred,
                "actual_lead_time_hours": lead_time
            })

        elif event_type == "hurricane_ri":
            sst = _rng.next_uniform(24, 31)
            ohc = _rng.next_uniform(20, 100)
            shear = _rng.next_uniform(5, 35)
            mld = _rng.next_uniform(20, 80)

            # Ground truth
            event_prob = 0.0
            if sst >= 27 and shear < 15 and ohc >= 50:
                event_prob = 0.7
            elif sst >= 28 and shear < 20:
                event_prob = 0.5
            elif sst >= 26.5 and shear < 10 and ohc >= 70:
                event_prob = 0.6

            if mld < 30:
                event_prob *= 0.5  # Shallow MLD inhibits

            event_occurred = _rng.next_uniform() < event_prob
            lead_time = _rng.next_uniform(6, 24) if event_occurred else 0

            events.append({
                "id": f"RI_{i:03d}",
                "type": "hurricane_ri",
                "sst": sst,
                "ohc": ohc,
                "shear": shear,
                "mld": mld,
                "event_occurred": event_occurred,
                "actual_lead_time_hours": lead_time
            })

        elif event_type == "gic":
            kp = _rng.next_uniform(2, 9)
            dbdt = _rng.next_uniform(10, 600)
            bz = _rng.next_uniform(-35, 10)
            density = _rng.next_uniform(2, 40)

            # Ground truth
            event_prob = 0.0
            if kp >= 7 and dbdt >= 200:
                event_prob = 0.9
            elif kp >= 6 and dbdt >= 100:
                event_prob = 0.6
            elif kp >= 5 and dbdt >= 150:
                event_prob = 0.5
            elif dbdt >= 300:  # Regional spike
                event_prob = 0.4

            event_occurred = _rng.next_uniform() < event_prob
            lead_time = _rng.next_uniform(2, 48) if event_occurred else 0

            events.append({
                "id": f"GIC_{i:03d}",
                "type": "gic",
                "kp": kp,
                "dbdt": dbdt,
                "bz": bz,
                "density": density,
                "event_occurred": event_occurred,
                "actual_lead_time_hours": lead_time
            })

    return events

# ============================================================================
# DETECTION FUNCTIONS (Simplified for testing)
# ============================================================================

def detect_flash_flood_v2(event: Dict, threshold: float = 0.40) -> Tuple[bool, float]:
    """Simplified v2 flash flood detection."""
    rain = event["rain_mm_hr"]
    sat = event["soil_saturation"]
    rise = event["rise_rate_cm_hr"]

    # v2 logic
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
    warning_issued = risk >= threshold

    return warning_issued, risk

def detect_tornado_v2(event: Dict, threshold: float = 0.25) -> Tuple[bool, float]:
    """Simplified v2 tornado detection."""
    cape = event["cape"]
    srh = event["srh"]
    shear = event["shear"]
    cin = event["cin"]

    # STP calculation
    stp = (min(cape/1500, 3) * min(srh/150, 3) * min(shear/20, 2) * 0.8)

    # CIN modifier
    if cin < 50:
        stp *= 1.2
    elif cin > 200:
        stp *= 0.5

    risk = 0.0
    if stp >= 4:
        risk = 0.50
    elif stp >= 1.5:
        risk = 0.35
    elif stp >= 0.5:
        risk = 0.20
    elif cape >= 1500 and srh >= 150:
        risk = 0.15

    warning_issued = risk >= threshold

    return warning_issued, risk

def detect_ri_v2(event: Dict, threshold: float = 0.30) -> Tuple[bool, float]:
    """Simplified v2 RI detection."""
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
    warning_issued = risk >= threshold

    return warning_issued, risk

def detect_gic_v2(event: Dict, threshold: float = 0.20) -> Tuple[bool, float]:
    """Simplified v2 GIC detection."""
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
    warning_issued = risk >= threshold

    return warning_issued, risk

# ============================================================================
# RUN VERIFICATION
# ============================================================================

def run_verification(event_type: str, n_events: int = 500):
    """Run verification for a given event type."""
    print(f"─" * 70)
    print(f"VERIFICATION: {event_type.upper()}")
    print(f"─" * 70)
    print()

    # Generate synthetic events
    events = generate_synthetic_events(n_events, event_type)

    # Select detector
    if event_type == "flash_flood":
        detector = detect_flash_flood_v2
    elif event_type == "tornado":
        detector = detect_tornado_v2
    elif event_type == "hurricane_ri":
        detector = detect_ri_v2
    elif event_type == "gic":
        detector = detect_gic_v2
    else:
        raise ValueError(f"Unknown event type: {event_type}")

    # Run detection and collect stats
    ct = ContingencyTable()
    lead_times = LeadTimeDistribution()
    reliability = ReliabilityDiagram()

    for event in events:
        warning_issued, risk = detector(event)
        event_occurred = event["event_occurred"]

        # Update contingency table
        if warning_issued and event_occurred:
            ct.hits += 1
            lead_times.add(event["actual_lead_time_hours"])
        elif warning_issued and not event_occurred:
            ct.false_alarms += 1
        elif not warning_issued and event_occurred:
            ct.misses += 1
        else:
            ct.correct_nulls += 1

        # Update reliability
        reliability.add(risk, event_occurred)

    # Print results
    print(ct.summary())
    print(lead_times.summary())
    print()
    print(reliability.summary())
    print()

    # Assessment
    print("ASSESSMENT:")
    pod = ct.pod()
    far = ct.far()
    csi = ct.csi()

    if pod >= 0.85:
        print(f"  ✓ POD {pod:.1%} meets target (≥85%)")
    else:
        print(f"  ⚠ POD {pod:.1%} below target (≥85%)")

    if far <= 0.30:
        print(f"  ✓ FAR {far:.1%} meets target (≤30%)")
    else:
        print(f"  ⚠ FAR {far:.1%} above target (≤30%)")

    if csi >= 0.50:
        print(f"  ✓ CSI {csi:.1%} meets target (≥50%)")
    else:
        print(f"  ⚠ CSI {csi:.1%} below target (≥50%)")

    print()

    return {
        "event_type": event_type,
        "n_events": n_events,
        "hits": ct.hits,
        "false_alarms": ct.false_alarms,
        "misses": ct.misses,
        "correct_nulls": ct.correct_nulls,
        "pod": round(ct.pod(), 4),
        "far": round(ct.far(), 4),
        "csi": round(ct.csi(), 4),
        "bias": round(ct.bias(), 4),
        "hss": round(ct.hss(), 4),
        "mean_lead_time": round(lead_times.mean(), 2),
        "brier_score": round(reliability.brier_score(), 4)
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    results = []

    for event_type in ["flash_flood", "tornado", "hurricane_ri", "gic"]:
        result = run_verification(event_type, n_events=500)
        results.append(result)

    # Summary table
    print("═" * 70)
    print("VERIFICATION SUMMARY")
    print("═" * 70)
    print()

    print("┌──────────────────┬────────┬────────┬────────┬────────┬────────┐")
    print("│ Event Type       │ POD    │ FAR    │ CSI    │ Bias   │ HSS    │")
    print("├──────────────────┼────────┼────────┼────────┼────────┼────────┤")

    for r in results:
        print(f"│ {r['event_type']:16} │ {r['pod']:5.1%}  │ {r['far']:5.1%}  │ {r['csi']:5.1%}  │ {r['bias']:5.2f}  │ {r['hss']:+5.3f} │")

    print("└──────────────────┴────────┴────────┴────────┴────────┴────────┘")
    print()

    print("Targets: POD ≥ 85%, FAR ≤ 30%, CSI ≥ 50%")
    print()

    # Overall assessment
    all_pod_met = all(r['pod'] >= 0.85 for r in results)
    all_far_met = all(r['far'] <= 0.30 for r in results)
    all_csi_met = all(r['csi'] >= 0.50 for r in results)

    if all_pod_met and all_far_met and all_csi_met:
        print("✓ ALL TARGETS MET - System verification successful")
    else:
        met_count = sum([all_pod_met, all_far_met, all_csi_met])
        print(f"⚠ {met_count}/3 targets met - System needs tuning")

    print()

    # Save results
    output = {
        "generated": datetime.now().isoformat(),
        "verification_results": results,
        "targets": {
            "pod": 0.85,
            "far": 0.30,
            "csi": 0.50
        }
    }

    with open('../data/verification_metrics.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("✓ Results saved to: ../data/verification_metrics.json")
    print()

if __name__ == "__main__":
    main()
