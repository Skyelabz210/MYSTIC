#!/usr/bin/env python3
"""
MYSTIC Ensemble Uncertainty Quantification

Instead of single deterministic predictions, provide probability ranges:
  "72h outlook: 35% ± 15%" → "6h warning: 85% ± 5%"

This addresses a key gap: users need to know prediction CONFIDENCE,
not just the point estimate.

Methods:
1. Monte Carlo perturbation of input parameters
2. Bootstrap resampling of training data
3. Bayesian updating as event approaches
4. Lead-time dependent confidence intervals

Key Insight:
- Early predictions have high uncertainty (wide bounds)
- As event approaches, uncertainty decreases (narrow bounds)
- This matches physical reality - chaos limits predictability
"""

import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import statistics

# QMNF: Import integer math and ShadowEntropy (replaces random module)
try:
    from qmnf_integer_math import isqrt, SCALE
    from mystic_advanced_math import ShadowEntropy
except ImportError:
    # Fallback definitions
    SCALE = 1_000_000

    def isqrt(n):
        if n < 0:
            raise ValueError("Square root of negative number")
        if n < 2:
            return n
        x = n
        y = (x + 1) // 2
        while y < x:
            x = y
            y = (x + n // x) // 2
        return x

    class ShadowEntropy:
        """Fallback deterministic PRNG."""
        def __init__(self, modulus=2147483647, seed=42):
            self.modulus = modulus
            self.state = seed % modulus

        def next_int(self, max_value=2**32):
            r = (3 * self.modulus) // 4
            self.state = ((r * self.state) % self.modulus *
                          ((self.modulus - self.state) % self.modulus)) % self.modulus
            return self.state % max_value

        def next_gaussian(self, mean=0, stddev=1000, scale=1000):
            uniform_sum = sum(self.next_int(scale) for _ in range(12))
            z = uniform_sum - 6 * scale
            return mean + (z * stddev) // scale

        def reset(self, seed=None):
            if seed is not None:
                self.state = seed % self.modulus

# Global ShadowEntropy instance (deterministic, reproducible)
_shadow_entropy = ShadowEntropy(modulus=2147483647, seed=42)

print("╔═══════════════════════════════════════════════════════════════════╗")
print("║         MYSTIC ENSEMBLE UNCERTAINTY QUANTIFICATION               ║")
print("║      Probability Bounds and Confidence Intervals                  ║")
print("╚═══════════════════════════════════════════════════════════════════╝")
print()

# ============================================================================
# PARAMETER UNCERTAINTY DEFINITIONS
# ============================================================================

@dataclass
class ParameterUncertainty:
    """Define uncertainty for each input parameter."""
    name: str
    measurement_error: float  # Relative error (e.g., 0.10 = 10%)
    model_error: float        # Model representation error
    temporal_variability: float  # How much it can change in observation window

# Define uncertainties for each hazard type
FLASH_FLOOD_UNCERTAINTIES = {
    "rain_mm_hr": ParameterUncertainty("rain_mm_hr", 0.15, 0.10, 0.30),
    "soil_saturation": ParameterUncertainty("soil_saturation", 0.20, 0.15, 0.10),
    "stream_cm": ParameterUncertainty("stream_cm", 0.05, 0.05, 0.20),
    "stream_change": ParameterUncertainty("stream_change", 0.10, 0.10, 0.40),
}

TORNADO_UNCERTAINTIES = {
    "cape": ParameterUncertainty("cape", 0.20, 0.15, 0.25),
    "srh": ParameterUncertainty("srh", 0.25, 0.20, 0.30),
    "shear": ParameterUncertainty("shear", 0.15, 0.10, 0.20),
    "cin": ParameterUncertainty("cin", 0.30, 0.25, 0.40),
}

RI_UNCERTAINTIES = {
    "sst": ParameterUncertainty("sst", 0.02, 0.05, 0.05),  # SST well-observed
    "ohc": ParameterUncertainty("ohc", 0.20, 0.25, 0.10),
    "shear": ParameterUncertainty("shear", 0.20, 0.15, 0.30),
    "mld": ParameterUncertainty("mld", 0.25, 0.30, 0.15),
}

GIC_UNCERTAINTIES = {
    "kp": ParameterUncertainty("kp", 0.10, 0.10, 0.20),
    "dbdt": ParameterUncertainty("dbdt", 0.20, 0.15, 0.50),  # High temporal variability
    "bz": ParameterUncertainty("bz", 0.15, 0.10, 0.40),
    "density": ParameterUncertainty("density", 0.25, 0.20, 0.35),
}

# ============================================================================
# MONTE CARLO ENSEMBLE
# ============================================================================

def perturb_parameters(base_values: Dict[str, float],
                       uncertainties: Dict[str, ParameterUncertainty],
                       n_perturbations: int = 100) -> List[Dict[str, float]]:
    """
    Generate ensemble of perturbed parameter sets.

    Uses Gaussian perturbations based on combined uncertainty.
    """
    ensemble = []

    for _ in range(n_perturbations):
        perturbed = {}
        for param, value in base_values.items():
            if param in uncertainties:
                u = uncertainties[param]
                # Combined uncertainty (root sum of squares) - QMNF integer arithmetic
                # Scale errors by SCALE for integer math
                err_m = int(u.measurement_error * SCALE)
                err_o = int(u.model_error * SCALE)
                err_t = int(u.temporal_variability * SCALE)
                sum_sq = err_m * err_m + err_o * err_o + err_t * err_t
                total_uncertainty_scaled = isqrt(sum_sq)  # In units of SCALE

                # Perturb with Gaussian using ShadowEntropy (deterministic!)
                # mean = SCALE (1.0), stddev = total_uncertainty_scaled
                perturbation_scaled = _shadow_entropy.next_gaussian(
                    mean=SCALE,
                    stddev=total_uncertainty_scaled,
                    scale=SCALE
                )
                perturbation = max(0, perturbation_scaled) / SCALE
                perturbed[param] = value * perturbation
            else:
                perturbed[param] = value
        ensemble.append(perturbed)

    return ensemble

def calculate_ensemble_statistics(probabilities: List[float]) -> Dict[str, float]:
    """
    Calculate statistics from ensemble of probability estimates.
    """
    if not probabilities:
        return {"mean": 0, "std": 0, "p10": 0, "p50": 0, "p90": 0}

    sorted_probs = sorted(probabilities)
    n = len(sorted_probs)

    return {
        "mean": statistics.mean(probabilities),
        "std": statistics.stdev(probabilities) if n > 1 else 0,
        "min": min(probabilities),
        "max": max(probabilities),
        "p10": sorted_probs[int(n * 0.10)],
        "p25": sorted_probs[int(n * 0.25)],
        "p50": sorted_probs[int(n * 0.50)],  # Median
        "p75": sorted_probs[int(n * 0.75)],
        "p90": sorted_probs[int(n * 0.90)],
    }

# ============================================================================
# LEAD-TIME DEPENDENT UNCERTAINTY
# ============================================================================

def lead_time_uncertainty_factor(lead_time_hours: float, hazard_type: str) -> float:
    """
    Calculate uncertainty multiplier based on lead time.

    Longer lead times = more uncertainty (Lorenz butterfly effect).

    Returns multiplier for standard deviation (1.0 = no change).
    """
    # Different hazards have different predictability horizons
    predictability_hours = {
        "flash_flood": 6,      # Very short predictability
        "tornado": 3,          # Even shorter
        "hurricane_ri": 48,    # Longer - large-scale patterns
        "gic": 24,             # Medium - ACE gives 1-hour warning
    }

    base_hours = predictability_hours.get(hazard_type, 12)

    # Uncertainty grows with sqrt of lead time ratio
    # At predictability horizon, multiplier = 2.0
    # Beyond horizon, grows faster
    # QMNF: Use isqrt with scaling
    ratio_scaled = int((lead_time_hours * SCALE) // base_hours)
    sqrt_ratio_scaled = isqrt(ratio_scaled * SCALE)  # sqrt(ratio) * SCALE
    multiplier = 1.0 + sqrt_ratio_scaled / SCALE

    return min(multiplier, 5.0)  # Cap at 5x uncertainty

def format_probability_with_uncertainty(mean: float, std: float,
                                         confidence_level: float = 0.90) -> str:
    """
    Format probability with confidence interval.

    Example: "45% (35%-55% at 90% confidence)"
    """
    # Z-score for confidence level
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence_level, 1.645)

    lower = max(0, mean - z * std)
    upper = min(1.0, mean + z * std)

    return f"{mean:.0%} ({lower:.0%}-{upper:.0%} at {confidence_level:.0%} confidence)"

# ============================================================================
# BAYESIAN UPDATING
# ============================================================================

@dataclass
class BayesianTracker:
    """
    Track probability evolution as event approaches.

    Implements simple Bayesian updating:
    P(event | new_data) ∝ P(new_data | event) × P(event | old_data)
    """
    prior_probability: float = 0.1  # Climatological base rate
    observations: List[Tuple[float, float]] = field(default_factory=list)  # (time, likelihood_ratio)

    def update(self, hours_to_event: float, likelihood_ratio: float):
        """
        Update probability based on new observation.

        likelihood_ratio = P(observation | event) / P(observation | no_event)
        """
        self.observations.append((hours_to_event, likelihood_ratio))

    def current_probability(self) -> float:
        """Calculate current posterior probability."""
        odds = self.prior_probability / (1 - self.prior_probability)

        for _, lr in self.observations:
            odds *= lr

        prob = odds / (1 + odds)
        return min(0.99, max(0.01, prob))

    def uncertainty(self) -> float:
        """Estimate uncertainty based on observation consistency."""
        if len(self.observations) < 2:
            return 0.30  # High uncertainty with few observations

        lrs = [lr for _, lr in self.observations]
        # Variance in likelihood ratios indicates uncertainty
        lr_std = statistics.stdev(lrs) if len(lrs) > 1 else 1.0
        return min(0.40, 0.10 + 0.05 * lr_std)

# ============================================================================
# ENSEMBLE DETECTION FUNCTIONS
# ============================================================================

def detect_flash_flood_risk(params: Dict[str, float]) -> float:
    """Calculate flash flood risk from parameters."""
    rain = params.get("rain_mm_hr", 0)
    sat = params.get("soil_saturation", 0.5)
    rise = params.get("stream_change", 0)

    effective_rain = rain * (1 + sat * 0.5)
    risk = 0.0

    if effective_rain >= 100:
        risk += 0.35
    elif effective_rain >= 65:
        risk += 0.25
    elif effective_rain >= 40:
        risk += 0.15

    if sat >= 0.8:
        risk += 0.20
    elif sat >= 0.6:
        risk += 0.10

    if rise >= 30:
        risk += 0.25
    elif rise >= 20:
        risk += 0.15

    return min(risk, 1.0)

def detect_tornado_risk(params: Dict[str, float]) -> float:
    """Calculate tornado risk from parameters."""
    cape = params.get("cape", 0)
    srh = params.get("srh", 0)
    shear = params.get("shear", 0)
    cin = params.get("cin", 100)

    stp = (min(cape/1500, 3) * min(srh/150, 3) * min(shear/20, 2) * 0.8)
    if cin < 50:
        stp *= 1.2
    elif cin > 200:
        stp *= 0.5

    if stp >= 4:
        return 0.50
    elif stp >= 1.5:
        return 0.35
    elif stp >= 0.5:
        return 0.20
    elif cape >= 1500 and srh >= 150:
        return 0.15
    return 0.05

def detect_ri_risk(params: Dict[str, float]) -> float:
    """Calculate RI risk from parameters."""
    sst = params.get("sst", 26)
    ohc = params.get("ohc", 40)
    shear = params.get("shear", 20)
    mld = params.get("mld", 40)

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
    elif shear >= 25:
        risk -= 0.15

    if mld >= 50:
        risk += 0.10
    elif mld < 30:
        risk -= 0.10

    return max(0, min(risk, 1.0))

def detect_gic_risk(params: Dict[str, float]) -> float:
    """Calculate GIC risk from parameters."""
    kp = params.get("kp", 3)
    dbdt = params.get("dbdt", 20)
    bz = params.get("bz", 0)
    density = params.get("density", 5)

    risk = 0.0

    if kp >= 8:
        risk += 0.35
    elif kp >= 7:
        risk += 0.25
    elif kp >= 6:
        risk += 0.15
    elif kp >= 5:
        risk += 0.08

    if dbdt >= 300:
        risk += 0.30
    elif dbdt >= 100:
        risk += 0.15
    elif dbdt >= 50:
        risk += 0.05

    if bz <= -15:
        risk += 0.15
    elif bz <= -10:
        risk += 0.10

    if density >= 20:
        risk += 0.10

    return min(risk, 1.0)

# ============================================================================
# ENSEMBLE PREDICTION
# ============================================================================

def ensemble_prediction(hazard_type: str,
                        base_params: Dict[str, float],
                        lead_time_hours: float = 6,
                        n_ensemble: int = 200) -> Dict:
    """
    Generate ensemble prediction with uncertainty bounds.

    Returns:
    - mean probability
    - standard deviation
    - confidence intervals
    - lead-time adjusted bounds
    """
    # Select uncertainty definitions and detector
    if hazard_type == "flash_flood":
        uncertainties = FLASH_FLOOD_UNCERTAINTIES
        detector = detect_flash_flood_risk
    elif hazard_type == "tornado":
        uncertainties = TORNADO_UNCERTAINTIES
        detector = detect_tornado_risk
    elif hazard_type == "hurricane_ri":
        uncertainties = RI_UNCERTAINTIES
        detector = detect_ri_risk
    elif hazard_type == "gic":
        uncertainties = GIC_UNCERTAINTIES
        detector = detect_gic_risk
    else:
        raise ValueError(f"Unknown hazard type: {hazard_type}")

    # Generate ensemble
    ensemble = perturb_parameters(base_params, uncertainties, n_ensemble)

    # Calculate probabilities for each ensemble member
    probabilities = [detector(params) for params in ensemble]

    # Get statistics
    stats = calculate_ensemble_statistics(probabilities)

    # Adjust for lead time
    lt_factor = lead_time_uncertainty_factor(lead_time_hours, hazard_type)
    adjusted_std = stats["std"] * lt_factor

    # Format output
    result = {
        "hazard_type": hazard_type,
        "lead_time_hours": lead_time_hours,
        "point_estimate": stats["mean"],
        "uncertainty_raw": stats["std"],
        "uncertainty_adjusted": adjusted_std,
        "confidence_90": {
            "lower": max(0, stats["mean"] - 1.645 * adjusted_std),
            "upper": min(1, stats["mean"] + 1.645 * adjusted_std)
        },
        "percentiles": {
            "p10": stats["p10"],
            "p25": stats["p25"],
            "p50": stats["p50"],
            "p75": stats["p75"],
            "p90": stats["p90"]
        },
        "formatted": format_probability_with_uncertainty(stats["mean"], adjusted_std)
    }

    return result

# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_ensemble():
    """Demonstrate ensemble uncertainty quantification."""

    print("═" * 70)
    print("DEMONSTRATION: ENSEMBLE UNCERTAINTY")
    print("═" * 70)
    print()

    # Flash flood example
    print("1. FLASH FLOOD - Moderate conditions")
    print()

    ff_params = {
        "rain_mm_hr": 55,
        "soil_saturation": 0.7,
        "stream_cm": 150,
        "stream_change": 18
    }

    for lead_time in [48, 24, 12, 6, 2]:
        result = ensemble_prediction("flash_flood", ff_params, lead_time)
        ci = result["confidence_90"]
        print(f"  T-{lead_time:2}h: {result['formatted']}")

    print()

    # Tornado example
    print("2. TORNADO - Favorable environment")
    print()

    tor_params = {
        "cape": 3000,
        "srh": 300,
        "shear": 45,
        "cin": 60
    }

    for lead_time in [12, 6, 3, 1]:
        result = ensemble_prediction("tornado", tor_params, lead_time)
        print(f"  T-{lead_time:2}h: {result['formatted']}")

    print()

    # Hurricane RI example
    print("3. HURRICANE RI - Marginal SST scenario")
    print()

    ri_params = {
        "sst": 27.0,
        "ohc": 65,
        "shear": 12,
        "mld": 45
    }

    for lead_time in [72, 48, 24, 12]:
        result = ensemble_prediction("hurricane_ri", ri_params, lead_time)
        print(f"  T-{lead_time:2}h: {result['formatted']}")

    print()

    # GIC example
    print("4. SPACE WEATHER GIC - Moderate storm")
    print()

    gic_params = {
        "kp": 6,
        "dbdt": 120,
        "bz": -12,
        "density": 15
    }

    for lead_time in [48, 24, 6, 1]:
        result = ensemble_prediction("gic", gic_params, lead_time)
        print(f"  T-{lead_time:2}h: {result['formatted']}")

    print()

def demonstrate_bayesian():
    """Demonstrate Bayesian updating."""

    print("═" * 70)
    print("DEMONSTRATION: BAYESIAN UPDATING")
    print("═" * 70)
    print()

    print("Scenario: Tracking flash flood probability as conditions evolve")
    print()

    tracker = BayesianTracker(prior_probability=0.05)  # 5% base rate

    # Simulate observations over time
    observations = [
        (-48, 1.2, "Soil moisture elevated"),
        (-36, 1.5, "Rain beginning"),
        (-24, 2.0, "Heavy rain developing"),
        (-12, 3.0, "Stream rising"),
        (-6, 5.0, "Rapid rise detected"),
        (-3, 8.0, "Near flood stage"),
        (-1, 15.0, "Flood imminent"),
    ]

    print("Time    │ Observation              │ P(flood) │ Change")
    print("────────┼──────────────────────────┼──────────┼────────")

    prev_prob = tracker.current_probability()
    for hours, lr, description in observations:
        tracker.update(hours, lr)
        prob = tracker.current_probability()
        change = prob - prev_prob
        print(f"T{hours:+3}h   │ {description:24} │ {prob:7.1%}  │ {change:+.1%}")
        prev_prob = prob

    print()
    print(f"Final probability: {tracker.current_probability():.1%}")
    print(f"Uncertainty estimate: ±{tracker.uncertainty():.1%}")
    print()

# ============================================================================
# MAIN
# ============================================================================

def main():
    _shadow_entropy.reset(42)  # Reproducible (QMNF: ShadowEntropy replaces random)

    demonstrate_ensemble()
    demonstrate_bayesian()

    # Summary
    print("═" * 70)
    print("ENSEMBLE UNCERTAINTY SUMMARY")
    print("═" * 70)
    print()

    print("KEY FEATURES IMPLEMENTED:")
    print("  1. Monte Carlo perturbation (200 ensemble members)")
    print("  2. Parameter-specific uncertainty (measurement + model + temporal)")
    print("  3. Lead-time dependent uncertainty scaling")
    print("  4. Bayesian probability updating")
    print("  5. Confidence interval formatting")
    print()

    print("UNCERTAINTY SOURCES:")
    print("  • Measurement error: Sensor/observation limits")
    print("  • Model error: Physics representation gaps")
    print("  • Temporal variability: How fast conditions change")
    print()

    print("PREDICTABILITY HORIZONS:")
    print("  • Flash flood: ~6 hours (short)")
    print("  • Tornado: ~3 hours (very short)")
    print("  • Hurricane RI: ~48 hours (medium)")
    print("  • GIC: ~24 hours (medium)")
    print()

    # Save results
    output = {
        "generated": datetime.now().isoformat(),
        "ensemble_size": 200,
        "confidence_level": 0.90,
        "predictability_horizons": {
            "flash_flood": 6,
            "tornado": 3,
            "hurricane_ri": 48,
            "gic": 24
        },
        "uncertainty_parameters": {
            "flash_flood": list(FLASH_FLOOD_UNCERTAINTIES.keys()),
            "tornado": list(TORNADO_UNCERTAINTIES.keys()),
            "hurricane_ri": list(RI_UNCERTAINTIES.keys()),
            "gic": list(GIC_UNCERTAINTIES.keys())
        }
    }

    with open('../data/ensemble_uncertainty.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("✓ Results saved to: ../data/ensemble_uncertainty.json")
    print()

if __name__ == "__main__":
    main()
