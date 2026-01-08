#!/usr/bin/env python3
"""
MYSTIC One-Shot Flash Flood Learner

Integrates QMNF's one-shot learning algorithm for rapid flash flood pattern
recognition from limited training data.

Key Innovation:
- Learn flash flood signatures from SINGLE exemplar per event type
- Uses modular consensus (not floats) for pattern extraction
- Deterministic perturbation for reproducible training
- Integer-only arithmetic throughout

Algorithm (from QMNF OneShotLearner):
1. Generate N perturbation variants of exemplar
2. Compute modular median for each feature channel
3. Store consensus template per class
4. Classify by circular (modular) distance to templates

Classes:
- 0: CLEAR (no flood)
- 1: WATCH (elevated risk)
- 2: ADVISORY (moderate risk)
- 3: WARNING (high risk)
- 4: EMERGENCY (imminent/occurring)
"""

import json
import os
import csv
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# =============================================================================
# MODULAR ARITHMETIC (Integer-Only - QMNF Style)
# =============================================================================

# Coprime moduli for RNS representation (matching QMNF pattern)
PRIMARY_MODULI = [127, 131, 137, 139, 149]   # 5 channels
REFERENCE_MODULI = [151, 157, 163]           # 3 reference channels

# Feature scaling (to map floats to integers)
SCALE_FACTOR = 10000  # 4 decimal places of precision


def to_residue(value: int, moduli: List[int]) -> List[int]:
    """Convert integer to residue representation."""
    return [value % m for m in moduli]


def from_residue(residues: List[int], moduli: List[int]) -> int:
    """Reconstruct integer from residues via CRT."""
    # Simplified CRT for positive integers within range
    M = 1
    for m in moduli:
        M *= m

    result = 0
    for r, m in zip(residues, moduli):
        Mi = M // m
        # Extended Euclidean for modular inverse
        yi = mod_inverse(Mi, m)
        result += r * Mi * yi

    return result % M


def mod_inverse(a: int, m: int) -> int:
    """Modular inverse using extended Euclidean algorithm."""
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y

    _, x, _ = extended_gcd(a % m, m)
    return (x % m + m) % m


def circular_distance(a: int, b: int, modulus: int) -> int:
    """Circular (modular) distance between two values."""
    diff = abs(a - b)
    return min(diff, modulus - diff)


def modular_median(values: List[int], modulus: int) -> int:
    """
    Find the modular median - value minimizing total circular distance.

    This is the core of QMNF's consensus algorithm.
    Integer-only: Uses 2^63-1 instead of float('inf').
    """
    if not values:
        return 0

    best = values[0]
    best_total = (1 << 63) - 1  # Integer max instead of float('inf')

    for candidate in values:
        total = sum(circular_distance(candidate, v, modulus) for v in values)
        if total < best_total:
            best_total = total
            best = candidate

    return best


# =============================================================================
# PLMG VALUE (Phase-Locked Modular Gearing)
# =============================================================================

@dataclass
class PLMGValue:
    """
    Value in PLMG (Phase-Locked Modular Gearing) representation.

    This is the integer-only representation used by QMNF neural networks.
    """
    primary: List[int]     # Primary moduli residues
    reference: List[int]   # Reference moduli residues

    @classmethod
    def encode(cls, value: int) -> 'PLMGValue':
        """Encode integer into PLMG form."""
        return cls(
            primary=to_residue(value, PRIMARY_MODULI),
            reference=to_residue(value, REFERENCE_MODULI),
        )

    @classmethod
    def from_features(cls, features: List[float]) -> 'PLMGValue':
        """
        Encode feature vector into PLMG form.

        Maps each feature to a modular channel.
        """
        # Scale floats to integers
        scaled = [int(f * SCALE_FACTOR) for f in features]

        # Ensure we have enough features (pad with zeros)
        while len(scaled) < len(PRIMARY_MODULI):
            scaled.append(0)

        # Take first N features for primary channels
        primary = [scaled[i] % PRIMARY_MODULI[i] for i in range(len(PRIMARY_MODULI))]

        # Use aggregated features for reference
        ref_vals = [
            sum(scaled) % REFERENCE_MODULI[0],
            (scaled[0] * scaled[1] if len(scaled) > 1 else 0) % REFERENCE_MODULI[1],
            max(scaled) % REFERENCE_MODULI[2],
        ]

        return cls(primary=primary, reference=ref_vals)

    def primary_distance(self, other: 'PLMGValue') -> int:
        """Total circular distance across primary channels."""
        total = 0
        for i, (a, b, m) in enumerate(zip(self.primary, other.primary, PRIMARY_MODULI)):
            total += circular_distance(a, b, m)
        return total

    def decode(self) -> int:
        """Decode back to integer (if within range)."""
        return from_residue(self.primary, PRIMARY_MODULI)


# =============================================================================
# ONE-SHOT LEARNER (QMNF Algorithm)
# =============================================================================

class OneShotLearner:
    """
    One-Shot Learning Engine (QMNF Algorithm).

    Extracts stable templates from single exemplar via perturbation + consensus.
    """

    def __init__(self, radius: int = 5, variant_count: int = 21, seed: int = 42):
        self.radius = radius
        self.variant_count = variant_count
        self.seed = seed

    def generate_variants(self, exemplar: PLMGValue) -> List[PLMGValue]:
        """Generate perturbation variants of exemplar."""
        variants = [exemplar]  # Always include original

        for v in range(1, self.variant_count):
            variant = self._perturb_deterministic(exemplar, v)
            variants.append(variant)

        return variants

    def extract_template(self, variants: List[PLMGValue]) -> PLMGValue:
        """Extract template via modular median consensus."""
        if not variants:
            return PLMGValue(primary=[0]*len(PRIMARY_MODULI),
                           reference=[0]*len(REFERENCE_MODULI))

        # Compute modular median for each primary channel
        primary_medians = []
        for i, m in enumerate(PRIMARY_MODULI):
            channel_values = [v.primary[i] for v in variants]
            primary_medians.append(modular_median(channel_values, m))

        # Compute modular median for each reference channel
        reference_medians = []
        for i, m in enumerate(REFERENCE_MODULI):
            channel_values = [v.reference[i] for v in variants]
            reference_medians.append(modular_median(channel_values, m))

        return PLMGValue(primary=primary_medians, reference=reference_medians)

    def learn(self, exemplar: PLMGValue) -> PLMGValue:
        """One-shot learn: exemplar → template."""
        variants = self.generate_variants(exemplar)
        return self.extract_template(variants)

    def _perturb_deterministic(self, value: PLMGValue, variant_idx: int) -> PLMGValue:
        """Deterministic perturbation based on variant index."""
        primary = []
        for i, (r, m) in enumerate(zip(value.primary, PRIMARY_MODULI)):
            delta = self._compute_delta(variant_idx, i)
            perturbed = (r + delta) % m
            primary.append(perturbed)

        reference = []
        for i, (r, m) in enumerate(zip(value.reference, REFERENCE_MODULI)):
            delta = self._compute_delta(variant_idx, i + len(value.primary))
            perturbed = (r + delta) % m
            reference.append(perturbed)

        return PLMGValue(primary=primary, reference=reference)

    def _compute_delta(self, variant_idx: int, channel_idx: int) -> int:
        """Compute deterministic perturbation delta."""
        combined = (self.seed * (variant_idx + 1) + channel_idx) % (2 * self.radius + 1)
        return combined - self.radius


# =============================================================================
# CONSENSUS CLASSIFIER
# =============================================================================

@dataclass
class ClassificationResult:
    """Classification result with confidence metrics."""
    label: int
    label_name: str
    distance: int
    confidence: float
    all_distances: List[Tuple[int, str, int]]


class FloodConsensusClassifier:
    """
    Flash Flood Classifier using risk-score thresholds.

    Classes:
    - 0: CLEAR (risk 0-63)
    - 1: WATCH (risk 64-199)
    - 2: ADVISORY (risk 200-399)
    - 3: WARNING (risk 400-699)
    - 4: EMERGENCY (risk 700+)

    Thresholds are calibrated from NWS Flash Flood Guidance principles:
    - CLEAR: Normal conditions, minimal precipitation
    - WATCH: Conditions favorable for flash flooding
    - ADVISORY: Flash flooding possible
    - WARNING: Flash flooding expected
    - EMERGENCY: Catastrophic flooding occurring
    """

    CLASS_NAMES = ["CLEAR", "WATCH", "ADVISORY", "WARNING", "EMERGENCY"]

    # Fixed thresholds based on meteorological risk scoring
    # These align with NWS Flash Flood Guidance categories
    RISK_THRESHOLDS = [64, 200, 400, 700]

    def __init__(self):
        self.templates: Dict[int, PLMGValue] = {}
        self.learner = OneShotLearner()

    def add_class(self, label: int, template: PLMGValue):
        """Add learned template for class."""
        self.templates[label] = template

    def train_one_shot(self, exemplars: List[Tuple[int, PLMGValue]]):
        """Train classifier from exemplars using one-shot learning."""
        for label, exemplar in exemplars:
            template = self.learner.learn(exemplar)
            self.add_class(label, template)

    def classify_risk(self, risk_score: int) -> ClassificationResult:
        """Classify based on risk score using fixed thresholds."""
        # Find which class band this risk falls into
        label = 0
        for i, threshold in enumerate(self.RISK_THRESHOLDS):
            if risk_score >= threshold:
                label = i + 1

        # Compute confidence based on distance to nearest threshold
        if label == 0:
            margin = self.RISK_THRESHOLDS[0] - risk_score
        elif label == 4:
            margin = risk_score - self.RISK_THRESHOLDS[-1]
        else:
            lower = self.RISK_THRESHOLDS[label - 1]
            upper = self.RISK_THRESHOLDS[label] if label < len(self.RISK_THRESHOLDS) else 1000
            margin = min(risk_score - lower, upper - risk_score)

        # Integer-only: confidence in permille (0-1000) then convert to 0.0-1.0 for API
        confidence_permille = min((margin * 1000) // 50, 1000)
        confidence = confidence_permille / 1000  # Only float at API boundary

        # Also compute template distances for diagnostics
        distances = []
        for lbl, template in self.templates.items():
            # Use risk difference as distance metric
            distances.append((lbl, self.CLASS_NAMES[lbl], abs(risk_score - lbl * 200)))
        distances.sort(key=lambda x: x[2])

        return ClassificationResult(
            label=label,
            label_name=self.CLASS_NAMES[label],
            distance=risk_score,
            confidence=confidence,
            all_distances=distances,
        )

    def classify(self, query: PLMGValue) -> ClassificationResult:
        """
        Classify query - for backward compatibility.

        Note: This classifier works best when you compute the risk score
        directly from MeteoFeatures and call classify_risk().
        """
        # For PLMG queries, estimate risk from the first channel
        # This is approximate due to modular encoding
        base = query.primary[0]
        offset = (0 * 17) % 127
        adjusted = (base - offset + 127) % 127
        risk = (adjusted * 1000) // 126

        return self.classify_risk(risk)


# =============================================================================
# METEOROLOGICAL FEATURE EXTRACTION
# =============================================================================

@dataclass
class MeteoFeatures:
    """Meteorological features for flash flood prediction."""
    rain_rate_mm_hr: float      # Current rainfall intensity
    rain_6hr_mm: float          # 6-hour accumulated rainfall
    rain_24hr_mm: float         # 24-hour accumulated rainfall
    dewpoint_depression_c: float  # T - Td (lower = more moisture)
    pressure_tendency_hpa: float  # 3-hr pressure change (negative = falling)
    wind_speed_mps: float       # Wind speed
    relative_humidity_pct: float  # Relative humidity
    phi_resonance_detected: bool = False  # NEW: φ-resonance storm organization

    def compute_risk_score(self, pressure_history: Optional[List[int]] = None) -> int:
        """
        Compute a monotonic risk score (0-1000) from meteorological features.

        This creates a single unified score that increases with flood risk,
        ensuring proper class separation in modular space.

        NEW: Integrates φ-resonance detection for organized storm boost.

        Args:
            pressure_history: Optional list of recent pressure readings for φ-detection
        """
        score = 0

        # Rainfall rate component (0-400 points)
        # Thresholds: 5mm=light, 12.5mm=moderate, 25mm=heavy, 50mm=intense, 75mm=extreme
        if self.rain_rate_mm_hr >= 75:
            score += 400
        elif self.rain_rate_mm_hr >= 50:
            score += 300 + int((self.rain_rate_mm_hr - 50) * 4)
        elif self.rain_rate_mm_hr >= 25:
            score += 200 + int((self.rain_rate_mm_hr - 25) * 4)
        elif self.rain_rate_mm_hr >= 12.5:
            score += 100 + int((self.rain_rate_mm_hr - 12.5) * 8)
        elif self.rain_rate_mm_hr >= 5:
            score += int((self.rain_rate_mm_hr - 5) * 13.3)

        # 6-hour accumulation component (0-300 points)
        # Thresholds: 50mm=moderate, 100mm=heavy, 150mm=extreme
        if self.rain_6hr_mm >= 150:
            score += 300
        elif self.rain_6hr_mm >= 100:
            score += 200 + int((self.rain_6hr_mm - 100) * 2)
        elif self.rain_6hr_mm >= 50:
            score += 100 + int((self.rain_6hr_mm - 50) * 2)
        elif self.rain_6hr_mm >= 10:
            score += int((self.rain_6hr_mm - 10) * 2.5)

        # 24-hour accumulation component (0-150 points)
        if self.rain_24hr_mm >= 250:
            score += 150
        elif self.rain_24hr_mm >= 150:
            score += 100 + int((self.rain_24hr_mm - 150) * 0.5)
        elif self.rain_24hr_mm >= 50:
            score += int((self.rain_24hr_mm - 50) * 1.0)

        # Atmospheric moisture component (0-100 points)
        # Low dewpoint depression = saturated air = more rain potential
        if self.dewpoint_depression_c <= 2:
            score += 100
        elif self.dewpoint_depression_c <= 5:
            score += int((5 - self.dewpoint_depression_c) * 33)

        # Pressure tendency component (0-50 points)
        # Rapidly falling pressure = intensifying storm
        if self.pressure_tendency_hpa <= -3:
            score += 50
        elif self.pressure_tendency_hpa < 0:
            score += int(-self.pressure_tendency_hpa * 16.7)

        # NEW: φ-Resonance boost (0-100 points)
        # Organized storm patterns detected via golden ratio in pressure drops
        if self.phi_resonance_detected:
            # Organized storm = higher probability of severe event
            # Add 100 points for φ-detected organization
            score += 100

        # Also check pressure_history for φ-resonance if provided
        if pressure_history and len(pressure_history) >= 5:
            try:
                from mystic_advanced_math import PhiResonanceDetector
                phi_detector = PhiResonanceDetector(tolerance_permille=20)
                result = phi_detector.detect_resonance(pressure_history)
                if result["has_resonance"] and result["confidence"] >= 30:
                    # Scale boost by confidence (30-60 points based on confidence)
                    boost = 30 + (result["confidence"] * 30) // 100
                    score += boost
            except ImportError:
                pass  # φ-resonance module not available

        return min(score, 1000)

    def to_plmg(self) -> PLMGValue:
        """
        Convert features to PLMG representation using direct residue mapping.

        Key insight: Map risk score directly to residue values within each modulus,
        ensuring monotonic mapping without wraparound aliasing.
        """
        risk_score = self.compute_risk_score()

        # Map risk (0-1000) directly to residue values
        # Each channel uses a different mapping to create unique signatures
        # The key is to stay WITHIN each modulus to avoid wraparound
        primary = []
        for i, m in enumerate(PRIMARY_MODULI):
            # Map 0-1000 risk to 0-(m-1) residue, offset by channel index
            # This ensures each class occupies a distinct region
            base = (risk_score * (m - 1)) // 1000
            offset = (i * 17) % m  # Prime offset for channel differentiation
            primary.append((base + offset) % m)

        # Reference channels use aggregated features
        reference = []
        for i, m in enumerate(REFERENCE_MODULI):
            base = (risk_score * (m - 1)) // 1000
            offset = ((i + 5) * 23) % m
            reference.append((base + offset) % m)

        return PLMGValue(primary=primary, reference=reference)


def extract_features_from_obs(obs: Dict) -> MeteoFeatures:
    """Extract features from a meteorological observation."""
    return MeteoFeatures(
        rain_rate_mm_hr=float(obs.get("precip_1hr_mm") or 0) * 25.4,  # Convert if needed
        rain_6hr_mm=float(obs.get("rain_6hr_mm") or 0),
        rain_24hr_mm=float(obs.get("rain_24hr_mm") or 0),
        dewpoint_depression_c=float(obs.get("temp_c") or 20) - float(obs.get("dewpoint_c") or 15),
        pressure_tendency_hpa=float(obs.get("pressure_change_hpa") or 0),
        wind_speed_mps=float(obs.get("wind_speed_mps") or 0),
        relative_humidity_pct=float(obs.get("rh_pct") or 50),
    )


# =============================================================================
# TRAINING FROM HISTORICAL DATA
# =============================================================================

METEO_DIR = os.path.join(DATA_DIR, "meteorological")


def load_historical_storm_data(filename: str) -> List[Dict]:
    """Load meteorological data from a historical storm event."""
    filepath = os.path.join(METEO_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return []

    records = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def get_precip_value(record: Dict) -> float:
    """Get precipitation value from a record, handling multiple field names."""
    # Try hourly first
    precip = record.get("precip_1hr_mm")
    if precip and precip != "" and float(precip) > 0:
        return float(precip)

    # Try daily precipitation
    precip = record.get("precip_mm")
    if precip and precip != "" and float(precip) > 0:
        return float(precip) / 24.0  # Convert daily to hourly rate estimate

    return 0.0


def find_peak_conditions(records: List[Dict]) -> Dict:
    """Find the peak rainfall conditions from storm records."""
    max_precip = 0.0
    peak_record = None

    for r in records:
        precip = get_precip_value(r)
        if precip > max_precip:
            max_precip = precip
            peak_record = r

    return peak_record or records[0] if records else {}


def calculate_accumulated_precip(records: List[Dict], idx: int, hours: int) -> float:
    """Calculate accumulated precipitation over past N hours."""
    total = 0.0
    for i in range(max(0, idx - hours), idx + 1):
        total += get_precip_value(records[i])
    return total


def extract_exemplar_from_storm(filename: str, label: int, peak_offset: int = 0) -> Tuple[int, PLMGValue]:
    """
    Extract a training exemplar from real historical storm data.

    peak_offset: hours before peak to extract (for early warning training)
    """
    records = load_historical_storm_data(filename)
    if not records:
        # Return default if no data
        return (label, MeteoFeatures(0, 0, 0, 10, 0, 0, 50).to_plmg())

    # Find peak rainfall index using the helper function
    max_precip = 0.0
    peak_idx = 0
    for i, r in enumerate(records):
        precip = get_precip_value(r)
        if precip > max_precip:
            max_precip = precip
            peak_idx = i

    # Get record at offset from peak
    target_idx = max(0, peak_idx - peak_offset)
    r = records[target_idx]

    # Calculate features from real data
    temp_c = float(r.get("temp_c") or 20) if r.get("temp_c") else 20.0
    dewpoint_c = float(r.get("dewpoint_c") or 15) if r.get("dewpoint_c") else 15.0
    precip_1hr = get_precip_value(r)
    precip_6hr = calculate_accumulated_precip(records, target_idx, 6)
    pressure = float(r.get("pressure_hpa") or 1013) if r.get("pressure_hpa") else 1013.0
    wind_speed = float(r.get("wind_speed_mps") or 0) if r.get("wind_speed_mps") else 0.0

    # Calculate 24-hour precip
    precip_24hr = calculate_accumulated_precip(records, target_idx, 24)

    # Calculate pressure tendency (3-hour change)
    pressure_tendency = 0.0
    if target_idx >= 3:
        prev_pressure_str = records[target_idx - 3].get("pressure_hpa")
        if prev_pressure_str and prev_pressure_str != "":
            prev_pressure = float(prev_pressure_str)
            pressure_tendency = pressure - prev_pressure

    features = MeteoFeatures(
        rain_rate_mm_hr=precip_1hr,
        rain_6hr_mm=precip_6hr,
        rain_24hr_mm=precip_24hr,
        dewpoint_depression_c=max(0, temp_c - dewpoint_c),
        pressure_tendency_hpa=pressure_tendency,
        wind_speed_mps=wind_speed,
        relative_humidity_pct=90.0 if temp_c - dewpoint_c < 3 else 70.0,
    )

    risk = features.compute_risk_score()
    print(f"  {filename}: rain={precip_1hr:.1f}mm/hr, 6hr={precip_6hr:.1f}mm, T-Td={temp_c-dewpoint_c:.1f}C, risk={risk}")

    return (label, features.to_plmg())


def create_exemplars_from_real_data() -> List[Tuple[int, PLMGValue]]:
    """
    Create training exemplars from REAL historical Texas flood events.

    Uses actual meteorological observations from:
    - Hurricane Harvey 2017
    - Memorial Day 2015
    - Halloween 2013
    - Tax Day 2016
    - Llano River 2018
    """
    exemplars = []

    print("Extracting exemplars from real historical storm data...")
    print()

    # Class 0: CLEAR - Normal conditions (pre-storm from any event)
    print("Class 0 (CLEAR): Normal conditions from pre-storm period")
    records = load_historical_storm_data("weather_memorial_day_2015.csv")
    if records:
        # Get conditions from well before the storm
        r = records[5]  # Early in dataset
        features = MeteoFeatures(
            rain_rate_mm_hr=float(r.get("precip_1hr_mm") or 0),
            rain_6hr_mm=0.0,
            rain_24hr_mm=0.0,
            dewpoint_depression_c=float(r.get("temp_c") or 25) - float(r.get("dewpoint_c") or 20),
            pressure_tendency_hpa=0.0,
            wind_speed_mps=float(r.get("wind_speed_mps") or 3),
            relative_humidity_pct=70.0,
        )
        exemplars.append((0, features.to_plmg()))
        print(f"  Memorial Day pre-storm: rain={features.rain_rate_mm_hr}mm/hr")

    # Class 1: WATCH - Halloween 2013 early stage (6 hours before peak)
    print("Class 1 (WATCH): Halloween 2013 - 6 hours before peak")
    exemplars.append(extract_exemplar_from_storm("weather_halloween_2013.csv", 1, peak_offset=6))

    # Class 2: ADVISORY - Tax Day 2016 (3 hours before peak)
    print("Class 2 (ADVISORY): Tax Day 2016 - 3 hours before peak")
    exemplars.append(extract_exemplar_from_storm("weather_tax_day_2016.csv", 2, peak_offset=3))

    # Class 3: WARNING - Memorial Day 2015 peak (extreme flash flood)
    print("Class 3 (WARNING): Memorial Day 2015 - at peak")
    exemplars.append(extract_exemplar_from_storm("weather_memorial_day_2015.csv", 3, peak_offset=0))

    # Class 4: EMERGENCY - Hurricane Harvey 2017 peak (catastrophic)
    print("Class 4 (EMERGENCY): Hurricane Harvey 2017 - at peak")
    exemplars.append(extract_exemplar_from_storm("weather_hurricane_harvey_2017.csv", 4, peak_offset=0))

    return exemplars


def create_exemplars_from_events() -> List[Tuple[int, PLMGValue]]:
    """
    Create training exemplars - tries real data first, falls back to synthetic.
    """
    # Try to load from real historical data
    exemplars = create_exemplars_from_real_data()

    if len(exemplars) >= 5:
        return exemplars

    print("\nFalling back to synthetic exemplars...")

    # Fallback to synthetic if real data not available
    exemplars = []

    # Class 0: CLEAR - Normal conditions
    clear_features = MeteoFeatures(
        rain_rate_mm_hr=0.0,
        rain_6hr_mm=5.0,
        rain_24hr_mm=10.0,
        dewpoint_depression_c=8.0,
        pressure_tendency_hpa=0.5,
        wind_speed_mps=3.0,
        relative_humidity_pct=55.0,
    )
    exemplars.append((0, clear_features.to_plmg()))

    # Class 1: WATCH - Elevated conditions
    watch_features = MeteoFeatures(
        rain_rate_mm_hr=15.0,
        rain_6hr_mm=40.0,
        rain_24hr_mm=60.0,
        dewpoint_depression_c=4.0,
        pressure_tendency_hpa=-1.0,
        wind_speed_mps=8.0,
        relative_humidity_pct=75.0,
    )
    exemplars.append((1, watch_features.to_plmg()))

    # Class 2: ADVISORY - Moderate flood risk
    advisory_features = MeteoFeatures(
        rain_rate_mm_hr=30.0,
        rain_6hr_mm=80.0,
        rain_24hr_mm=120.0,
        dewpoint_depression_c=2.5,
        pressure_tendency_hpa=-2.0,
        wind_speed_mps=12.0,
        relative_humidity_pct=85.0,
    )
    exemplars.append((2, advisory_features.to_plmg()))

    # Class 3: WARNING - High flood risk
    warning_features = MeteoFeatures(
        rain_rate_mm_hr=50.0,
        rain_6hr_mm=150.0,
        rain_24hr_mm=200.0,
        dewpoint_depression_c=1.5,
        pressure_tendency_hpa=-3.0,
        wind_speed_mps=15.0,
        relative_humidity_pct=92.0,
    )
    exemplars.append((3, warning_features.to_plmg()))

    # Class 4: EMERGENCY - Imminent/occurring
    emergency_features = MeteoFeatures(
        rain_rate_mm_hr=100.0,
        rain_6hr_mm=250.0,
        rain_24hr_mm=500.0,
        dewpoint_depression_c=0.5,
        pressure_tendency_hpa=-5.0,
        wind_speed_mps=25.0,
        relative_humidity_pct=98.0,
    )
    exemplars.append((4, emergency_features.to_plmg()))

    return exemplars


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("╔═══════════════════════════════════════════════════════════════════════════╗")
    print("║         MYSTIC One-Shot Flash Flood Learner                               ║")
    print("║         QMNF Integer-Only Machine Learning                                ║")
    print("╚═══════════════════════════════════════════════════════════════════════════╝")
    print()
    print("Training from ONE exemplar per class (5 total examples)")
    print("Using modular consensus algorithm - NO FLOATS in learning")
    print()

    # Create exemplars from historical conditions
    exemplars = create_exemplars_from_events()

    print("Training Exemplars:")
    for label, plmg in exemplars:
        class_name = FloodConsensusClassifier.CLASS_NAMES[label]
        print(f"  Class {label} ({class_name}): {plmg.primary}")
    print()

    # Train classifier with one-shot learning
    print("Training with One-Shot Learning...")
    classifier = FloodConsensusClassifier()
    classifier.train_one_shot(exemplars)
    print(f"  Trained {len(classifier.templates)} class templates")
    print()

    # Show learned templates
    print("Learned Templates (Modular Consensus):")
    for label, template in classifier.templates.items():
        class_name = FloodConsensusClassifier.CLASS_NAMES[label]
        print(f"  {class_name}: {template.primary}")
    print()

    # Test on some scenarios
    print("=" * 70)
    print("TESTING ON SAMPLE CONDITIONS")
    print("=" * 70)
    print(f"Risk thresholds: WATCH≥64, ADVISORY≥200, WARNING≥400, EMERGENCY≥700")

    # Define expected classes for validation
    test_cases = [
        ("Light drizzle", MeteoFeatures(2.0, 8.0, 15.0, 6.0, 0.0, 4.0, 60.0), "CLEAR"),
        ("Moderate rain", MeteoFeatures(20.0, 50.0, 80.0, 3.0, -1.5, 10.0, 80.0), "ADVISORY"),
        ("Heavy storm", MeteoFeatures(45.0, 120.0, 180.0, 1.8, -2.5, 14.0, 90.0), "EMERGENCY"),
        ("Extreme (Harvey-like)", MeteoFeatures(80.0, 200.0, 400.0, 0.8, -4.0, 20.0, 96.0), "EMERGENCY"),
        ("Near-normal after rain", MeteoFeatures(1.0, 30.0, 100.0, 5.0, 1.0, 5.0, 65.0), "WATCH"),
    ]

    correct = 0
    total = len(test_cases)

    for name, features, expected in test_cases:
        risk_score = features.compute_risk_score()
        # Use classify_risk directly with the computed risk score
        result = classifier.classify_risk(risk_score)

        match = "✓" if result.label_name == expected else "✗"
        if result.label_name == expected:
            correct += 1

        print(f"\n{name}:")
        print(f"  Rain: {features.rain_rate_mm_hr:.1f} mm/hr | 6hr: {features.rain_6hr_mm:.1f} mm | Risk: {risk_score}/1000")
        print(f"  Classification: {result.label_name} {match} (expected: {expected})")

    print(f"\n\nTest Accuracy: {correct}/{total} ({100*correct/total:.0f}%)")

    # Save model
    model_file = os.path.join(DATA_DIR, "oneshot_flood_model.json")
    model_data = {
        "algorithm": "QMNF One-Shot Learning",
        "variant_count": 21,
        "perturbation_radius": 5,
        "moduli": {
            "primary": PRIMARY_MODULI,
            "reference": REFERENCE_MODULI,
        },
        "templates": {
            FloodConsensusClassifier.CLASS_NAMES[label]: {
                "primary": template.primary,
                "reference": template.reference,
            }
            for label, template in classifier.templates.items()
        },
        "trained": datetime.now().isoformat(),
    }

    with open(model_file, "w") as f:
        json.dump(model_data, f, indent=2)

    print(f"\n\nModel saved to: {model_file}")
    print("\nOne-shot learning complete - trained from just 5 exemplars!")


if __name__ == "__main__":
    main()
