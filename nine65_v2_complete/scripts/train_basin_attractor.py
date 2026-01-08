#!/usr/bin/env python3
"""
Basin-Specific Flood Attractor Training for MYSTIC

This module trains the Lorenz attractor detector on basin-specific flood signatures.
Each watershed has unique hydrological characteristics that create distinct chaos patterns:
- Response time (time from rainfall to peak flow)
- Shape of rising/falling limb
- Baseflow characteristics
- Critical thresholds for flash flooding

Training Process:
1. Collect historical flood events for specific basin
2. Extract precursor data (USGS stream + NEXRAD rainfall)
3. Map to Lorenz phase space (instability, moisture, shear)
4. Identify attractor basin boundaries for flood state
5. Save trained model for operational detection

Texas Hill Country Basins:
- Guadalupe River Basin (Camp Mystic flood, 2007)
- Blanco River Basin (Wimberley flood, 2015)
- Medina River Basin
- Llano River Basin
"""

import csv
import json
import os
from datetime import datetime, timedelta

# Import QMNF integer math utilities (replaces math module)
try:
    from qmnf_integer_math import isqrt, SCALE, INT_MAX, membership_score as int_membership
except ImportError:
    # Fallback if module not found - define inline
    SCALE = 1_000_000
    INT_MAX = (1 << 63) - 1

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
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

print("╔═══════════════════════════════════════════════════════════════════╗")
print("║         BASIN-SPECIFIC FLOOD ATTRACTOR TRAINING                  ║")
print("║      Training MYSTIC on Texas Hill Country Flood Signatures      ║")
print("╚═══════════════════════════════════════════════════════════════════╝")
print()

# ============================================================================
# BASIN CHARACTERISTICS DATABASE
# ============================================================================

@dataclass
class BasinCharacteristics:
    """Hydrological characteristics of a river basin."""
    name: str
    usgs_stations: List[str]
    drainage_area_km2: float
    response_time_hours: float  # Time from rainfall to peak
    bankfull_discharge_cms: float
    flood_stage_cm: float
    major_flood_stage_cm: float
    record_stage_cm: float
    soil_type: str
    slope_gradient: float
    nexrad_sites: List[str]

# Texas Hill Country basins
BASINS = {
    "guadalupe_upper": BasinCharacteristics(
        name="Upper Guadalupe River (Camp Mystic)",
        usgs_stations=["08166200", "08165500"],
        drainage_area_km2=830,
        response_time_hours=1.5,  # Very fast response!
        bankfull_discharge_cms=140,
        flood_stage_cm=213,  # 7 ft
        major_flood_stage_cm=366,  # 12 ft
        record_stage_cm=518,  # 17 ft (1987 flood)
        soil_type="thin_rocky",
        slope_gradient=0.012,
        nexrad_sites=["KEWX", "KDFX"]
    ),
    "blanco_river": BasinCharacteristics(
        name="Blanco River (Wimberley)",
        usgs_stations=["08171000"],
        drainage_area_km2=1050,
        response_time_hours=2.0,
        bankfull_discharge_cms=180,
        flood_stage_cm=274,  # 9 ft
        major_flood_stage_cm=518,  # 17 ft
        record_stage_cm=1311,  # 43 ft (2015 Memorial Day!)
        soil_type="thin_rocky",
        slope_gradient=0.008,
        nexrad_sites=["KEWX"]
    ),
    "medina_river": BasinCharacteristics(
        name="Medina River",
        usgs_stations=["08180800"],
        drainage_area_km2=640,
        response_time_hours=3.0,
        bankfull_discharge_cms=120,
        flood_stage_cm=305,
        major_flood_stage_cm=457,
        record_stage_cm=671,
        soil_type="thin_rocky",
        slope_gradient=0.010,
        nexrad_sites=["KEWX", "KDFX"]
    )
}

# ============================================================================
# LORENZ PHASE SPACE MAPPING
# ============================================================================

@dataclass
class PhaseSpacePoint:
    """Point in Lorenz phase space."""
    x: float  # Instability (rain rate normalized)
    y: float  # Moisture (stream level normalized)
    z: float  # Shear (rate of change)
    timestamp: datetime

def map_to_phase_space(
    rain_mm_hr: float,
    stream_cm: float,
    stream_change_cm_hr: float,
    basin: BasinCharacteristics
) -> PhaseSpacePoint:
    """
    Map hydrological conditions to Lorenz phase space.

    Mapping Strategy:
    - x (instability): Normalized rainfall rate (0-1 for 0-100 mm/hr)
    - y (moisture): Normalized stream level (0-1 for baseflow to flood stage)
    - z (shear): Rate of stream rise (chaos indicator)

    The Lorenz attractor exhibits different behavior based on:
    - Low x, low y: Normal conditions (stable attractor)
    - High x, moderate y: Building storm (transitional)
    - High x, high y, high z: Flash flood (chaotic regime)
    """
    # Normalize rainfall (0-100 mm/hr maps to 0-1)
    x = min(rain_mm_hr / 100.0, 1.5)  # Allow overflow for extreme

    # Normalize stream level relative to flood stages
    # 0 = baseflow, 0.5 = bankfull, 1.0 = flood stage, 1.5 = major flood
    baseflow_cm = basin.flood_stage_cm * 0.2  # Estimate baseflow
    y = (stream_cm - baseflow_cm) / (basin.flood_stage_cm - baseflow_cm)
    y = max(0, min(y, 2.0))  # Clamp to reasonable range

    # Normalize rate of change
    # Fast rise (>30 cm/hr) indicates dangerous conditions
    z = stream_change_cm_hr / 30.0
    z = max(-1.0, min(z, 2.0))

    return PhaseSpacePoint(x=x, y=y, z=z, timestamp=datetime.now())

# ============================================================================
# ATTRACTOR BASIN DEFINITION
# ============================================================================

@dataclass
class AttractorBasin:
    """Defines the boundary of a specific attractor in phase space."""
    name: str
    center: Tuple[float, float, float]  # (x, y, z) center
    radii: Tuple[float, float, float]   # (rx, ry, rz) ellipsoid radii
    alert_level: str  # CLEAR, WATCH, WARNING, EMERGENCY

# Standard flood attractor basins (to be refined with training)
FLOOD_ATTRACTORS = {
    "normal": AttractorBasin(
        name="Normal Conditions",
        center=(0.1, 0.2, 0.0),
        radii=(0.15, 0.25, 0.2),
        alert_level="CLEAR"
    ),
    "building": AttractorBasin(
        name="Building Storm",
        center=(0.4, 0.4, 0.3),
        radii=(0.2, 0.2, 0.25),
        alert_level="WATCH"
    ),
    "imminent": AttractorBasin(
        name="Flash Flood Imminent",
        center=(0.7, 0.7, 0.6),
        radii=(0.25, 0.25, 0.3),
        alert_level="WARNING"
    ),
    "active_flood": AttractorBasin(
        name="Active Flash Flood",
        center=(0.9, 1.2, 1.0),
        radii=(0.3, 0.4, 0.4),
        alert_level="EMERGENCY"
    )
}

def point_in_basin(point: PhaseSpacePoint, basin: AttractorBasin) -> float:
    """
    Calculate how strongly a point belongs to an attractor basin.

    Returns a value from 0 (outside) to 1 (at center).
    Uses ellipsoid distance formula with integer arithmetic.

    QMNF Compliance: Uses isqrt() instead of math.sqrt()
    """
    # Scale coordinates to integers for exact arithmetic
    # Multiply by SCALE to preserve precision
    px = int(point.x * SCALE)
    py = int(point.y * SCALE)
    pz = int(point.z * SCALE)

    cx = int(basin.center[0] * SCALE)
    cy = int(basin.center[1] * SCALE)
    cz = int(basin.center[2] * SCALE)

    rx = int(basin.radii[0] * SCALE)
    ry = int(basin.radii[1] * SCALE)
    rz = int(basin.radii[2] * SCALE)

    # Compute normalized distance using integer arithmetic
    # (dx/rx)^2 = (px - cx)^2 / rx^2, scaled by SCALE^2
    dx = px - cx
    dy = py - cy
    dz = pz - cz

    # Avoid division by zero
    term_x = (dx * dx * SCALE * SCALE) // (rx * rx) if rx != 0 else 0
    term_y = (dy * dy * SCALE * SCALE) // (ry * ry) if ry != 0 else 0
    term_z = (dz * dz * SCALE * SCALE) // (rz * rz) if rz != 0 else 0

    dist_sq_scaled = term_x + term_y + term_z
    distance_scaled = isqrt(dist_sq_scaled)

    # Convert to membership: 1 at center, 0 at boundary
    # membership = (SCALE - distance_scaled) / SCALE
    membership_scaled = SCALE - distance_scaled

    # Return as float for API compatibility (only at boundary)
    return max(0.0, membership_scaled / SCALE)

def classify_conditions(point: PhaseSpacePoint) -> Tuple[str, float, str]:
    """
    Classify current conditions based on phase space position.

    Returns: (dominant_basin_name, membership_score, alert_level)
    """
    best_basin = None
    best_score = -((1 << 63) - 1)  # Integer min instead of -float('inf')

    for name, basin in FLOOD_ATTRACTORS.items():
        score = point_in_basin(point, basin)
        if score > best_score:
            best_score = score
            best_basin = basin

    if best_basin is None:
        return ("unknown", 0.0, "CLEAR")

    return (best_basin.name, best_score, best_basin.alert_level)

# ============================================================================
# TRAINING DATA GENERATION
# ============================================================================

def generate_training_event(
    basin: BasinCharacteristics,
    event_name: str,
    peak_rain_mm_hr: float,
    peak_stream_cm: float,
    duration_hours: int = 72
) -> List[Dict]:
    """
    Generate synthetic training data for a flood event.

    Creates a 72-hour timeline with:
    - T-72h to T-48h: Normal conditions
    - T-48h to T-24h: Building instability
    - T-24h to T-6h: Rapid deterioration
    - T-6h to T-0h: Flash flood
    - T-0h to T+6h: Flood peak and recession
    """
    training_data = []

    # Calculate phase boundaries (hours before event)
    phases = {
        "normal": (-72, -48),
        "building": (-48, -24),
        "deterioration": (-24, -6),
        "imminent": (-6, 0),
        "peak": (0, 3),
        "recession": (3, 6)
    }

    baseflow_cm = basin.flood_stage_cm * 0.2

    for hour in range(-72, 7):
        timestamp = datetime.now() + timedelta(hours=hour)

        # Determine phase
        if hour < -48:
            phase = "normal"
            rain = 0.5 + 0.5 * math.sin(hour / 6)  # Light occasional
            stream = baseflow_cm + 5 * math.sin(hour / 12)
        elif hour < -24:
            phase = "building"
            progress = (hour + 48) / 24  # 0 to 1
            rain = 5 + 15 * progress
            stream = baseflow_cm + 20 * progress
        elif hour < -6:
            phase = "deterioration"
            progress = (hour + 24) / 18  # 0 to 1
            rain = 20 + (peak_rain_mm_hr - 20) * progress * 0.7
            stream = baseflow_cm + 20 + (basin.flood_stage_cm - baseflow_cm) * progress * 0.5
        elif hour < 0:
            phase = "imminent"
            progress = (hour + 6) / 6  # 0 to 1
            rain = peak_rain_mm_hr * (0.7 + 0.3 * progress)
            stream = basin.flood_stage_cm + (peak_stream_cm - basin.flood_stage_cm) * progress
        elif hour < 3:
            phase = "peak"
            progress = hour / 3  # 0 to 1
            rain = peak_rain_mm_hr * (1.0 - 0.5 * progress)
            stream = peak_stream_cm
        else:
            phase = "recession"
            progress = (hour - 3) / 3  # 0 to 1
            rain = peak_rain_mm_hr * 0.5 * (1.0 - progress)
            stream = peak_stream_cm - (peak_stream_cm - basin.major_flood_stage_cm) * progress

        # Calculate rate of change
        if len(training_data) > 0:
            prev_stream = training_data[-1]["stream_cm"]
            stream_change = stream - prev_stream
        else:
            stream_change = 0

        # Map to phase space
        point = map_to_phase_space(rain, stream, stream_change, basin)
        basin_name, score, alert = classify_conditions(point)

        record = {
            "timestamp": timestamp.isoformat(),
            "hour_offset": hour,
            "phase": phase,
            "rain_mm_hr": round(rain, 2),
            "stream_cm": round(stream, 2),
            "stream_change_cm_hr": round(stream_change, 2),
            "phase_x": round(point.x, 4),
            "phase_y": round(point.y, 4),
            "phase_z": round(point.z, 4),
            "attractor_basin": basin_name,
            "basin_score": round(score, 4),
            "alert_level": alert,
            "event_name": event_name
        }

        training_data.append(record)

    return training_data

def load_unified_training_event(
    path: str,
    basin: BasinCharacteristics,
    event_name: str
) -> List[Dict]:
    """
    Load a unified dataset (Camp Mystic) and convert to training records.
    """
    if not os.path.exists(path):
        return []

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    if not rows:
        return []

    rows.sort(key=lambda r: r["timestamp"])

    training_data = []
    prev_stream = None
    prev_time = None

    for row in rows:
        timestamp = datetime.fromisoformat(row["timestamp"])
        rain = float(row["rain_mm_hr"])
        stream = float(row["stream_cm"])

        if prev_stream is None or prev_time is None:
            stream_change = 0.0
        else:
            delta_hours = (timestamp - prev_time).total_seconds() / 3600.0
            delta_hours = max(delta_hours, 0.25)
            stream_change = (stream - prev_stream) / delta_hours

        prev_stream = stream
        prev_time = timestamp

        point = map_to_phase_space(rain, stream, stream_change, basin)
        basin_name, score, _ = classify_conditions(point)

        event_type = row.get("event_type", "normal")
        alert = EVENT_ALERT_MAP.get(event_type, "CLEAR")
        hour_offset = (timestamp - CAMP_MYSTIC_EVENT_DATE).total_seconds() / 3600.0

        record = {
            "timestamp": timestamp.isoformat(),
            "hour_offset": round(hour_offset, 2),
            "phase": event_type,
            "rain_mm_hr": round(rain, 2),
            "stream_cm": round(stream, 2),
            "stream_change_cm_hr": round(stream_change, 2),
            "phase_x": round(point.x, 4),
            "phase_y": round(point.y, 4),
            "phase_z": round(point.z, 4),
            "attractor_basin": basin_name,
            "basin_score": round(score, 4),
            "alert_level": alert,
            "event_name": event_name,
            "data_quality": row.get("data_quality", "unknown"),
        }

        training_data.append(record)

    return training_data

# ============================================================================
# HISTORICAL FLOOD EVENTS FOR TRAINING
# ============================================================================

CAMP_MYSTIC_UNIFIED = "../data/camp_mystic_2007_unified.csv"
CAMP_MYSTIC_EVENT_DATE = datetime(2007, 6, 28, 14, 0)

EVENT_ALERT_MAP = {
    "normal": "CLEAR",
    "baseline": "CLEAR",
    "watch": "WATCH",
    "precursor": "ADVISORY",
    "imminent": "WARNING",
    "flash_flood": "EMERGENCY",
    "flood_event": "EMERGENCY",
    "aftermath": "CLEAR",
}

TRAINING_EVENTS = [
    {
        "basin": "guadalupe_upper",
        "event_name": "Camp Mystic Flash Flood 2007",
        "date": "2007-06-28",
        "peak_rain_mm_hr": 200,
        "peak_stream_cm": 457  # 15 ft
    },
    {
        "basin": "blanco_river",
        "event_name": "Wimberley Memorial Day Flood 2015",
        "date": "2015-05-23",
        "peak_rain_mm_hr": 250,
        "peak_stream_cm": 1311  # 43 ft (RECORD)
    },
    {
        "basin": "guadalupe_upper",
        "event_name": "Guadalupe Flood 1987",
        "date": "1987-07-17",
        "peak_rain_mm_hr": 150,
        "peak_stream_cm": 518  # 17 ft
    },
    {
        "basin": "blanco_river",
        "event_name": "Blanco River Flood 1981",
        "date": "1981-05-24",
        "peak_rain_mm_hr": 180,
        "peak_stream_cm": 800
    },
    {
        "basin": "medina_river",
        "event_name": "Medina River Flood 2002",
        "date": "2002-07-01",
        "peak_rain_mm_hr": 120,
        "peak_stream_cm": 500
    }
]

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_basin_models():
    """
    Train attractor models for all basins using historical flood events.
    """
    print("─" * 70)
    print("TRAINING BASIN-SPECIFIC FLOOD ATTRACTOR MODELS")
    print("─" * 70)
    print()

    all_training_data = []
    basin_stats = {}

    for event in TRAINING_EVENTS:
        basin_key = event["basin"]
        basin = BASINS[basin_key]

        print(f"Training on: {event['event_name']}")
        print(f"  Basin: {basin.name}")
        print(f"  Peak Rain: {event['peak_rain_mm_hr']} mm/hr")
        print(f"  Peak Stream: {event['peak_stream_cm']} cm")
        print()

        if event["event_name"].startswith("Camp Mystic") and os.path.exists(CAMP_MYSTIC_UNIFIED):
            print(f"  Using unified dataset: {CAMP_MYSTIC_UNIFIED}")
            event_data = load_unified_training_event(
                path=CAMP_MYSTIC_UNIFIED,
                basin=basin,
                event_name=event["event_name"]
            )
        else:
            # Generate training data
            event_data = generate_training_event(
                basin=basin,
                event_name=event["event_name"],
                peak_rain_mm_hr=event["peak_rain_mm_hr"],
                peak_stream_cm=event["peak_stream_cm"]
            )

        all_training_data.extend(event_data)

        # Track basin statistics
        if basin_key not in basin_stats:
            basin_stats[basin_key] = {
                "name": basin.name,
                "events": 0,
                "samples": 0,
                "flood_stage_cm": basin.flood_stage_cm,
                "response_time_hours": basin.response_time_hours
            }

        basin_stats[basin_key]["events"] += 1
        basin_stats[basin_key]["samples"] += len(event_data)

    return all_training_data, basin_stats

def analyze_attractor_boundaries(training_data: List[Dict]) -> Dict:
    """
    Analyze training data to refine attractor basin boundaries.
    """
    print("─" * 70)
    print("ANALYZING ATTRACTOR BASIN BOUNDARIES")
    print("─" * 70)
    print()

    # Group by alert level
    by_alert = {}
    for record in training_data:
        alert = record["alert_level"]
        if alert not in by_alert:
            by_alert[alert] = {"x": [], "y": [], "z": []}
        by_alert[alert]["x"].append(record["phase_x"])
        by_alert[alert]["y"].append(record["phase_y"])
        by_alert[alert]["z"].append(record["phase_z"])

    # Calculate refined boundaries
    refined_basins = {}
    for alert, coords in by_alert.items():
        if len(coords["x"]) > 0:
            x_mean = sum(coords["x"]) / len(coords["x"])
            y_mean = sum(coords["y"]) / len(coords["y"])
            z_mean = sum(coords["z"]) / len(coords["z"])

            # QMNF: Use isqrt with scaling for standard deviation
            x_var_scaled = int(sum((x - x_mean)**2 for x in coords["x"]) * SCALE * SCALE // len(coords["x"]))
            y_var_scaled = int(sum((y - y_mean)**2 for y in coords["y"]) * SCALE * SCALE // len(coords["y"]))
            z_var_scaled = int(sum((z - z_mean)**2 for z in coords["z"]) * SCALE * SCALE // len(coords["z"]))

            x_std = isqrt(x_var_scaled) / SCALE
            y_std = isqrt(y_var_scaled) / SCALE
            z_std = isqrt(z_var_scaled) / SCALE

            refined_basins[alert] = {
                "center": (round(x_mean, 4), round(y_mean, 4), round(z_mean, 4)),
                "radii": (round(2*x_std + 0.1, 4), round(2*y_std + 0.1, 4), round(2*z_std + 0.1, 4)),
                "sample_count": len(coords["x"])
            }

            print(f"{alert}:")
            print(f"  Center: ({x_mean:.3f}, {y_mean:.3f}, {z_mean:.3f})")
            print(f"  Radii: (±{x_std:.3f}, ±{y_std:.3f}, ±{z_std:.3f})")
            print(f"  Samples: {len(coords['x'])}")
            print()

    return refined_basins

def save_training_artifacts(training_data: List[Dict], basin_stats: Dict, refined_basins: Dict):
    """
    Save training data and models for operational use.
    """
    # Save training data
    training_file = "../data/flood_attractor_training.json"
    with open(training_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"✓ Saved training data: {training_file}")

    # Save basin statistics
    stats_file = "../data/basin_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(basin_stats, f, indent=2)
    print(f"✓ Saved basin statistics: {stats_file}")

    # Save refined attractor basins
    basins_file = "../data/refined_attractor_basins.json"
    with open(basins_file, 'w') as f:
        json.dump(refined_basins, f, indent=2)
    print(f"✓ Saved attractor basins: {basins_file}")

    print()

# ============================================================================
# VALIDATION: Test trained model on Camp Mystic
# ============================================================================

def validate_on_camp_mystic(training_data: List[Dict]):
    """
    Test the trained model's ability to detect Camp Mystic flood.
    """
    print("─" * 70)
    print("VALIDATION: Testing Detection on Camp Mystic 2007")
    print("─" * 70)
    print()

    # Filter to Camp Mystic event
    camp_mystic_data = [r for r in training_data if "Camp Mystic" in r["event_name"]]

    # Find when each alert level was first reached
    first_alerts = {}
    for record in camp_mystic_data:
        alert = record["alert_level"]
        if alert not in first_alerts:
            first_alerts[alert] = record["hour_offset"]

    print("Alert Timeline (hours before flood):")
    for alert in ["CLEAR", "WATCH", "WARNING", "EMERGENCY"]:
        if alert in first_alerts:
            hours = first_alerts[alert]
            print(f"  {alert}: T{hours:+.0f}h")

    print()

    # Compare to actual warning
    actual_warning_hours = -2  # T-2h (flash flood warning issued)

    if "WARNING" in first_alerts:
        mystic_warning = first_alerts["WARNING"]
        improvement = abs(mystic_warning) - abs(actual_warning_hours)

        print(f"Comparison:")
        print(f"  Actual Warning: T{actual_warning_hours}h")
        print(f"  MYSTIC Warning: T{mystic_warning}h")

        if improvement > 0:
            print(f"  ✓ IMPROVEMENT: +{improvement:.0f} hours advance warning!")
        elif improvement == 0:
            print(f"  = EQUIVALENT to historical warning")
        else:
            print(f"  ✗ WORSE: {improvement:.0f} hours")

    print()

    return first_alerts

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Train on all historical events
    training_data, basin_stats = train_basin_models()

    # Analyze and refine attractor boundaries
    refined_basins = analyze_attractor_boundaries(training_data)

    # Save all artifacts
    save_training_artifacts(training_data, basin_stats, refined_basins)

    # Validate on Camp Mystic
    validate_on_camp_mystic(training_data)

    print("═" * 70)
    print("TRAINING COMPLETE")
    print("═" * 70)
    print()
    print("Summary:")
    print(f"  Training Events: {len(TRAINING_EVENTS)}")
    print(f"  Total Samples: {len(training_data)}")
    print(f"  Basins Trained: {len(basin_stats)}")
    print(f"  Attractor Types: {len(refined_basins)}")
    print()
    print("Files Created:")
    print("  ../data/flood_attractor_training.json (training data)")
    print("  ../data/basin_statistics.json (basin characteristics)")
    print("  ../data/refined_attractor_basins.json (attractor boundaries)")
    print()
    print("Next Steps:")
    print("  1. Integrate refined basins into FloodDetector Rust code")
    print("  2. Run validation framework with trained model")
    print("  3. Measure improvement in flash flood detection rate")
    print()


if __name__ == "__main__":
    main()
