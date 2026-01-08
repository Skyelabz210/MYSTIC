#!/usr/bin/env python3
"""
MYSTIC Unified Multi-Scale Pipeline

Integrates ALL data sources (terrestrial → cosmic) into a single MYSTIC training dataset.
Maps disparate phenomena to unified chaos signature space for pattern detection.

Architecture:
  - Each scale maps to Lorenz phase space dimensions
  - Cross-scale correlations detected via chaos signatures
  - Unified CSV format for MYSTIC ingestion

Example correlations we can now detect:
  - Seismic activity + lunar tidal forces
  - Space weather + atmospheric pressure anomalies
  - Ocean temperature + planetary alignments
  - Geomagnetic storms + weather pattern shifts
"""

import json
import csv
from datetime import datetime, timedelta
import urllib.request
import math

print("╔═══════════════════════════════════════════════════════════════════╗")
print("║         MYSTIC UNIFIED MULTI-SCALE PIPELINE                        ║")
print("║      Integrating Terrestrial, Atmospheric, Space & Cosmic Data     ║")
print("╚═══════════════════════════════════════════════════════════════════╝")
print()

# Load the summary from omniscient fetch
try:
    with open('../data/omniscient_data_summary.json', 'r') as f:
        summary = json.load(f)
    print(f"✓ Loaded data summary from: {summary['timestamp']}")
except:
    print("⚠ Run fetch_all_data_sources.py first")
    summary = {}

print()

# ============================================================================
# UNIFIED MYSTIC FORMAT
# ============================================================================

def nested_get(data, keys, default=None):
    cursor = data
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def to_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def create_unified_csv():
    """
    Create unified CSV with ALL scales integrated.

    MYSTIC format extended:
      timestamp, station_id,
      # Weather (Scale 1-2)
      temp_c, dewpoint_c, pressure_hpa, wind_mps, rain_mm_hr,
      # Terrestrial (Scale 1)
      soil_pct, stream_cm, seismic_magnitude, seismic_distance_km,
      # Oceanic (Scale 3)
      ocean_temp_c, wave_height_m, tide_level_cm,
      # Space Weather (Scale 4)
      solar_xray_flux, geomagnetic_kp, solar_wind_speed,
      # Planetary (Scale 5)
      lunar_phase, tidal_force_index,
      # Cosmic (Scale 6)
      cosmic_ray_flux,
      # Classification
      event_type
    """

    print("─" * 70)
    print("CREATING UNIFIED MULTI-SCALE DATASET")
    print("─" * 70)
    print()

    output_file = "../data/unified_multiscale_training.csv"

    forecast_period = nested_get(summary, ["scale_1_terrestrial", "nws_forecast", "period"], {})
    temp_base = to_float(forecast_period.get("temp_c"), 25.0)
    dewpoint_base = to_float(forecast_period.get("dewpoint_c"), temp_base - 5.0)
    wind_base = to_float(forecast_period.get("wind_mps"), 5.0)
    pressure_base = to_float(forecast_period.get("pressure_hpa"), 1013.0)

    seismic_mag_base = to_float(
        nested_get(summary, ["scale_1_terrestrial", "earthquakes", "nearest", "mag"]), 2.5
    )
    seismic_dist_base = to_float(
        nested_get(summary, ["scale_1_terrestrial", "earthquakes", "nearest", "distance_km"]), 150.0
    )

    ocean_temp_base = to_float(
        nested_get(summary, ["scale_3_oceanic", "buoy", "data", "water_temp_c"]), 23.8
    )
    wave_height_base = to_float(
        nested_get(summary, ["scale_3_oceanic", "buoy", "data", "wave_height_m"]), 1.0
    )
    tide_level_base = to_float(
        nested_get(summary, ["scale_3_oceanic", "tides", "water_level_cm"]), 100.0
    )

    geomagnetic_kp_base = to_float(
        nested_get(summary, ["scale_4_space_weather", "geomagnetic_kp", "kp"]), 4.0
    )
    solar_xray_base = to_float(
        nested_get(summary, ["scale_4_space_weather", "solar", "xray_flux"]), 1e-6
    )
    solar_wind_speed_km_s = to_float(
        nested_get(summary, ["scale_4_space_weather", "solar", "solar_wind", "speed_km_s"]), 400.0
    )
    solar_wind_base = solar_wind_speed_km_s * 1000.0

    lunar_phase_base = to_float(
        nested_get(summary, ["scale_5_planetary", "moon_phase", "phase"]), 0.087
    )
    tidal_force_base = to_float(
        nested_get(summary, ["scale_5_planetary", "moon_phase", "tidal_force_index"]), None
    )
    if tidal_force_base is None:
        tidal_force_base = abs(math.sin(lunar_phase_base * 2 * math.pi))

    cosmic_ray_base = to_float(
        nested_get(summary, ["scale_6_cosmic", "proton_flux", "flux"]), 6500.0
    )

    fieldnames = [
        # Core
        'timestamp', 'station_id',
        # Weather/Atmospheric
        'temp_c', 'dewpoint_c', 'pressure_hpa', 'wind_mps', 'rain_mm_hr',
        # Terrestrial
        'soil_pct', 'stream_cm', 'seismic_mag', 'seismic_dist_km',
        # Oceanic
        'ocean_temp_c', 'wave_height_m', 'tide_level_cm',
        # Space Weather
        'solar_xray', 'geomagnetic_kp', 'solar_wind_mps',
        # Planetary
        'lunar_phase', 'tidal_force',
        # Cosmic
        'cosmic_ray_flux',
        # Classification
        'event_type'
    ]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Generate synthetic multi-scale event (demonstration)
        # In production, this would merge real-time feeds

        now = datetime.now()

        # Simulate 24 hours of unified data
        for hour in range(24):
            timestamp = now - timedelta(hours=24-hour)

            # SCALE 1-2: Weather (baseline + diurnal cycle)
            hour_of_day = timestamp.hour
            temp = temp_base + 5.0 * math.sin((hour_of_day - 6) * math.pi / 12)
            pressure = pressure_base - 3.0 * math.sin(hour_of_day * math.pi / 12)

            # SCALE 1: Seismic (recent USGS data)
            # Closest earthquake to timestamp (simplified)
            seismic_mag = seismic_mag_base
            seismic_dist = seismic_dist_base

            # SCALE 3: Ocean (NOAA buoy data)
            ocean_temp = ocean_temp_base
            wave_height = wave_height_base + 0.5 * math.sin(hour_of_day * math.pi / 6)
            tide_level = tide_level_base + 50.0 * math.sin((hour_of_day - 3) * math.pi / 6)

            # SCALE 4: Space Weather
            # Real Kp index: 4.0 (from fetch)
            geomagnetic_kp = geomagnetic_kp_base
            solar_xray = solar_xray_base
            solar_wind = solar_wind_base

            # SCALE 5: Planetary
            # Real lunar phase: 0.087 (from fetch)
            lunar_phase = lunar_phase_base
            # Tidal force proxy (simplified - moon + sun)
            tidal_force = tidal_force_base

            # SCALE 6: Cosmic
            cosmic_ray = cosmic_ray_base

            writer.writerow({
                'timestamp': timestamp.isoformat(),
                'station_id': 1,
                'temp_c': temp,
                'dewpoint_c': dewpoint_base,
                'pressure_hpa': pressure,
                'wind_mps': wind_base,
                'rain_mm_hr': 0.0,
                'soil_pct': 40.0,
                'stream_cm': 100.0,
                'seismic_mag': seismic_mag,
                'seismic_dist_km': seismic_dist,
                'ocean_temp_c': ocean_temp,
                'wave_height_m': wave_height,
                'tide_level_cm': tide_level,
                'solar_xray': solar_xray,
                'geomagnetic_kp': geomagnetic_kp,
                'solar_wind_mps': solar_wind,
                'lunar_phase': lunar_phase,
                'tidal_force': tidal_force,
                'cosmic_ray_flux': cosmic_ray,
                'event_type': 'normal'
            })

    print(f"✓ Created unified dataset: {output_file}")
    print(f"  Records: 24 (hourly for demonstration)")
    print(f"  Fields: {len(fieldnames)}")
    print()


# ============================================================================
# MULTI-SCALE LORENZ MAPPING
# ============================================================================

def explain_multiscale_mapping():
    """
    Explain how multi-scale data maps to Lorenz phase space.
    """
    print("─" * 70)
    print("MULTI-SCALE LORENZ PHASE SPACE MAPPING")
    print("─" * 70)
    print()

    print("Traditional MYSTIC (weather only):")
    print("  x = Atmospheric instability (CAPE proxy)")
    print("  y = Moisture flux convergence")
    print("  z = Vertical wind shear")
    print()

    print("Extended MYSTIC (omniscient):")
    print()
    print("PRIMARY MAPPING (weather-dominant events):")
    print("  x = Atmospheric instability + seismic stress indicator")
    print("  y = Moisture flux + ocean heat content")
    print("  z = Wind shear + geomagnetic perturbation")
    print()

    print("SECONDARY DIMENSIONS (cross-scale correlations):")
    print("  Chaos signature modulated by:")
    print("    - Lunar tidal force (affects baseline chaos level)")
    print("    - Solar activity (space weather → atmospheric coupling)")
    print("    - Seismic proximity (crustal stress → atmospheric anomalies)")
    print("    - Cosmic ray flux (cloud formation mechanism)")
    print()

    print("CORRELATION EXAMPLES:")
    print("  1. Full moon + high Kp index → Enhanced geomagnetic storm")
    print("     (affects auroral precipitation → atmospheric dynamics)")
    print()
    print("  2. Major earthquake + low pressure system")
    print("     (crustal degassing + atmospheric instability → precursors?)")
    print()
    print("  3. Solar CME + stratospheric warming")
    print("     (space weather → sudden stratospheric warming events)")
    print()
    print("  4. Spring tide + heavy rainfall")
    print("     (tidal + weather → compound flooding risk)")
    print()


# ============================================================================
# PATTERN DETECTION SCENARIOS
# ============================================================================

def demonstrate_detection_scenarios():
    """
    Show example multi-scale patterns MYSTIC can now detect.
    """
    print("─" * 70)
    print("MULTI-SCALE DETECTION SCENARIOS")
    print("─" * 70)
    print()

    scenarios = [
        {
            "name": "Compound Flood Event",
            "scales": ["Weather", "Oceanic", "Planetary"],
            "signature": {
                "heavy_rain": ">100 mm/hr",
                "storm_surge": "High waves + spring tide",
                "lunar_phase": "Full/new moon (spring tide)",
            },
            "lead_time": "2-6 hours",
            "example": "Hurricane + king tide = extreme coastal flooding"
        },
        {
            "name": "Geomagnetic Storm Impact",
            "scales": ["Space Weather", "Atmospheric"],
            "signature": {
                "kp_index": ">6 (storm)",
                "pressure_anomaly": "Sudden stratospheric warming",
                "solar_wind": ">600 km/s"
            },
            "lead_time": "12-24 hours",
            "example": "Solar CME → geomagnetic storm → atmospheric effects"
        },
        {
            "name": "Earthquake Precursor Detection",
            "scales": ["Seismic", "Atmospheric", "Planetary"],
            "signature": {
                "seismic_swarm": "Multiple M2-3 events",
                "atmospheric_anomaly": "Pressure/temperature changes",
                "tidal_stress": "Maximum tidal force"
            },
            "lead_time": "Hours to days (speculative)",
            "example": "Tidal triggering of earthquakes (controversial but studied)"
        },
        {
            "name": "Severe Weather + Space Weather",
            "scales": ["Weather", "Space Weather", "Cosmic"],
            "signature": {
                "atmospheric_instability": "High CAPE",
                "cosmic_ray_flux": "Low (Forbush decrease)",
                "solar_activity": "Solar flare"
            },
            "lead_time": "6-12 hours",
            "example": "Galactic cosmic ray modulation affects cloud formation"
        }
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['name']}")
        print(f"   Scales: {', '.join(scenario['scales'])}")
        print(f"   Signature:")
        for key, val in scenario['signature'].items():
            print(f"     - {key}: {val}")
        print(f"   Lead time: {scenario['lead_time']}")
        print(f"   Example: {scenario['example']}")
        print()


# ============================================================================
# INTEGRATION ARCHITECTURE
# ============================================================================

def show_architecture():
    """
    Visualize the unified pipeline architecture.
    """
    print("─" * 70)
    print("UNIFIED PIPELINE ARCHITECTURE")
    print("─" * 70)
    print()

    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│                    DATA SOURCES (APIs)                          │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print("│  USGS: Stream gauges, earthquakes                              │")
    print("│  NOAA: Weather stations, buoys, satellites, space weather      │")
    print("│  NASA: Planetary ephemerides, solar data                       │")
    print("│  NMDB: Cosmic ray monitors                                     │")
    print("└─────────────────────────────────────────────────────────────────┘")
    print("                            ↓")
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│              PYTHON INGESTION LAYER                             │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print("│  - Fetch from APIs (every 15 min / hourly / daily)             │")
    print("│  - Unit conversion (all to metric)                             │")
    print("│  - Timestamp synchronization                                    │")
    print("│  - Quality control / gap filling                               │")
    print("└─────────────────────────────────────────────────────────────────┘")
    print("                            ↓")
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│            UNIFIED CSV FORMAT                                   │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print("│  timestamp, station_id, temp_c, ..., seismic_mag, ...,         │")
    print("│  ocean_temp_c, ..., geomagnetic_kp, ..., lunar_phase, ...,     │")
    print("│  cosmic_ray_flux, event_type                                    │")
    print("└─────────────────────────────────────────────────────────────────┘")
    print("                            ↓")
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│              RUST / MYSTIC ENGINE                               │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print("│  1. Load CSV → WeatherState (extended)")
    print("│  2. Map to Lorenz phase space (multi-dimensional)")
    print("│  3. Compute exact chaos signatures")
    print("│  4. Detect attractor basin entry")
    print("│  5. Issue multi-scale alerts")
    print("└─────────────────────────────────────────────────────────────────┘")
    print("                            ↓")
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│                  ALERT OUTPUTS                                  │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print("│  - Flash flood warning (2-6 hours)")
    print("│  - Geomagnetic storm alert (12-24 hours)")
    print("│  - Compound event warning (multi-scale)")
    print("│  - Seismic activity notification")
    print("└─────────────────────────────────────────────────────────────────┘")
    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    create_unified_csv()
    explain_multiscale_mapping()
    demonstrate_detection_scenarios()
    show_architecture()

    print("═" * 70)
    print("UNIFIED PIPELINE READY")
    print("═" * 70)
    print()
    print("Created files:")
    print("  ✓ unified_multiscale_training.csv")
    print()
    print("Next steps:")
    print("  1. Extend MYSTIC WeatherState to handle all scales")
    print("  2. Implement multi-dimensional Lorenz mapping")
    print("  3. Train on historical multi-scale events")
    print("  4. Deploy real-time multi-scale monitoring")
    print()
    print("Example usage:")
    print("  python3 train_multiscale_detector.py")
    print()


if __name__ == "__main__":
    main()
