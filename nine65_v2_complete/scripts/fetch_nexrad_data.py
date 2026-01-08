#!/usr/bin/env python3
"""
NEXRAD Radar Data Integration for MYSTIC Flash Flood Detection

Fetches NOAA NEXRAD Level II radar data (reflectivity → rainfall rate conversion)
to provide the missing rainfall intensity measurements identified in validation testing.

NEXRAD Coverage:
- ~160 WSR-88D radar sites across USA
- 5-10 minute updates
- Reflectivity (dBZ) → Rain rate (mm/hr) conversion
- Range: 230 km (coverage), 460 km (detection)

Data Sources:
1. NOAA National Centers for Environmental Information (NCEI)
   - Historical Level II archive
   - https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00345

2. NOAA Weather and Climate Toolkit
   - Real-time Level III products (base reflectivity)
   - https://www.ncdc.noaa.gov/nexradinv/

3. AWS NEXRAD on S3 (Real-time)
   - Bucket: noaa-nexrad-level2
   - Public access, requester pays
   - https://registry.opendata.aws/noaa-nexrad/

Marshall-Palmer Z-R Relationship:
  Z = 200 * R^1.6
  where:
    Z = reflectivity (mm^6/m^3)
    R = rainfall rate (mm/hr)
    dBZ = 10 * log10(Z)

  Solving for R:
    R = (10^(dBZ/10) / 200)^(1/1.6)
    R ≈ (Z / 200)^0.625
"""

import json
import urllib.request
import math
from datetime import datetime, timedelta

print("╔═══════════════════════════════════════════════════════════════════╗")
print("║         NEXRAD RADAR DATA INTEGRATION                            ║")
print("║      Rainfall Intensity for Flash Flood Detection                ║")
print("╚═══════════════════════════════════════════════════════════════════╝")
print()

# ============================================================================
# NEXRAD SITE DATABASE
# ============================================================================

# Texas Hill Country NEXRAD sites (for Camp Mystic, Wimberley floods)
NEXRAD_SITES = {
    "KEWX": {
        "name": "San Antonio/Austin, TX",
        "location": [29.704, -98.028],
        "elevation_m": 188,
        "coverage": ["Camp Mystic", "Wimberley", "Texas Hill Country"],
        "commissioned": "1995-03-31"
    },
    "KGRK": {
        "name": "Central Texas (Fort Hood)",
        "location": [30.722, -97.383],
        "elevation_m": 164,
        "coverage": ["Austin", "Central Texas"],
        "commissioned": "1995-09-30"
    },
    "KDFX": {
        "name": "Laughlin AFB, TX",
        "location": [29.273, -100.280],
        "elevation_m": 339,
        "coverage": ["West Texas Hill Country"],
        "commissioned": "1995-09-30"
    }
}

# ============================================================================
# REFLECTIVITY TO RAINFALL CONVERSION
# ============================================================================

def dbz_to_rain_rate(dbz):
    """
    Convert NEXRAD reflectivity (dBZ) to rainfall rate (mm/hr).

    Uses Marshall-Palmer Z-R relationship:
      Z = 200 * R^1.6
      R = (10^(dBZ/10) / 200)^(1/1.6)

    Args:
        dbz: Reflectivity in decibels (dBZ), typically 0-70

    Returns:
        Rainfall rate in mm/hr

    Interpretation:
        <20 dBZ: Light rain (<2.5 mm/hr)
        20-40 dBZ: Moderate rain (2.5-10 mm/hr)
        40-50 dBZ: Heavy rain (10-50 mm/hr)
        >50 dBZ: Extreme rain (>50 mm/hr, flash flood threat)
        >60 dBZ: Large hail or extreme precipitation
    """
    if dbz <= 0:
        return 0.0

    # Convert dBZ to Z (reflectivity factor)
    Z = 10 ** (dbz / 10.0)

    # Marshall-Palmer: R = (Z/200)^(1/1.6) = (Z/200)^0.625
    R = (Z / 200.0) ** (1.0 / 1.6)

    return R


def classify_precipitation_type(dbz, temp_c=None):
    """
    Classify precipitation type based on reflectivity and temperature.

    Args:
        dbz: Reflectivity in dBZ
        temp_c: Temperature in Celsius (optional)

    Returns:
        Tuple of (intensity, type, flash_flood_risk)
    """
    R = dbz_to_rain_rate(dbz)

    # Intensity classification
    if R < 0.25:
        intensity = "trace"
    elif R < 2.5:
        intensity = "light"
    elif R < 10:
        intensity = "moderate"
    elif R < 50:
        intensity = "heavy"
    else:
        intensity = "extreme"

    # Precipitation type (requires temperature)
    if temp_c is not None:
        if temp_c <= 0:
            precip_type = "snow" if dbz < 55 else "sleet/ice"
        elif temp_c > 0 and dbz > 55:
            precip_type = "rain_with_hail"
        else:
            precip_type = "rain"
    else:
        precip_type = "rain" if dbz < 55 else "rain_with_hail"

    # Flash flood risk assessment
    if R >= 50:
        flash_flood_risk = "EXTREME"
    elif R >= 25:
        flash_flood_risk = "HIGH"
    elif R >= 10:
        flash_flood_risk = "MODERATE"
    else:
        flash_flood_risk = "LOW"

    return (intensity, precip_type, flash_flood_risk)


# ============================================================================
# NEXRAD DATA FETCHING (DEMONSTRATION)
# ============================================================================

def fetch_nexrad_realtime_simulation(site_id, timestamp):
    """
    Simulate NEXRAD data fetch for demonstration.

    In production, this would:
    1. Connect to AWS S3: s3://noaa-nexrad-level2/
    2. Download Level II data file for timestamp
    3. Parse binary radar data (requires nexradpy or pyart library)
    4. Extract base reflectivity product
    5. Convert to rainfall rate grid

    For now, we'll simulate based on known flood events.
    """
    print(f"  Fetching NEXRAD data:")
    print(f"    Site: {site_id} ({NEXRAD_SITES[site_id]['name']})")
    print(f"    Timestamp: {timestamp}")
    print(f"    Source: AWS S3 (simulated)")
    print()

    # Simulate reflectivity values based on timestamp proximity to flood
    # In reality, this would come from actual radar scans

    # Example: Camp Mystic flood (2007-06-28 14:00)
    camp_mystic_time = datetime.fromisoformat("2007-06-28T14:00:00")
    time_to_flood = (camp_mystic_time - timestamp).total_seconds() / 3600

    if -6 <= time_to_flood <= 0:
        # During flood event: extreme reflectivity
        max_dbz = 60.0 - (abs(time_to_flood) * 5)  # Peak at T-0
    elif -24 <= time_to_flood < -6:
        # Building: moderate to heavy
        max_dbz = 30.0 + (6 - abs(time_to_flood - (-6))) * 2
    elif -72 <= time_to_flood < -24:
        # Precursor: light to moderate
        max_dbz = 15.0 + (24 - abs(time_to_flood - (-24))) * 0.5
    else:
        # Normal conditions
        max_dbz = 10.0

    # Simulate radar scan (simplified grid)
    scan_data = {
        "site": site_id,
        "timestamp": timestamp.isoformat(),
        "elevation_angle": 0.5,  # degrees (lowest tilt)
        "max_reflectivity_dbz": max_dbz,
        "max_rain_rate_mm_hr": dbz_to_rain_rate(max_dbz),
        "coverage_km": 230,
        "data_quality": "simulated"
    }

    intensity, precip_type, risk = classify_precipitation_type(
        max_dbz,
        temp_c=25.0  # Summer in Texas
    )

    scan_data["intensity"] = intensity
    scan_data["precip_type"] = precip_type
    scan_data["flash_flood_risk"] = risk

    print(f"  NEXRAD Scan Results:")
    print(f"    Max Reflectivity: {max_dbz:.1f} dBZ")
    print(f"    Max Rain Rate: {scan_data['max_rain_rate_mm_hr']:.1f} mm/hr")
    print(f"    Intensity: {intensity}")
    print(f"    Flash Flood Risk: {risk}")
    print()

    return scan_data


# ============================================================================
# INTEGRATION WITH MYSTIC FORMAT
# ============================================================================

def nexrad_to_mystic_format(scan_data, station_id="NEXRAD"):
    """
    Convert NEXRAD scan data to MYSTIC CSV format.

    Maps:
      rain_mm_hr ← max_rain_rate_mm_hr (from reflectivity conversion)
      event_type ← flash_flood_risk classification
    """
    return {
        "timestamp": scan_data["timestamp"],
        "station_id": station_id,
        "rain_mm_hr": scan_data["max_rain_rate_mm_hr"],
        "event_type": scan_data["flash_flood_risk"].lower(),
        "source": "NEXRAD",
        "site": scan_data["site"]
    }


# ============================================================================
# DEMONSTRATION: CAMP MYSTIC FLOOD NEXRAD DATA
# ============================================================================

def simulate_camp_mystic_nexrad():
    """
    Simulate NEXRAD radar scans for Camp Mystic flood event.
    """
    print("─" * 70)
    print("DEMONSTRATION: Camp Mystic Flash Flood NEXRAD Reconstruction")
    print("─" * 70)
    print()

    event_time = datetime.fromisoformat("2007-06-28T14:00:00")
    site = "KEWX"  # San Antonio radar (covers Kerr County)

    # Simulate scans at key times
    scan_times = [
        event_time - timedelta(hours=72),  # T-72h (precursor)
        event_time - timedelta(hours=24),  # T-24h (building)
        event_time - timedelta(hours=6),   # T-6h (imminent)
        event_time - timedelta(hours=2),   # T-2h (warning issued)
        event_time - timedelta(hours=1),   # T-1h (rapid intensification)
        event_time,                         # T-0 (flood peak)
    ]

    nexrad_data = []

    for scan_time in scan_times:
        print(f"Scan: {scan_time.isoformat()}")
        scan = fetch_nexrad_realtime_simulation(site, scan_time)
        mystic_row = nexrad_to_mystic_format(scan, station_id=site)
        nexrad_data.append(mystic_row)
        print()

    return nexrad_data


# ============================================================================
# CONVERSION TABLE (REFERENCE)
# ============================================================================

def print_conversion_table():
    """
    Print dBZ to rainfall rate conversion table for reference.
    """
    print("─" * 70)
    print("NEXRAD REFLECTIVITY TO RAINFALL RATE CONVERSION TABLE")
    print("─" * 70)
    print()
    print("dBZ  | Rain Rate | Intensity    | Flash Flood Risk | Description")
    print("-----+-----------+--------------+------------------+------------------")

    test_dbz = [0, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

    for dbz in test_dbz:
        R = dbz_to_rain_rate(dbz)
        intensity, ptype, risk = classify_precipitation_type(dbz, temp_c=20)

        if R < 1:
            desc = "Drizzle/Mist"
        elif R < 5:
            desc = "Light rain"
        elif R < 10:
            desc = "Moderate rain"
        elif R < 25:
            desc = "Heavy rain"
        elif R < 50:
            desc = "Very heavy rain"
        else:
            desc = "Extreme rainfall"

        print(f"{dbz:4d} | {R:8.2f}  | {intensity:12s} | {risk:16s} | {desc}")

    print()


# ============================================================================
# API DOCUMENTATION
# ============================================================================

def print_api_info():
    """
    Print information about NEXRAD data access APIs.
    """
    print("─" * 70)
    print("NEXRAD DATA ACCESS METHODS")
    print("─" * 70)
    print()

    print("1. AWS S3 (Real-time, Public Access)")
    print("   Bucket: s3://noaa-nexrad-level2/")
    print("   Format: YYYY/MM/DD/SITE/SITEYYYYMMDD_HHMMSSx")
    print("   Example: s3://noaa-nexrad-level2/2024/12/22/KEWX/KEWX20241222_153742_V06")
    print("   Cost: Requester pays S3 transfer (~$0.09/GB)")
    print()

    print("2. NOAA NCEI Archive (Historical)")
    print("   URL: https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00345")
    print("   Coverage: 1991-present")
    print("   Format: Level II (binary) or Level III (derived products)")
    print("   Cost: Free, but slow retrieval")
    print()

    print("3. Iowa State NEXRAD Archive")
    print("   URL: https://mesonet.agron.iastate.edu/archive/data/")
    print("   Coverage: Real-time + archive")
    print("   Format: Level III base reflectivity (easier to process)")
    print("   Cost: Free")
    print()

    print("4. Required Python Libraries:")
    print("   - boto3: AWS S3 access")
    print("   - pyart: Python ARM Radar Toolkit (NEXRAD parsing)")
    print("   - nexradpy: NOAA NEXRAD Python library")
    print("   - wradlib: Weather Radar Library (alternative)")
    print()

    print("Installation:")
    print("   pip install boto3 arm-pyart nexradpy")
    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Print conversion table
    print_conversion_table()

    # Simulate Camp Mystic NEXRAD data
    nexrad_data = simulate_camp_mystic_nexrad()

    # Save to JSON
    output_file = "../data/nexrad_camp_mystic_simulated.json"
    with open(output_file, 'w') as f:
        json.dump(nexrad_data, f, indent=2)

    print("─" * 70)
    print(f"✓ Saved NEXRAD simulation to: {output_file}")
    print()

    # Print API info
    print_api_info()

    print("═" * 70)
    print("NEXT STEPS FOR PRODUCTION INTEGRATION")
    print("═" * 70)
    print()
    print("1. Install NEXRAD processing libraries:")
    print("   pip install boto3 arm-pyart nexradpy")
    print()
    print("2. Set up AWS credentials for S3 access (if using real-time):")
    print("   aws configure")
    print()
    print("3. Modify fetch_nexrad_realtime_simulation() to use real data:")
    print("   - Download Level II file from S3")
    print("   - Parse with pyart.io.read_nexrad_archive()")
    print("   - Extract base reflectivity field")
    print("   - Apply Marshall-Palmer conversion")
    print()
    print("4. Integrate into MYSTIC pipeline:")
    print("   - Add NEXRAD rain_mm_hr to unified CSV")
    print("   - Train FloodDetector on NEXRAD + USGS stream data")
    print("   - Re-run validation tests")
    print()


if __name__ == "__main__":
    main()
