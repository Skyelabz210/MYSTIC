#!/usr/bin/env python3
"""
MYSTIC Historical Disaster Database

Compiles major disasters across all categories for validation testing:
- Flash floods
- Hurricanes/tropical storms
- Tornadoes
- Earthquakes (with reported precursors)
- Geomagnetic storms (with reported impacts)
- Compound events (multi-scale)

For each event, we need:
1. Event date/time (when it occurred)
2. Earliest warning/report (T-minus to event)
3. Precursor data (hours/days before)
4. Impact severity
5. Available data sources for reconstruction
"""

import json
from datetime import datetime

print("╔═══════════════════════════════════════════════════════════════════╗")
print("║         MYSTIC HISTORICAL DISASTER DATABASE                       ║")
print("║      Validation Events for Testing Predictive Capability          ║")
print("╚═══════════════════════════════════════════════════════════════════╝")
print()

# ============================================================================
# DISASTER DATABASE
# ============================================================================

disasters = {
    "flash_floods": [
        {
            "name": "Camp Mystic Flash Flood",
            "date": "2007-06-28T14:00:00",
            "location": "Kerr County, Texas, USA",
            "coordinates": [29.95, -99.34],
            "deaths": 3,
            "earliest_warning": "T-2h (flash flood warning issued)",
            "precursor_period": "72 hours (heavy rain upstream)",
            "severity": 8,
            "data_sources": ["USGS stream gauges", "NEXRAD radar", "NWS reports"],
            "notes": "Rapid river rise from 2ft to 15ft in <1 hour. Summer camp evacuated too late.",
            "usgs_stations": ["08166200", "08165500"],
            "testable": True
        },
        {
            "name": "Memorial Day Flood (Wimberley)",
            "date": "2015-05-23T22:00:00",
            "location": "Wimberley, Texas, USA",
            "coordinates": [29.98, -98.09],
            "deaths": 13,
            "earliest_warning": "T-4h (flash flood watch)",
            "precursor_period": "48 hours (training thunderstorms)",
            "severity": 9,
            "data_sources": ["USGS Blanco River gauge", "NEXRAD radar", "NWS"],
            "notes": "Blanco River rose 40 feet in 3 hours. Record-breaking flood.",
            "usgs_stations": ["08171000"],
            "testable": True
        },
        {
            "name": "Ellicott City Flash Flood",
            "date": "2016-07-30T19:30:00",
            "location": "Ellicott City, Maryland, USA",
            "coordinates": [39.27, -76.80],
            "deaths": 2,
            "earliest_warning": "T-1h (severe thunderstorm warning)",
            "precursor_period": "6 hours (training storms)",
            "severity": 7,
            "data_sources": ["NEXRAD", "NWS", "USGS"],
            "notes": "6+ inches rain in 2 hours. Historic downtown devastated.",
            "testable": True
        },
        {
            "name": "Kinston Flash Flood (Hurricane Matthew)",
            "date": "2016-10-10T06:00:00",
            "location": "Kinston, North Carolina, USA",
            "coordinates": [35.26, -77.58],
            "deaths": 4,
            "earliest_warning": "T-12h (hurricane flood warnings)",
            "precursor_period": "48 hours (Hurricane Matthew approach)",
            "severity": 8,
            "data_sources": ["NHC advisories", "USGS", "NEXRAD"],
            "notes": "Neuse River flooding from Hurricane Matthew. Delayed crest.",
            "testable": True
        }
    ],

    "hurricanes": [
        {
            "name": "Hurricane Harvey",
            "date": "2017-08-25T03:00:00",
            "location": "Rockport, Texas, USA (landfall)",
            "coordinates": [28.05, -97.05],
            "deaths": 107,
            "earliest_warning": "T-120h (NHC tropical storm watch)",
            "precursor_period": "5 days (tropical wave tracking)",
            "severity": 10,
            "data_sources": ["NHC", "NOAA satellites", "buoys", "NEXRAD"],
            "notes": "Cat 4 landfall. Historic rainfall (60+ inches). $125B damage.",
            "max_winds": 130,
            "min_pressure": 937,
            "testable": True
        },
        {
            "name": "Hurricane Katrina",
            "date": "2005-08-29T06:10:00",
            "location": "Louisiana, USA",
            "coordinates": [29.35, -89.60],
            "deaths": 1833,
            "earliest_warning": "T-72h (hurricane watch)",
            "precursor_period": "7 days (tracking from Bahamas)",
            "severity": 10,
            "data_sources": ["NHC", "NOAA", "buoys"],
            "notes": "Cat 5 in Gulf. Cat 3 at landfall. Catastrophic storm surge.",
            "max_winds": 175,
            "min_pressure": 902,
            "testable": True
        },
        {
            "name": "Hurricane Maria",
            "date": "2017-09-20T06:15:00",
            "location": "Puerto Rico",
            "coordinates": [18.20, -66.50],
            "deaths": 2975,
            "earliest_warning": "T-96h (tropical storm watch)",
            "precursor_period": "5 days",
            "severity": 10,
            "data_sources": ["NHC", "GOES satellites", "reconnaissance"],
            "notes": "Cat 5 direct hit. Total infrastructure collapse.",
            "max_winds": 175,
            "min_pressure": 908,
            "testable": True
        }
    ],

    "earthquakes": [
        {
            "name": "Tohoku Earthquake & Tsunami",
            "date": "2011-03-11T14:46:00",
            "location": "Off Honshu, Japan",
            "coordinates": [38.30, 142.37],
            "deaths": 15899,
            "earliest_warning": "T-0s (no precursor warning, P-wave detection only)",
            "precursor_period": "Disputed (possible electromagnetic anomalies days before)",
            "severity": 10,
            "magnitude": 9.1,
            "depth_km": 29,
            "data_sources": ["USGS", "JMA", "ionospheric monitors"],
            "notes": "Precursor research: Ionospheric disturbances reported. Highly controversial.",
            "testable": False  # No consensus on precursors
        },
        {
            "name": "Haiti Earthquake",
            "date": "2010-01-12T21:53:00",
            "location": "Port-au-Prince, Haiti",
            "coordinates": [18.46, -72.53],
            "deaths": 316000,
            "earliest_warning": "T-0s (no warning)",
            "precursor_period": "None documented",
            "severity": 10,
            "magnitude": 7.0,
            "depth_km": 13,
            "data_sources": ["USGS"],
            "notes": "Shallow strike-slip on Enriquillo-Plantain Garden fault.",
            "testable": False
        },
        {
            "name": "L'Aquila Earthquake",
            "date": "2009-04-06T01:32:00",
            "location": "L'Aquila, Italy",
            "coordinates": [42.35, 13.38],
            "deaths": 309,
            "earliest_warning": "T-0s (controversial 'swarm' observations)",
            "precursor_period": "Months (foreshock swarm, radon emissions reported)",
            "severity": 7,
            "magnitude": 6.3,
            "depth_km": 8.8,
            "data_sources": ["USGS", "INGV Italy", "radon monitoring"],
            "notes": "Famous for prosecution of scientists. Precursor debate.",
            "testable": True  # Has documented precursors (radon, swarms)
        }
    ],

    "geomagnetic_storms": [
        {
            "name": "Halloween Geomagnetic Storm",
            "date": "2003-10-29T06:00:00",
            "location": "Global (space weather event)",
            "coordinates": [0, 0],
            "impact": "Satellite damage, power grid fluctuations, aurora to Texas",
            "earliest_warning": "T-48h (solar flare X17.2 observed)",
            "precursor_period": "72 hours (active region tracking)",
            "severity": 9,
            "kp_max": 9,
            "data_sources": ["NOAA SWPC", "ACE spacecraft", "magnetometers"],
            "notes": "One of largest storms in recorded history. CME travel time: ~19 hours.",
            "testable": True
        },
        {
            "name": "Carrington Event (Historical)",
            "date": "1859-09-01T11:00:00",
            "location": "Global",
            "coordinates": [0, 0],
            "impact": "Telegraph system failures, aurora to Caribbean",
            "earliest_warning": "T-17h (solar flare observed by Carrington)",
            "precursor_period": "24 hours",
            "severity": 10,
            "kp_max": 9,
            "data_sources": ["Historical records", "ice cores", "tree rings"],
            "notes": "Largest geomagnetic storm in recorded history. Would cripple modern tech.",
            "testable": False  # No modern data
        },
        {
            "name": "Quebec Blackout Storm",
            "date": "1989-03-13T02:45:00",
            "location": "Quebec, Canada (power grid)",
            "coordinates": [45.50, -73.57],
            "impact": "6 million without power for 9 hours",
            "earliest_warning": "T-24h (solar flare observed)",
            "precursor_period": "48 hours",
            "severity": 8,
            "kp_max": 9,
            "data_sources": ["NOAA", "Canadian magnetometers"],
            "notes": "Geomagnetically Induced Currents (GIC) collapsed grid.",
            "testable": True
        }
    ],

    "compound_events": [
        {
            "name": "Hurricane Harvey + King Tide",
            "date": "2017-08-25T03:00:00",
            "location": "Texas Coast",
            "coordinates": [28.05, -97.05],
            "scales": ["hurricane", "oceanic", "planetary"],
            "deaths": 107,
            "earliest_warning": "T-120h (hurricane), T-weeks (astronomical tide prediction)",
            "precursor_period": "5 days (hurricane), months (tide tables)",
            "severity": 10,
            "data_sources": ["NHC", "NOAA tides", "lunar ephemeris"],
            "notes": "Hurricane coincided with perigean spring tide. Extreme storm surge.",
            "testable": True
        },
        {
            "name": "Tohoku Earthquake + Tsunami + Nuclear",
            "date": "2011-03-11T14:46:00",
            "location": "Japan",
            "coordinates": [38.30, 142.37],
            "scales": ["seismic", "oceanic", "industrial"],
            "deaths": 15899,
            "earliest_warning": "T-0 (earthquake), T-10min (tsunami warnings)",
            "precursor_period": "Disputed",
            "severity": 10,
            "data_sources": ["USGS", "JMA", "DART buoys"],
            "notes": "Cascading disaster: M9.1 quake → 40m tsunami → Fukushima meltdown.",
            "testable": False
        }
    ],

    "tornado_outbreaks": [
        {
            "name": "2011 Super Outbreak",
            "date": "2011-04-27T15:00:00",
            "location": "Southeastern USA",
            "coordinates": [33.5, -87.5],
            "deaths": 321,
            "earliest_warning": "T-72h (severe weather outlook)",
            "precursor_period": "5 days (pattern recognition)",
            "severity": 10,
            "ef5_count": 4,
            "total_tornadoes": 360,
            "data_sources": ["NWS SPC", "NEXRAD", "storm reports"],
            "notes": "Largest outbreak since 1974. Multiple EF5 tornadoes.",
            "testable": True
        },
        {
            "name": "Joplin Tornado",
            "date": "2011-05-22T17:34:00",
            "location": "Joplin, Missouri, USA",
            "coordinates": [37.08, -94.51],
            "deaths": 161,
            "earliest_warning": "T-20min (tornado warning)",
            "precursor_period": "24 hours (severe outlook)",
            "severity": 9,
            "ef_rating": "EF5",
            "data_sources": ["NWS", "NEXRAD", "damage surveys"],
            "notes": "Deadliest single tornado since 1950. $2.8B damage.",
            "testable": True
        }
    ]
}

# ============================================================================
# SAVE DATABASE
# ============================================================================

output_file = "../data/historical_disaster_database.json"

with open(output_file, 'w') as f:
    json.dump(disasters, f, indent=2)

print(f"✓ Created disaster database: {output_file}")
print()

# ============================================================================
# STATISTICS
# ============================================================================

def print_statistics():
    print("─" * 70)
    print("DATABASE STATISTICS")
    print("─" * 70)
    print()

    total_events = sum(len(v) for v in disasters.values())
    total_deaths = 0
    testable_count = 0

    for category, events in disasters.items():
        count = len(events)
        deaths = sum(e.get('deaths', 0) for e in events)
        testable = sum(1 for e in events if e.get('testable', False))

        total_deaths += deaths
        testable_count += testable

        print(f"{category.upper().replace('_', ' ')}:")
        print(f"  Events: {count}")
        print(f"  Deaths: {deaths:,}")
        print(f"  Testable: {testable}/{count}")
        print()

    print("TOTALS:")
    print(f"  Categories: {len(disasters)}")
    print(f"  Total Events: {total_events}")
    print(f"  Total Deaths: {total_deaths:,}")
    print(f"  Testable Events: {testable_count}/{total_events}")
    print()

print_statistics()

# ============================================================================
# TESTABLE EVENTS SUMMARY
# ============================================================================

def print_testable_events():
    print("─" * 70)
    print("TESTABLE EVENTS (For MYSTIC Validation)")
    print("─" * 70)
    print()

    testable = []
    for category, events in disasters.items():
        for event in events:
            if event.get('testable', False):
                testable.append({
                    'category': category,
                    'name': event['name'],
                    'date': event['date'],
                    'warning_time': event['earliest_warning'],
                    'severity': event['severity'],
                    'data_sources': event['data_sources']
                })

    for i, event in enumerate(testable, 1):
        print(f"{i}. {event['name']} ({event['category']})")
        print(f"   Date: {event['date']}")
        print(f"   Earliest Warning: {event['warning_time']}")
        print(f"   Severity: {event['severity']}/10")
        print(f"   Data: {', '.join(event['data_sources'])}")
        print()

    print(f"Total testable events: {len(testable)}")
    print()

print_testable_events()

# ============================================================================
# MYSTIC CAPABILITY GAPS
# ============================================================================

def identify_gaps():
    """
    Identify what MYSTIC needs to add to test each category.
    """
    print("─" * 70)
    print("MYSTIC CAPABILITY GAPS & REQUIREMENTS")
    print("─" * 70)
    print()

    gaps = {
        "Flash Floods": {
            "current": ["Stream gauge data", "Lorenz chaos detection", "Atmospheric mapping"],
            "needed": ["NEXRAD radar rainfall intensity", "Soil saturation models", "Basin-specific training"],
            "priority": "HIGH - Core MYSTIC application"
        },
        "Hurricanes": {
            "current": ["Ocean buoy data", "Atmospheric pressure", "Geomagnetic (indirect)"],
            "needed": ["NHC track/intensity data", "SST fields", "Wind shear analysis", "Rapid intensification predictors"],
            "priority": "MEDIUM - Requires tropical meteorology expertise"
        },
        "Earthquakes": {
            "current": ["USGS seismic feed", "Tidal force calculations"],
            "needed": ["Radon monitoring", "Electromagnetic sensors", "Ionospheric TEC data", "Foreshock analysis"],
            "priority": "LOW - Precursor science controversial/unproven"
        },
        "Geomagnetic Storms": {
            "current": ["NOAA SWPC data", "Kp index", "Solar wind", "CME detection"],
            "needed": ["Magnetometer network", "Ionospheric models", "Power grid coupling"],
            "priority": "MEDIUM - Well-understood physics, good data"
        },
        "Tornadoes": {
            "current": ["NEXRAD access", "Atmospheric instability proxies"],
            "needed": ["Supercell detection algorithms", "Mesocyclone tracking", "Tornado genesis predictors"],
            "priority": "MEDIUM - Requires radar algorithm development"
        },
        "Compound Events": {
            "current": ["Multi-scale data integration (operational)"],
            "needed": ["Cross-correlation algorithms", "Multi-scale attractor training"],
            "priority": "HIGH - MYSTIC's unique strength"
        }
    }

    for category, info in gaps.items():
        print(f"{category}:")
        print(f"  Current Capabilities:")
        for item in info['current']:
            print(f"    ✓ {item}")
        print(f"  Needed:")
        for item in info['needed']:
            print(f"    ⚠ {item}")
        print(f"  Priority: {info['priority']}")
        print()

identify_gaps()

print("═" * 70)
print("NEXT STEP: Create validation test framework")
print("  python3 create_validation_framework.py")
print("═" * 70)
