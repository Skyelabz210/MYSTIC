# MYSTIC Predictive Gauntlet - Historical Validation Design

## Purpose

Run an extensive, repeatable test suite that pushes MYSTIC through real, archived
historical events and verifies predictions against documented outcomes. The
objective is to validate multi-variable fusion, trend awareness, and attractor
classification using record-backed data.

## Goals

- Validate risk levels and hazard types against historical records.
- Measure lead-time detection (risk escalation before peak impact).
- Compare single-variable vs multi-variable fusion performance.
- Provide reproducible, offline datasets for deterministic validation.

## Test Phases

### Phase A: Archived Offline Gauntlet (Reproducible)

Source: CSV archives in `nine65_v2_complete/data/`.

- USGS streamflow and gage height: `data/historical/usgs_*.csv`
- Meteorological data: `data/meteorological/weather_*.csv`

Outcome: Fully deterministic, no network required.

### Phase B: Live Historical Gauntlet (API-backed)

Source: `historical_data_loader.py` (Open-Meteo + USGS IV/DV).

Outcome: Validates pipeline against live source behavior and current API
responses.

### Phase C: Reconstructed Pattern Gauntlet (Synthetic)

Source: `historical_validation.py` patterns derived from NOAA/NWS reports.

Outcome: Sanity check for risk thresholds across archetypal events.

## Metrics

- Risk Level Accuracy: % of events matching expected risk band.
- Hazard Type Accuracy: % matching expected hazard type (if specified).
- Score Threshold: risk_score >= expected_min_score.
- Lead-Time Success: risk_score >= lead_min_score before peak impact.
- Multi-Variable Lift: delta in accuracy vs single-variable runs.

## Pass/Fail Gates

- Risk accuracy >= 85% for Phase A.
- Lead-time success >= 70% for high/critical events.
- Hazard type accuracy >= 70% for Phase A.
- Phase B must be within 10% of Phase A accuracy.

## Event Set (Phase A)

Each event is validated against public records (USGS, NOAA, NWS). Expected
risk levels map to documented severity (fatalities, declared disasters, or
extreme hydrometeorological impacts).

| ID | Event | Expected Hazard | Expected Risk | Records |
|----|-------|-----------------|---------------|---------|
| camp_mystic_2007 | Camp Mystic Flood (2007) | FLASH_FLOOD | CRITICAL | USGS, local reports |
| halloween_2013 | Halloween Floods (2013) | FLASH_FLOOD | HIGH | USGS, NOAA |
| harvey_2017 | Hurricane Harvey (2017) | HURRICANE | CRITICAL | NOAA NHC, USGS |
| llano_2018 | Llano River Flood (2018) | FLASH_FLOOD | HIGH | USGS, NWS |
| memorial_day_2015 | Memorial Day Floods (2015) | FLASH_FLOOD | CRITICAL | USGS, NOAA |
| memorial_day_2016 | Memorial Day Floods (2016) | FLASH_FLOOD | HIGH | USGS, NOAA |
| tax_day_2016 | Tax Day Flood (2016) | FLASH_FLOOD | HIGH | USGS, NOAA |
| imelda_2019 | Tropical Storm Imelda (2019) | HURRICANE | CRITICAL | NOAA NHC, USGS |

## Event Inputs

- Streamflow (discharge_cfs): scaled to integer (cfs * 100)
- Gage height (ft): scaled to integer (ft * 100)
- Pressure (hPa): scaled to integer (hPa * 10)
- Precipitation (mm): scaled to integer (mm * 100)
- Wind speed (m/s): converted to km/h * 10
- Temperature (C): scaled to integer (C * 100)
- Humidity: derived from dewpoint/temp when available

## Execution

Use `predictive_gauntlet.py` with the event config file to run Phase A.

Example:

```
python3 predictive_gauntlet.py --events predictive_gauntlet_events.json --output predictive_gauntlet_report.json
```

Live Phase B:

```
python3 predictive_gauntlet_live.py --output predictive_gauntlet_live_report.json
```

## Outputs

- `predictive_gauntlet_report.json`: structured results, per-event details
- Console summary: accuracy, lead-time metrics, and failures

## Latest Results (2026-01-08)

- Events tested: 8
- Risk accuracy: 100%
- Score accuracy: 100%
- Hazard accuracy: 100%
- Lead-time success: 100%

Report: `predictive_gauntlet_report.json`

## Phase B Results (Live APIs, 2026-01-08)

- Events tested: 6
- Risk accuracy: 100%
- Score accuracy: 100%
- Hazard accuracy: 100%
- Lead-time success: 100%

Report: `predictive_gauntlet_live_report.json`
