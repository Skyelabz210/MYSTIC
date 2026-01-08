---
name: mystic-free-data-integration
description: Implement free-source data ingestion and wiring for MYSTIC
---

# Plan

Build a complete, free-source ingestion layer for MYSTIC (Texas flash flooding focus) by turning the current catalog-only fetcher into a real data collector, wiring it into the unified pipeline, and documenting access tokens. Key gaps to close: several feeds only print metadata (no numeric ingest), `omniscient_data_summary.json` lacks real values, token/env handling is undocumented, and time/unit harmonization is inconsistent.

## Requirements
- Ingest as many free sources as possible across meteorological, hydrologic, seismic, oceanic, solar, geomagnetic, and cosmic domains.
- Use optional free tokens where they unlock richer data:
  - NOAA CDO token (free) for GHCN daily stations: https://www.ncdc.noaa.gov/cdo-web/token
  - NASA Earthdata login (free) for NLDAS/GLDAS/SMAP: https://urs.earthdata.nasa.gov
- Preserve offline mode and provide graceful fallbacks when tokens are missing.

## Scope
- In: NOAA/NWS/USGS/CO-OPS/NDBC/SWPC/NEXRAD/GOES/NASA/Oulu cosmic rays ingestion, unified summary fields, pipeline wiring, documentation updates.
- Out: paid providers, custom hardware sensor ingestion, large-scale storage/ETL beyond JSON summary.

## Files and entry points
- `scripts/fetch_all_data_sources.py`
- `scripts/create_unified_pipeline.py`
- `scripts/fetch_camp_mystic_2007.py`
- `scripts/build_camp_mystic_unified.py`
- `scripts/run_camp_mystic_pipeline.py`
- `data/omniscient_data_summary.json`
- `WHAT_YOU_NEED.md`

## Data model / API changes
- Expand `data/omniscient_data_summary.json` with numeric fields (e.g., `nws_alert_count`, `nexrad_latest_key`, `solar_xray_flux`, `kp_index`, `solar_wind_speed`, `cosmic_ray_flux`, buoy/tide metrics).
- Add env var usage notes: `NOAA_CDO_TOKEN`, `NASA_EARTHDATA_USER`, `NASA_EARTHDATA_PASS`, optional `SYNOPTIC_TOKEN`.

## Action items
[ ] Audit current fetcher outputs and identify stub-only sections.
[ ] Implement NOAA CDO station + daily observations (tokened) and wire into summary.
[ ] Add NWS active alerts, NEXRAD latest key, and NOAA GOES/GRIB catalog pointers (no-token).
[ ] Implement CO-OPS tides/currents pull and NDBC buoy parsing into summary.
[ ] Implement SWPC solar/geomagnetic feeds (alerts, X-ray flux, Kp index) with numeric extraction.
[ ] Add solar wind (e.g., SWPC real-time solar wind JSON) into summary.
[ ] Add cosmic ray feed (e.g., Oulu neutron monitor) and parse latest value.
[ ] Wire summary fields into `scripts/create_unified_pipeline.py` so downstream uses real values when present.
[ ] Add offline/no-token fallbacks and clear status fields for all sources.
[ ] Update `WHAT_YOU_NEED.md` with token signup steps and env var usage.

## Testing and validation
- Run `python3 scripts/fetch_all_data_sources.py` and verify `data/omniscient_data_summary.json` contains numeric values.
- Run `python3 scripts/create_unified_pipeline.py` and confirm data fields are ingested.
- Optional: run `python3 scripts/run_camp_mystic_pipeline.py --offline` to validate fallback paths.

## Risks and edge cases
- API rate limits or temporary outages; need cached fallback or status flags.
- Unit mismatches (knots vs m/s, Celsius vs Kelvin, mm vs inches).
- Some feeds are near-real-time only (not historical for 2007); must treat as live features or fall back to archives/synthetic.

## Open questions
- Confirm the exact lat/lon radius for Camp Mystic ingestion (default now is San Antonio area).
- Confirm which optional tokens you can provide (NOAA CDO, NASA Earthdata, Synoptic).
- Do you want raw feed snapshots stored to disk or summary-only?
