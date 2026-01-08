---
name: mystic-early-warning-execution
description: Task-based plan for integrating free data sources into MYSTIC early warning
---

# Plan

Build a concrete, task-based execution plan to integrate all free data sources (meteorological, hydrologic, seismic, space, and cosmic) into the MYSTIC early warning pipeline for the Camp Mystic flash flood focus. Emphasis is on real ingestion, token management, and a unified dataset that the existing chaos/attractor logic can consume.

## Requirements
- Integrate as many free data sources as possible (NOAA/NWS/USGS/NOAA SWPC/JPL Horizons/NDBC/CO-OPS/NEXRAD, etc.).
- Use tokens where required; tolerate missing tokens without failing the pipeline.
- Persist a unified summary for downstream ingestion and training.
- Keep MYSTIC’s existing data formats stable; only extend where needed.
- Maintain offline/synthetic fallback paths.

## Scope
- In: Python ingestion scripts, unified pipeline mapping, token/access documentation, summary schema.
- Out: Re-architecting Rust core or rewriting detection algorithms; paid APIs; external alerting/SMS integrations.

## Files and entry points
- scripts/fetch_all_data_sources.py
- scripts/create_unified_pipeline.py
- WHAT_YOU_NEED.md
- data/omniscient_data_summary.json (generated)

## Data model / API changes
- Extend `omniscient_data_summary.json` to include structured per-scale metrics (status, latest values, station IDs, timestamps).
- Keep `unified_multiscale_training.csv` field list stable, but populate values from real summary data when available.

## Action items
[ ] Confirm target Camp Mystic coordinates/stations for NOAA/NWS/CO-OPS/NEXRAD and set defaults in the ingestion script.
[ ] Implement NOAA/NWS/NEXRAD ingestion with real JSON/XML parsing and add status-safe return objects.
[ ] Implement CO-OPS tides/water level ingestion and NDBC buoy parsing; capture latest ocean metrics.
[ ] Implement SWPC space-weather ingestion (Kp, solar wind plasma/mag, GOES X-ray flux) and store latest values.
[ ] Implement JPL Horizons query for lunar distance/phase proxy; compute tidal-force index.
[ ] Add cosmic-ray feed ingestion (free neutron monitor or Oulu station) with robust parsing.
[ ] Update unified pipeline to map summary values into CSV rows with clear defaults when data is missing.
[ ] Update `WHAT_YOU_NEED.md` with free-token access steps and required environment variables.
[ ] Add a quick-run validation sequence (fetch → unify → optional Camp Mystic pipeline) and document it.

## Testing and validation
- `python3 scripts/fetch_all_data_sources.py`
- `python3 scripts/create_unified_pipeline.py`
- Optional: `python3 scripts/run_camp_mystic_pipeline.py --offline` (verifies end-to-end without network)

## Risks and edge cases
- External APIs may rate-limit or change responses; ingestion must degrade gracefully.
- Some feeds are location- or station-specific; incorrect IDs could yield empty data.
- Some sources (NASA Earthdata) require registration; missing tokens should not block the pipeline.

## Open questions
- Confirm preferred Camp Mystic coordinates and the NEXRAD/CO-OPS station IDs to hard-code as defaults.
