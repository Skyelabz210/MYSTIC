# MYSTIC Weather System - Quick Start Guide

In memory of Camp Mystic. No more tragedies.

---

## What You Have Now

- Python MYSTIC V3 predictors (`mystic_v3_integrated.py`, `mystic_v3_tuned.py`, `mystic_v3_production.py`)
- DataHub fetchers for USGS/Open-Meteo/NOAA (`data_sources_extended.py`)
- Rust NINE65/MYSTIC v2 demo in `nine65_v2_complete`
- No bundled historical datasets in this repo

---

## Run the Rust Demo (No Network Required)

```bash
cd ./nine65_v2_complete
cargo run --release --bin mystic_demo --features v2
```

---

## Run the Python Validation (Synthetic Patterns)

`historical_validation.py` uses reconstructed patterns for validation.

```bash
python3 historical_validation.py
```

---

## Run the Live Pipeline (Network Required)

```bash
python3 mystic_live_pipeline.py --lat 30.05 --lon -99.17
```

---

## Data Ingestion

- `mystic_live_pipeline.py` uses live feeds through `MYSTICDataHub`.
- `historical_data_loader.py` can fetch real historical data windows.
- `data_sources.py` and `data_sources_extended.py` provide programmatic access.

---

## Training (Rust Pipeline)

```bash
cd ./nine65_v2_complete
python3 scripts/train_flood_detector.py
cargo run --release --bin train_mystic --features v2
```

---

## Next Steps

1. Gather real historical event windows for flood events.
2. Label event windows if you are training the Rust detector.
3. Run live or historical validation for specific locations.

