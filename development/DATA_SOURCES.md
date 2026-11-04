# Data sources, outputs, and reproducibility

## Public data inputs

| Input | Source | Used for |
| --- | --- | --- |
| NYC Citi Bike trip histories, **October 2025–September 2026** | [Citi Bike System Data](https://citibikenyc.com/system-data) (monthly trip-data archives; original project used NYC trip files, excluding `JC` Jersey City files) | Departures, returns, rider categories, bike categories, ride durations, borough flows |
| Citi Bike GBFS station information and station status | [GBFS discovery endpoint](https://gbfs.citibikenyc.com/gbfs/2.3/gbfs.json) | Station IDs, coordinates, and available bicycles/docks (live or saved snapshot) |
| NYC 2020 Neighborhood Tabulation Areas (NTAs) | [NYC Open Data, 2020 NTA boundaries](https://data.cityofnewyork.us/d/9nt8-h7nd) | Neighborhood polygons, borough assignment, station-to-zone mapping |
| Historical and forecast weather | [Open-Meteo historical archive](https://archive-api.open-meteo.com/v1/archive) and [forecast API](https://api.open-meteo.com/v1/forecast) | Weather features for forecasting, descriptive weather charts, future weather scenarios |

Historical sources and the selected time window reflect the saved development artifacts. The package does not mirror third-party raw datasets or guarantee that their future downloads will have exactly the same contents.

## What is and is not provided

**Included at application root:** compressed monthly zone-hourly trip summaries, rider-category summaries, trip-duration histograms, NYC zone boundaries, station-to-NTA mapping, historical weather, directional-flow aggregates, daily weather association data, trained departure/return models, and a saved GBFS status snapshot.

**Included in `development/`:** Stage 1–5 processing/training source code, original execution notes, a subset of saved quality and model comparison reports, a historical station-information snapshot, and numerical consistency QA.

**Not included:** roughly 45 million individual ride records, large Stage 1 station-hour departure/return aggregates, original raw month folders, some intermediary audit files, or snapshots of previously issued weather forecasts. These inputs are unnecessary for running the demo, but the original training pipeline cannot be rerun end-to-end from this ZIP alone.

## Historical run order (reference only)

1. `prepare_citibike.py` was originally run at the **project root**, with extracted monthly CSVs in `raw_data/`, creating `processed_data/departures_YYYYMM.csv.gz`, `returns_YYYYMM.csv.gz`, `daily_usage_YYYYMM.csv.gz`, and duration files.
2. `build_geography_weather.py` was run at the root with `reference_data/station_information_snapshot.json`, creating zone maps, station mappings and hourly observed weather.
3. **Use `build_forecasts_stage3_v2.py`**, not the zero-match historical first attempt. It reads Stage 1 station aggregates plus Stage 2 data, performs matching/validation, benchmarks forecasts, writes the final zone-hour summaries, reports, and `models/*.joblib`.
4. The first Streamlit prototype then used these prepared data and models. The final root-level app runs directly from the included outputs and does not retrain models.
5. The later scripts for borough movements and weather charts generated the final `processed_data/borough_directional_flows.csv.gz` and `processed_data/weather_sensitivity_daily.csv` files.

This is an archaeological description of the recorded workflow, not a tested turnkey installation procedure. Consult the stage-specific original notes and read each script's input expectations before attempting reconstruction.
