# Development archive — Citi Bike Fleet Availability Planner

This folder documents how the final application was developed. **You do not need to run anything here to launch the application.** The working Streamlit entry point is `../../app_v2_3.py` relative to the `development/` folder's subdirectories, or `../app_v2_3.py` directly from this README.

The source scripts from Stages 1–4 and Stage 5 were recovered from the actual development archives, **not reconstructed**. The original instructions are retained and labeled `README_ORIGINAL.txt`. The adapted one-day verification runner is clearly identified as an additional convenience script.

| Stage | Folder | What you can inspect | Key recorded evidence |
| --- | --- | --- | --- |
| 1 | [`01_data_preparation/`](01_data_preparation/) | Memory-conscious monthly trip-CSV processing | `recorded_outputs/quality_summary.csv`, `manifest.json` |
| 2 | [`02_geography_weather/`](02_geography_weather/) | NYC NTA station matching and Open-Meteo history | `recorded_outputs/stage2_report.json`; archived station-information snapshot |
| 3 | [`03_model_training/`](03_model_training/) | Corrected model-building script; first failed matching attempt | `recorded_outputs/model_comparison.csv`, `station_matching_audit.csv`, `stage3_report.json` |
| 4 | [`04_initial_application/`](04_initial_application/) | First working Streamlit UI and planning calculations | Historical `app.py`, `planner_core.py` |
| 5 | [`05_dashboard_extensions/`](05_dashboard_extensions/) | Manhattan/outer-borough directional flows and weather association charts | Two preprocessing scripts and a station-ID alias file; outputs in top-level `processed_data/` |
| 6 | [`06_validation/`](06_validation/) | Original and final one-day calculation comparison | Saved QA reports and `verify_one_day_consistency.py` |

## How to read the archive

1. Start at Stage 1 to see how the trip records were prepared. The original raw trip data are **not** in this package.
2. In Stage 2, the original NYC geography/weather builder and a station-information snapshot explain zone assignment and weather integration. The produced weather and zone files already appear in the root `processed_data/` folder and are not duplicated here.
3. Stage 3 has the **selected corrected** trainer `build_forecasts_stage3_v2.py`, plus a labelled `historical_first_attempt/` showing the station-ID matching error. The model comparison comes from the chronological August–September 2026 holdout recorded at the time. Saved model files already appear in root `models/`.
4. Stage 4 contains the first application. The final code is the root-level `app_v2_3.py`, `planner_core_v2.py` and `dashboard_additions.py`.
5. Stage 5 contains the later feature-construction scripts. Its original borough script expects `scripts/prepare_borough_flows.py` **at the project root** and the alias file at `processed_data/station_short_names.csv`. To rerun, place those files in those historical locations and supply the original trip CSVs; do not expect it to run directly from this archive folder. The weather-chart script likewise expects to reside under root `scripts/`.
6. Stage 6 includes an original historical comparison script, saved test outputs, and a small **adapted** verification runner that can be run in place from this repository's root with installed dependencies: `python development/06_validation/verify_one_day_consistency.py`. It checks one fixed historical day using the same saved models and station snapshot for both algorithm versions. It does not test the browser UI or live API feeds.

## Reproducing the entire pipeline

These are preserved historical scripts, **not a one-command reproducible pipeline**. The older Stage 1–4 source code expects scripts at the root of an original `CitiBike_Project/` workspace, with source files and outputs in their historical locations. Some intermediate station-hour aggregates are not available in this archive. See [`DATA_SOURCES.md`](DATA_SOURCES.md) for original public data sources, missing inputs, and output locations. The saved reports document the original results; the complete model-training pipeline has not been rerun as part of this packaging operation.

## Source provenance

- Stages 1–4: `CitiBike_Stages_1_to_4_Development_Archive.zip` (original script bytes retained).
- Stage 5: `CitiBike_V2_3_Enhanced_Refined(1).zip` (`scripts/` and station short-name mapping).
- Stage 6: `CitiBike_V23_Handoff_Tools.zip` (original comparison script and reference module), plus saved earlier numerical QA reports; `verify_one_day_consistency.py` is an explicitly adapted runner.

No raw trip files, private production credentials, or transient `.venv`/`__pycache__` folders are included.
