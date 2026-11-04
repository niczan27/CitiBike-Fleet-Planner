CITI BIKE STAGE 3 V2 - FIX FOR 0% HISTORICAL STATION MATCHING

WHAT WENT WRONG
The earlier Stage 3 mapped historical trip station codes only against GBFS station_id
(often UUID or long integer). Citi Bike's historical monthly CSVs also commonly use
GBFS short_name codes, such as 5216.06. This caused zero matched trips and meaningless
zero-error models. The v2 script reads both identifier types from your original Stage 2
station snapshot and stops before training if fewer than 80% of rides are matched.

DO THIS
1. Download build_forecasts_stage3_v2.py and put it directly inside CitiBike_Project.
2. Keep raw_data/, processed_data/, reference_data/ and models/ where they are.
3. In Terminal, while already in CitiBike_Project, run:

   .venv/bin/python build_forecasts_stage3_v2.py

No library installation and no re-running Stage 1 or Stage 2 are required.
This script can overwrite incorrect Stage 3 outputs, but NOT your raw datasets or
Stage 1/2 inputs.

SUCCESS CHECK
Look for 'departures: ... usable (XX%)' and 'returns: ... usable (XX%)'.
Both must be nonzero and preferably over 95%. Then compare models.
Do NOT use the old model comparison, old zero-filled zone_hourly files, or previously
saved forecasts.

AFTER IT FINISHES SEND
processed_data/stage3_report.json
processed_data/station_matching_audit.csv
models/model_comparison.csv
If coverage below 95%, also send
processed_data/unknown_historical_station_ids.csv

If the new script stops due to coverage under 80%, send the latter two audit CSVs.
Do not manually combine the original CSVs; v2 reads the existing gzip aggregates.
