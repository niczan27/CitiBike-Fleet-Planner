CITI BIKE PROJECT - STAGE 3: HISTORICAL STATION MATCHING + FORECAST MODEL COMPARISON

WHAT TO DO
1. Download build_forecasts_stage3.py.
2. Place it DIRECTLY inside CitiBike_Project (at the same level as prepare_citibike.py).
3. Do NOT put it in a new subfolder. Keep raw_data/ and processed_data/ unchanged.

From your Mac Terminal, while in CitiBike_Project, run:

  .venv/bin/python -m pip install scikit-learn joblib
  .venv/bin/python build_forecasts_stage3.py

The second command reads the outputs of the previous two stages, NOT the 2 GB zipped collection.
The script does not change or delete any of your original files.

WHAT IT DOES
A. Loads monthly hourly station departures/returns prepared in Stage 1.
B. Matches historical station IDs to the NYC NTA directory from Stage 2.
C. Calculates matching coverage by month; saves missing station IDs for investigation.
D. Builds compact hourly zone data, including separate member/casual groups.
E. Benchmarks four approaches for departures and returns separately:
      - average from historical zone / bicycle / weekday / hour
      - Poisson regression using calendar and weather
      - boosted trees without weather
      - boosted trees including weather
F. Validates on August and September 2026, not used to fit the tested models.
G. Selects each target's best model by mean absolute error and refits it using
   all 12 available months for use in the application.

NEW OUTPUTS
processed_data/station_matching_audit.csv
processed_data/unknown_historical_station_ids.csv
processed_data/zone_hourly_YYYYMM.csv.gz  (12 files)
processed_data/zone_hourly_member_YYYYMM.csv.gz  (12 files)
processed_data/stage3_report.json
models/model_comparison.csv
models/departures_model.joblib
models/returns_model.joblib
models/model_features.json

SEND BACK THESE SMALL FILES
  processed_data/stage3_report.json
  processed_data/station_matching_audit.csv
  models/model_comparison.csv
If the terminal warns that less than 95% of historical station records matched,
also upload processed_data/unknown_historical_station_ids.csv.

IMPORTANT LIMITATIONS
- These are counts of completed rides, not attempted/unfulfilled rentals.
- Current station locations do not cover all historical/retired station IDs.
  The coverage audit tells us whether to recover some old station coordinates.
- Daylight-saving time can repeat a local hour in November. Stage 1 collapsed
  duplicate clock hours, so Stage 3 averages any duplicated weather hours.
- No location-specific weather variation is modeled yet.
- The model-selection training uses sampled rows for laptop performance; the
  baseline is calculated using all earlier rows. Compare metrics accordingly.
- This stage prepares and trains forecasting models. The Streamlit dashboard,
  current station-status API, and rebalancing calculation come next.
