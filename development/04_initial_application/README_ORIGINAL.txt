CITI BIKE STAGE 4 — INTERACTIVE APPLICATION

What you now have
- app.py: two connected dashboards in Streamlit.
- planner_core.py: forecast-input, inventory-risk and transfer calculations.
- reference_data/station_status_snapshot.json: saved Oct 30 snapshot used ONLY if live Citi Bike GBFS is unavailable. A snapshot is not live.

1) Move app.py and planner_core.py into the ROOT of CitiBike_Project,
   next to build_forecasts_stage3_v2.py and prepare_citibike.py.

2) Move ONLY station_status_snapshot.json into the existing
   CitiBike_Project/reference_data/ folder. Do not replace your existing
   station_information_snapshot.json.

3) Open Terminal inside CitiBike_Project. Run:

   .venv/bin/python -m pip install streamlit "plotly>=6" 
   .venv/bin/python -m streamlit run app.py

   Streamlit opens the app at http://localhost:8501 in your browser.

4) Existing local dependencies from Stages 1–3 are automatically reused:
   processed_data/zone_hourly_member_YYYYMM.csv.gz
   processed_data/zone_hourly_YYYYMM.csv.gz
   processed_data/duration_histogram_YYYYMM.csv.gz
   processed_data/station_zones.csv
   processed_data/nta_station_summary.csv
   processed_data/nta_zones.geojson
   processed_data/weather_hourly_202510_202609.csv
   models/departures_model.joblib and returns_model.joblib

5) Suggested presentation demo
   (a) Understand usage: choose Sep 2026, all bike types; change to weekdays.
   (b) Select a neighborhood; show hourly peaks and the zone inventory map.
   (c) Plan availability: 'Tomorrow'; check that station feed and weather are
       LIVE. If not, explain the labelled snapshot / seasonal-weather fallback.
   (d) Adjust transfer budget and highlight the suggested moves.

TROUBLESHOOTING
- 'missing files': verify you placed files at project ROOT, not a nested ZIP folder.
- A live API may reject requests. Fallback is explicitly labelled.
- Streamlit launches locally, not as a public website. Keep Terminal running.
- Do not upload or copy the 12 monthly raw CSV archives to this package.

WHAT IS AND IS NOT IMPLEMENTED
Implemented: historical NTA activity, bike/member filters, day/hour filters,
map, hour/weekday patterns, citywide monthly trend, weather-based live future
predictions, station inventory, deterministic stock/dock risk, constrained
nearby-zone transfer proposals, CSV export, stale-feed notices.

Not implemented: generative LLM assistant, origin-to-destination flows,
calibrated probability intervals, live re-training, van routing, and station-
level verification of feasibility. The manager remains responsible for moves.

MODEL & DATA CAVEATS
- Stage 3 holdout metrics (Aug–Sep 2026) used observed historical weather.
- Each boosted model was fitted on a sample of 200,000 aggregated zone-hour
  observations derived from ~45 million original completed rides.
- Historical matching of NYC station references was ~98.25% departures and
  ~98.24% returns, with retired IDs making up some remaining matches.
- 'nyc_stations: 4822' in Stage 3 report counts aliases; actual mapped stations:
  2,411, across 131 NTAs.
- Stock targets are based on EXPECTED net flows + a user-controlled reserve.
  They are not statistically validated 90% service levels and do not detect
  unmet demand.
- NTA-aggregated docks can hide physical station-level constraints.
- One NY central weather location; transitions to and from daylight-saving
  time use 24 clock hours for consistency with Stage 3 training.

All original files and trained models remain unchanged by the app.
