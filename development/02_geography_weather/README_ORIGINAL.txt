CITI BIKE PROJECT - STAGE 2: OFFICIAL NYC GEOGRAPHY + HISTORICAL WEATHER

WHAT TO COPY
Unzip CitiBike_stage2.zip directly into your CitiBike_Project folder.
The contents must be:
  CitiBike_Project/build_geography_weather.py
  CitiBike_Project/reference_data/station_information_snapshot.json

Your existing raw_data/, processed_data/, and prepare_citibike.py stay untouched.

HOW TO RUN
In Terminal (from inside CitiBike_Project), enter these commands:

  .venv/bin/python -m pip install shapely
  .venv/bin/python build_geography_weather.py

The script downloads official 2020 NYC Neighborhood Tabulation Areas (NTAs)
from NYC Open Data and hourly historical weather from Open-Meteo.
You need an internet connection. No API key is required.

WHAT IS CREATED
  processed_data/station_zones.csv
    Station ID -> official NTA, borough, community district area, coordinates.
    Stations outside NYC are marked in_nyc=0.
  processed_data/nta_zones.geojson
    Polygons for mapping every official NTA (including areas with no stations).
  processed_data/nta_station_summary.csv
    Number of snapshot stations in each neighborhood.
  processed_data/weather_hourly_202510_202609.csv
    Hourly precipitation, temperature, wind, snow, etc. Oct 2025-Sep 2026.
  processed_data/stage2_report.json
    Quick audit of area / weather coverage.

IMPORTANT
This does NOT yet train a model or merge hourly trips into geographic zones.
Some historic station IDs may no longer appear in the current snapshot. We will
check this separately and recover historical locations as needed.

If the boundary API is temporarily unavailable, manually download the GeoJSON
from https://data.cityofnewyork.us/d/9nt8-h7nd and save it as
reference_data/nta_2020.geojson, then rerun the script.

The weather is initially measured at one representative NYC coordinate, to
keep the first version straightforward. Forecast weather uses a separate live
Open-Meteo endpoint in the final application.
