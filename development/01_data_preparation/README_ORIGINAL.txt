CITI BIKE — FIRST DATA PREPARATION STEP
=======================================
This is a local, month-by-month processing script for the Citi Bike project.
It reads multiple CSV files from each monthly folder and creates compact,
compressed hourly summary files. The original ride files are not changed.

1) On your Mac, create one folder named CitiBike_Project.

2) Put prepare_citibike.py inside CitiBike_Project.

3) Create a folder named raw_data inside CitiBike_Project.
   Inside raw_data, place the 12 UNZIPPED monthly Citi Bike folders.
   Do not manually combine their CSV parts.
   Do not place the combined 2 GB ZIP in raw_data as a substitute for CSVs.
   The script searches subfolders recursively for the individual CSVs.

   Expected structure:

   CitiBike_Project/
     prepare_citibike.py
     raw_data/
       202510-citibike-tripdata/
         ...CSV parts...
       202511-citibike-tripdata/
         ...CSV parts...
       ...
       202609-citibike-tripdata/
         ...CSV parts...

4) Open Terminal. Type 'cd ' (with a space), drag the CitiBike_Project
   folder from Finder into Terminal, then press Return.

5) If Python 3 is already installed, set up a small dedicated environment:

   python3 -m venv .venv
   .venv/bin/python -m pip install pandas

6) Run the processor:

   .venv/bin/python prepare_citibike.py

7) The script creates processed_data/ with, for each month:

   departures_YYYYMM.csv.gz         Station-hour departures by bike type/member
   returns_YYYYMM.csv.gz            Station-hour returns by bike type/member
   daily_usage_YYYYMM.csv.gz        Daily rides by bike type/member
   duration_histogram_YYYYMM.csv.gz Approximate trip-duration distribution

   It also creates quality_summary.csv and manifest.json.

8) Check quality_summary.csv, especially csv_files, valid_trips,
   invalid timestamps/durations, and missing station IDs.
   These files are the FIRST preparation outputs, not the finished model.
   The next step is matching station IDs, assigning NYC operating zones,
   adding hourly weather/calendar features, and training the models.

Important:
- Do not include files starting with JC; they are from Jersey City.
- Do not delete the original monthly CSV files.
- The script expects the usual Citi Bike 2025–2026 trip-data columns.
- No weather download or historical availability snapshots are needed to
  run this first stage.
- Exact duplicate ride IDs across files are not yet removed; this will be
  audited before final training.
- Local clock timestamps cannot distinguish repeated 1 a.m. DST hours;
  this will be addressed in the forecasting data preparation stage.
