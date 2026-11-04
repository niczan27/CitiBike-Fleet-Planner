#!/usr/bin/env python3
"""Create a fixed Oct 2025--Sep 2026 daily weather/ridership summary.

Requires only the portable package's compact zone-hourly files and weather CSV.
Rebuild with: python3 scripts/prepare_weather_charts.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'processed_data'

def build() -> pd.DataFrame:
    weather = pd.read_csv(DATA / 'weather_hourly_202510_202609.csv', parse_dates=['hour'])
    weather['day'] = weather['hour'].dt.normalize()
    daily_weather = weather.groupby('day', as_index=False).agg(
        temperature_max_c=('temperature_2m', 'max'),
        rain_mm=('rain', 'sum'), snowfall_cm=('snowfall', 'sum'),
        wind_max_kmh=('wind_speed_10m', 'max'),
    )
    daily_trips = []
    for file in sorted(DATA.glob('zone_hourly_20????.csv.gz')):
        trips = pd.read_csv(file, usecols=['hour', 'departures'], compression='gzip')
        trips['day'] = pd.to_datetime(trips['hour'].str[:10], errors='coerce')
        daily_trips.append(trips.groupby('day', as_index=False)['departures'].sum())
    if not daily_trips:
        raise FileNotFoundError('No compact zone_hourly_YYYYMM.csv.gz files found')
    counts = pd.concat(daily_trips, ignore_index=True).groupby('day', as_index=False)['departures'].sum()
    merged = daily_weather.merge(counts, on='day', how='inner', validate='one_to_one').sort_values('day')
    merged['weekday'] = merged['day'].dt.dayofweek
    if len(merged) < 300:
        raise ValueError(f'Unexpectedly few matched ride/weather dates: {len(merged)}')
    out = DATA / 'weather_sensitivity_daily.csv'
    merged.to_csv(out, index=False, float_format='%.3f')
    print(f'Created {out.relative_to(ROOT)}: {len(merged)} days, total {merged.departures.sum():,} recorded NYC departures')
    return merged

if __name__ == '__main__':
    build()
