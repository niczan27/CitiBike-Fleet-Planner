#!/usr/bin/env python3
"""Stage 2: map Citi Bike station locations to NYC's official 2020 NTAs,
and retrieve hourly historical weather for Oct 2025--Sep 2026.

Run from CitiBike_Project, after installing shapely:
  .venv/bin/python -m pip install shapely
  .venv/bin/python build_geography_weather.py

Inputs:
  reference_data/station_information_snapshot.json (bundled)
  2020 NYC NTAs GeoJSON (downloaded automatically, with manual-file fallback)

Outputs in processed_data/:
  station_zones.csv
  nta_zones.geojson
  nta_station_summary.csv
  weather_hourly_202510_202609.csv
  stage2_report.json
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    from shapely.geometry import Point, shape
except ImportError:
    raise SystemExit("Install the mapping dependency: .venv/bin/python -m pip install shapely")

ROOT = Path(__file__).resolve().parent
REF = ROOT / 'reference_data'
OUT = ROOT / 'processed_data'
OUT.mkdir(exist_ok=True)
REF.mkdir(exist_ok=True)
STATION_FILE = REF / 'station_information_snapshot.json'
GEO_FILE = REF / 'nta_2020.geojson'

# NYC Department of City Planning, official 2020 Neighborhood Tabulation Areas:
# https://data.cityofnewyork.us/d/9nt8-h7nd
NTA_URLS = [
    'https://data.cityofnewyork.us/resource/9nt8-h7nd.geojson?%24limit=500',
    'https://data.cityofnewyork.us/api/v3/views/9nt8-h7nd/query.geojson?accessType=DOWNLOAD',
]
WEATHER_URL = 'https://archive-api.open-meteo.com/v1/archive'
START_DATE, END_DATE = '2025-10-01', '2026-09-30'
WEATHER_FIELDS = [
    'temperature_2m', 'apparent_temperature', 'precipitation', 'rain',
    'snowfall', 'snow_depth', 'wind_speed_10m', 'wind_gusts_10m', 'weather_code',
]


def read_url(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={
        'User-Agent': 'CitiBike-UniversityProject/1.0 (public open-data client)',
        'Accept': 'application/json, application/geo+json',
    })
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def prop(p: dict, key: str, default=''):
    """Socrata field names can differ in capitalization across exports."""
    lower = {str(k).lower(): v for k, v in p.items()}
    return lower.get(key.lower(), default)


def get_geojson() -> dict:
    if GEO_FILE.exists():
        print('Using downloaded boundaries:', GEO_FILE)
        return json.loads(GEO_FILE.read_text(encoding='utf-8'))
    errors = []
    for url in NTA_URLS:
        try:
            print('Downloading 2020 NYC NTA boundaries...')
            payload = json.loads(read_url(url))
            if not isinstance(payload.get('features'), list) or len(payload['features']) < 200:
                raise ValueError('Unexpected GeoJSON structure or too few areas')
            GEO_FILE.write_text(json.dumps(payload), encoding='utf-8')
            return payload
        except (urllib.error.URLError, ValueError, KeyError, TimeoutError) as exc:
            errors.append(str(exc))
    raise RuntimeError(
        'Could not download official NYC NTA geography. Manually download GeoJSON '
        'from https://data.cityofnewyork.us/d/9nt8-h7nd and save it as '
        'reference_data/nta_2020.geojson. Errors: ' + '; '.join(errors)
    )


def prepare_zones() -> dict:
    if not STATION_FILE.exists():
        raise FileNotFoundError('Missing reference_data/station_information_snapshot.json')
    stations = json.loads(STATION_FILE.read_text(encoding='utf-8'))['data']['stations']
    geojson = get_geojson()
    features = geojson['features']
    areas = []
    for f in features:
        props = f.get('properties', {})
        code = str(prop(props, 'nta2020'))
        name = str(prop(props, 'ntaname'))
        borough = str(prop(props, 'boroname'))
        cdta = str(prop(props, 'cdta2020'))
        ntype = str(prop(props, 'ntatype'))
        geom = shape(f['geometry'])
        if not geom.is_empty:
            areas.append((geom, code, name, borough, cdta, ntype))
    if len(areas) < 200:
        raise ValueError(f'Expected 200+ official NTAs; found {len(areas)}')

    station_rows = []
    for s in stations:
        point = Point(float(s['lon']), float(s['lat']))
        matches = [a for a in areas if a[0].covers(point)]
        # Stations genuinely outside NYC (Jersey City, Hoboken) must remain unmatched.
        match = matches[0] if matches else None
        station_rows.append({
            'station_id': str(s['station_id']),
            'station_name': s['name'],
            'lat': s['lat'], 'lon': s['lon'],
            'region_id': s.get('region_id', ''),
            'capacity_snapshot': s.get('capacity', ''),
            'in_nyc': int(match is not None),
            'nta_code': match[1] if match else '',
            'nta_name': match[2] if match else '',
            'borough': match[3] if match else '',
            'cdta_code': match[4] if match else '',
            'nta_type': match[5] if match else '',
        })

    outpath = OUT / 'station_zones.csv'
    with outpath.open('w', newline='', encoding='utf-8') as o:
        wr = csv.DictWriter(o, fieldnames=station_rows[0].keys())
        wr.writeheader()
        wr.writerows(station_rows)

    # Preserve all official polygons (including zones with zero stations),
    # allowing a consistent background map in the application.
    (OUT / 'nta_zones.geojson').write_text(json.dumps(geojson), encoding='utf-8')
    count = Counter(r['nta_code'] for r in station_rows if r['in_nyc'])
    boroughs = {}
    summary = []
    for geom, code, name, borough, cdta, ntype in areas:
        if code in boroughs:
            continue
        boroughs[code] = borough
        summary.append({'nta_code': code, 'nta_name': name, 'borough': borough,
                        'cdta_code': cdta, 'nta_type': ntype,
                        'stations_in_snapshot': count[code]})
    with (OUT / 'nta_station_summary.csv').open('w', newline='', encoding='utf-8') as o:
        wr = csv.DictWriter(o, fieldnames=summary[0].keys())
        wr.writeheader()
        wr.writerows(sorted(summary, key=lambda x: (x['borough'], x['nta_code'])))

    matched = sum(r['in_nyc'] for r in station_rows)
    print(f'Station snapshot: {len(stations)} total; {matched} inside NYC; '
          f'{len(stations)-matched} outside NYC or unmatched.')
    print(f'NTAs with stations: {sum(v > 0 for v in count.values())}; '
          f'NTAs with 1-2 stations: {sum(0 < v < 3 for v in count.values())}.')
    return {
        'source_station_count': len(stations),
        'matched_nyc_station_count': matched,
        'excluded_or_unmatched_station_count': len(stations) - matched,
        'official_nta_polygon_count': len(areas),
        'nta_with_stations': sum(v > 0 for v in count.values()),
        'nta_with_1_or_2_stations': sum(0 < v < 3 for v in count.values()),
        'note': 'Snapshot coverage is not the same as historical station-ID coverage. Audit historical matches later.',
    }


def prepare_weather() -> dict:
    out = OUT / 'weather_hourly_202510_202609.csv'
    if out.exists() and out.stat().st_size > 300_000:
        print('Historical weather file exists, using it:', out)
        with out.open(encoding='utf-8') as f:
            rows = sum(1 for _ in f)-1
        return {'weather_rows': rows, 'weather_file': out.name, 'reused_cache': True}
    args = urllib.parse.urlencode({
        'latitude': 40.744, 'longitude': -73.978,
        'start_date': START_DATE, 'end_date': END_DATE,
        'hourly': ','.join(WEATHER_FIELDS),
        'timezone': 'America/New_York',
        'temperature_unit': 'celsius', 'wind_speed_unit': 'kmh',
        'precipitation_unit': 'mm',
    })
    print('Downloading NYC historical weather (hourly, full 12 months)...')
    response = json.loads(read_url(WEATHER_URL + '?' + args, timeout=180))
    if response.get('error'):
        raise RuntimeError('Open-Meteo returned: ' + str(response.get('reason')))
    h = response.get('hourly', {})
    t = h.get('time', [])
    if len(t) < 8500:
        raise RuntimeError('Weather response has too few hourly observations; got ' + str(len(t)))
    for field in WEATHER_FIELDS:
        if field not in h or len(h[field]) != len(t):
            raise RuntimeError(f'Missing or incomplete weather field: {field}')
    with out.open('w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['hour'] + WEATHER_FIELDS)
        for i, hour in enumerate(t):
            wr.writerow([hour.replace('T', ' ') + ':00'] + [h[k][i] for k in WEATHER_FIELDS])
    print(f'Saved {len(t)} hourly weather observations to', out)
    return {
        'weather_rows': len(t), 'weather_file': out.name,
        'weather_coordinates': [40.744, -73.978],
        'weather_timezone': 'America/New_York',
        'weather_source': WEATHER_URL,
        'weather_range': [START_DATE, END_DATE],
    }


def main() -> None:
    print('Stage 2: geography and weather')
    zones = prepare_zones()
    weather = prepare_weather()
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'nyc_boundary_dataset': '2020 Neighborhood Tabulation Areas (NYC DCP, Socrata 9nt8-h7nd)',
        'geography': zones,
        'weather': weather,
        'cautions': [
            'NTA boundaries are statistical zones, not bike-share service territories.',
            'Historical retired/unmatched stations may require coordinate-based recovery.',
            'Weather represents one central NYC coordinate, not microclimate variation by borough.',
            'DST repeated 1am hour remains ambiguous in initial historical trip aggregates.',
            'No trip forecasting models have been trained by this script.',
        ],
    }
    (OUT / 'stage2_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Done. Inspect processed_data/stage2_report.json and nta_station_summary.csv.')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('ERROR:', str(e), file=sys.stderr)
        raise SystemExit(1)
