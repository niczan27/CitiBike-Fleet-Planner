#!/usr/bin/env python3
"""Offline, reproducible smoke/consistency test for Citi Bike V1 and V2.3.

Run from repository root: python development/06_validation/verify_one_day_consistency.py
Needs V1 planner_core.py and V2.3 planner_core_v2.py, saved model/data files.
This does not run the browser interface or make network requests.
"""
from __future__ import annotations
import importlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
OUT = HERE / 'last_run_one_day_qa.json'


def compare():
    old = importlib.import_module('planner_core_v1_reference')
    new = importlib.import_module('planner_core_v2')
    required = [
        'development/06_validation/planner_core_v1_reference.py', 'planner_core_v2.py', 'models/departures_model.joblib', 'models/returns_model.joblib',
        'processed_data/weather_hourly_202510_202609.csv', 'processed_data/station_zones.csv',
        'processed_data/nta_station_summary.csv', 'reference_data/station_status_snapshot.json',
    ]
    missing = [p for p in required if not (ROOT/p).is_file()]
    if missing:
        raise FileNotFoundError('Missing: '+', '.join(missing))
    model_dep = joblib.load(ROOT/'models/departures_model.joblib')
    model_ret = joblib.load(ROOT/'models/returns_model.joblib')
    stations = pd.read_csv(ROOT/'processed_data/station_zones.csv', dtype={'station_id': 'string', 'nta_code': 'string'}, keep_default_na=False)
    zones = pd.read_csv(ROOT/'processed_data/nta_station_summary.csv', dtype={'nta_code': 'string'}, keep_default_na=False)
    center = stations.loc[pd.to_numeric(stations['in_nyc'], errors='coerce').eq(1)].groupby('nta_code', as_index=False).agg(
        latitude=('lat', 'mean'), longitude=('lon', 'mean'))
    zones = zones.loc[zones['stations_in_snapshot']>0].merge(center,on='nta_code',how='left')
    history = pd.read_csv(ROOT/'processed_data/weather_hourly_202510_202609.csv')
    history['hour'] = pd.to_datetime(history['hour'])
    # Historical date ensures the comparison is deterministic and works offline.
    sample_day = date(2026, 9, 15)
    weather = history.loc[history['hour'].dt.date == sample_day].copy()
    if len(weather) < 24:
        raise ValueError('Missing weather for the offline sample date')
    forecast_times_old = old.calendar_weather_features(weather, [sample_day])
    forecast_times_new = new.calendar_weather_features(weather, [sample_day])
    pd.testing.assert_frame_equal(forecast_times_old, forecast_times_new)
    first_dep = old.predict_counts(model_dep, forecast_times_old).rename(columns={'predicted': 'departures'})
    first_ret = old.predict_counts(model_ret, forecast_times_old).rename(columns={'predicted': 'returns'})
    second_dep = new.predict_counts(model_dep, forecast_times_new).rename(columns={'predicted': 'departures'})
    second_ret = new.predict_counts(model_ret, forecast_times_new).rename(columns={'predicted': 'returns'})
    pd.testing.assert_frame_equal(first_dep, second_dep)
    pd.testing.assert_frame_equal(first_ret, second_ret)
    predicted = first_dep.merge(first_ret, on=['hour','nta_code','rideable_type'], validate='one_to_one')
    status = json.loads((ROOT/'reference_data/station_status_snapshot.json').read_text(encoding='utf-8'))
    station1, inventory1 = old.station_inventory(stations, status)
    station2, inventory2 = new.station_inventory(stations, status)
    pd.testing.assert_frame_equal(station1, station2)
    pd.testing.assert_frame_equal(inventory1, inventory2)
    targets1, docks1 = old.required_inventory(predicted, inventory1, reserve_fraction=0.05)
    targets2, docks2 = new.required_inventory(predicted, inventory2, reserve_fraction=0.05)
    for c in ['expected_departures','expected_returns','peak_depletion','buffer', 'target_bikes','available_bikes','shortfall','surplus']:
        a, b = (targets1.sort_values(['nta_code','rideable_type'])[c].reset_index(drop=True),
                targets2.sort_values(['nta_code','rideable_type'])[c].reset_index(drop=True))
        pd.testing.assert_series_equal(a, b, check_names=False)
    for c in ['dock_target','dock_shortfall','slots_for_transfers']:
        a,b = (docks1.sort_values('nta_code')[c].reset_index(drop=True),
               docks2.sort_values('nta_code')[c].reset_index(drop=True))
        pd.testing.assert_series_equal(a, b, check_names=False)
    plan1 = old.suggest_transfers(targets1, docks1, zones, budget=500, max_km=4.0, same_borough=True)
    plan2 = new.suggest_transfers(targets2, docks2, zones, budget=500, max_km=4.0, same_borough=True)
    pd.testing.assert_frame_equal(plan1, plan2)
    summary = {
        'comparison_date': str(sample_day),
        'weather_source': 'same stored historical observed weather for both; no live API',
        'station_source': 'same saved GBFS status snapshot for both',
        'model_files': 'same saved departure/return joblib models for both',
        'settings': {'planning_days': 1, 'buffer_pct': 5, 'budget':500,'max_km':4.0,'same_borough':True},
        'hourly_forecasts_equal': True, 'inventory_equal': True,
        'one_day_stock_targets_equal': True, 'one_day_dock_risks_equal': True,
        'relocation_recommendations_equal': True,
        'result_kpis': {
            'departures': round(float(targets1.expected_departures.sum())),
            'returns': round(float(targets1.expected_returns.sum())),
            'bike_need_both_types': int(targets1.shortfall.sum()),
            'dock_risk_zones': int((docks1.dock_shortfall>0).sum()),
            'recommended_moves': int(plan1.Quantity.sum()) if not plan1.empty else 0,
        },
        'important_caveats': [
            'This is a controlled offline one-day comparison, not a full browser UI test.',
            'V2.3 intentionally changed the buffer formula for multi-day planning; multi-day targets can differ.',
            'Live weather forecasts and current bike availability differ over time; do not compare screenshots taken at different times.',
            'Historical map/table display interactions and hosting performance are not tested by this script.',
        ]
    }
    OUT.write_text(json.dumps(summary, indent=2)+'\n',encoding='utf-8')
    print('PASS: same hourly forecasts, inventory, one-day targets, dock risks, and transfer plan for V1/V2.3.')
    print('Wrote:', OUT)
    print('KPI results:', summary['result_kpis'])

if __name__ == '__main__':
    try:
        compare()
    except Exception as exc:
        print('FAILED or could not finish QA:', type(exc).__name__, str(exc))
        raise
