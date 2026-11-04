#!/usr/bin/env python3
"""Citi Bike Fleet Availability Planner — locally hosted Streamlit app.

Run from the project folder: .venv/bin/python -m streamlit run app.py
Requires finished Stage 1–3 pipeline artifacts in ./processed_data and ./models.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from planner_core import (
    WEATHER_FIELDS, BIKE_TYPES, calendar_weather_features, predict_counts,
    station_inventory, required_inventory, suggest_transfers,
)

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / 'processed_data'
MODELS = ROOT / 'models'
REF = ROOT / 'reference_data'
GBFS_DISCOVERY = 'https://gbfs.citibikenyc.com/gbfs/2.3/gbfs.json'
NY_TZ = ZoneInfo('America/New_York')
BIKE_LABELS = {'classic_bike': 'Classic', 'electric_bike': 'Electric'}

st.set_page_config(page_title='Citi Bike | Fleet Availability Planner', page_icon='🚲', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container{padding-top:1.4rem;padding-bottom:2rem;max-width:1600px}
[data-testid="stMetricValue"]{font-size:1.7rem}
[data-testid="stSidebar"]{background:#f7f9fb}
h1,h2,h3{letter-spacing:-.02em}
</style>''', unsafe_allow_html=True)


def fail_if_missing():
    required = [
        PROCESSED / 'station_zones.csv', PROCESSED / 'nta_zones.geojson',
        PROCESSED / 'nta_station_summary.csv', PROCESSED / 'weather_hourly_202510_202609.csv',
        MODELS / 'departures_model.joblib', MODELS / 'returns_model.joblib',
        REF / 'station_status_snapshot.json',
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing:
        st.error('Required files are missing from CitiBike_Project: ' + ', '.join(missing))
        st.info('Keep the Stage 1–3 processed_data/ and models/ folders. Move the new Stage 4 files into the project root, preserving the reference_data/ folder.')
        st.stop()


@st.cache_data(show_spinner=False)
def read_geography():
    stations = pd.read_csv(PROCESSED / 'station_zones.csv', dtype={'station_id': 'string', 'nta_code': 'string'}, keep_default_na=False)
    zones = pd.read_csv(PROCESSED / 'nta_station_summary.csv', dtype={'nta_code': 'string'}, keep_default_na=False)
    zones = zones.loc[zones['stations_in_snapshot'] > 0].copy()
    geo = json.loads((PROCESSED / 'nta_zones.geojson').read_text(encoding='utf-8'))
    center = stations.loc[pd.to_numeric(stations['in_nyc'], errors='coerce').eq(1)].groupby('nta_code', as_index=False).agg(
        latitude=('lat', 'mean'), longitude=('lon', 'mean'))
    zones = zones.merge(center, on='nta_code', how='left')
    return stations, zones, geo


@st.cache_resource(show_spinner=False)
def get_models():
    return joblib.load(MODELS / 'departures_model.joblib'), joblib.load(MODELS / 'returns_model.joblib')


@st.cache_data(show_spinner=False)
def load_weather_history():
    return pd.read_csv(PROCESSED / 'weather_hourly_202510_202609.csv')


@st.cache_data(show_spinner='Loading selected historical months...', max_entries=3)
def load_history(months: tuple[str, ...]):
    parts = []
    cols = ['hour', 'nta_code', 'rideable_type', 'member_casual', 'departures', 'returns']
    for m in months:
        path = PROCESSED / f'zone_hourly_member_{m}.csv.gz'
        if not path.exists():
            raise FileNotFoundError(path.name)
        part = pd.read_csv(path, usecols=cols, dtype={
            'hour': 'string', 'nta_code': 'string', 'rideable_type': 'category', 'member_casual': 'category',
            'departures': 'int32', 'returns': 'int32'}, compression='gzip')
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)


@st.cache_data(show_spinner=False)
def monthly_overview():
    result = []
    for path in sorted(PROCESSED.glob('zone_hourly_20????.csv.gz')):
        name = path.stem.split('.')[0].rsplit('_', 1)[-1]
        if len(name) != 6:
            continue
        df = pd.read_csv(path, usecols=['rideable_type', 'departures', 'returns'], compression='gzip')
        agg = df.groupby('rideable_type')[['departures', 'returns']].sum()
        for bike, r in agg.iterrows():
            result.append({'month': f'{name[:4]}-{name[4:]}', 'rideable_type': BIKE_LABELS.get(bike, bike),
                           'departures': int(r['departures']), 'returns': int(r['returns'])})
    return pd.DataFrame(result)


@st.cache_data(show_spinner=False)
def median_duration(months: tuple[str, ...], bike: str, member: str):
    pieces = []
    for m in months:
        path = PROCESSED / f'duration_histogram_{m}.csv.gz'
        if path.exists():
            pieces.append(pd.read_csv(path, compression='gzip'))
    if not pieces:
        return None
    d = pd.concat(pieces, ignore_index=True)
    if bike != 'Both':
        d = d.loc[d['rideable_type'].eq(bike)]
    if member != 'All':
        d = d.loc[d['member_casual'].eq(member)]
    if d.empty:
        return None
    d = d.groupby('duration_minutes_rounded', as_index=False)['rides'].sum().sort_values('duration_minutes_rounded')
    mid = d['rides'].sum() / 2
    return int(d.loc[d['rides'].cumsum().ge(mid), 'duration_minutes_rounded'].iloc[0])


def api_json(url: str, timeout: int = 12) -> dict:
    req = urllib.request.Request(url, headers={
        'User-Agent': 'CitiBike-Academic-FleetPlanner/1.0', 'Accept': 'application/json',
    })
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


@st.cache_data(ttl=120, show_spinner=False)
def live_station_status():
    """Try official discovery first; fall back explicitly to saved reference data."""
    try:
        d = api_json(GBFS_DISCOVERY)
        langs = d.get('data', {})
        feeds = langs.get('en', {}).get('feeds') or next(
            (val.get('feeds') for val in langs.values() if isinstance(val, dict) and 'feeds' in val), [])
        matches = [f for f in feeds if f.get('name') == 'station_status']
        if not matches:
            raise ValueError('Discovery feed did not list station_status')
        status = api_json(matches[0]['url'])
        if not status.get('data', {}).get('stations'):
            raise ValueError('No station records in live response')
        updated = int(status.get('last_updated', 0) or 0)
        age_minutes = (datetime.now(timezone.utc).timestamp() - updated) / 60 if updated else None
        is_recent = age_minutes is not None and age_minutes <= 5
        return status, ('Live feed' if is_recent else 'Feed responded but last_updated may be stale'), updated, is_recent
    except Exception as exc:
        fallback = json.loads((REF / 'station_status_snapshot.json').read_text(encoding='utf-8'))
        return fallback, f'Saved snapshot — live API unavailable ({type(exc).__name__})', int(fallback.get('last_updated', 0) or 0), False


@st.cache_data(ttl=900, show_spinner=False)
def weather_for_dates(days: tuple[str, ...]):
    """Try real forecast; if unavailable, use historical month/hour averages, labelled."""
    requested = [date.fromisoformat(x) for x in days]
    params = urllib.parse.urlencode({
        'latitude': 40.744, 'longitude': -73.978,
        'hourly': ','.join(WEATHER_FIELDS), 'timezone': 'America/New_York',
        'start_date': requested[0].isoformat(), 'end_date': requested[-1].isoformat(),
        'wind_speed_unit': 'kmh', 'temperature_unit': 'celsius', 'precipitation_unit': 'mm',
    })
    try:
        payload = api_json('https://api.open-meteo.com/v1/forecast?' + params, timeout=18)
        if payload.get('error'):
            raise ValueError(payload.get('reason', 'Forecast API error'))
        hourly = payload['hourly']
        df = pd.DataFrame({'hour': hourly['time']})
        for field in WEATHER_FIELDS:
            df[field] = hourly[field]
        df['hour'] = pd.to_datetime(df['hour'])
        for field in WEATHER_FIELDS:
            df[field] = pd.to_numeric(df[field], errors='coerce')
        # For a partially unavailable forecast field, use a clearly disclosed
        # climatological fallback instead of training on NaNs.
        if df[WEATHER_FIELDS].isna().any().any():
            hist = load_weather_history()
            hist['hour'] = pd.to_datetime(hist['hour'])
            means = hist.groupby([hist['hour'].dt.month, hist['hour'].dt.hour])[WEATHER_FIELDS].mean()
            for i in range(len(df)):
                row = df.iloc[i]
                for field in WEATHER_FIELDS:
                    if pd.isna(df.at[i, field]):
                        df.at[i, field] = float(means.loc[(row['hour'].month, row['hour'].hour), field])
            return df, 'Open-Meteo forecast (missing fields estimated from historical month/hour averages)'
        return df, 'Open-Meteo forecast'
    except Exception as exc:
        hist = load_weather_history()
        hist['hour'] = pd.to_datetime(hist['hour'])
        grouped = hist.groupby([hist['hour'].dt.month, hist['hour'].dt.hour])[WEATHER_FIELDS].mean()
        hours = pd.date_range(requested[0].isoformat(), periods=24 * len(requested), freq='h')
        rows = []
        for hr in hours:
            key = (hr.month, hr.hour)
            # A past year can have data in the same month; otherwise use overall mean by hour.
            if key in grouped.index:
                vals = grouped.loc[key]
            else:
                vals = hist.loc[hist['hour'].dt.hour.eq(hr.hour), WEATHER_FIELDS].mean()
            rows.append({'hour': hr, **{f: float(vals[f]) for f in WEATHER_FIELDS}})
        return pd.DataFrame(rows), f'Seasonal-weather fallback — NOT a live forecast ({type(exc).__name__})'


def plot_zones(geo, df, field, title, selected_zone=None, diverging=False):
    mapped = df.copy()
    vals = pd.to_numeric(mapped[field], errors='coerce').fillna(0)
    mapped[field] = vals
    fig = px.choropleth_map(
        mapped, geojson=geo, locations='nta_code', featureidkey='properties.nta2020',
        color=field, hover_name='nta_name', hover_data={'nta_code': True, field: ':.1f'},
        color_continuous_scale='RdBu' if diverging else 'YlGnBu',
        color_continuous_midpoint=0 if diverging else None,
        center={'lat': 40.735, 'lon': -73.95}, zoom=10, opacity=.75,
        map_style='carto-positron', height=555,
    )
    fig.update_traces(marker_line_width=.45, marker_line_color='#4b5563')
    fig.update_layout(margin=dict(l=0, r=0, t=26, b=0), title=title,
                      coloraxis_colorbar=dict(title='', thickness=12))
    return fig


def all_months():
    return pd.period_range('2025-10', '2026-09', freq='M').strftime('%Y%m').tolist()


def historical_tab(stations, zones, geo):
    st.subheader('Understand usage')
    st.caption('Observed completed Citi Bike trips · NYC zones · October 2025 – September 2026')
    a, b, c, d = st.columns([1.6, 1.2, 1, 1.2])
    with a:
        range_value = st.date_input('Historical date range', (date(2026, 9, 1), date(2026, 9, 30)),
                                    min_value=date(2025, 10, 1), max_value=date(2026, 9, 30))
    with b:
        bike_option = st.selectbox('Bicycle', ['Both', 'classic_bike', 'electric_bike'],
                                   format_func=lambda v: 'Both types' if v == 'Both' else BIKE_LABELS[v], key='usage_bike')
    with c:
        member_option = st.selectbox('Rider category', ['All', 'member', 'casual', 'unknown'], key='usage_member')
    with d:
        day_filter = st.selectbox('Days', ['All days', 'Weekdays', 'Weekends'], key='usage_weekdays')
    tcol1, tcol2 = st.columns([2, 3])
    with tcol1:
        hour_window = st.slider('Hour of day (New York local)', 0, 23, (0, 23), key='usage_hours')
    with tcol2:
        zone_options = {'Citywide': 'All zones'}
        for row in zones.itertuples():
            zone_options[f'{row.nta_name} · {row.borough} ({row.nta_code})'] = row.nta_code
        choice = st.selectbox('Inspect a zone', list(zone_options), key='usage_zone')
        selected_code = zone_options[choice]
    if not isinstance(range_value, (tuple, list)) or len(range_value) != 2:
        st.info('Select both a start and an end date.')
        return
    begin, end = range_value
    months = tuple(p.strftime('%Y%m') for p in pd.period_range(begin.strftime('%Y-%m'), end.strftime('%Y-%m'), freq='M'))
    with st.spinner('Reading saved, compact historical activity...'):
        history = load_history(months)
    history = history.loc[history['hour'].str[:10].between(begin.isoformat(), end.isoformat())].copy()
    if history.empty:
        st.warning('No recorded activity for that selection.')
        return
    history['hour_of_day'] = history['hour'].str[11:13].astype(int)
    history = history.loc[history['hour_of_day'].between(*hour_window)].copy()
    if bike_option != 'Both':
        history = history.loc[history['rideable_type'].eq(bike_option)]
    if member_option != 'All':
        history = history.loc[history['member_casual'].eq(member_option)]
    history['datetime'] = pd.to_datetime(history['hour'])
    history['weekday'] = history['datetime'].dt.dayofweek
    if day_filter == 'Weekdays':
        history = history.loc[history['weekday'] < 5]
    elif day_filter == 'Weekends':
        history = history.loc[history['weekday'] >= 5]
    if history.empty:
        st.warning('No recorded activity for that selection.')
        return
    zone_totals = history.groupby('nta_code', observed=True)[['departures', 'returns']].sum().reset_index()
    if selected_code != 'All zones':
        focused = history.loc[history['nta_code'].eq(selected_code)]
    else:
        focused = history
    total_dep, total_ret = int(focused['departures'].sum()), int(focused['returns'].sum())
    n_days = max(1, int(focused['datetime'].dt.normalize().nunique()))
    # Daily averages use all calendar days in the date window when unfiltered.
    if day_filter == 'All days':
        n_days = max(1, (end-begin).days+1)
    else:
        calendar = pd.date_range(begin, end, freq='D')
        n_days = max(1, int(sum(calendar.dayofweek < 5) if day_filter == 'Weekdays' else sum(calendar.dayofweek >= 5)))
    median = median_duration(months, bike_option, member_option) if selected_code == 'All zones' else None
    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Departures', f'{total_dep:,}')
    k2.metric('Avg departures / selected day', f'{total_dep / n_days:,.0f}')
    k3.metric('Net outflow', f'{total_dep-total_ret:+,}')
    k4.metric('Approx. ride duration', f'{median} min' if median is not None else 'Not available')
    st.caption('Duration is a rounded, network-wide estimate for the selected whole months and bike/member filters; it is not available by zone, hour or partial-month range.')
    display = zones[['nta_code', 'nta_name', 'borough']].merge(zone_totals, on='nta_code', how='left').fillna({'departures':0,'returns':0})
    display['net_outflow'] = display['departures'] - display['returns']
    display[['departures','returns']] = display[['departures','returns']].astype(float)
    left, right = st.columns([1.6, 1])
    with left:
        metric = st.radio('Map metric', ['departures', 'returns', 'net_outflow'],
                          format_func=lambda s: {'departures':'Departures','returns':'Returns','net_outflow':'Net outflow'}[s],
                          horizontal=True, key='hist_map_metric')
        st.plotly_chart(plot_zones(geo, display, metric, 'Historical ride activity by neighborhood', diverging=(metric == 'net_outflow')), use_container_width=True)
        st.caption('Map areas are official 2020 NTAs. White/empty areas have no mapped station activity. Click-free zone selection is above the map.')
    with right:
        st.markdown('**Highest-activity neighborhoods**')
        leaders = display.sort_values('departures', ascending=False).head(12)
        st.dataframe(leaders[['nta_name', 'borough', 'departures', 'returns']].rename(columns={
            'nta_name':'Neighborhood','borough':'Borough','departures':'Departures','returns':'Returns'}),
            hide_index=True, use_container_width=True, height=390)
        if selected_code != 'All zones':
            st.caption(f'Selected neighborhood: {choice}')
    col_a, col_b = st.columns(2)
    with col_a:
        profile = focused.groupby('hour_of_day', observed=True)[['departures','returns']].sum().reset_index()
        fig = px.line(profile, x='hour_of_day', y=['departures','returns'], markers=True,
                      labels={'value':'Trips','hour_of_day':'Hour','variable':'Flow'}, title='Hourly profile')
        fig.update_layout(legend_orientation='h', margin=dict(l=0,r=0,t=44,b=5), height=330)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        heat = focused.groupby(['weekday','hour_of_day'], observed=True)['departures'].sum().reset_index()
        mat = heat.pivot(index='weekday', columns='hour_of_day', values='departures').reindex(range(7)).fillna(0)
        mat.index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        fig = px.imshow(mat, labels={'x':'Hour of day','y':'Day','color':'Departures'},
                        aspect='auto', color_continuous_scale='Blues', title='Departure heatmap')
        fig.update_layout(margin=dict(l=0,r=0,t=44,b=5), height=330)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('**Monthly trend (all NYC zones and all rider categories)**')
    overall = monthly_overview()
    if not overall.empty:
        fig = px.bar(overall, x='month', y='departures', color='rideable_type', barmode='stack',
                     labels={'month':'Month','departures':'Departures','rideable_type':'Bicycle'},
                     color_discrete_map={'Classic':'#1d7997','Electric':'#e0a23c'})
        fig.update_layout(margin=dict(l=0,r=0,t=12,b=10), height=290)
        st.plotly_chart(fig, use_container_width=True)
    if selected_code != 'All zones':
        st.markdown('**Stations within the selected neighborhood**')
        subset = stations.loc[(stations['nta_code'] == selected_code) & (pd.to_numeric(stations['in_nyc'], errors='coerce') == 1)]
        st.dataframe(subset[['station_name', 'station_id', 'capacity_snapshot']].rename(columns={
            'station_name':'Station','station_id':'Station ID','capacity_snapshot':'Snapshot capacity'}),
            hide_index=True, use_container_width=True, height=230)
    st.caption('Historical data represent completed rides, not unfulfilled rental attempts or true historical fleet utilization.')


def planning_tab(stations, zones, geo):
    st.subheader('Plan availability')
    st.caption('Forecast → compare with current station allocation → review relocation proposals')
    ny_today = datetime.now(NY_TZ).date()
    tomorrow = ny_today + timedelta(days=1)
    lead, modecol, sizecol = st.columns([1.2, 1, 1.2])
    with lead:
        horizon = st.selectbox('Planning horizon', ['Tomorrow', 'Next 2 days', 'Next 7 days', 'Custom day'])
    with modecol:
        bike_option = st.selectbox('Prioritize bicycle', ['Both', 'classic_bike', 'electric_bike'],
                                   format_func=lambda v: 'Both types' if v == 'Both' else BIKE_LABELS[v], key='plan_bike')
    with sizecol:
        reserve_percent = st.slider('Planning buffer (% of daily departures)', min_value=0, max_value=20, value=5, step=1)
    if horizon == 'Custom day':
        chosen = st.date_input('Date to evaluate', tomorrow, min_value=tomorrow, max_value=ny_today+timedelta(days=7))
        days = [chosen]
    else:
        window = {'Tomorrow':1, 'Next 2 days':2, 'Next 7 days':7}[horizon]
        days = [tomorrow+timedelta(days=i) for i in range(window)]
    dates_txt = tuple(x.isoformat() for x in days)
    with st.spinner('Loading forecasts and station availability...'):
        model_dep, model_ret = get_models()
        if model_dep['zone_codes'] != model_ret['zone_codes'] or model_dep['bike_types'] != model_ret['bike_types']:
            st.error('Departure and return models disagree about zone or bicycle type coding.')
            st.stop()
        wraw, weather_source = weather_for_dates(dates_txt)
        inputs = calendar_weather_features(wraw, days)
        departures = predict_counts(model_dep, inputs).rename(columns={'predicted':'departures'})
        returns = predict_counts(model_ret, inputs).rename(columns={'predicted':'returns'})
        prediction = departures.merge(returns, on=['hour','nta_code','rideable_type'], validate='one_to_one')
        status, status_source, updated, live_flag = live_station_status()
        station_status, inventory = station_inventory(stations, status)
    status_text = (datetime.fromtimestamp(updated, NY_TZ).strftime('%b %d, %Y %H:%M NY') if updated > 1000000000 else 'Unknown timestamp')
    if not live_flag:
        st.warning(f'Availability is not confirmed live. Source: {status_source}. Feed timestamp: {status_text}. Results are **illustrative what-if scenarios**, not dispatch instructions.')
    else:
        st.success(f'Current station feed loaded · {status_text}')
    st.caption(f'Weather source: {weather_source}. Prediction period: {days[0]} through {days[-1]}. All times are New York local clock hours.')
    if 'fallback' in weather_source.lower():
        st.warning('Live weather forecast could not be retrieved. Estimated seasonal weather is being used; the forecasts are demonstrative.')
    if st.button('Refresh station availability and weather'):
        live_station_status.clear()
        weather_for_dates.clear()
        st.rerun()
    targets, dock_needs = required_inventory(prediction, inventory, reserve_fraction=reserve_percent/100)
    target_map = targets.groupby('nta_code', as_index=False).agg(shortfall=('shortfall','sum'), surplus=('surplus','sum'),
                          departures=('expected_departures','sum'), returns=('expected_returns','sum'))
    merged = zones.merge(target_map, on='nta_code', how='left').merge(dock_needs[['nta_code','dock_shortfall','slots_for_transfers']], on='nta_code', how='left')
    for col in ['shortfall','surplus','departures','returns','dock_shortfall','slots_for_transfers']:
        merged[col] = merged[col].fillna(0)
    selection = targets if bike_option == 'Both' else targets.loc[targets['rideable_type'].eq(bike_option)]
    chosen_needs = selection.groupby('nta_code', as_index=False).agg(shortfall=('shortfall','sum'), surplus=('surplus','sum'),
                              departures=('expected_departures','sum'), returns=('expected_returns','sum'))
    merged = merged.drop(columns=['shortfall','surplus','departures','returns']).merge(chosen_needs, on='nta_code', how='left')
    for col in ['shortfall','surplus','departures','returns']:
        merged[col] = merged[col].fillna(0)
    a,b,c,d = st.columns(4)
    a.metric('Expected departures', f'{selection.expected_departures.sum():,.0f}')
    b.metric('Expected returns', f'{selection.expected_returns.sum():,.0f}')
    c.metric('Estimated bike shortfall', f'{selection.shortfall.sum():,.0f}')
    d.metric('Areas with dock-space risk', f'{int((dock_needs.dock_shortfall>0).sum()):,}')
    st.caption('Shortfalls and dock risk are deterministic net-flow scenarios under a selected buffer, **not measured lost customers or calibrated probabilities**.')
    map_col, table_col = st.columns([1.5,1])
    with map_col:
        metric = st.radio('Map view', ['shortfall','surplus','dock_shortfall'],
                          format_func=lambda k: {'shortfall':'Bicycle shortfall','surplus':'Transferable bicycle surplus','dock_shortfall':'Dock-space risk'}[k],
                          horizontal=True)
        st.plotly_chart(plot_zones(geo, merged, metric, 'Expected inventory pressure by neighborhood'), use_container_width=True)
    with table_col:
        st.markdown('**Priority neighborhoods**')
        shortlist = merged.sort_values('shortfall', ascending=False).head(20)
        st.dataframe(shortlist[['nta_name','borough','shortfall','dock_shortfall']].rename(columns={
            'nta_name':'Neighborhood','borough':'Borough','shortfall':'Bike need','dock_shortfall':'Dock risk'}),
            hide_index=True, use_container_width=True, height=375)
        sel = st.selectbox('Inspect forecast trajectory',
                           options=merged['nta_code'].tolist(),
                           format_func=lambda code: f"{merged.loc[merged.nta_code.eq(code),'nta_name'].iloc[0]} · {code}")
        type_sel = st.selectbox('Bike for trajectory', BIKE_TYPES, format_func=lambda t: BIKE_LABELS[t])
        trace = prediction.loc[prediction['nta_code'].eq(sel) & prediction['rideable_type'].eq(type_sel)].copy()
        trace['cumulative_outflow'] = (trace['departures']-trace['returns']).cumsum()
        start_count = int(inventory.loc[inventory['nta_code'].eq(sel), 'classic' if type_sel=='classic_bike' else 'electric'].sum())
        trace['projected_bikes'] = start_count-trace['cumulative_outflow']
        fig = px.line(trace, x='hour', y='projected_bikes', title='Projected stock (no transfers)', labels={'hour':'Hour','projected_bikes':'Bikes'})
        fig.add_hline(y=0, line_color='red', line_dash='dash')
        fig.update_layout(margin=dict(l=0,r=0,t=42,b=20), height=240)
        st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.markdown('### Relocation scenarios')
    l, m, r = st.columns(3)
    with l:
        budget = st.slider('Bicycles available to relocate', 0, 500, 100, step=10)
    with m:
        distance = st.slider('Maximum straight-line km', 1.0, 10.0, 4.0, 0.5)
    with r:
        same_borough = st.checkbox('Only within the same borough', value=True)
    plan = suggest_transfers(selection, dock_needs, zones, budget=budget, max_km=distance, same_borough=same_borough)
    left, right = st.columns([1,3])
    with left:
        st.metric('Suggested moves', int(plan['Quantity'].sum()) if not plan.empty else 0)
        st.metric('Remaining estimated shortage', max(0, int(selection['shortfall'].sum())-(int(plan['Quantity'].sum()) if not plan.empty else 0)))
    with right:
        if plan.empty:
            st.info('No transfers meet the current reserve, free-dock, distance and movement limits. Widen distance or review the scenario. Some shortages may require additional fleet or operational action.')
        else:
            st.dataframe(plan[['From zone','To zone','Bicycle','Quantity','Straight-line km']], hide_index=True,
                         use_container_width=True, height=330)
            st.download_button('Export suggested transfers as CSV', data=plan.to_csv(index=False).encode(),
                               file_name='citibike_suggested_transfers.csv', mime='text/csv')
    st.info('Decision support only: the app does not dispatch vehicles, model van routes or certify station-level docking feasibility. An operator must verify availability and approve moves.')
    with st.expander('Current stations in selected neighborhood'):
        s0 = station_status.loc[station_status['nta_code'].eq(sel)].copy()
        if not s0.empty:
            st.dataframe(s0[['station_name','classic','electric','docks','status_found','is_renting','is_returning']],
                         hide_index=True, use_container_width=True)
    with st.expander('About the forecasts and limitations'):
        st.write('Models: departures and returns use separate gradient-boosted trees trained on hourly NYC NTA observations. Weather, hour, weekday, season, geography and bicycle type are inputs. Predictions are converted into approximate stock targets using cumulative returns minus departures.')
        st.write('The reported Stage 3 holdout errors are aggregate averages. The model was tested with observed historical weather, not archived weather forecasts. The buffer is a controllable planning assumption, not a statistically calibrated confidence level.')
        st.write('Station inventory is current only when the live feed is available and fresh. Snapshot comparisons with tomorrow are explicitly hypothetical; stations can change between planning and dispatch. Dock-space calculations are aggregated across stations within a zone.')
        st.write('For daylight-saving change days, this version follows the training data’s 24-clock-hour approximation rather than tracking separate repeated 1 a.m. hours.')


def main():
    fail_if_missing()
    st.markdown('## 🚲 Citi Bike · Fleet Availability Planner')
    st.caption('A decision-support prototype for NYC bike-share operations · built from public Citi Bike and Open-Meteo data · not an official Lyft product')
    stations, zones, geo = read_geography()
    dep, ret = get_models()
    if dep.get('model_name') != 'boosted_trees_with_weather' or ret.get('model_name') != 'boosted_trees_with_weather':
        st.warning('Saved model differs from the Stage 3 selected weather-boosted model. The app will still use the saved model package.')
    with st.sidebar:
        st.markdown('### Operations cockpit')
        st.write('**Historical training window:** Oct 2025 – Sep 2026')
        st.write(f'**NYC operating zones:** {len(zones)}')
        st.write('**Two bike types:** classic + electric')
        st.divider()
        st.caption('Workflow: understand usage → predict departures and returns → compare to station inventory → consider nearby transfers.')
        st.caption('Weather: Open-Meteo · Station status: Citi Bike GBFS · Boundaries: NYC DCP')
    tab_usage, tab_plan = st.tabs(['1 · Understand usage', '2 · Plan availability'])
    with tab_usage:
        historical_tab(stations, zones, geo)
    with tab_plan:
        planning_tab(stations, zones, geo)
    st.divider()
    st.caption('Academic prototype. Completed rides are not a measure of unmet demand. Recommendations must be checked by an operations manager.')


if __name__ == '__main__':
    main()
