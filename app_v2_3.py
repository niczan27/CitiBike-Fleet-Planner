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

from planner_core_v2 import (
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

st.set_page_config(page_title='Citi Bike | Fleet Availability Planner V2.3', page_icon='🚲', layout='wide', initial_sidebar_state='expanded')
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
        # Request one coherent forecast window from the present day. Specifying
        # forecast_days together with a custom start/end date has been associated
        # with HTTP errors on some Open-Meteo responses.
        'forecast_days': 10,
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
        df = df.loc[df['hour'].dt.date.isin(set(requested))].copy().reset_index(drop=True)
        if df.empty or not set(requested).issubset(set(df['hour'].dt.date)):
            raise ValueError('Requested dates are not fully covered by the live weather response')
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



# V2: presentation-layer changes, retaining original trained models and data.
# It deliberately does not import or modify app.py or planner_core.py.


def refresh_snapshots() -> None:
    """Manager-controlled API refresh, executed before a Streamlit rerun."""
    live_station_status.clear()
    weather_for_dates.clear()
    st.session_state.pop('v22_station_snapshot', None)
    st.session_state.pop('v22_weather_snapshot', None)
    st.session_state.pop('v22_weather_window', None)
    # Reanchor 'Tomorrow' only on an explicit refresh, not at every rerun.
    st.session_state['v22_planning_today'] = datetime.now(NY_TZ).date()


def all_months():
    return pd.period_range('2025-10', '2026-09', freq='M').strftime('%Y%m').tolist()


def zone_label(zones: pd.DataFrame, code: str) -> str:
    found = zones.loc[zones['nta_code'].astype(str).eq(str(code))]
    if found.empty:
        return 'NYC total' if code == 'All zones' else str(code)
    row = found.iloc[0]
    return f"{row['nta_name']} · {row['borough']}"


def change_zone(zone_key: str, new_code: str) -> None:
    if str(new_code) != str(st.session_state.get(zone_key, 'All zones')):
        st.session_state[zone_key] = str(new_code)
        st.rerun()


def zone_picker(zones: pd.DataFrame, key: str, title: str, allow_citywide=False) -> str:
    valid = zones['nta_code'].astype(str).tolist()
    if allow_citywide:
        valid = ['All zones'] + valid
    if key not in st.session_state or st.session_state[key] not in valid:
        st.session_state[key] = valid[0]
    current = st.session_state[key]
    # Dynamic widget key is intentional: ranking-table clicks update active selection
    # before a fresh widget is rendered, without modifying an instantiated widget.
    chosen = st.selectbox(title, valid, index=valid.index(current),
                          format_func=lambda c: 'Citywide' if c == 'All zones' else zone_label(zones, c),
                          key=f'{key}_picker_{current}')
    if chosen != current:
        change_zone(key, chosen)
    return current


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
    # Preserve smooth pan/zoom from V1 even after filters and row selections.
    fig.update_layout(margin=dict(l=0, r=0, t=26, b=0), title=title,
                      coloraxis_colorbar=dict(title='', thickness=12),
                      uirevision='citibike_v1_map')
    # Highlight selected polygon without replacing the native Plotly map.
    if selected_zone and str(selected_zone) in set(mapped['nta_code'].astype(str)):
        fig.add_trace(go.Choroplethmap(
            geojson=geo, locations=[str(selected_zone)],
            featureidkey='properties.nta2020', z=[1],
            colorscale=[[0,'#e87516'],[1,'#e87516']],
            marker={'opacity':0.03, 'line':{'width':3.2,'color':'#e87516'}},
            showscale=False, hoverinfo='skip', showlegend=False,
        ))
    return fig



def table_selection_positions(df: pd.DataFrame, selected_code: str) -> list[int]:
    """A persisted selection applies only when its area appears in this table."""
    codes = df['nta_code'].astype(str).tolist()
    return [codes.index(str(selected_code))] if str(selected_code) in codes else []


def _on_neighborhood_table_select(widget_key: str, codes: tuple[str, ...],
                                  zone_key: str, table_key: str) -> None:
    """Streamlit calls this BEFORE rerunning, avoiding a second UI refresh.

    Keep the same dataframe widget key for clicks. Switching the dropdown will
    create a new key, with the appropriate checked row preselected (if visible).
    """
    state = st.session_state.get(widget_key, {})
    selected = state.get('selection', {}).get('rows', []) if hasattr(state, 'get') else []
    if selected and 0 <= int(selected[0]) < len(codes):
        code = codes[int(selected[0])]
        st.session_state[zone_key] = code
        st.session_state[f'{table_key}_rendered_code'] = code


def selectable_zone_table(df: pd.DataFrame, cols: list[str], renames: dict[str,str],
                          selected_code: str, selection_key: str, table_key: str,
                          height: int = 395) -> None:
    if df.empty:
        st.info('No neighborhoods for this selection.')
        return
    view = df[cols].rename(columns=renames)
    codes = tuple(df['nta_code'].astype(str))
    # When the dropdown changes to a listed zone, its red selection marker moves.
    # When it changes to an UNLISTED zone, the table marker is cleared. A direct
    # click does not recreate the widget, so its native check remains visible.
    signature = (codes, tuple(cols))
    rendered_zone_key = f'{table_key}_rendered_code'
    signature_key = f'{table_key}_signature'
    generation_key = f'{table_key}_generation'
    if (st.session_state.get(rendered_zone_key) != str(selected_code)
            or st.session_state.get(signature_key) != signature):
        st.session_state[generation_key] = int(st.session_state.get(generation_key, 0)) + 1
        st.session_state[rendered_zone_key] = str(selected_code)
        st.session_state[signature_key] = signature
    widget_key = f'{table_key}_selection_{st.session_state.get(generation_key, 0)}'
    selected_rows = table_selection_positions(df, selected_code)
    try:
        st.dataframe(
            view, hide_index=True, width='stretch', height=height,
            on_select=lambda: _on_neighborhood_table_select(
                widget_key, codes, selection_key, table_key),
            selection_mode='single-row',
            selection_default={'selection': {'rows': selected_rows}},
            key=widget_key,
        )
    except TypeError:
        # Backwards-compatible table for older Streamlit releases. The selected
        # area remains visible in a dedicated status column in this fallback.
        fallback = view.copy()
        fallback.insert(0, 'Selected', ['✓' if code == str(selected_code) else '' for code in codes])
        st.dataframe(fallback, hide_index=True, width='stretch', height=height,
                     on_select=lambda: _on_neighborhood_table_select(
                         widget_key, codes, selection_key, table_key),
                     selection_mode='single-row', key=widget_key)


def historical_tab(stations: pd.DataFrame, zones: pd.DataFrame, geo: dict) -> None:
    st.subheader('Understand usage')
    st.caption('Completed Citi Bike trips · NYC · October 2025 – September 2026')
    a, b, c, d = st.columns([1.6, 1.2, 1, 1.2])
    with a:
        range_value = st.date_input('Historical date range', (date(2026, 9, 1), date(2026, 9, 30)),
                                    min_value=date(2025, 10, 1), max_value=date(2026, 9, 30), key='v2_hist_dates')
    with b:
        bike_option = st.selectbox('Bicycle', ['Both', 'classic_bike', 'electric_bike'],
                                   format_func=lambda v: 'Both types' if v == 'Both' else BIKE_LABELS[v], key='v2_usage_bike')
    with c:
        member_option = st.selectbox('Rider category', ['All','member','casual','unknown'], key='v2_usage_member')
    with d:
        day_filter = st.selectbox('Days', ['All days','Weekdays','Weekends'], key='v2_usage_weekdays')
    c1, c2 = st.columns([1.3, 2])
    with c1:
        hour_window = st.slider('Hour of day (New York local)', 0, 23, (0,23), key='v2_usage_hours')
    with c2:
        selected_code = zone_picker(zones, 'v2_usage_active_zone', 'Inspect neighborhood', allow_citywide=True)
    if not isinstance(range_value, (tuple, list)) or len(range_value) != 2:
        st.info('Select both a start and end date.')
        return
    begin, end = range_value
    months = tuple(p.strftime('%Y%m') for p in pd.period_range(begin.strftime('%Y-%m'),end.strftime('%Y-%m'),freq='M'))
    with st.spinner('Loading compact historical activity...'):
        history = load_history(months)
    history = history.loc[history['hour'].str[:10].between(begin.isoformat(), end.isoformat())].copy()
    if history.empty:
        st.warning('No activity for that selection.')
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
        st.warning('No activity for that selection.')
        return
    totals = history.groupby('nta_code', observed=True)[['departures','returns']].sum().reset_index()
    focused = history if selected_code == 'All zones' else history.loc[history['nta_code'].eq(selected_code)]
    n_days = (end - begin).days + 1
    if day_filter != 'All days':
        dates = pd.date_range(begin, end, freq='D')
        n_days = int(sum(dates.dayofweek < 5) if day_filter == 'Weekdays' else sum(dates.dayofweek >= 5))
    n_days = max(1, n_days)
    dep_count, ret_count = int(focused['departures'].sum()), int(focused['returns'].sum())
    median = median_duration(months, bike_option, member_option) if selected_code == 'All zones' else None
    st.markdown('**Historical activity — ' + ('NYC total' if selected_code == 'All zones' else zone_label(zones, selected_code)) + '**')
    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Departures · selected area', f'{dep_count:,}')
    k2.metric('Avg departures / selected day', f'{dep_count / n_days:,.0f}')
    k3.metric('Net outflow · selected area', f'{dep_count-ret_count:+,}')
    k4.metric('Approx. ride duration · NYC', f'{median} min' if median is not None else 'N/A for a zone')
    st.caption('First three KPIs use selected neighborhood, dates, bike, rider, days and hours. Duration uses only NYC-wide selected months + bike/rider filters, and is unavailable for a single zone.')
    display = zones[['nta_code','nta_name','borough']].merge(totals, on='nta_code', how='left')
    for v in ['departures','returns']:
        display[v] = pd.to_numeric(display[v], errors='coerce').fillna(0)
    display['net_outflow'] = display['departures'] - display['returns']
    # Controls above the two display panels: both panels now start on the same row.
    selector_left, selector_right = st.columns([1.6, 1])
    with selector_left:
        metric = st.radio('Map metric', ['departures','returns','net_outflow'],
                          format_func=lambda v: {'departures':'Departures','returns':'Returns','net_outflow':'Net outflow'}[v],
                          horizontal=True, key='v2_usage_map_mode')
    with selector_right:
        st.caption('Select a neighborhood in the ranking table or the dropdown.')
    left,right=st.columns([1.6,1])
    with left:
        st.markdown('**NYC neighborhood activity**')
        fig = plot_zones(geo, display, metric, '',
                         selected_zone=selected_code if selected_code!='All zones' else None,
                         diverging=(metric=='net_outflow'))
        st.plotly_chart(fig, use_container_width=True, key='v22_usage_plotly',
                        config={'scrollZoom':True,'displaylogo':False})
        st.caption('Original V1-style Plotly map. Select a neighborhood using the ranking table or dropdown; orange outline indicates the selected zone.')
    with right:
        st.markdown('**Neighborhood activity ranking · NYC-wide**')
        leaders = display.sort_values('departures', ascending=False).head(16).reset_index(drop=True)
        # Focus the narrow right column on the active measure so no values
        # disappear behind horizontal scrolling.
        ranking_cols = ['nta_name', 'borough', metric]
        selectable_zone_table(leaders, ranking_cols,
                              {'nta_name':'Neighborhood','borough':'Borough',
                               'departures':'Departures','returns':'Returns','net_outflow':'Net outflow'},
                              selected_code,'v2_usage_active_zone','v22_usage_ranking',height=555)
        if st.button('Reset to citywide',key='v21_hist_reset'):
            change_zone('v2_usage_active_zone','All zones')
    st.markdown('**Hourly patterns — ' + ('NYC total' if selected_code == 'All zones' else zone_label(zones,selected_code)) + '**')
    col_a,col_b=st.columns(2)
    with col_a:
        profile = focused.groupby('hour_of_day',observed=True)[['departures','returns']].sum().reset_index()
        fig = px.line(profile,x='hour_of_day',y=['departures','returns'],markers=True,
                      labels={'value':'Trips (sum over selected days)','hour_of_day':'Hour','variable':'Flow'},title='Hourly profile')
        fig.update_layout(legend_orientation='h',margin=dict(l=0,r=0,t=44,b=5),height=330)
        st.plotly_chart(fig,use_container_width=True)
    with col_b:
        heat = focused.groupby(['weekday','hour_of_day'],observed=True)['departures'].sum().reset_index()
        mat = heat.pivot(index='weekday',columns='hour_of_day',values='departures').reindex(range(7)).fillna(0)
        mat.index=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        fig=px.imshow(mat,labels={'x':'Hour of day','y':'Day','color':'Departures'},aspect='auto',
                      color_continuous_scale='Blues',title='Departure heatmap')
        fig.update_layout(margin=dict(l=0,r=0,t=44,b=5),height=330)
        st.plotly_chart(fig,use_container_width=True)
    st.markdown('**Monthly trend · NYC-wide, Oct 2025–Sep 2026 (fixed reference)**')
    overall=monthly_overview()
    if not overall.empty:
        fig=px.bar(overall,x='month',y='departures',color='rideable_type',barmode='stack',
                   labels={'month':'Month','departures':'Departures','rideable_type':'Bicycle'},
                   color_discrete_map={'Classic':'#1d7997','Electric':'#e0a23c'})
        fig.update_layout(margin=dict(l=0,r=0,t=12,b=10),height=290)
        st.plotly_chart(fig,use_container_width=True)
    st.caption('The map and ranking always cover NYC within the date/rider/bike filters. The monthly reference chart does not change with filters. Historical trips are completed rides, not unmet demand.')


def planning_tab(stations: pd.DataFrame, zones: pd.DataFrame, geo: dict) -> None:
    st.subheader('Plan availability')
    st.caption('Forecast → stock target → compare to current allocation → consider transfers')
    if 'v22_planning_today' not in st.session_state:
        st.session_state['v22_planning_today'] = datetime.now(NY_TZ).date()
    ny_today=st.session_state['v22_planning_today']
    tomorrow=ny_today+timedelta(days=1)
    a,b,c=st.columns([1.2,1,1.2])
    with a:
        horizon=st.selectbox('Planning horizon',['Tomorrow','Next 2 days','Next 7 days','Custom day'],key='v2_plan_horizon')
    with b:
        bike_option=st.selectbox('Prioritize bicycle',['Both','classic_bike','electric_bike'],
                                 format_func=lambda v:'Both types' if v=='Both' else BIKE_LABELS[v],key='v2_plan_bike')
    with c:
        reserve_percent=st.slider('Planning safety buffer (% of predicted daily departures)',0,20,5,1,key='v2_plan_buffer')
    st.caption('Safety buffer: for each neighborhood and bike type, reserve at least 2 bikes or the chosen percentage of departures on the busiest forecast day, whichever is higher. This reserve increases the target starting stock; it is not a confidence level.')
    if horizon=='Custom day':
        chosen=st.date_input('Date to evaluate',tomorrow,min_value=tomorrow,max_value=ny_today+timedelta(days=7),key='v2_custom_date')
        days=[chosen]
    else:
        window={'Tomorrow':1,'Next 2 days':2,'Next 7 days':7}[horizon]
        days=[tomorrow+timedelta(days=i) for i in range(window)]
    dates_txt=tuple(d.isoformat() for d in days)
    has_dst_change = any(
        datetime.combine(d, datetime.min.time()).replace(tzinfo=NY_TZ).utcoffset()
        != datetime.combine(d + timedelta(days=1), datetime.min.time()).replace(tzinfo=NY_TZ).utcoffset()
        for d in days
    )
    # A single compact status row; diagnostics and the DST caveat go below in
    # the existing expandable methodology section.
    refresh_col, station_col, weather_col = st.columns([1.15, 1.25, 1.4], vertical_alignment='center')
    with refresh_col:
        st.button('Refresh availability and weather', key='v22_refresh',
                  on_click=refresh_snapshots)
    with station_col:
        station_status_slot = st.empty()
    with weather_col:
        weather_status_slot = st.empty()

    future_days = [tomorrow + timedelta(days=i) for i in range(8)]
    future_keys = tuple(d.isoformat() for d in future_days)
    if (st.session_state.get('v22_weather_window') != future_keys
            or 'v22_weather_snapshot' not in st.session_state):
        with st.spinner('Loading weather snapshot...'):
            data, label = weather_for_dates(future_keys)
        st.session_state['v22_weather_snapshot'] = (data, label, datetime.now(timezone.utc))
        st.session_state['v22_weather_window'] = future_keys
    if 'v22_station_snapshot' not in st.session_state:
        with st.spinner('Loading station availability snapshot...'):
            result = live_station_status()
        st.session_state['v22_station_snapshot'] = (*result, datetime.now(timezone.utc))

    weather_all, weather_source, weather_retrieved = st.session_state['v22_weather_snapshot']
    status,status_source,updated,live_flag,status_retrieved = st.session_state['v22_station_snapshot']
    with st.spinner('Calculating future activity from the saved data snapshots...'):
        model_dep,model_ret=get_models()
        if model_dep['zone_codes'] != model_ret['zone_codes'] or model_dep['bike_types'] != model_ret['bike_types']:
            st.error('Model zone/bicycle encodings are incompatible.')
            st.stop()
        inputs=calendar_weather_features(weather_all,days)
        departures=predict_counts(model_dep,inputs).rename(columns={'predicted':'departures'})
        returns=predict_counts(model_ret,inputs).rename(columns={'predicted':'returns'})
        prediction=departures.merge(returns,on=['hour','nta_code','rideable_type'],validate='one_to_one')
        station_status,inventory=station_inventory(stations,status)
    status_text=datetime.fromtimestamp(updated,NY_TZ).strftime('%b %d, %Y %H:%M NY') if updated>1000000000 else 'Unknown'
    status_fetched = status_retrieved.astimezone(NY_TZ).strftime('%b %d, %Y %H:%M NY')
    # Keep freshness visible without three explanatory paragraphs or alerts.
    station_flag = 'Live' if live_flag else 'Saved / unverified'
    station_status_slot.caption(f'**Stations:** {station_flag} · {status_text}')
    fetched_at = weather_retrieved.astimezone(NY_TZ).strftime('%b %d, %Y %H:%M NY')
    weather_fallback = 'fallback' in weather_source.lower()
    weather_flag = '⚠ Historical fallback' if weather_fallback else 'Forecast'
    weather_status_slot.caption(f'**Weather:** {weather_flag} · {fetched_at}')
    targets,dock_needs=required_inventory(prediction,inventory,reserve_fraction=reserve_percent/100)
    selection=targets if bike_option=='Both' else targets.loc[targets['rideable_type'].eq(bike_option)]
    chosen_needs=selection.groupby('nta_code',as_index=False).agg(
        shortfall=('shortfall','sum'),surplus=('surplus','sum'),
        departures=('expected_departures','sum'),returns=('expected_returns','sum'))
    merged=zones.merge(chosen_needs,on='nta_code',how='left').merge(
        dock_needs[['nta_code','dock_shortfall','slots_for_transfers']],on='nta_code',how='left')
    for col in ['shortfall','surplus','departures','returns','dock_shortfall','slots_for_transfers']:
        merged[col]=pd.to_numeric(merged[col],errors='coerce').fillna(0)
    st.markdown('**NYC-wide forecast and inventory pressure — selected planning period**')
    k1,k2,k3,k4=st.columns(4)
    k1.metric('Expected departures · selected bike',f'{selection.expected_departures.sum():,.0f}')
    k2.metric('Expected returns · selected bike',f'{selection.expected_returns.sum():,.0f}')
    k3.metric('Additional bike need · selected bike',f'{selection.shortfall.sum():,.0f}')
    k4.metric('Areas with dock-space risk · both types',f'{int((dock_needs.dock_shortfall>0).sum()):,}')
    st.caption('All four KPIs are NYC-wide and do not change when a neighborhood is selected. Dock-space risk counts neighborhoods with a positive estimated dock-space shortfall, across both bike types. These are simulation outputs, not observed lost rentals.')
    st.markdown('**Geographic priorities**')
    map_controls, table_controls = st.columns([1.55, 1])
    with map_controls:
        map_metric = st.radio('Map view',['shortfall','surplus','dock_shortfall'],
                              format_func=lambda k:{'shortfall':'Bike need','surplus':'Bike surplus','dock_shortfall':'Dock risk'}[k],
                              horizontal=True,key='v2_plan_map_mode')
    with table_controls:
        table_mode = st.radio('Table view',['shortfall','surplus','dock_shortfall'],
                              format_func=lambda k:{'shortfall':'Bike need','surplus':'Bike surplus','dock_shortfall':'Dock risk'}[k],
                              horizontal=True,key='v2_plan_table_mode')
    if 'v2_plan_active_zone' not in st.session_state:
        st.session_state['v2_plan_active_zone']=str(merged.sort_values('shortfall',ascending=False).iloc[0]['nta_code'])
    selected_code=st.session_state['v2_plan_active_zone']
    if selected_code not in set(merged['nta_code'].astype(str)):
        selected_code=str(merged.iloc[0]['nta_code'])
        st.session_state['v2_plan_active_zone']=selected_code
    # These panels are deliberately in one new row after the controls.
    # Their headings and 555px contents begin at the same vertical position.
    left,right=st.columns([1.55,1])
    with left:
        st.markdown('**Forecast inventory pressure**')
        fig = plot_zones(geo, merged, map_metric, '', selected_zone=selected_code)
        st.plotly_chart(fig, use_container_width=True, key='v22_plan_plotly',
                        config={'scrollZoom':True,'displaylogo':False})
        st.caption('Plotly map: zoom/pan/hover freely. Select an area in the table, then switch the map to Bike surplus to identify nearby donors.')
    with right:
        if table_mode=='shortfall':
            rank_title='Largest additional bike requirements'
            ranking=merged.sort_values('shortfall',ascending=False).head(20).reset_index(drop=True)
        elif table_mode=='surplus':
            rank_title='Largest transferable bike surpluses'
            ranking=merged.sort_values('surplus',ascending=False).head(20).reset_index(drop=True)
        else:
            rank_title='Largest dock-space risks'
            ranking=merged.sort_values('dock_shortfall',ascending=False).head(20).reset_index(drop=True)
        st.markdown('**'+rank_title+'**')
        # Only the chosen metric is visible, so the table remains readable.
        selectable_zone_table(ranking,['nta_name','borough',table_mode],
                              {'nta_name':'Neighborhood','borough':'Borough',
                               'shortfall':'Bike need','surplus':'Bike surplus',
                               'dock_shortfall':'Dock risk (spaces)'},
                              selected_code,'v2_plan_active_zone',f'v22_plan_ranking_{table_mode}',height=555)
    st.divider()
    st.markdown('### Selected neighborhood — inventory requirements')
    c1,c2=st.columns([2,1])
    with c1:
        selected_code=zone_picker(zones,'v2_plan_active_zone','Inspect forecast trajectory')
    with c2:
        if bike_option=='Both':
            type_sel=st.selectbox('Bike for trajectory',BIKE_TYPES,
                                  format_func=lambda t:BIKE_LABELS[t],key='v2_trace_bike')
        else:
            type_sel=bike_option
            st.caption('Trajectory type follows the selected bike type: **'+BIKE_LABELS[type_sel]+'**')
    chosen_rows=targets.loc[targets['nta_code'].eq(selected_code) & targets['rideable_type'].eq(type_sel)]
    dock_rows=dock_needs.loc[dock_needs['nta_code'].eq(selected_code)]
    inv_rows=inventory.loc[inventory['nta_code'].eq(selected_code)]
    if chosen_rows.empty or dock_rows.empty or inv_rows.empty:
        st.warning('No forecasting/inventory data for this neighborhood.')
        return
    chosen=chosen_rows.iloc[0]
    docks=dock_rows.iloc[0]
    trace=prediction.loc[prediction['nta_code'].eq(selected_code)&prediction['rideable_type'].eq(type_sel)].copy()
    trace=trace.sort_values('hour')
    trace['cumulative_outflow']=(trace['departures']-trace['returns']).cumsum()
    trace['projected_bikes']=int(chosen['available_bikes'])-trace['cumulative_outflow']
    safety_reserve=int(chosen['buffer'])
    graphcol,detailscol=st.columns([1.6,1])
    with graphcol:
        fig=px.line(trace,x='hour',y='projected_bikes',
                    title=f'Projected {BIKE_LABELS[type_sel].lower()} stock · no transfers',
                    labels={'hour':'Local hour','projected_bikes':'Bicycles'})
        fig.add_hline(y=0,line_dash='dash',line_color='#c93535',
                      annotation_text='0 bikes',annotation_position='bottom right')
        fig.add_hline(y=safety_reserve,line_dash='dot',line_color='#e4a12b',
                      annotation_text=f'Safety reserve: {safety_reserve}',annotation_position='top right')
        fig.update_layout(margin=dict(l=0,r=0,t=42,b=30),height=350)
        st.plotly_chart(fig,use_container_width=True)
        minimum=min(float(chosen['available_bikes']),float(trace['projected_bikes'].min()))
        st.caption(f'Minimum projected stock: {minimum:,.0f} bicycles. A need may exist even if stock stays above zero, because the forecast must also retain the safety reserve.')
    with detailscol:
        st.caption(zone_label(zones,selected_code)+' · '+BIKE_LABELS[type_sel])
        u1,u2=st.columns(2)
        u1.metric('Current available bikes',f"{int(chosen['available_bikes']):,}")
        u2.metric('Target starting stock',f"{int(chosen['target_bikes']):,}")
        u3,u4=st.columns(2)
        u3.metric('Peak predicted depletion',f"{float(chosen['peak_depletion']):,.0f}")
        u4.metric(f'Safety reserve ({reserve_percent}%)',f"{safety_reserve:,}")
        st.metric('Additional bike need',f"{int(chosen['shortfall']):,}")
        st.divider()
        v1,v2=st.columns(2)
        v1.metric('Empty docks now · both types',f"{int(docks['docks']):,}")
        v2.metric('Space for initial transfers',f"{int(docks['slots_for_transfers']):,}")
        st.caption(f"Dock-space shortfall risk: {int(docks['dock_shortfall']):,} spaces. Remaining need beyond initial receiving capacity: {max(0,int(chosen['shortfall'])-int(docks['slots_for_transfers'])):,} bikes, before donor/distance constraints.")
    st.caption('Target starting stock = maximum predicted cumulative net outflow + safety reserve. Space for initial transfers = current empty docks − allowance for peak net arrivals of both bike types − 2 reserved empty docks, floored at zero. This scenario assumes current inventory at the future period start.')
    with st.expander('Stations in selected neighborhood — current available fleet, docks and capacity'):
        subset=station_status.loc[station_status['nta_code'].eq(selected_code)].copy()
        if not subset.empty:
            for col in ['capacity_snapshot','classic','electric','docks']:
                subset[col]=pd.to_numeric(subset[col],errors='coerce').fillna(0).astype(int)
            st.dataframe(subset[['station_name','capacity_snapshot','classic','electric','docks','status_found','is_renting','is_returning']].rename(columns={
                'station_name':'Station','capacity_snapshot':'Directory capacity','classic':'Classic bikes',
                'electric':'E-bikes','docks':'Empty docks','status_found':'Status available',
                'is_renting':'Renting enabled','is_returning':'Returns enabled'}),
                hide_index=True,use_container_width=True,height=290)
        st.caption('Directory capacities and station availability are different fields. Aggregated zone-level feasibility does not guarantee any particular station can accept a proposed transfer.')
    st.divider()
    st.markdown('### Relocation scenarios')
    b,d,s=st.columns(3)
    with b:
        budget=int(st.number_input('Maximum bicycles available to relocate',min_value=0,value=500,step=50,
                                   help='Operational constraint chosen by the manager, not a model-generated limit.',
                                   key='v2_move_budget'))
    with d:
        distance=st.slider('Maximum straight-line km',1.0,10.0,4.0,.5,key='v2_move_dist')
    with s:
        same_borough=st.checkbox('Only within same borough',value=True,key='v2_same_borough')
    plan=suggest_transfers(selection,dock_needs,zones,budget=budget,max_km=distance,same_borough=same_borough)
    moved=int(plan['Quantity'].sum()) if not plan.empty else 0
    shortage=int(selection['shortfall'].sum())
    remain=max(0,shortage-moved)
    achieved=100*moved/shortage if shortage else 0
    k1,k2,k3,k4=st.columns(4)
    k1.metric('Initial estimated need',f'{shortage:,}')
    k2.metric('Suggested moves',f'{moved:,}')
    k3.metric('Remaining estimated need',f'{remain:,}')
    k4.metric('Need addressed (scenario)',f'{achieved:.1f}%')
    if plan.empty:
        st.info('No eligible zone transfers under the current constraints. Consider adjusting the budget or distance; other physical constraints may still prevent moves.')
    else:
        st.dataframe(plan[['From zone','To zone','Bicycle','Quantity','Straight-line km']],
                     hide_index=True,use_container_width=True,height=325)
        st.download_button('Export suggested transfers as CSV',data=plan.to_csv(index=False).encode('utf-8'),
                           file_name='citibike_v2_suggested_transfers.csv',mime='text/csv')
    st.caption('Increasing the budget cannot overcome insufficient donor surplus, reserved dock space, or distance/borough limits. Quantities are neighborhood-level estimates: actual station-level dispatch requires human verification.')
    with st.expander('About forecasts, buffers and limitations'):
        st.write('Two separate Stage 3 boosted-tree models forecast hourly completed departures and returns using calendar, location, bike type and Open-Meteo weather. For each zone and type, peak predicted depletion plus a safety reserve produces target starting stock.')
        st.write('The 0–20% reserve is a planning assumption, not a confidence interval. In V2, a multi-day period uses the busiest day as the reserve base, not all departures summed across the horizon.')
        st.write('Current availability is tested as a hypothetical starting inventory for the selected future period. The stock trajectory can go negative algebraically; actual inventory cannot. Dock-space risk measures a potential deficiency of empty docks across both types, not a probability.')
        st.write('The model comparison uses past observed weather rather than archived historical forecasts. Station inventory may be a saved snapshot if GBFS is inaccessible. Zone totals cannot certify station-level feasibility or van routing.')
        st.write('Station status and weather are retrieved once per session and refreshed only using the Refresh button. Current observed stock is treated as hypothetical starting stock for the selected future period.')
        st.write(f'Station source: {status_source}. Station data timestamp: {status_text}; retrieved: {status_fetched}.')
        st.write(f'Weather source: {weather_source}. Retrieved: {fetched_at}. Planning dates: {days[0]} to {days[-1]}.')
        if weather_fallback:
            st.write('The weather API response was unavailable, so seasonal historical averages were substituted. These are NOT live weather forecasts.')
        if has_dst_change:
            st.write('Daylight-saving transition falls within the selected planning period. This model approximates each date with 24 clock hours, while the transition day has 23 or 25 actual hours.')
        st.write('The project uses the original Stage 3 models. No retraining or rewriting of the historical data was performed.')


def main() -> None:
    fail_if_missing()
    st.markdown('## 🚲 Citi Bike · Fleet Availability Planner')
    st.caption('V2.3 interface · public NYC Citi Bike + Open-Meteo data · decision-support prototype, not an official Lyft product')
    stations,zones,geo=read_geography()
    if 'nta_code' in zones.columns:
        zones['nta_code']=zones['nta_code'].astype(str)
    dep,ret=get_models()
    if dep.get('model_name')!='boosted_trees_with_weather' or ret.get('model_name')!='boosted_trees_with_weather':
        st.warning('The saved model differs from the selected Stage 3 weather-boosted model.')
    with st.sidebar:
        st.markdown('### Operations cockpit · V2.3')
        st.write('**Historical data:** Oct 2025–Sep 2026')
        st.write(f'**NYC operating zones:** {len(zones)}')
        st.write('**Bike types:** classic and electric')
        st.divider()
        st.caption('Understand usage → predict activity → compare to available stock → review transfer opportunities.')
        st.caption('V2.3 preserves earlier versions. Maps use V1 Plotly and selection uses clickable ranking rows/dropdowns.')
    tab_a,tab_b=st.tabs(['1 · Understand usage','2 · Plan availability'])
    with tab_a:
        historical_tab(stations,zones,geo)
    with tab_b:
        planning_tab(stations,zones,geo)
    st.divider()
    st.caption('Academic decision-support prototype. Completed rides are not unmet demand. Operations teams must verify transfers, personnel, transport and station constraints.')


if __name__ == '__main__':
    main()
