"""Add-on visualizations for the portable V2.3 app.

Isolated from the existing models, planning logic and existing charts.
"""
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go


PALETTE = {'Temperature': '#D67A45', 'Rain': '#209D91', 'Snow': '#657CB4', 'Wind': '#64748B'}


def build_weather_series(df: pd.DataFrame):
    """Produce readable exploratory bins. Each series includes zero-rain/snow days.

    Similar-day restrictions reduce (but do not eliminate) confounding.
    These are associations, NOT model feature importance or causal estimates.
    """
    df = df.copy()
    df['day'] = pd.to_datetime(df['day'])
    weekdays = df.loc[df['day'].dt.dayofweek < 5].copy()
    dry = weekdays.loc[(weekdays['rain_mm'] < 1) & (weekdays['snowfall_cm'] == 0)].copy()
    warm = weekdays.loc[weekdays['temperature_max_c'] >= 15].copy()
    cold = weekdays.loc[weekdays['temperature_max_c'] <= 5].copy()

    cases = {
        'Temperature': (dry, 'temperature_max_c', [-15,-10,-5,0,5,10,15,20,25,30,35,45],
                        ['-15–-10','-10–-5','-5–0','0–5','5–10','10–15','15–20','20–25','25–30','30–35','35–45'],
                        'Max daily temperature (°C)', 'Weekdays with <1 mm rain and no snowfall'),
        'Rain': (warm, 'rain_mm', [-0.001,0,2,5,10,20,1000],
                 ['0','0–2','2–5','5–10','10–20','>20'], 'Daily rain (mm)', 'Weekdays reaching at least 15°C'),
        'Snow': (cold, 'snowfall_cm', [-0.001,0,1,3,8,1000],
                 ['0','0–1','1–3','3–8','>8'], 'Daily snowfall (cm)', 'Weekdays with max temperature ≤5°C'),
        'Wind': (dry, 'wind_max_kmh', [-0.001,8,12,16,20,25,1000],
                 ['0–8','8–12','12–16','16–20','20–25','>25'], 'Maximum wind (km/h)', 'Weekdays with <1 mm rain and no snowfall'),
    }
    series = {}
    for title, (source, column, edges, labels, axis, subset) in cases.items():
        source['bin'] = pd.cut(source[column], edges, labels=labels, include_lowest=True)
        grouped = source.groupby('bin', observed=True)['departures'].agg(['mean','size']).reset_index()
        grouped = grouped.loc[grouped['size'].ge(2)].copy()
        grouped['bin'] = grouped['bin'].astype(str)
        series[title] = dict(data=grouped, x_title=axis, criteria=subset, n_days=len(source))
    return series


def render_weather_sensitivity(processed: Path):
    import streamlit as st

    source = processed / 'weather_sensitivity_daily.csv'
    st.markdown('**Weather sensitivity · NYC-wide (fixed reference)**')
    st.caption('Vertical scale: average daily departures (trips/day) · Horizontal scale: weather-condition bins')
    if not source.is_file():
        st.warning('Weather summary is missing from this package. Re-extract the complete ZIP archive.')
        return

    @st.cache_data(show_spinner=False)
    def _read(path: str):
        return pd.read_csv(path, parse_dates=['day'])

    daily = _read(str(source))
    series = build_weather_series(daily)
    cols = st.columns(4)
    for col, title in zip(cols, ['Temperature', 'Rain', 'Snow', 'Wind']):
        meta = series[title]
        values = meta['data']
        with col:
            st.caption(f'**{title}** · {meta["n_days"]} comparable days')
            fig = go.Figure()
            color = PALETTE[title]
            if not values.empty:
                # Consistent line+marker styling, including snow. Snow is drawn
                # dashed to emphasize the coarse bins and small sample.
                fig.add_trace(go.Scatter(
                    x=values['bin'], y=values['mean'], mode='lines+markers',
                    line=dict(color=color, width=2.5,
                              dash='dot' if title == 'Snow' else 'solid'),
                    marker=dict(size=7, color=color),
                    customdata=values[['size']].values,
                    hovertemplate='%{x}<br>Average daily departures: %{y:,.0f}'
                                  '<br>Days in bin: %{customdata[0]}<extra></extra>'
                ))
            fig.update_layout(
                height=235, margin=dict(l=38, r=5, t=8, b=12),
                showlegend=False,
                font=dict(size=11),
                xaxis=dict(title=None, type='category', tickangle=-35,
                           tickfont=dict(size=9), showgrid=False,
                           automargin=True),
                yaxis=dict(title=None, tickfont=dict(size=10), tickformat='~s',
                           rangemode='tozero', gridcolor='#e9eef2',
                           zeroline=False, automargin=True),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True,
                            key=f'weather_fixed_{title.lower()}',
                            config={'displayModeBar': False, 'displaylogo': False})
            st.caption(meta['x_title'])
    with st.popover('How to interpret these weather charts'):
        st.caption(
            'Each point is the average number of recorded NYC departures per calendar day '
            'within that weather bin. Bins with fewer than two days are omitted. '
            'The charts use different subsets of weekdays to reduce confounding: '
            'temperature and wind use relatively dry, snow-free days; '
            'rain uses days with maximum temperature at least 15°C; '
            'snow uses days with maximum temperature at most 5°C. '
            'Samples and vertical scales differ across charts. '
            'These descriptive associations are not isolated causal weather effects.'
        )

def prepare_borough_profiles(data: pd.DataFrame, begin, end, bike, member, day_filter, target_borough):
    """Compute hourly cross-borough trip averages on selected calendar dates."""
    d = data.copy()
    d['hour'] = pd.to_datetime(d['hour'], errors='coerce')
    d = d.loc[d['hour'].notna() & d['hour'].dt.date.between(begin, end)] if hasattr(d['hour'].dt.date, 'between') else d.loc[d['hour'].notna() & (d['hour'].dt.date >= begin) & (d['hour'].dt.date <= end)]
    if bike != 'Both':
        d = d.loc[d['rideable_type'].eq(bike)]
    if member != 'All':
        d = d.loc[d['member_casual'].eq(member)]
    if day_filter == 'Weekdays':
        d = d.loc[d['hour'].dt.dayofweek < 5]
    if day_filter == 'Weekends':
        d = d.loc[d['hour'].dt.dayofweek >= 5]
    if target_borough != 'All outer boroughs':
        d = d.loc[(d['origin_borough'].eq(target_borough)) | (d['destination_borough'].eq(target_borough))]
    d['direction'] = d['origin_borough'].eq('Manhattan').map({True: 'Manhattan → outer boroughs', False: 'Outer boroughs → Manhattan'})
    if d.empty:
        return pd.DataFrame(columns=['hour_of_day','direction','avg_trips'])
    d['hour_of_day'] = d['hour'].dt.hour
    total_days = pd.date_range(begin, end, freq='D')
    if day_filter == 'Weekdays':
        total_days = total_days[total_days.dayofweek < 5]
    elif day_filter == 'Weekends':
        total_days = total_days[total_days.dayofweek >= 5]
    total_days = max(1,len(total_days))
    totals = d.groupby(['hour_of_day','direction'], as_index=False)['trips'].sum()
    totals['avg_trips'] = totals['trips'] / total_days
    frame = pd.MultiIndex.from_product([range(24),
                   ['Manhattan → outer boroughs','Outer boroughs → Manhattan']],
                   names=['hour_of_day','direction']).to_frame(index=False)
    return frame.merge(totals[['hour_of_day','direction','avg_trips']], how='left').fillna({'avg_trips': 0})


def render_borough_flow_if_available(processed: Path, begin, end, bike, member, day_filter):
    import streamlit as st

    path = processed / 'borough_directional_flows.csv.gz'
    if not path.exists():
        st.caption('Optional borough-direction chart is not yet available: it needs a compact origin–destination summary from the raw trip files. All other app functionality works without it.')
        return

    @st.cache_data(show_spinner=False)
    def _read(path_str: str):
        return pd.read_csv(path_str, compression='gzip', dtype={
            'origin_borough': 'string', 'destination_borough': 'string',
            'rideable_type': 'string', 'member_casual': 'string'})

    data = _read(str(path))
    st.markdown('**Manhattan ↔ outer boroughs · hourly directional flows**')
    opts = ['All outer boroughs'] + sorted(
        set(data['origin_borough']).union(data['destination_borough']) - {'Manhattan'})
    target = st.selectbox('Compare Manhattan with:', opts, key='v23_addon_borough_choice')
    df = prepare_borough_profiles(data, begin, end, bike, member, day_filter, target)
    if df.empty or not df['avg_trips'].sum():
        st.info('No matching cross-borough trips for the current historical selection.')
        return

    # A horizontal description avoids the clipped vertical label in the original chart.
    st.caption('Average hourly trips per selected calendar day')
    fig = go.Figure()
    other = 'outer boroughs' if target == 'All outer boroughs' else target
    for direction, color, visible_label in [
        ('Manhattan → outer boroughs', '#2778D4', f'Manhattan → {other}'),
        ('Outer boroughs → Manhattan', '#E77738', f'{other} → Manhattan'),
    ]:
        part = df.loc[df['direction'].eq(direction)].sort_values('hour_of_day')
        fig.add_trace(go.Scatter(
            x=part['hour_of_day'], y=part['avg_trips'], name=visible_label,
            mode='lines', line=dict(color=color, width=3),
            hovertemplate='%{x}:00<br>%{y:,.1f} trips/hour per selected day<extra></extra>'
        ))
    fig.update_layout(
        height=355,
        margin=dict(l=55, r=20, t=8, b=60),
        font=dict(size=12),
        xaxis=dict(title='Hour of day (New York local)', dtick=2, range=[0, 23],
                   automargin=True),
        yaxis=dict(title=None, tickformat=',.0f', rangemode='tozero',
                   automargin=True, gridcolor='#e9eef2', zeroline=False),
        legend=dict(orientation='h', y=-0.30, x=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True, key='v23_addon_borough_fig',
                    config={'displaylogo': False})
    st.caption('Uses historical date range, bicycle type, rider category and weekday/weekend filters; '
               'shows all 24 hours and both directions regardless of selected neighborhood/hour slider. '
               'Counts reflect matched origin and destination stations (unmatched trips excluded).')
