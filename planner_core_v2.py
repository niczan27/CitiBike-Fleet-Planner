"""Pure calculations for the Citi Bike manager-facing prototype.

This module intentionally does not import Streamlit, so it can be unit tested.
Original data and trained model files are not modified.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Mapping
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

BIKE_TYPES = ["classic_bike", "electric_bike"]
WEATHER_FIELDS = [
    "temperature_2m", "apparent_temperature", "precipitation", "rain",
    "snowfall", "snow_depth", "wind_speed_10m", "wind_gusts_10m",
]
FEATURE_NAMES = [
    "zone_index", "bike_index", "hour_of_day", "day_of_week", "month",
    "federal_holiday_proxy", *WEATHER_FIELDS,
]


def calendar_weather_features(weather: pd.DataFrame, dates: list[date]) -> pd.DataFrame:
    """Return 24 rows for each requested local date in Stage 3 feature order.

    Weather dates with repeated DST hour are averaged into a single clock hour,
    consistent with the Stage 3 training approximation.
    """
    if not dates:
        raise ValueError("At least one planning date is required")
    hours = pd.date_range(dates[0].isoformat(), periods=24 * len(dates), freq="h")
    # Calendar days may cross a DST change. Our training uses 24 nominal local
    # clock labels, not fully timezone-aware 23/25-hour days.
    w = weather.copy()
    w["hour"] = pd.to_datetime(w["hour"], errors="coerce")
    w = w.dropna(subset=["hour"])
    for field in WEATHER_FIELDS:
        if field not in w.columns:
            w[field] = np.nan
        w[field] = pd.to_numeric(w[field], errors="coerce")
    w = w.groupby("hour", as_index=False)[WEATHER_FIELDS].mean()
    required = pd.DataFrame({"hour": hours})
    merged = required.merge(w, on="hour", how="left", validate="one_to_one")
    if merged[WEATHER_FIELDS].isna().any().any():
        missing = merged[merged[WEATHER_FIELDS].isna().any(axis=1)]["hour"]
        raise ValueError(f"Forecast weather incomplete for {len(missing)} hour(s), e.g. {missing.iloc[0]}")
    cal = USFederalHolidayCalendar().holidays(start=hours.min().normalize(), end=hours.max().normalize())
    merged["hour_of_day"] = hours.hour
    merged["day_of_week"] = hours.dayofweek
    merged["month"] = hours.month
    merged["federal_holiday_proxy"] = hours.normalize().isin(cal).astype(int)
    return merged


def predict_counts(model_package: Mapping, time_weather: pd.DataFrame) -> pd.DataFrame:
    """Generate hourly zone x bike type predictions with the saved Stage 3 model."""
    zones = model_package["zone_codes"]
    bike_types = model_package["bike_types"]
    ntime, nz, nb = len(time_weather), len(zones), len(bike_types)
    if not ntime or not nz or not nb:
        raise ValueError("Empty model zones, bike types, or forecast hours")
    if model_package.get("feature_names") != FEATURE_NAMES:
        raise ValueError("The saved model's feature schema does not match this app. Recheck model version.")
    x = np.empty((ntime * nz * nb, len(FEATURE_NAMES)), dtype=np.float32)
    x[:, 0] = np.tile(np.repeat(np.arange(nz), nb), ntime)
    x[:, 1] = np.tile(np.arange(nb), ntime * nz)
    for i, name in enumerate(FEATURE_NAMES[2:], start=2):
        x[:, i] = np.repeat(time_weather[name].to_numpy(dtype=np.float32), nz * nb)
    columns = model_package["selected_feature_columns"]
    if model_package["method"] == "baseline":
        from_key = ((x[:, 0].astype(int) * 2 + x[:, 1].astype(int)) * 7 + x[:, 3].astype(int)) * 24 + x[:, 2].astype(int)
        predicted = model_package["estimator"][from_key]
    else:
        predicted = model_package["estimator"].predict(x[:, columns])
    predicted = np.maximum(0, np.asarray(predicted, dtype=float))
    if not np.isfinite(predicted).all():
        raise RuntimeError("Forecast produced invalid non-finite predictions")
    return pd.DataFrame({
        "hour": np.repeat(time_weather["hour"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(), nz * nb),
        "nta_code": np.tile(np.repeat(np.asarray(zones), nb), ntime),
        "rideable_type": np.tile(np.asarray(bike_types), ntime * nz),
        "predicted": predicted,
    })


def station_inventory(station_zones: pd.DataFrame, status: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct station and zone inventory from the status API or explicit snapshot."""
    s = station_zones.copy()
    s["station_id"] = s["station_id"].astype(str).str.strip()
    s = s.loc[(pd.to_numeric(s["in_nyc"], errors="coerce") == 1) & s["nta_code"].notna()].copy()
    rows = []
    for item in status.get("data", {}).get("stations", []):
        d = {v.get("vehicle_type_id"): int(v.get("count", 0)) for v in item.get("vehicle_types_available", [])}
        can_rent = bool(item.get("is_installed", 0)) and bool(item.get("is_renting", 0))
        can_return = bool(item.get("is_installed", 0)) and bool(item.get("is_returning", 0))
        rows.append({
            "station_id": str(item.get("station_id", "")),
            "classic": max(0, d.get("1", 0)) if can_rent else 0,
            "electric": max(0, d.get("2", item.get("num_ebikes_available", 0))) if can_rent else 0,
            "docks": max(0, int(item.get("num_docks_available", 0) or 0)) if can_return else 0,
            "is_renting": int(can_rent), "is_returning": int(can_return),
            "last_reported": int(item.get("last_reported", 0) or 0),
        })
    st = pd.DataFrame(rows)
    if st.empty:
        raise ValueError("No station status records received")
    if st["station_id"].duplicated().any():
        raise ValueError("Duplicate station IDs in station status feed")
    stations = s.merge(st, on="station_id", how="left", validate="one_to_one", indicator=True)
    stations["status_found"] = stations["_merge"].eq("both")
    stations = stations.drop(columns=["_merge"])
    for col in ["classic", "electric", "docks", "is_renting", "is_returning"]:
        stations[col] = stations[col].fillna(0).astype(int)
    for col in ["lat", "lon"]:
        stations[col] = pd.to_numeric(stations[col], errors="coerce")
    zones = stations.groupby("nta_code", as_index=False).agg(
        classic=("classic", "sum"), electric=("electric", "sum"), docks=("docks", "sum"),
        total_stations=("station_id", "size"), reporting_stations=("status_found", "sum"),
        latitude=("lat", "mean"), longitude=("lon", "mean"),
    )
    return stations, zones


def required_inventory(forecast: pd.DataFrame, inventories: pd.DataFrame, reserve_fraction: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimated starting stocks needed to avoid forecast net-flow depletion.

    This is a deterministic scenario, not a probability of service level.
    Returns a zone+type table and a zone-level dock table.
    """
    f = forecast.copy().sort_values(["nta_code", "rideable_type", "hour"])
    required = {"nta_code", "rideable_type", "departures", "returns", "hour"}
    if not required.issubset(f.columns):
        raise ValueError(f"Missing forecast columns: {required - set(f.columns)}")
    for col in ("departures", "returns"):
        f[col] = pd.to_numeric(f[col], errors="coerce").fillna(0).clip(lower=0)
    f["net_out"] = f["departures"] - f["returns"]
    f["cumulative_out"] = f.groupby(["nta_code", "rideable_type"], sort=False)["net_out"].cumsum()
    g = f.groupby(["nta_code", "rideable_type"], as_index=False).agg(
        expected_departures=("departures", "sum"), expected_returns=("returns", "sum"),
        peak_depletion=("cumulative_out", "max"),
    )
    g["peak_depletion"] = g["peak_depletion"].clip(lower=0)
    # For multi-day horizons use the busiest single day's predicted departures
    # as the reserve base. Summing all days would artificially multiply the buffer.
    daily = f.copy()
    daily["service_date"] = daily["hour"].astype(str).str.slice(0, 10)
    daily = daily.groupby(["nta_code", "rideable_type", "service_date"], as_index=False)["departures"].sum()
    max_daily = daily.groupby(["nta_code", "rideable_type"], as_index=False)["departures"].max().rename(columns={"departures": "peak_daily_departures"})
    g = g.merge(max_daily, on=["nta_code", "rideable_type"], how="left", validate="one_to_one")
    g["buffer"] = np.maximum(2, np.ceil(g["peak_daily_departures"] * reserve_fraction)).astype(int)
    g["target_bikes"] = np.ceil(g["peak_depletion"] + g["buffer"]).astype(int)
    supply = inventories.set_index("nta_code")[["classic", "electric"]]
    g["available_bikes"] = [
        int(supply.at[z, "classic" if b == "classic_bike" else "electric"]) if z in supply.index else 0
        for z, b in zip(g["nta_code"], g["rideable_type"])
    ]
    g["shortfall"] = (g["target_bikes"] - g["available_bikes"]).clip(lower=0).astype(int)
    g["surplus"] = (g["available_bikes"] - g["target_bikes"]).clip(lower=0).astype(int)
    both = f.groupby(["nta_code", "hour"], as_index=False)[["departures", "returns"]].sum()
    both = both.sort_values(["nta_code", "hour"])
    both["cum_arrivals"] = (both["returns"] - both["departures"]).groupby(both["nta_code"]).cumsum()
    d = both.groupby("nta_code", as_index=False).agg(peak_net_arrivals=("cum_arrivals", "max"))
    d["peak_net_arrivals"] = d["peak_net_arrivals"].clip(lower=0)
    d = d.merge(inventories[["nta_code", "docks"]], on="nta_code", how="left")
    d["docks"] = d["docks"].fillna(0).astype(int)
    d["dock_target"] = np.ceil(d["peak_net_arrivals"] + 2).astype(int)
    d["dock_shortfall"] = (d["dock_target"] - d["docks"]).clip(lower=0).astype(int)
    d["slots_for_transfers"] = (d["docks"] - d["dock_target"]).clip(lower=0).astype(int)
    return g, d


def haversine_km(lat1, lon1, lat2, lon2):
    a1, a2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(a1) * math.cos(a2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(a)))


def suggest_transfers(needs: pd.DataFrame, docks: pd.DataFrame, zones: pd.DataFrame,
                      budget: int = 100, max_km: float = 4.0, same_borough: bool = True) -> pd.DataFrame:
    """Greedy nearby-zone transfer plan, constrained by stock and dock slack."""
    merged = needs.merge(zones[["nta_code", "nta_name", "borough", "latitude", "longitude"]], on="nta_code", how="left")
    dest_slots = dict(zip(docks["nta_code"], docks["slots_for_transfers"].astype(int)))
    receivers = merged.loc[merged["shortfall"] > 0].sort_values("shortfall", ascending=False).copy()
    donors = merged.loc[merged["surplus"] > 0].copy()
    donor_spare = {(r.nta_code, r.rideable_type): int(r.surplus) for r in donors.itertuples()}
    remaining = max(0, int(budget))
    transfers = []
    for receiver in receivers.itertuples():
        if remaining <= 0:
            break
        need = min(int(receiver.shortfall), int(dest_slots.get(receiver.nta_code, 0)))
        if need <= 0 or pd.isna(receiver.latitude):
            continue
        options = []
        for donor in donors.itertuples():
            if donor.rideable_type != receiver.rideable_type or donor.nta_code == receiver.nta_code:
                continue
            if same_borough and donor.borough != receiver.borough:
                continue
            if pd.isna(donor.latitude):
                continue
            distance = haversine_km(receiver.latitude, receiver.longitude, donor.latitude, donor.longitude)
            if distance <= max_km:
                options.append((distance, donor))
        for distance, donor in sorted(options, key=lambda item: item[0]):
            spare = donor_spare.get((donor.nta_code, donor.rideable_type), 0)
            quantity = min(need, spare, remaining)
            if quantity <= 0:
                continue
            transfers.append({
                "From zone": str(donor.nta_name), "To zone": str(receiver.nta_name),
                "Bicycle": "Electric" if receiver.rideable_type == "electric_bike" else "Classic",
                "Quantity": int(quantity), "Straight-line km": round(distance, 2),
                "Origin code": donor.nta_code, "Destination code": receiver.nta_code,
            })
            donor_spare[(donor.nta_code, donor.rideable_type)] -= quantity
            dest_slots[receiver.nta_code] -= quantity
            need -= quantity
            remaining -= quantity
            if need <= 0 or remaining <= 0:
                break
    return pd.DataFrame(transfers, columns=[
        "From zone", "To zone", "Bicycle", "Quantity", "Straight-line km", "Origin code", "Destination code"
    ])
