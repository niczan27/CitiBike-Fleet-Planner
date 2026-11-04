#!/usr/bin/env python3
"""Citi Bike Stage 3: audited station-hour mapping + compact zone files + forecast models.

Run from CitiBike_Project:
  .venv/bin/python -m pip install scikit-learn joblib
  .venv/bin/python build_forecasts_stage3.py

Requirements: pandas, numpy, scipy, scikit-learn, joblib.
Original raw files and Stage 1/2 products are never modified.
"""
from __future__ import annotations

import json
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "processed_data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MONTHS = pd.period_range("2025-10", "2026-09", freq="M").strftime("%Y%m").tolist()
VALID_BIKES = ["classic_bike", "electric_bike"]
MEMBER_VALUES = ["member", "casual", "unknown"]
# Keep the last two complete months strictly out of training when choosing models.
VALIDATION_START = "2026-08-01"
RANDOM_SEED = 42
POISSON_TRAIN_ROWS = 120_000
BOOST_TRAIN_ROWS = 200_000
BOOST_MAX_ITER = 110
READ_CHUNK = 220_000
MIN_STATION_MATCH_WARNING = 0.95

WEATHER_FIELDS = [
    "temperature_2m", "apparent_temperature", "precipitation", "rain",
    "snowfall", "snow_depth", "wind_speed_10m", "wind_gusts_10m",
]
FEATURE_NAMES = [
    "zone_index", "bike_index", "hour_of_day", "day_of_week", "month",
    "federal_holiday_proxy", "temperature_2m", "apparent_temperature",
    "precipitation", "rain", "snowfall", "snow_depth",
    "wind_speed_10m", "wind_gusts_10m",
]
FEATURE_SHORT = [0, 1, 2, 3, 4, 5]
FEATURE_ALL = list(range(len(FEATURE_NAMES)))
CATEGORICAL_FEATURES = list(range(6))


def require(name: str) -> Path:
    path = OUT / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Complete Stages 1 and 2 first.")
    return path


def normalize_station_id(value):
    """Normalize known shortened numeric Citi Bike station identifiers.

    Historical CSVs may contain e.g. 5216.1 while GBFS short_name is
    5216.10. Always try the exact string first; this is a fallback only.
    UUIDs and long numeric GBFS IDs remain unchanged.
    """
    s = str(value).strip()
    if s.count(".") == 1:
        stem, suffix = s.split(".")
        if stem.isdigit() and len(stem) <= 5 and suffix.isdigit() and len(suffix) == 1:
            return stem + "." + suffix + "0"
    return s


def load_metadata():
    s = pd.read_csv(require("station_zones.csv"), dtype={"station_id": "string", "nta_code": "string"}, keep_default_na=False)
    if s["station_id"].duplicated().any():
        raise ValueError("Duplicate station IDs in station_zones.csv. Resolve before proceeding.")
    s["in_nyc"] = pd.to_numeric(s["in_nyc"], errors="coerce").fillna(0).astype(int)
    active = s[(s["in_nyc"] == 1) & (s["nta_code"] != "")].copy()
    zones = sorted(active["nta_code"].unique().tolist())
    if len(zones) < 10:
        raise ValueError("Too few NYC zones mapped. Check Stage 2 outputs.")

    # Stage 2 exported only the GBFS station_id (often a UUID or long integer).
    # Citi Bike's historical ride CSVs commonly use the GBFS short_name,
    # e.g. 5216.06. Without joining both ID types, coverage is ~0%.
    info_file = ROOT / "reference_data" / "station_information_snapshot.json"
    if not info_file.exists():
        raise FileNotFoundError(f"Missing {info_file}. This comes from your Stage 2 ZIP.")
    stations = json.loads(info_file.read_text(encoding="utf-8"))["data"]["stations"]
    short_by_id = {str(row["station_id"]).strip(): str(row.get("short_name", "")).strip()
                   for row in stations}

    # Preserve the exact station ID / short name whenever unambiguous.
    zone_by_id = dict(zip(s["station_id"].astype(str).str.strip(), s["nta_code"].astype(str)))
    in_nyc_by_id = dict(zip(s["station_id"].astype(str).str.strip(), s["in_nyc"].astype(int)))
    known_aliases = defaultdict(set)
    active_aliases = defaultdict(set)
    for sid, code in zone_by_id.items():
        short = short_by_id.get(sid, "")
        for alias in (sid, short):
            if not alias:
                continue
            known_aliases[alias].add(sid)
            if in_nyc_by_id[sid] == 1 and code:
                active_aliases[alias].add(code)
    ambiguous = {alias for alias, ids in known_aliases.items() if len(ids) > 1}
    if ambiguous:
        print(f"  WARNING: {len(ambiguous)} duplicated GBFS identifier aliases; excluding ambiguous aliases.", flush=True)
    station_lookup = {alias: next(iter(codes)) for alias, codes in active_aliases.items()
                      if alias not in ambiguous and len(codes) == 1}
    all_stations = {alias for alias in known_aliases if alias not in ambiguous}
    if not station_lookup:
        raise ValueError("Station identifier lookup is empty; check Stage 2 station snapshot.")
    print(f"  Built {len(station_lookup):,} NYC station aliases (UUID/GBFS station_id plus short_name).", flush=True)
    return s, zones, station_lookup, all_stations


def weather_and_calendar():
    path = require("weather_hourly_202510_202609.csv")
    weather = pd.read_csv(path, dtype={"hour": "string"})
    missing = [f for f in ["hour", *WEATHER_FIELDS] if f not in weather.columns]
    if missing:
        raise ValueError(f"Weather file lacks fields: {missing}")
    weather["hour"] = pd.to_datetime(weather["hour"], errors="coerce")
    if weather["hour"].isna().any():
        raise ValueError("Unparseable hourly timestamps in weather file.")
    original_rows = len(weather)
    # On the fall-back DST day local 01:00 repeats. Stage 1's local-hour trip keys
    # cannot distinguish those hours; average duplicated weather rows, rather than
    # creating a many-to-many join that would multiply the trip counts.
    weather = weather.groupby("hour", as_index=False)[WEATHER_FIELDS].mean(numeric_only=True)
    weather = weather.sort_values("hour").reset_index(drop=True)
    if weather["hour"].duplicated().any():
        raise ValueError("Duplicate local weather hour remains after grouping.")
    if len(weather) < 8500:
        raise ValueError(f"Expected around 8760 weather hours, found {len(weather)}")
    for f in WEATHER_FIELDS:
        weather[f] = pd.to_numeric(weather[f], errors="coerce")
    null_before = {f: int(weather[f].isna().sum()) for f in WEATHER_FIELDS}
    for f in WEATHER_FIELDS:
        # Snow variables can have naturally missing readings; zero for missing snow.
        # Other weather fields use the observed median, and missingness is reported.
        replacement = 0.0 if f in {"snowfall", "snow_depth"} else weather[f].median()
        if pd.isna(replacement):
            replacement = 0.0
        weather[f] = weather[f].fillna(float(replacement))

    times = pd.DatetimeIndex(weather["hour"])
    from pandas.tseries.holiday import USFederalHolidayCalendar
    federal = USFederalHolidayCalendar().holidays(start=times.min().normalize(), end=times.max().normalize())
    time_features = pd.DataFrame({
        "hour": times,
        "hour_of_day": times.hour,
        "day_of_week": times.dayofweek,
        "month": times.month,
        "federal_holiday_proxy": times.normalize().isin(federal).astype(int),
    })
    time_features = pd.concat([time_features, weather[WEATHER_FIELDS]], axis=1)
    return time_features, original_rows, null_before


def accumulate_events(prefix, months, time_keys, zone_keys, station_lookup, all_stations,
                      data, audit_rows, unmatched_counter):
    """Add station-hour-member counts directly into small integer arrays."""
    time_to_idx = {hour: i for i, hour in enumerate(time_keys)}
    zone_to_idx = {z: i for i, z in enumerate(zone_keys)}
    bike_to_idx = {b: i for i, b in enumerate(VALID_BIKES)}
    mem_to_idx = {m: i for i, m in enumerate(MEMBER_VALUES)}
    seen_any = False

    for month in months:
        file = require(f"{prefix}_{month}.csv.gz")
        month_total = mapped = outside_nyc = id_absent = outside_hours = invalid = 0
        print(f"  {prefix:10} {month} ...", flush=True)
        for chunk in pd.read_csv(file, compression="gzip", chunksize=READ_CHUNK,
                                 dtype={"station_id": "string", "rideable_type": "string",
                                        "member_casual": "string", "hour": "string"}):
            seen_any = True
            chunk["rides"] = pd.to_numeric(chunk["rides"], errors="coerce").fillna(0).astype(np.int64)
            # Event counts, not file rows, are the basis for match percentages.
            month_total += int(chunk["rides"].sum())
            sid = chunk["station_id"].fillna("").str.strip()
            z = sid.map(station_lookup)
            has_id = sid.isin(all_stations)
            # Fallback only for a shortened numeric suffix, such as 5216.1 -> 5216.10.
            # Do not modify an ID that already matches exactly.
            unresolved = ~has_id
            if unresolved.any():
                normalized = sid[unresolved].map(normalize_station_id)
                z.loc[unresolved] = normalized.map(station_lookup)
                has_id.loc[unresolved] = normalized.isin(all_stations).to_numpy()
            unknown_mask = ~has_id
            id_absent += int(chunk.loc[unknown_mask, "rides"].sum())
            outside_mask = has_id & z.isna()
            outside_nyc += int(chunk.loc[outside_mask, "rides"].sum())
            if unknown_mask.any():
                for key, value in chunk.loc[unknown_mask].groupby(sid[unknown_mask])["rides"].sum().items():
                    unmatched_counter[(prefix, str(key))] += int(value)
            t_idx = chunk["hour"].map(time_to_idx)
            for valid_z_idx in (z.map(zone_to_idx),):
                b_idx = chunk["rideable_type"].map(bike_to_idx)
                m_idx = chunk["member_casual"].fillna("unknown").str.strip().str.lower().map(mem_to_idx).fillna(2)
                valid = valid_z_idx.notna() & t_idx.notna() & b_idx.notna() & (chunk["rides"] > 0)
                # Report records outside the 12-month weather interval separately.
                outside_hours += int(chunk.loc[z.notna() & t_idx.isna(), "rides"].sum())
                invalid += int(chunk.loc[z.notna() & t_idx.notna() & b_idx.isna(), "rides"].sum())
                if valid.any():
                    v = chunk.loc[valid, "rides"].to_numpy(dtype=np.int64)
                    mapped += int(v.sum())
                    np.add.at(data, (t_idx[valid].to_numpy(dtype=int), valid_z_idx[valid].to_numpy(dtype=int),
                                     b_idx[valid].to_numpy(dtype=int), m_idx[valid].to_numpy(dtype=int)), v)
        audit_rows.append({
            "direction": prefix, "month": month, "events_in_aggregates": month_total,
            "usable_mapped_events": mapped, "usable_fraction": mapped / month_total if month_total else 0,
            "station_id_not_in_snapshot": id_absent,
            "station_in_snapshot_outside_nyc": outside_nyc,
            "mapped_but_outside_weather_window": outside_hours,
            "mapped_but_invalid_bike_type": invalid,
        })
    if not seen_any:
        raise RuntimeError(f"No {prefix} events were loaded.")


def save_compact_views(times, zones, dep4, ret4):
    hour_strings = times.strftime("%Y-%m-%d %H:00:00")
    for month in MONTHS:
        time_idx = np.where(times.strftime("%Y%m") == month)[0]
        if not len(time_idx):
            continue
        n_t, n_z, n_b, n_m = len(time_idx), len(zones), 2, 3
        sub_d, sub_r = dep4[time_idx], ret4[time_idx]
        dense_depart = sub_d.sum(axis=3).reshape(-1)
        dense_return = sub_r.sum(axis=3).reshape(-1)
        hourly = pd.DataFrame({
            "hour": np.repeat(hour_strings[time_idx], n_z*n_b),
            "nta_code": np.tile(np.repeat(zones, n_b), n_t),
            "rideable_type": np.tile(VALID_BIKES, n_t*n_z),
            "departures": dense_depart,
            "returns": dense_return,
        })
        hourly.to_csv(OUT / f"zone_hourly_{month}.csv.gz", index=False, compression="gzip")
        # Keep member/casual breakdown for the application's historical filters.
        nonzero = np.nonzero(sub_d + sub_r)
        members = pd.DataFrame({
            "hour": hour_strings[time_idx][nonzero[0]],
            "nta_code": np.asarray(zones)[nonzero[1]],
            "rideable_type": np.asarray(VALID_BIKES)[nonzero[2]],
            "member_casual": np.asarray(MEMBER_VALUES)[nonzero[3]],
            "departures": sub_d[nonzero],
            "returns": sub_r[nonzero],
        })
        members.to_csv(OUT / f"zone_hourly_member_{month}.csv.gz", index=False, compression="gzip")
        print(f"  Saved zone/hour files for {month} ({len(hourly):,} rows; {len(members):,} nonzero member groups)", flush=True)


def feature_matrix(tf, zones):
    ntime = len(tf)
    nz = len(zones)
    nb = len(VALID_BIKES)
    nrows = ntime * nz * nb
    x = np.empty((nrows, len(FEATURE_NAMES)), dtype=np.float32)
    x[:, 0] = np.tile(np.repeat(np.arange(nz), nb), ntime)
    x[:, 1] = np.tile(np.arange(nb), ntime*nz)
    for i, field in enumerate(FEATURE_NAMES[2:], start=2):
        x[:, i] = np.repeat(tf[field].to_numpy(dtype=np.float32), nz*nb)
    return x


def metric(y_true, y_pred):
    p = np.maximum(0, np.asarray(y_pred, dtype=np.float64))
    return {
        "mae": float(mean_absolute_error(y_true, p)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, p))),
        "bias_trips_per_zone_hour": float(np.mean(p - y_true)),
    }


def baseline_fit(x, y):
    # Mean rides for each (zone, bicycle type, weekday, hour) cell.
    # This is a transparent, weather-free comparator.
    z, b, hr, dow = [x[:, k].astype(np.int32) for k in (0,1,2,3)]
    keys = ((z*2+b)*7+dow)*24+hr
    total = int(keys.max())+1
    cnt = np.bincount(keys, minlength=total)
    sums = np.bincount(keys, weights=y, minlength=total)
    baseline = np.divide(sums, cnt, out=np.zeros(total), where=cnt>0)
    return baseline


def baseline_predict(model, x):
    z, b, hr, dow = [x[:, k].astype(np.int32) for k in (0,1,2,3)]
    k = ((z*2+b)*7+dow)*24+hr
    return model[k]


def glm_pipeline():
    # One-hot zone, bike, hour, weekday, month, federal holiday + scaled weather.
    # Sparse preprocessing keeps regression training practical on laptops.
    return Pipeline([
        ("transform", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("weather", StandardScaler(), list(range(6, len(FEATURE_NAMES)))),
        ], sparse_threshold=1.0)),
        ("glm", PoissonRegressor(alpha=0.015, max_iter=55, tol=1e-4)),
    ])


def boosted_model(with_weather):
    return HistGradientBoostingRegressor(
        loss="poisson", max_iter=BOOST_MAX_ITER, learning_rate=0.085,
        max_leaf_nodes=25, min_samples_leaf=75, l2_regularization=4.0,
        categorical_features=CATEGORICAL_FEATURES,
        early_stopping=False, random_state=RANDOM_SEED,
    )


def choose_indices(pool, maximum, seed):
    if len(pool) <= maximum:
        return pool
    return np.sort(np.random.default_rng(seed).choice(pool, size=maximum, replace=False))


def run_models(x, y_depart, y_return, times, zones):
    # Split by clock date, not randomly by record, to avoid mixing future hours into training.
    time_train = (times < pd.Timestamp(VALIDATION_START)).to_numpy() if isinstance(times, pd.Series) else np.asarray(times < pd.Timestamp(VALIDATION_START))
    n_per_hour = len(zones)*2
    row_train = np.repeat(time_train, n_per_hour)
    train_ix = np.flatnonzero(row_train)
    test_ix = np.flatnonzero(~row_train)
    if not len(train_ix) or not len(test_ix):
        raise RuntimeError("Cannot create chronological training / validation split.")
    x_test = x[test_ix]
    if len(test_ix) < 15_000:
        raise RuntimeError("Unexpectedly small validation period.")
    print(f"Train: {len(train_ix):,} zone/bike/hour records; validation: {len(test_ix):,} records.", flush=True)
    print("Comparing baseline, Poisson regression, boosted trees with and without weather...", flush=True)
    glm_ix = choose_indices(train_ix, POISSON_TRAIN_ROWS, RANDOM_SEED)
    boost_ix = choose_indices(train_ix, BOOST_TRAIN_ROWS, RANDOM_SEED+1)
    report_rows = []
    winners = {}
    saved = {}
    for direction, y in [("departures", y_depart), ("returns", y_return)]:
        print(f"\nForecast target: {direction}", flush=True)
        y = y.astype(np.float32, copy=False)
        y_train, y_test = y[train_ix], y[test_ix]
        candidates = {}
        bline = baseline_fit(x[train_ix], y_train)
        p0 = baseline_predict(bline, x_test)
        candidates["historical_hourly_average"] = (bline, metric(y_test, p0), "baseline")
        print("  Baseline MAE:", round(candidates["historical_hourly_average"][1]["mae"],3), flush=True)

        glm = glm_pipeline()
        try:
            t0 = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                glm.fit(x[glm_ix], y[glm_ix])
            res = metric(y_test, glm.predict(x_test))
            candidates["poisson_regression_weather"] = (glm, res, "poisson")
            print(f"  Poisson MAE: {res['mae']:.3f} ({time.perf_counter()-t0:.1f}s)", flush=True)
            if caught:
                print("   Warning:", str(caught[-1].message)[:220], flush=True)
        except Exception as exc:
            print("  Poisson training failed:", str(exc)[:250], flush=True)
            report_rows.append({"target": direction, "model": "poisson_regression_weather",
                                "status": "failed", "reason": str(exc)[:250]})

        for label, cols in [("boosted_trees_without_weather", FEATURE_SHORT),
                            ("boosted_trees_with_weather", FEATURE_ALL)]:
            try:
                t0 = time.perf_counter()
                model = boosted_model(len(cols)>len(FEATURE_SHORT))
                model.fit(x[boost_ix][:, cols], y[boost_ix])
                pred = model.predict(x_test[:, cols])
                res = metric(y_test, pred)
                candidates[label] = (model, res, "boost")
                print(f"  {label} MAE: {res['mae']:.3f} ({time.perf_counter()-t0:.1f}s)", flush=True)
            except Exception as exc:
                print("  Model training failed:", label, str(exc)[:250], flush=True)
                report_rows.append({"target":direction,"model":label,"status":"failed","reason":str(exc)[:250]})

        for label, (model, res, kind) in candidates.items():
            report_rows.append({"target": direction, "model": label, "status":"ok", **res})

        best = min(candidates, key=lambda name: candidates[name][1]["mae"])
        print("  Selected:", best, flush=True)
        winners[direction] = best
        kind = candidates[best][2]
        # Refit winning method using all available months after its comparison.
        all_ix = np.arange(len(y))
        if kind == "baseline":
            final = baseline_fit(x, y)
            cols = FEATURE_SHORT
        elif kind == "poisson":
            final = glm_pipeline()
            final_ix = choose_indices(all_ix, POISSON_TRAIN_ROWS, RANDOM_SEED+8)
            final.fit(x[final_ix], y[final_ix])
            cols = FEATURE_ALL
        else:
            cols = FEATURE_ALL if best.endswith("with_weather") else FEATURE_SHORT
            final_ix = choose_indices(all_ix, BOOST_TRAIN_ROWS, RANDOM_SEED+8)
            final = boosted_model(len(cols)>len(FEATURE_SHORT))
            final.fit(x[final_ix][:, cols], y[final_ix])
        out = MODEL_DIR / f"{direction}_model.joblib"
        joblib.dump({"model_name":best,"method":kind,"selected_feature_columns":cols,
                     "feature_names":FEATURE_NAMES,"estimator":final,"zone_codes":zones,
                     "bike_types":VALID_BIKES},out,compress=3)
        saved[direction] = out.name
        print("  Saved:", out, flush=True)
    scores = pd.DataFrame(report_rows)
    scores.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
    return winners, saved, report_rows


def main():
    print("Stage 3: station-hour joining and forecast model comparison", flush=True)
    s, zones, station_lookup, all_stations = load_metadata()
    tf, original_weather_rows, weather_missing = weather_and_calendar()
    times = pd.DatetimeIndex(tf["hour"])
    time_keys = times.strftime("%Y-%m-%d %H:00:00").tolist()
    shape = (len(times), len(zones), 2, len(MEMBER_VALUES))
    print(f"Using {len(zones)} NTAs and {len(times)} unique local-clock weather hours.", flush=True)
    dep = np.zeros(shape, dtype=np.int32)
    ret = np.zeros(shape, dtype=np.int32)
    audit = []
    unknown_counter = Counter()
    accumulate_events("departures", MONTHS, time_keys, zones, station_lookup, all_stations, dep, audit, unknown_counter)
    accumulate_events("returns", MONTHS, time_keys, zones, station_lookup, all_stations, ret, audit, unknown_counter)
    df_audit = pd.DataFrame(audit)
    df_audit.to_csv(OUT / "station_matching_audit.csv", index=False)
    unknown = [{"direction":k[0],"station_id":k[1],"rides":v} for k,v in unknown_counter.most_common()]
    pd.DataFrame(unknown, columns=["direction","station_id","rides"]).to_csv(
        OUT / "unknown_historical_station_ids.csv", index=False)
    totals = df_audit.groupby("direction")[["events_in_aggregates","usable_mapped_events"]].sum()
    station_coverage = {}
    for direction, v in totals.iterrows():
        frac = float(v["usable_mapped_events"]/max(v["events_in_aggregates"],1))
        station_coverage[direction] = frac
        print(f"  {direction}: {v['usable_mapped_events']:,} of {v['events_in_aggregates']:,} events usable ({frac:.2%}).", flush=True)
        if frac < MIN_STATION_MATCH_WARNING:
            print(f"  WARNING: {direction} mapped event coverage is below {MIN_STATION_MATCH_WARNING:.0%}.",flush=True)
            print("  Review unknown_historical_station_ids.csv before using forecasts operationally.",flush=True)

    # Never train on all-zero targets or overwrite valid model artifacts after
    # a station-ID mismatch. Save audit CSVs above so a problem can be diagnosed.
    minimum_coverage = 0.80
    if min(station_coverage.values()) < minimum_coverage:
        print("\nTOP UNMATCHED HISTORICAL STATION IDs:", flush=True)
        for r in unknown[:12]:
            print(f"  {r['direction']}: {r['station_id']} ({r['rides']:,} events)", flush=True)
        raise RuntimeError(
            f"Station matching failed: <{minimum_coverage:.0%} of trips matched. "
            "No valid models have been trained. Upload "
            "processed_data/unknown_historical_station_ids.csv and "
            "processed_data/station_matching_audit.csv for diagnosis."
        )
    if not dep.any() or not ret.any():
        raise RuntimeError("Zero departure or return target: refusing to train invalid models.")

    save_compact_views(times, zones, dep, ret)
    matrix = feature_matrix(tf, zones)
    dep_y = dep.sum(axis=3).reshape(-1)
    ret_y = ret.sum(axis=3).reshape(-1)
    winner, saved, comparisons = run_models(matrix, dep_y, ret_y, times, zones)
    details = {
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "station_snapshot_rows":int(len(s)),
        "nyc_ntas":len(zones),"nyc_stations":len(station_lookup),
        "weather_original_rows":original_weather_rows,
        "weather_unique_local_clock_hours":len(times),
        "candidate_training_sample_limits":{"poisson":POISSON_TRAIN_ROWS,"boosted_trees":BOOST_TRAIN_ROWS},
        "weather_missing_values_before_filling":weather_missing,
        "event_station_mapping_coverage":station_coverage,
        "training_period":"2025-10-01 to 2026-07-31",
        "holdout_period":"2026-08-01 to 2026-09-30",
        "selected_models":winner,
        "model_files":saved,
        "all_models_refit_on_full_history":True,
        "id_matching_method":"GBFS station_id + short_name with cautious numeric-suffix fallback",
        "operational_limitations":[
            "Targets represent completed trips, not attempted rentals or unmet demand.",
            "NTA zone forecasts exclude trips with historical station IDs not matched to the current station snapshot.",
            "Historical returns/departures may reflect relocation and maintenance not visible in trip records.",
            "DST fall repeated hour is combined in local-hour historical trip counts; weather for duplicate hour is averaged.",
            "Open-Meteo historical weather at a central NYC point is not an archive of forecasts issued at each past date.",
            "Public holidays feature is a US federal holiday proxy, not a complete NYC event/school calendar.",
            "Predictions must be converted to inventory targets with dock constraints in the app; this script does not relocate bikes.",
        ],
    }
    (OUT/"stage3_report.json").write_text(json.dumps(details,indent=2),encoding="utf-8")
    (MODEL_DIR/"model_features.json").write_text(json.dumps({
        "feature_names":FEATURE_NAMES,"bike_types":VALID_BIKES,"zone_codes":zones,
        "holiday_calendar":"US federal holiday proxy",
        "models":saved,"winners":winner,
    },indent=2),encoding="utf-8")
    print("\nDone. Please upload:",flush=True)
    print("  processed_data/stage3_report.json",flush=True)
    print("  processed_data/station_matching_audit.csv",flush=True)
    print("  models/model_comparison.csv",flush=True)
    print("If any direction has <95% coverage, also upload:",flush=True)
    print("  processed_data/unknown_historical_station_ids.csv",flush=True)


if __name__=="__main__":
    main()
