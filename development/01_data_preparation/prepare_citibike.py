#!/usr/bin/env python3
"""Prepare Citi Bike's nested monthly CSV files without loading a full year into memory.

Usage:
    python3 prepare_citibike.py

Expected layout:
    CitiBike_Project/
       prepare_citibike.py
       raw_data/
          202510-citibike-tripdata/  (CSV parts inside)
          ...
       processed_data/  (created automatically)

Requires: pandas (python3 -m pip install pandas)
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw_data"
OUTPUT = ROOT / "processed_data"
FIRST_MONTH = "202510"
LAST_MONTH = "202609"
CHUNK_SIZE = 150_000
VALID_TYPES = {"classic_bike", "electric_bike"}
COLS = [
    "ride_id", "rideable_type", "started_at", "ended_at",
    "start_station_id", "end_station_id", "member_casual",
]
REQUIRED = {"rideable_type", "started_at", "ended_at", "start_station_id", "end_station_id"}


def month_from_path(path: Path) -> str | None:
    # Monthly folders and their individual parts normally contain YYYYMM in their names.
    for part in reversed(path.relative_to(RAW).parts):
        match = re.search(r"(?<!\d)(20\d{2})(0[1-9]|1[0-2])(?!\d)", part)
        if match:
            return "".join(match.groups())
    return None


def write_counts(path: Path, counts: Counter, header: list[str]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header + ["rides"])
        for key, value in sorted(counts.items()):
            writer.writerow([*key, value])


def prepare_month(month: str, paths: list[Path]) -> dict:
    dep = Counter()
    ret = Counter()
    daily = Counter()
    duration_histogram = Counter()
    scanned = kept = bad_date_or_duration = outside_month = invalid_bike = 0
    missing_departure_station = missing_return_station = 0
    earliest = f"{month[:4]}-{month[4:]}-01"
    next_month = pd.Timestamp(earliest) + pd.offsets.MonthBegin(1)
    start_month = pd.Timestamp(earliest)

    for i, file in enumerate(paths, start=1):
        print(f"  [{i}/{len(paths)}] Reading {file.name}", flush=True)
        with open(file, "r", encoding="utf-8-sig", newline="") as handle:
            headers = set(next(csv.reader(handle)))
        if not REQUIRED.issubset(headers):
            raise ValueError(f"Missing columns in {file}: {sorted(REQUIRED - headers)}")
        selected = [c for c in COLS if c in headers]
        dtype = {c: "string" for c in selected if c not in {"started_at", "ended_at"}}
        for chunk in pd.read_csv(
            file, usecols=selected, dtype=dtype,
            chunksize=CHUNK_SIZE, low_memory=False, encoding="utf-8-sig",
        ):
            scanned += len(chunk)
            started = pd.to_datetime(chunk["started_at"], errors="coerce", format="mixed")
            ended = pd.to_datetime(chunk["ended_at"], errors="coerce", format="mixed")
            duration_min = (ended - started).dt.total_seconds() / 60.0
            valid_time = started.notna() & ended.notna() & duration_min.gt(0) & duration_min.le(24 * 60)
            bad_date_or_duration += int((~valid_time).sum())
            valid_month = started.ge(start_month) & started.lt(next_month)
            outside_month += int((valid_time & ~valid_month).sum())
            bike_type = chunk["rideable_type"].str.strip().str.lower()
            valid_bike = bike_type.isin(VALID_TYPES)
            invalid_bike += int((valid_time & valid_month & ~valid_bike).sum())
            good = valid_time & valid_month & valid_bike
            if not good.any():
                continue
            x = chunk.loc[good].copy()
            x["started_at"] = started.loc[good]
            x["ended_at"] = ended.loc[good]
            x["duration_minutes"] = duration_min.loc[good]
            x["rideable_type"] = bike_type.loc[good]
            if "member_casual" not in x:
                x["member_casual"] = "unknown"
            x["member_casual"] = x["member_casual"].fillna("unknown").str.strip().str.lower()
            for field in ("start_station_id", "end_station_id"):
                x[field] = x[field].str.strip().replace("", pd.NA)
            kept += len(x)

            # Day-level usage summary for dashboard trends and filtering.
            daily_df = pd.DataFrame({
                "date": x["started_at"].dt.strftime("%Y-%m-%d"),
                "rideable_type": x["rideable_type"],
                "member_casual": x["member_casual"],
            })
            daily.update(daily_df.groupby(["date", "rideable_type", "member_casual"]).size().to_dict())

            # Minutes rounded to the nearest minute: approximate ride-duration medians.
            duration_df = pd.DataFrame({
                "rideable_type": x["rideable_type"],
                "member_casual": x["member_casual"],
                "duration_minutes_rounded": x["duration_minutes"].round().clip(0, 1440).astype(int),
            })
            duration_histogram.update(duration_df.groupby([
                "rideable_type", "member_casual", "duration_minutes_rounded"
            ]).size().to_dict())

            for station_field, time_field, target in [
                ("start_station_id", "started_at", dep),
                ("end_station_id", "ended_at", ret),
            ]:
                has_station = x[station_field].notna()
                missing = int((~has_station).sum())
                if station_field == "start_station_id":
                    missing_departure_station += missing
                else:
                    missing_return_station += missing
                if not has_station.any():
                    continue
                y = x.loc[has_station]
                event_hour = y[time_field].dt.strftime("%Y-%m-%d %H:00:00")
                # The return may cross into the following month; keep its true hour.
                summary = pd.DataFrame({
                    "hour": event_hour,
                    "station_id": y[station_field],
                    "rideable_type": y["rideable_type"],
                    "member_casual": y["member_casual"],
                })
                target.update(summary.groupby([
                    "hour", "station_id", "rideable_type", "member_casual"
                ]).size().to_dict())

    write_counts(OUTPUT / f"departures_{month}.csv.gz", dep,
                 ["hour", "station_id", "rideable_type", "member_casual"])
    write_counts(OUTPUT / f"returns_{month}.csv.gz", ret,
                 ["hour", "station_id", "rideable_type", "member_casual"])
    write_counts(OUTPUT / f"daily_usage_{month}.csv.gz", daily,
                 ["date", "rideable_type", "member_casual"])
    write_counts(OUTPUT / f"duration_histogram_{month}.csv.gz", duration_histogram,
                 ["rideable_type", "member_casual", "duration_minutes_rounded"])
    return dict(month=month, csv_files=len(paths), scanned_rows=scanned,
                valid_trips=kept, invalid_timestamp_or_duration=bad_date_or_duration,
                outside_stated_month=outside_month, invalid_bike_type=invalid_bike,
                missing_start_station_id=missing_departure_station,
                missing_end_station_id=missing_return_station,
                hourly_departure_groups=len(dep), hourly_return_groups=len(ret))


def main() -> None:
    if not RAW.is_dir():
        raise SystemExit(f"Folder not found: {RAW}\nCreate raw_data next to this script and put all monthly folders inside it.")
    OUTPUT.mkdir(exist_ok=True)
    by_month = defaultdict(list)
    ignored = []
    for path in sorted(RAW.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".csv":
            continue
        if any(part.upper().startswith("JC-") for part in path.relative_to(RAW).parts):
            ignored.append(str(path.relative_to(RAW)))
            continue
        month = month_from_path(path)
        if month is not None and FIRST_MONTH <= month <= LAST_MONTH:
            by_month[month].append(path)
        else:
            ignored.append(str(path.relative_to(RAW)))
    if not by_month:
        raise SystemExit("No NYC monthly CSV files found in raw_data. Check your folder names.")
    months_expected = pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M").strftime("%Y%m").tolist()
    missing = sorted(set(months_expected) - set(by_month))
    if missing:
        print("WARNING: These months have no CSVs:", ", ".join(missing), flush=True)
    print(f"Found {sum(map(len,by_month.values()))} CSV files in {len(by_month)} months.", flush=True)
    summary = []
    for month, paths in sorted(by_month.items()):
        print(f"\nProcessing {month} ({len(paths)} CSV part(s))", flush=True)
        result = prepare_month(month, paths)
        summary.append(result)
        print(f"  Valid trips: {result['valid_trips']:,}; hourly departure groups: {result['hourly_departure_groups']:,}", flush=True)
    pd.DataFrame(summary).to_csv(OUTPUT / "quality_summary.csv", index=False)
    manifest = {
        "first_month": FIRST_MONTH, "last_month": LAST_MONTH,
        "months_processed": list(sorted(by_month)),
        "missing_months": missing, "ignored_csvs": ignored,
        "notes": [
            "These files aggregate observed completed rides, not unmet rental demand.",
            "The return timestamp can be in a different month from the ride start.",
            "Hourly timestamps are local clock readings supplied by Citi Bike; DST repeated hours are not disambiguated.",
            "Station IDs and availability feeds must be audited before station/zone matching.",
            "Raw CSVs are preserved and can be reused for origin-destination analysis and duplicate checks.",
            "Trip IDs are not deduplicated across CSVs in this first-pass aggregation.",
        ],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nDone. Output: " + str(OUTPUT), flush=True)
    print("Review quality_summary.csv for missing dates/stations before modeling.", flush=True)


if __name__ == "__main__":
    main()
