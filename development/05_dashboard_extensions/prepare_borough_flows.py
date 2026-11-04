#!/usr/bin/env python3
"""Build compact actual Manhattan/outer-borough trip-flow data.

FIX: Citi Bike ride-level CSVs normally identify stations with SHORT NAMES
(e.g. 5216.06), while station_zones.csv stores the internal station IDs.
The new station_short_names.csv bridges the two ID systems.

Diagnostic first (reads just one short chunk, not the full 8 GB):
  python3 scripts/prepare_borough_flows.py --raw-dir "..." --diagnose

Full run:
  python3 scripts/prepare_borough_flows.py --raw-dir "..."
"""
import argparse
import re
import zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NEEDED = ['started_at','start_station_id','end_station_id','rideable_type','member_casual']
MONTH_RE = re.compile(r'20\d{4}')


def discover_files(folder: Path):
    # Avoid counting an extracted CSV and its ZIP copy in the same month.
    zipped = sorted(folder.rglob('*.zip'))
    months = {m.group(0) for p in zipped if (m := MONTH_RE.search(p.name))}
    csvs = [p for p in sorted(folder.rglob('*.csv'))
            if not ((m := MONTH_RE.search(p.name)) and m.group(0) in months)]
    return zipped + csvs


def iter_chunks(file: Path, chunksize: int):
    readargs = dict(usecols=lambda c: c in NEEDED,
                    dtype={'start_station_id':'string','end_station_id':'string',
                           'rideable_type':'string','member_casual':'string'},
                    chunksize=chunksize, low_memory=False)
    if file.suffix.lower() == '.zip':
        with zipfile.ZipFile(file) as archive:
            for name in archive.namelist():
                if name.lower().endswith('.csv') and not name.startswith('__MACOSX/'):
                    with archive.open(name) as stream:
                        yield from pd.read_csv(stream, **readargs)
    else:
        yield from pd.read_csv(file, **readargs)


def create_borough_map():
    zones = pd.read_csv(ROOT/'processed_data'/'station_zones.csv',
                        dtype={'station_id':'string'}, keep_default_na=False)
    zones = zones.loc[(zones['in_nyc'].astype(str) == '1') & (zones['borough'] != '')].copy()
    lookup = zones.drop_duplicates('station_id').set_index('station_id')['borough'].to_dict()
    aliases_path = ROOT/'processed_data'/'station_short_names.csv'
    if not aliases_path.exists():
        raise FileNotFoundError(f'Missing {aliases_path}. Install the borough fix patch first.')
    aliases = pd.read_csv(aliases_path, dtype='string', keep_default_na=False)
    alias_matches = aliases.loc[aliases['station_id'].isin(lookup)].copy()
    short_map = alias_matches.drop_duplicates('short_name').set_index('short_name')['station_id'].map(lookup).to_dict()
    collision = set(lookup).intersection(short_map)
    for alias in collision:
        if lookup[alias] != short_map[alias]:
            raise ValueError(f'Conflicting station aliases: {alias}')
    return {**lookup, **short_map}, len(lookup), len(short_map)


def match_boroughs(batch, borough_map):
    start = batch['start_station_id'].astype('string').str.strip()
    end = batch['end_station_id'].astype('string').str.strip()
    return start.map(borough_map), end.map(borough_map)


def build(raw_dir: Path, output: Path, chunksize: int, diagnose: bool = False):
    files = discover_files(raw_dir)
    if not files:
        raise FileNotFoundError(f'No raw CSVs or ZIPs found under {raw_dir}')
    mapping, id_count, short_count = create_borough_map()
    print(f'Found {len(files)} raw CSV/ZIP files; mapping includes {id_count} NYC station IDs and {short_count} public short codes.', flush=True)
    if diagnose:
        for file in files:
            for batch in iter_chunks(file, min(chunksize, 10000)):
                if not set(NEEDED).issubset(batch.columns):
                    raise ValueError(f'Unexpected columns in {file}: {list(batch.columns)}')
                origin, dest = match_boroughs(batch, mapping)
                both = origin.notna() & dest.notna()
                cross = both & (origin.eq('Manhattan') ^ dest.eq('Manhattan')).fillna(False)
                print('Diagnostic sample:', file.name)
                print(f'Rows sampled: {len(batch):,}, both stations matched: {int(both.sum()):,} ({both.mean():.1%}), Manhattan crossings: {int(cross.sum()):,}')
                print('Example start_station_id:', batch['start_station_id'].dropna().head(5).tolist())
                print('Example end_station_id:', batch['end_station_id'].dropna().head(5).tolist())
                if not both.any():
                    raise ValueError('Zero station matches in diagnostic. Please share the diagnostic output; do not run the full dataset.')
                print('Diagnostic PASSED. Run again without --diagnose to build all months.')
                return
        raise ValueError('No rows found in raw files')

    results = []
    total_rows = matched = matched_cross = 0
    for file in files:
        print('Reading', file, flush=True)
        n_chunks = file_rows = file_matched = file_cross = 0
        for batch in iter_chunks(file, chunksize):
            if not set(NEEDED).issubset(batch.columns):
                raise ValueError(f'Expected columns missing from {file}: {set(NEEDED) - set(batch.columns)}')
            n_chunks += 1
            n = len(batch)
            file_rows += n
            total_rows += n
            origin, destination = match_boroughs(batch, mapping)
            ok = origin.notna() & destination.notna()
            hit = int(ok.sum())
            matched += hit
            file_matched += hit
            if n_chunks == 1 and hit == 0:
                examples = batch['start_station_id'].dropna().head(5).tolist()
                raise ValueError(f'No station IDs matched in first chunk of {file.name}. Examples: {examples}. Check aliases before processing further.')
            wanted = (origin.eq('Manhattan') ^ destination.eq('Manhattan')).fillna(False) & ok
            if not wanted.any():
                continue
            filtered = batch.loc[wanted].copy()
            filtered['origin_borough'] = origin.loc[wanted].values
            filtered['destination_borough'] = destination.loc[wanted].values
            filtered['hour'] = pd.to_datetime(filtered['started_at'], errors='coerce').dt.floor('h')
            filtered = filtered.dropna(subset=['hour'])
            file_cross += len(filtered)
            matched_cross += len(filtered)
            if filtered.empty:
                continue
            grp = filtered.groupby(['hour','origin_borough','destination_borough',
                                    'rideable_type','member_casual'], observed=True).size().rename('trips').reset_index()
            results.append(grp)
        print(f'  chunks: {n_chunks}, trips: {file_rows:,}, both stations matched: {file_matched:,} ({file_matched/max(file_rows,1):.1%}), cross-borough: {file_cross:,}', flush=True)
    if not results:
        raise ValueError(f'No valid cross-borough trips found. Matched NYC station pairs: {matched:,} of {total_rows:,}.')
    aggregated = pd.concat(results, ignore_index=True)
    aggregated = aggregated.groupby(['hour','origin_borough','destination_borough',
                                     'rideable_type','member_casual'], as_index=False, observed=True)['trips'].sum()
    # Keep the published comparison period aligned with the app's 12-month history.
    aggregated = aggregated.loc[
        aggregated['hour'].ge(pd.Timestamp('2025-10-01')) &
        aggregated['hour'].lt(pd.Timestamp('2026-10-01'))
    ].copy()
    aggregated = aggregated.sort_values(['hour','origin_borough','destination_borough',
                                         'rideable_type','member_casual'])
    output.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(output, index=False, compression='gzip')
    print(f'Output: {output} ({len(aggregated):,} summarized rows)')
    print(f'Raw rides: {total_rows:,}; both NYC station IDs matched: {matched:,} ({matched/max(total_rows,1):.1%}); cross-borough rides: {matched_cross:,}.')
    print('Historical stations absent from the current directory remain excluded (no imputed matches).')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=ROOT/'processed_data'/'borough_directional_flows.csv.gz')
    ap.add_argument('--chunksize', type=int, default=150000)
    ap.add_argument('--diagnose', action='store_true', help='Check the first 10,000 rides without processing all 12 months')
    args = ap.parse_args()
    build(args.raw_dir, args.output, args.chunksize, args.diagnose)
