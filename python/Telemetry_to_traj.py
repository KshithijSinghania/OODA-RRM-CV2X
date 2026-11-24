#!/usr/bin/env python3
"""0.1 s SUMO telemetry -> ns-3 trajectory file (1 s samples by default).
Streams the big CSV in chunks, so memory stays small.
Usage: python3 telemetry01_to_traj.py big_file.csv traj_S4.csv [--step 1.0] [--scenario NAME_SUBSTRING]
Columns used: time, vehicle_id, x, y, speed, acceleration, angle (scenario_id/scenario_name only for --scenario)."""
import sys, argparse, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument('src'); ap.add_argument('dst')
ap.add_argument('--step', type=float, default=1.0, help='output sampling interval in s (multiple of 0.1)')
ap.add_argument('--scenario', default=None, help='keep rows whose scenario_id or scenario_name contains this text')
a = ap.parse_args()
k = max(1, round(a.step * 10))
hdr = pd.read_csv(a.src, nrows=0).columns.str.strip().tolist()
need = ['time', 'vehicle_id', 'x', 'y', 'speed', 'acceleration', 'angle']
miss = [c for c in need if c not in hdr]
if miss: sys.exit(f'missing columns {miss}; header is {hdr}')
use = need + [c for c in ('scenario_id', 'scenario_name') if c in hdr]
out, first, last, n_in = [], {}, {}, 0
for ch in pd.read_csv(a.src, usecols=use, chunksize=1_000_000):
    n_in += len(ch)
    if a.scenario:
        m = pd.Series(False, index=ch.index)
        for c in ('scenario_id', 'scenario_name'):
            if c in ch: m |= ch[c].astype(str).str.contains(a.scenario, case=False, na=False)
        ch = ch[m]
    ch = ch.copy(); ch['tick'] = (ch.time * 10).round().astype(int)
    for vid, g in ch.groupby('vehicle_id', sort=False):
        first.setdefault(vid, g.iloc[0]); last[vid] = g.iloc[-1]
    out.append(ch[ch.tick % k == 0][need])
    print(f'  read {n_in:,} rows', end='\r')
df = pd.concat(out)
ends = pd.DataFrame([r[need] for r in list(first.values()) + list(last.values())])  # spawn + final point
df = pd.concat([df, ends]).drop_duplicates(['vehicle_id', 'time']).sort_values(['vehicle_id', 'time'])
df = df.rename(columns={'acceleration': 'accel'})
df['time'] = df.time.round(3); df[['x', 'y']] = df[['x', 'y']].round(2)
df[['vehicle_id', 'time', 'x', 'y', 'speed', 'angle', 'accel']].to_csv(a.dst, index=False)
print(f'\n{n_in:,} rows read -> {len(df):,} rows, {df.vehicle_id.nunique()} vehicles, t={df.time.min()}..{df.time.max()}')