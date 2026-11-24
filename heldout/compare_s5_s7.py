#!/usr/bin/env python3
"""Are S5 and S7 the same before the ambulance enters (150 s)?  Compares the Observe inputs window by window.
  python compare_s5_s7.py --a ..\\OODA_S5\\data --b ..\\OODA_S7\\data
Vehicle-side fields (count, speed) come from SUMO; radio-side fields (loss, delay, throughput) come from ns-3.
If vehicle fields match but radio fields differ, the difference is ns-3 randomness, not different traffic."""
import argparse, glob, json, os
ap = argparse.ArgumentParser(); ap.add_argument('--a', required=True); ap.add_argument('--b', required=True); ap.add_argument('--amb_start', type=float, default=150.0); ap.add_argument('--amb_end', type=float, default=625.0); x = ap.parse_args()
def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, 'observe_S?.jsonl'))):
        for l in open(f, encoding='utf-8'):
            r = json.loads(l); out[r['id'].split('-', 1)[1]] = json.loads(r['messages'][1]['content'])
    return out
A, B = load(x.a), load(x.b); keys = sorted(set(A) & set(B)); print(f'windows: A {len(A)}, B {len(B)}, in both {len(keys)}')
VEH = {'vehicles': lambda i: i['n_active_vehicles'], 'speed': lambda i: i['speed_mps_mean']}
RAD = {'loss': lambda i: i['loss_pct']['mean'], 'delay': lambda i: i['delay_ms']['mean'], 'throughput': lambda i: i['throughput_mbps_mean']}
same = lambda p, q: abs((p or 0) - (q or 0)) <= 0.005
for name, sel in (('before ambulance (t < %g s)' % x.amb_start, lambda t: t < x.amb_start), ('ambulance present', lambda t: x.amb_start <= t < x.amb_end), ('after ambulance (t >= %g s)' % x.amb_end, lambda t: t >= x.amb_end)):
    ks = [k for k in keys if sel(A[k]['window_s'][0])]
    if not ks: continue
    print(f'\n{name}: {len(ks)} windows, share of windows with identical value:')
    for nm, fn in {**VEH, **RAD}.items(): print(f'   {nm:12s} {100.0 * sum(same(fn(A[k]), fn(B[k])) for k in ks) / len(ks):6.1f}%')
