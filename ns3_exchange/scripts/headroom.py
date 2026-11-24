r"""Plan step 1: does ADAPTIVE (per-cell) static power beat the best single flat power? Uses only existing ns-3 static-power runs (cells files).
  cd ~/exchange/results && python3 ~/exchange/scripts/headroom.py
Loss per (cell, window) = lost/tx from <prefix>_cells_5s.csv (warmup rows dropped). Runs are grouped by power OFFSET in dB vs default (pooled over seeds).
Incomplete runs (fewer windows than the longest) are skipped. 2-fold CROSS-VALIDATION over time (choose settings on one half of the windows, evaluate on the other)
removes selection bias.  Strategies: default (0 dB) / best single flat offset / best offset PER CELL.
CAVEAT: a per-cell mix of offsets was never simulated; neighbouring cells interact (interference), so the per-cell number is an INDICATOR, not a prediction."""
import csv, os, sys, argparse, collections
import numpy as np
RUNS = {  # offset dB -> candidate prefixes (missing files are skipped)
    6: ['sw_high'], 0: ['sw_base', 'sw_base_seed2', 'sw_base_seed3', 'hp_base'], -3: ['hp_m3'],
    -6: ['sw_low', 'hp_m6_s2', 'hp_m6_s3'], -9: ['hp_m9_s2', 'hp_m9_s3'], -10.75: ['hp_static11_s2', 'hp_static11_s3'],
    -13: ['b2_static13', 'hp_static13_s2', 'hp_static13_s3'], -15: ['hp_m15'], -18: ['hp_m18']}
ap = argparse.ArgumentParser(); ap.add_argument('--dir', default='.'); ap.add_argument('--min_pkts', type=float, default=2000, help='min packets per cell (default run) to be judged separately')
a = ap.parse_args()
def load(p):
    d = {}
    for r in csv.DictReader(open(os.path.join(a.dir, p + '_cells_5s.csv'))):
        if r['warmup'] != '0': continue
        tx = float(r['tx_packets'])
        if tx > 0: d[(int(r['cell']), float(r['window_start']))] = (tx, max(0.0, tx - float(r['rx_packets'])))
    return d
data, nruns, skipped = {}, {}, []
loaded = {off: [(p, load(p)) for p in ps if os.path.exists(os.path.join(a.dir, p + '_cells_5s.csv'))] for off, ps in RUNS.items()}
nw = max((len({k[1] for k in d}) for L in loaded.values() for _, d in L), default=0)
for off, L in loaded.items():
    ok = []
    for p, d in L:
        if len({k[1] for k in d}) < nw: skipped.append(p); continue
        ok.append((p, d))
    if not ok: continue
    agg = collections.defaultdict(lambda: [0.0, 0.0])
    for p, d in ok:
        for k, (tx, lo) in d.items(): agg[k][0] += tx; agg[k][1] += lo
    data[off] = dict(agg); nruns[off] = [p for p, _ in ok]
if 0 not in data: sys.exit('need at least the default (0 dB) runs: sw_base ...')
offs = sorted(data); print('offsets/runs used:'); [print(f'  {o:>7} dB : {nruns[o]}') for o in offs]
if skipped: print('skipped (incomplete):', skipped)
wins = sorted({k[1] for k in data[0]}); cells = sorted({k[0] for k in data[0]})
half = {'A': set(wins[:len(wins)//2]), 'B': set(wins[len(wins)//2:])}
def rate(off, c, ws):
    tx = sum(data[off].get((c, w), (0, 0))[0] for w in ws); lo = sum(data[off].get((c, w), (0, 0))[1] for w in ws)
    return lo / tx if tx else np.nan
def wt(c, ws): # weight = default-run traffic per run
    return sum(data[0].get((c, w), (0, 0))[0] for w in ws) / len(nruns[0])
allw = set(wins)
print(f'\nwindows: {len(wins)}  cells with traffic: {len(cells)}')
print('\n== Whole network, all windows: loss % by offset (pooled over seeds) ==')
for o in offs:
    W = sum(wt(c, allw) for c in cells); print(f'  {o:>7} dB : {100*sum(wt(c, allw)*np.nan_to_num(rate(o, c, allw)) for c in cells)/W:6.3f}   ({len(nruns[o])} run(s))')
big = [c for c in cells if wt(c, allw) * len(nruns[0]) >= a.min_pkts]
print(f'\n== Per cell, all windows: loss % by offset (cells with >= {a.min_pkts:.0f} packets in the default runs) ==')
print('cell  pkts/run ' + ''.join(f'{o:>8}' for o in offs) + '   best')
for c in big:
    r = [100 * rate(o, c, allw) for o in offs]; print(f'{c:>4} {wt(c, allw):9.0f} ' + ''.join(f'{x:8.2f}' for x in r) + f'   {offs[int(np.nanargmin(r))]}')
def net(assign, ws):  # assign: cell -> offset ; weights from default traffic
    W = sum(wt(c, ws) for c in cells); return 100 * sum(wt(c, ws) * np.nan_to_num(rate(assign[c], c, ws)) for c in cells) / W
print('\n== 2-fold cross-validation over time (choose on one half, evaluate on the other) ==')
res = collections.defaultdict(list)
for sel, ev in (('A', 'B'), ('B', 'A')):
    S, E = half[sel], half[ev]
    gl = min(offs, key=lambda o: net({c: o for c in cells}, S))
    pc = {}
    for c in cells:
        pc[c] = min(offs, key=lambda o: np.nan_to_num(rate(o, c, S), nan=9) ) if c in big else gl
    res['default'].append(net({c: 0 for c in cells}, E)); res['best flat'].append(net({c: gl for c in cells}, E)); res['per-cell best'].append(net(pc, E))
    print(f'  choose on half {sel}, evaluate on half {ev}: best flat = {gl} dB; per-cell choices = {sorted(set(pc.values()))}; cells changed vs flat: {sum(pc[c] != gl for c in cells)}')
    print(f'     eval loss %: default {res["default"][-1]:.3f}   best flat {res["best flat"][-1]:.3f}   per-cell {res["per-cell best"][-1]:.3f}   gain per-cell vs flat {res["best flat"][-1]-res["per-cell best"][-1]:+.3f} pp')
g = np.mean(res['best flat']) - np.mean(res['per-cell best'])
print(f'\nMEAN of folds: default {np.mean(res["default"]):.3f}  best flat {np.mean(res["best flat"]):.3f}  per-cell {np.mean(res["per-cell best"]):.3f}   => per-cell gain over best flat: {g:+.3f} pp')
print('Reading: seed noise at the network level is about 0.24 pp. A gain well under ~0.3 pp (or not the same sign in both folds) = no usable headroom for per-cell power.')
