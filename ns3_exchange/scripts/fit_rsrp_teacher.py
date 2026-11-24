r"""Fit the RSRP teacher on S7 ONLY (S5 stays unseen):  cd ~/exchange/results && python3 ~/exchange/scripts/fit_rsrp_teacher.py
Idea (research: regime-conditioned labels): a cell whose served vehicles have WEAK signal is coverage-limited -> cut little;
a cell with STRONG signal is interference-limited -> cut deep.  Feature per (cell, 5 s window) = 10th percentile of the served
vehicles' RSRP at DEFAULT power (from the default runs), so it does not depend on the power being tested.
For each feature bin and each flat offset, loss is pooled over all S7 runs with that offset; the rule picks the lowest-loss offset.
Limitation (stated): flat runs move neighbours together with the cell, so own and neighbour effects are not separated.
Writes rsrp_teacher.json (used by ctrl_server.py --policy rsrp)."""
import csv, os, json, math, argparse, collections
ap = argparse.ArgumentParser(); ap.add_argument('--nbins', type=int, default=4); ap.add_argument('--min_ues', type=int, default=3)
ap.add_argument('--out', default='rsrp_teacher.json'); a = ap.parse_args()
S7 = {6: ['sw_high'], 0: ['sw_base', 'sw_base_seed2', 'sw_base_seed3', 'hp_base'], -3: ['hp_m3'], -6: ['sw_low', 'hp_m6_s2', 'hp_m6_s3'],
      -9: ['hp_m9_s2', 'hp_m9_s3'], -10.75: ['hp_static11_s2', 'hp_static11_s3'], -13: ['b2_static13', 'hp_static13_s2', 'hp_static13_s3'],
      -15: ['hp_m15'], -18: ['hp_m18']}
S5_DEFAULT = ['s5_base', 's5_base_s2', 's5_base_s3']
def pct(v, q):
    v = sorted(v); i = (len(v) - 1) * q; lo = int(i); hi = min(lo + 1, len(v) - 1); return v[lo] + (v[hi] - v[lo]) * (i - lo)
def load(p):
    f = p + '_ns3_5s.csv'
    if not os.path.exists(f): return None
    d = collections.defaultdict(lambda: [0.0, 0.0, []])   # (cell, window) -> tx, lost, rsrp list
    for r in csv.DictReader(open(f)):
        if r['warmup'] != '0' or float(r['tx_packets']) <= 0: continue
        k = (int(float(r['serving_cell'])), float(r['window_start'])); e = d[k]
        e[0] += float(r['tx_packets']); e[1] += float(r['lost_packets'])
        x = float(r['rsrp_dbm']) if r['rsrp_dbm'] not in ('', 'nan', '-nan') else float('nan')
        if math.isfinite(x): e[2].append(x)
    return d
def feature(rs): return (pct(rs, 0.10) if len(rs) >= a.min_ues else sum(rs) / len(rs)) if rs else None
runs = {o: [(p, load(p)) for p in ps] for o, ps in S7.items()}
runs = {o: [(p, d) for p, d in L if d] for o, L in runs.items()}; runs = {o: L for o, L in runs.items() if L}
print('S7 runs used:', {o: [p for p, _ in L] for o, L in runs.items()})
# feature from default runs (pooled RSRP lists)
feat = {}
for _, d in runs[0]:
    for k, e in d.items(): feat.setdefault(k, []).extend(e[2])
feat = {k: feature(v) for k, v in feat.items() if v}
vals = sorted(feat.values()); edges = [pct(vals, i / a.nbins) for i in range(1, a.nbins)]
def b(x): return sum(x > e for e in edges)
wins = sorted({k[1] for k in feat}); half = {'A': set(wins[:len(wins) // 2]), 'B': set(wins[len(wins) // 2:])}
def table(ws):
    T = collections.defaultdict(lambda: [0.0, 0.0])   # (bin, offset) -> tx, lost (pooled, per-run averaged by counting runs)
    for o, L in runs.items():
        for _, d in L:
            for k, e in d.items():
                if k in feat and k[1] in ws: t = T[(b(feat[k]), o)]; t[0] += e[0] / len(L); t[1] += e[1] / len(L)
    return T
offs = sorted(runs); T = table(set(wins))
print(f'\nfeature = p10 RSRP of served vehicles at default power; bin edges (dBm): {[round(e, 1) for e in edges]}')
print('loss % by bin (rows) and flat offset (columns), S7 pooled:')
print('bin  range(dBm)        ' + ''.join(f'{o:>8}' for o in offs) + '    best')
rule = []
for i in range(a.nbins):
    lo = '-inf' if i == 0 else f'{edges[i-1]:.1f}'; hi = '+inf' if i == a.nbins - 1 else f'{edges[i]:.1f}'
    r = {o: 100 * T[(i, o)][1] / T[(i, o)][0] for o in offs if T[(i, o)][0] > 0}
    best = min(r, key=r.get); rule.append(best)
    print(f'{i:3d}  {lo:>7}..{hi:<7}  ' + ''.join(f'{r.get(o, float("nan")):8.2f}' for o in offs) + f'   {best}')
# cross-validation over time halves: rule vs best flat, both chosen on one half
print('\n2-fold time cross-validation (choose on one half, evaluate on the other), S7, indicator only (mixed offsets never simulated):')
for sel, ev in (('A', 'B'), ('B', 'A')):
    Ts, Te = table(half[sel]), table(half[ev])
    rb = [min((o for o in offs if Ts[(i, o)][0] > 0), key=lambda o: Ts[(i, o)][1] / Ts[(i, o)][0]) for i in range(a.nbins)]
    tot = lambda T, o: sum(T[(i, o)][1] for i in range(a.nbins)) / max(1e-9, sum(T[(i, o)][0] for i in range(a.nbins)))
    flat = min(offs, key=lambda o: tot(Ts, o))
    rule_e = sum(Te[(i, rb[i])][1] for i in range(a.nbins)) / sum(Te[(i, rb[i])][0] for i in range(a.nbins))
    print(f'  choose {sel} -> eval {ev}: rule offsets {rb}  rule {100*rule_e:.3f}%  best flat ({flat} dB) {100*tot(Te, flat):.3f}%  default {100*tot(Te, 0):.3f}%')
# S5 check (no fitting): how S5 cells fall into the bins
f5 = {}
for p in S5_DEFAULT:
    d = load(p)
    if d:
        for k, e in d.items(): f5.setdefault(k, []).extend(e[2])
if f5:
    c5 = collections.Counter(b(feature(v)) for v in f5.values() if v); c7 = collections.Counter(b(x) for x in feat.values())
    print('\nshare of cell-windows per bin   S7: ' + '  '.join(f'bin{i} {100*c7[i]/sum(c7.values()):.0f}%' for i in range(a.nbins))
          + '   |   S5 (unseen): ' + '  '.join(f'bin{i} {100*c5[i]/sum(c5.values()):.0f}%' for i in range(a.nbins)))
    print('implied mean offset per cell-window   S7: %.1f dB   S5: %.1f dB' % (sum(rule[i] * c7[i] for i in c7) / sum(c7.values()), sum(rule[i] * c5[i] for i in c5) / sum(c5.values())))
json.dump({'feature': 'p10_rsrp_at_default_dbm', 'edges': edges, 'offsets': rule, 'min_ues': a.min_ues, 'fitted_on': 'S7'}, open(a.out, 'w'), indent=1)
print(f'\nwrote {a.out}: edges {[round(e,1) for e in edges]}  offsets {rule}')
