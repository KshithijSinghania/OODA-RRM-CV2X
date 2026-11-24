#!/usr/bin/env python3
"""Break an ooda_loop trace down by the GOLD scenario tag (so the ambulance windows can be read on their own).
  python analyze_trace.py --trace loop_S7_3llm_full.jsonl --data_dir ..\\OODA_S7\\data
Uses only the trace and data\\orient_S?.jsonl. No GPU, no models."""
import argparse, collections, glob, json, os
ap = argparse.ArgumentParser(); ap.add_argument('--trace', required=True); ap.add_argument('--data_dir', required=True); a = ap.parse_args()
gold = {}
for f in sorted(glob.glob(os.path.join(a.data_dir, 'orient_S?.jsonl'))):
    for l in open(f, encoding='utf-8'):
        r = json.loads(l); gold.setdefault(r['obs_id'], r['scenario_tag'])
T = [json.loads(l) for l in open(a.trace, encoding='utf-8')]
rows = [t for t in T if 'chain_ok' in t and t['obs_id'] in gold]
print(f'{len(T)} trace lines, {len(rows)} complete windows with a gold tag ({len(T) - len(rows)} stopped early or unmatched)')
G = collections.defaultdict(list)
for t in rows: G[gold[t['obs_id']]].append(t)
pc = lambda x, n: f'{100.0 * x / max(n, 1):5.1f}%'
print(f'\n{"gold scenario":22s} {"windows":>8s} {"tag right":>10s} {"PRB==CPLEX":>11s} {"Act==gold":>10s} {"chain OK":>9s} {"escalates":>10s}')
for g in sorted(G, key=lambda k: -len(G[k])) + ['ALL']:
    R = rows if g == 'ALL' else G[g]; n = len(R)
    print(f'{g:22s} {n:8d} {pc(sum(t["scenario_tag"] == gold[t["obs_id"]] for t in R), n):>10s} {pc(sum(bool(t["prb_matches_cplex"]) for t in R), n):>11s} '
          f'{pc(sum(bool(t["act_matches_gold"]) for t in R), n):>10s} {pc(sum(bool(t["chain_ok"]) for t in R), n):>9s} {pc(sum(bool(t["escalation"]["escalate"]) for t in R), n):>10s}')
print('\nconfusion (rows = gold tag, columns = predicted tag):')
C = collections.Counter((gold[t['obs_id']], t['scenario_tag']) for t in rows); tags = sorted({k[0] for k in C} | {k[1] for k in C})
print(f'{"":22s}' + ''.join(f'{x:>20s}' for x in tags))
for g in tags: print(f'{g:22s}' + ''.join(f'{C[(g, p)]:20d}' for p in tags))
E_, M_ = 'emergency_preempt', None
em = [t for t in rows if gold[t['obs_id']] == E_]; ot = [t for t in rows if gold[t['obs_id']] != E_]
if em or any(t['scenario_tag'] == E_ for t in rows):
    missed = sum(t['scenario_tag'] != E_ for t in em); false = sum(t['scenario_tag'] == E_ for t in ot)
    print(f'\nEMERGENCY: {len(em)} gold emergency windows -> missed (predicted something else): {missed} ({pc(missed, len(em))}); '
          f'false alarms among {len(ot)} other windows: {false} ({pc(false, len(ot))})')
