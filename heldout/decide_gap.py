#!/usr/bin/env python3
r"""How far is Decide from the CPLEX gold policy, and does it matter?
  python decide_gap.py --trace loop_S7_cov2.jsonl --data_dir ..\OODA_S7\data
Joins trace rows to gold Decide records by policy_id (the glue copies the gold id into the trace policy). No GPU."""
import argparse, collections, glob, json, os, statistics as st
ap = argparse.ArgumentParser(); ap.add_argument('--trace', required=True); ap.add_argument('--data_dir', required=True); a = ap.parse_args()
tag = {}
for f in glob.glob(os.path.join(a.data_dir, 'orient_S?.jsonl')):
    for l in open(f, encoding='utf-8'):
        r = json.loads(l); tag.setdefault(r['obs_id'], r['scenario_tag'])
gold = {}
files = glob.glob(os.path.join(a.data_dir, 'decide_S?.jsonl'))
if not files: raise SystemExit('no decide_S?.jsonl in ' + a.data_dir + ': found ' + str(os.listdir(a.data_dir)))
for f in files:
    for l in open(f, encoding='utf-8'):
        r = json.loads(l)
        try: p = json.loads(r['messages'][2]['content'])
        except Exception: continue
        if isinstance(p, dict) and 'policy_id' in p: gold[p['policy_id']] = p
print('gold decide policies:', len(gold))
rows = []
for l in open(a.trace, encoding='utf-8'):
    t = json.loads(l)
    if 'chain_ok' not in t or not t.get('policy') or t['obs_id'] not in tag: continue
    g = gold.get(t['policy']['policy_id'])
    if g: rows.append((tag[t['obs_id']], t, g))
print('joined rows:', len(rows))
if not rows: raise SystemExit('nothing joined: policy_id not found in gold decide files')
def split(p): q = p['prb_priority']; return (q['urllc_pct'], q['embb_pct'], q['mmtc_pct'])
for grp in sorted({r[0] for r in rows}) + ['ALL']:
    R = [r for r in rows if grp == 'ALL' or r[0] == grp]; n = len(R)
    l1 = [sum(abs(x - y) for x, y in zip(split(t['policy']), split(g))) for _, t, g in R]
    du = [split(t['policy'])[0] - split(g)[0] for _, t, g in R]
    f = lambda c: f'{100.0 * sum(map(bool, c)) / n:5.1f}%'
    print(f'\n== {grp}  ({n} windows)')
    print('  PRB split exact           ', f([x == 0 for x in l1]))
    print('  PRB L1 gap <= 5 points    ', f([x <= 5 for x in l1]), '  <= 10:', f([x <= 10 for x in l1]))
    print('  median / 90th pct L1 gap  ', st.median(l1), '/', sorted(l1)[int(0.9 * (n - 1))])
    print('  URLLC share: mean diff vs gold %+.1f, Decide LOWER than gold by >5: %s' % (st.mean(du), f([d < -5 for d in du])))
    print('  priority_order same       ', f([t['policy']['prb_priority']['priority_order'] == g['prb_priority']['priority_order'] for _, t, g in R]))
    print('  preemption same           ', f([t['policy']['preemption'] == g['preemption'] for _, t, g in R]))
    print('  power budget same         ', f([t['policy']['power_budget_dbm'] == g['power_budget_dbm'] for _, t, g in R]))
    print('  strategy same             ', f([t['policy']['interference_coordination']['strategy'] == g['interference_coordination']['strategy'] for _, t, g in R]))
    print('  MEC shares same           ', f([t['policy']['mec_offloading']['compute_share_pct'] == g['mec_offloading']['compute_share_pct'] for _, t, g in R]))
    print('  Decide feasible vs Orient ', f([t.get('decide_feasible') for _, t, g in R]))
