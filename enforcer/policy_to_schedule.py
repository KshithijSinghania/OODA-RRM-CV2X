#!/usr/bin/env python3
r"""Turn OODA policies into the ns-3 per-site TX power schedule (--powerSchedule).
Power mapping (one TxPower per gNB, policies have a budget per class):
   p_avg = sum_k share_k * 10^(level_k/10)      (share = prb_priority pct/100, level = power_budget_dbm[class])
   delta_dB = 10*log10(p_avg / 10^(23/10))      (23 dBm = full power in the analytic model; <= 0)
   tx_dbm = base + delta_dB   (base = --txMacro for site 0, --txRsu for sites 1..11, the ns-3 defaults 37 / 30)
A policy computed from window [a, a+5) is applied from a+shift (default shift=5: causal, like delay_ticks=50 in the analytic model).
Sources: --dec decide_S7.jsonl (gold CPLEX policies, variant v0) or --trace loop_S7_dcov2.jsonl (LLM chain; records with obs_id + policy).
usage: python3 policy_to_schedule.py --obs observe_S7.jsonl --dec decide_S7.jsonl --t0 90 --t1 200 --out sched_gold.csv
       python3 policy_to_schedule.py --obs observe_S7.jsonl --trace loop_S7_dcov2.jsonl --t0 90 --t1 200 --out sched_chain.csv
       add --scale 0 for a CONTROL schedule with every delta forced to 0 (should reproduce the base run exactly)."""
import json, re, math, argparse, collections
ap = argparse.ArgumentParser()
ap.add_argument('--obs', required=True); ap.add_argument('--dec'); ap.add_argument('--trace')
ap.add_argument('--t0', type=float, default=0); ap.add_argument('--t1', type=float, default=1e9)
ap.add_argument('--shift', type=float, default=5.0); ap.add_argument('--txMacro', type=float, default=37.0); ap.add_argument('--txRsu', type=float, default=30.0)
ap.add_argument('--full_dbm', type=float, default=23.0); ap.add_argument('--scale', type=float, default=1.0, help='multiply delta (0 = control, 1 = as is)')
ap.add_argument('--out', required=True); a = ap.parse_args()
jl = lambda f: [json.loads(l) for l in open(f) if l.strip()]
pol = {}
if a.dec:
    recs = jl(a.dec); has_v0 = any('-v0-' in r['id'] for r in recs)
    for r in recs:
        if has_v0 and '-v0-' not in r['id']: continue
        pol[r['obs_id']] = json.loads(r['messages'][2]['content'])
if a.trace:
    for r in jl(a.trace):
        p = r.get('policy'); oid = r.get('obs_id')
        if isinstance(p, str):
            try: p = json.loads(p)
            except Exception: p = None
        if oid and isinstance(p, dict) and 'prb_priority' in p: pol[oid] = p
if not pol: raise SystemExit('no policies found (check --dec/--trace and the field names obs_id/policy)')
def delta(p):
    x = p['prb_priority']; l = p['power_budget_dbm']
    sh = {'urllc': x['urllc_pct'], 'embb': x['embb_pct'], 'mmtc': x['mmtc_pct']}; tot = sum(sh.values()) or 1.0
    pavg = sum(sh[k] / tot * 10 ** (l[k] / 10) for k in sh)
    return 10 * math.log10(max(pavg, 1e-9) / 10 ** (a.full_dbm / 10))
rows, cells, ds = [], set(), []
for o in jl(a.obs):
    if o.get('layer') != 'rsu_edge' or o['id'] not in pol: continue
    ob = json.loads(o['messages'][2]['content']); w0 = ob['window_s'][0]
    if not (a.t0 <= w0 < a.t1): continue
    m = re.search(r'(\d+)\s*$', str(o['cell']))
    if not m: raise SystemExit(f"cannot read a site index from cell id {o['cell']!r}")
    site = int(m.group(1)); cells.add(str(o['cell']))
    d = a.scale * delta(pol[o['id']]); ds.append(d)
    tx = (a.txMacro if site == 0 else a.txRsu) + d
    rows.append((w0 + a.shift, site, tx, d))
rows.sort()
with open(a.out, 'w') as f:
    f.write('t_start,site,tx_dbm,delta_db\n')
    for t, s, tx, d in rows: f.write(f'{t:.3f},{s},{tx:.3f},{d:.3f}\n')
ds.sort(); n = len(ds)
print(f'wrote {a.out}: {n} entries, cells {sorted(cells)[:14]}  sites {sorted({r[1] for r in rows})}')
if n: print(f'delta dB: min {ds[0]:.2f}  p10 {ds[n//10]:.2f}  median {ds[n//2]:.2f}  p90 {ds[(9*n)//10]:.2f}  max {ds[-1]:.2f}   share of entries with |delta|<0.5 dB: {100*sum(abs(d)<0.5 for d in ds)/n:.0f}%')
