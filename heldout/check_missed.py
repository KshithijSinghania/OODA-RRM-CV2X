#!/usr/bin/env python3
"""List the emergency windows the Orient model missed, and what Observe had already reported for them.
  python check_missed.py --trace loop_S7_3llm_full.jsonl --data_dir ..\\OODA_S7\\data
If Observe already flagged an in-scope emergency vehicle in every missed window, a one-line code override fixes them."""
import argparse, collections, glob, json, os
ap = argparse.ArgumentParser(); ap.add_argument('--trace', required=True); ap.add_argument('--data_dir', required=True); a = ap.parse_args()
gold = {}
for f in sorted(glob.glob(os.path.join(a.data_dir, 'orient_S?.jsonl'))):
    for l in open(f, encoding='utf-8'): r = json.loads(l); gold.setdefault(r['obs_id'], r['scenario_tag'])
obs = {}
for f in sorted(glob.glob(os.path.join(a.data_dir, 'observe_S?.jsonl'))):
    for l in open(f, encoding='utf-8'): r = json.loads(l); obs[r['id']] = json.loads(r['messages'][1]['content'])
T = [json.loads(l) for l in open(a.trace, encoding='utf-8')]; T = [t for t in T if 'chain_ok' in t and t['obs_id'] in gold]
E = 'emergency_preempt'; miss = [t for t in T if gold[t['obs_id']] == E and t['scenario_tag'] != E]; hit = [t for t in T if gold[t['obs_id']] == E and t['scenario_tag'] == E]
def ev(t):
    e = (obs.get(t['obs_id']) or {}).get('emergency_vehicles') or []
    return [x for x in e if x.get('in_scope')], e
print(f'{len(hit)} emergency windows caught, {len(miss)} missed')
for name, S in (('caught', hit), ('missed', miss)):
    inscope = sum(bool(ev(t)[0]) for t in S); anyev = sum(bool(ev(t)[1]) for t in S)
    print(f'  {name:7s}: Observe reported an emergency vehicle in {anyev}/{len(S)} windows, in scope of the cell in {inscope}/{len(S)}')
print('\nmissed windows (obs_id, predicted tag, nearby-vehicle count from Observe):')
for t in miss[:30]:
    s, e = ev(t); print(f'  {t["obs_id"]:22s} predicted {t["scenario_tag"]:20s} in_scope={bool(s)} n_within_300m={[x.get("n_vehicles_within_300m") for x in e]}')
print('\nfalse alarms (predicted emergency, gold not):', sum(t['scenario_tag'] == E and gold[t['obs_id']] != E for t in T))
