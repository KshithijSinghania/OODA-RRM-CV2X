#!/usr/bin/env python3
"""Evaluate the deterministic Observe (observe_rules.py, no LLM, no GPU) against the gold ObserveSummary on the test split.
  python eval_observe_rules.py --data_dir data --schema_dir schemas"""
import argparse, collections, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jsonschema import validate, ValidationError
import observe_rules as OR
ap = argparse.ArgumentParser(); ap.add_argument('--data_dir', default='data'); ap.add_argument('--schema_dir', default='schemas'); ap.add_argument('--split', default='test'); a = ap.parse_args()
schema = json.load(open(os.path.join(a.schema_dir, 'observe_summary.schema.json')))
rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(a.data_dir, 'observe_S?.jsonl'))) for l in open(f, encoding='utf-8')]
rows = [r for r in rows if r['split'] == a.split]
hit = collections.Counter(); n = ok = exact = 0; t0 = time.time(); bad = []
def fields(o):
    return {'load_level': o['load_level'], 'qos severity': o['qos_status']['severity'], 'trend.qos': o['trend']['qos'], 'trend.load': o['trend']['load'], 'anomalies': o['anomalies'],
            'hotspots': o['hotspots'], 'special_events': o['special_events'], 'mobility': o['mobility'], 'v2x_messages': o['v2x_messages'], 'resource_state': o['resource_state'],
            'qos numeric fields': {k: v for k, v in o['qos_status'].items() if k != 'severity'}, 'trend deltas (numeric)': (o['trend']['delta_loss_pp'], o['trend']['delta_vehicles_pct']),
            'narrative': o['narrative']}
for r in rows:
    inp = json.loads(r['messages'][1]['content']); g = json.loads(r['messages'][2]['content']); p = OR.observe_rules(inp); n += 1
    try: validate(p, schema); ok += 1
    except ValidationError: pass
    exact += p == g; fp, fg = fields(p), fields(g)
    cat = all(fp[k] == fg[k] for k in ('load_level', 'qos severity', 'trend.qos', 'trend.load', 'anomalies')); hit['ALL 5 categorical decisions'] += cat
    for k in fp: hit[k] += fp[k] == fg[k]
    if not cat: bad.append(r['id'])
dt = (time.time() - t0) / max(n, 1)
print(f"observe (rules, no LLM) on split '{a.split}': n={n}  schema-valid {ok/n:.1%}  exact match of the full summary {exact/n:.1%}  ({dt*1000:.2f} ms per window)")
for k in sorted(hit, key=lambda k: hit[k] / n): print(f'  {k:<30} {hit[k]/n:.1%}')
print(f'  windows where a categorical decision differs from gold: {len(bad)}  e.g. {bad[:3]}')
print('  (numeric trend deltas / narrative differ from gold only because the input carries the previous-window means already rounded)')
