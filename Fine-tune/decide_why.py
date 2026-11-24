import json, glob, os, sys
tr, dd = sys.argv[1], sys.argv[2]
gold = {}
for f in glob.glob(os.path.join(dd, 'decide_S?.jsonl')):
    for l in open(f, encoding='utf-8'):
        r = json.loads(l); p = json.loads(r['messages'][2]['content']); gold[p['policy_id']] = (r, p)
sp = lambda p: tuple(p['prb_priority'][k] for k in ('urllc_pct','embb_pct','mmtc_pct'))
bad, ok = [], []
for l in open(tr, encoding='utf-8'):
    t = json.loads(l)
    if t.get('scenario_tag') != 'emergency_preempt' or not t.get('policy'): continue
    g = gold.get(t['policy']['policy_id'])
    if not g: continue
    d = sum(abs(a-b) for a, b in zip(sp(t['policy']), sp(g[1])))
    (bad if d > 10 else ok).append((d, t, g))
print('emergency windows: bad (L1>10):', len(bad), ' ok:', len(ok))
r0 = bad[0][2][0]
print('\nrecord keys:', list(r0.keys()))
u = json.loads(r0['messages'][1]['content'])
def shape(x, d=0):
    if isinstance(x, dict) and d < 3: return {k: shape(v, d+1) for k, v in x.items()}
    if isinstance(x, list): return [shape(x[0], d+1)] if x else []
    return type(x).__name__ if d >= 3 else x
print('decide input structure:', json.dumps(shape(u))[:1800])
for name, grp in (('BAD', bad[:4]), ('OK', ok[:2])):
    for d, t, g in grp:
        print('\n', name, t['obs_id'], 'L1 gap', d)
        print('  context  :', json.dumps(t['context']['constraint_overrides']), json.dumps(t['context']['qos_weights']))
        print('  pred     :', sp(t['policy']), t['policy']['power_budget_dbm'], t['policy']['interference_coordination']['strategy'])
        print('  gold     :', sp(g[1]), g[1]['power_budget_dbm'], g[1]['interference_coordination']['strategy'])
        print('  input    :', g[0]['messages'][1]['content'][:900])
