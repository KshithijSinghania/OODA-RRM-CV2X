import json, glob, os, sys, collections
sys.path.insert(0, r'..\OODA'); import compact2 as C2
cb = C2.load_codebook()
rows = []
for dd in sys.argv[1:]:
    for f in glob.glob(os.path.join(dd, 'decide_S?.jsonl')):
        for l in open(f, encoding='utf-8'):
            r = json.loads(l)
            try:
                u = json.loads(r['messages'][1]['content']); p = json.loads(r['messages'][2]['content'])
                code = C2.encode_decide2(p, cb); key = C2.decide_input2(u['context'], u['cell_state'])
            except Exception as e: continue
            rows.append((r['scenario_tag'], key, code, u))
print('records:', len(rows))
FIELDS = {'whole code': slice(0, 12), 'PRB split': slice(0, 6), 'power budget': slice(6, 9), 'strategy': slice(9, 10), 'MEC': slice(10, 12)}
def ceiling(rs, extra, sl):
    g = collections.defaultdict(collections.Counter)
    for tag, key, code, u in rs: g[(key, extra(u))][code[sl]] += 1
    return sum(max(c.values()) for c in g.values()) / len(rs), len(g)
cs = lambda u: u['cell_state']; rsd = lambda u: cs(u)['resource_state']
EXTRA = {'current prompt': lambda u: 0,
  '+ load_level': lambda u: cs(u)['load_level'],
  '+ offered_load_ratio(0.1)': lambda u: round(rsd(u)['offered_load_ratio'], 1),
  '+ interference_iot(1dB)': lambda u: round(rsd(u)['interference_iot_db']),
  '+ mec_load_ratio(0.1)': lambda u: round(rsd(u)['mec_load_ratio'], 1),
  '+ current prb_pct': lambda u: json.dumps(u['current_allocation']['prb_pct']),
  '+ current power_dbm': lambda u: json.dumps(u['current_allocation']['power_dbm']),
  '+ trend+qos severity': lambda u: (cs(u)['trend']['qos'], cs(u)['trend']['load'], cs(u)['qos']['severity']),
  '+ ALL of the above': lambda u: (cs(u)['load_level'], round(rsd(u)['offered_load_ratio'], 1), round(rsd(u)['interference_iot_db']), round(rsd(u)['mec_load_ratio'], 1), json.dumps(u['current_allocation']), cs(u)['trend']['qos'], cs(u)['qos']['severity'])}
for grp in ('emergency_preempt', 'intersection_dense', 'baseline'):
    rs = [r for r in rows if r[0] == grp]
    if not rs: continue
    print('\n=== %s (%d windows): best possible accuracy for ANY model that sees this much' % (grp, len(rs)))
    print('%-28s' % '' + ''.join('%14s' % k for k in FIELDS))
    for name, ex in EXTRA.items():
        print('%-28s' % name + ''.join('%13.1f%%' % (100 * ceiling(rs, ex, sl)[0]) for sl in FIELDS.values()))
