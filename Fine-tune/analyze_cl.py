r"""Compare a closed-loop run against the dataset-feedback control run (same episodes, same noise seeds -> paired).
  python analyze_cl.py --ds loop_S7_cl_dataset.jsonl --en loop_S7_cl_enforcer.jsonl"""
import argparse, json, collections
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--ds', required=True); ap.add_argument('--en', required=True); ap.add_argument('--boot', type=int, default=3000)
a = ap.parse_args()
jl = lambda p: [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
epd = {e['ep']: e for e in jl(a.ds + '.episodes.jsonl')}; epe = {e['ep']: e for e in jl(a.en + '.episodes.jsonl')}
common = sorted(set(epd) & set(epe)); print(f'{len(common)} paired episodes ({sum(epd[i]["emergency"] for i in common)} with an emergency window)')
KEYS = [('u_mean', 'URLLC unserved %', 100), ('over', 'ticks>budget %', 100), ('embb_sat', 'eMBB sat', 1), ('power', 'power', 1), ('rtrue', 'reward', 1)]
def agg(e, k): return float(np.mean([w[k] for w in e['windows'][1:]]))        # windows 1..W-1 are the ones run under a chain policy
rng = np.random.default_rng(0)
def ci(d):
    d = np.asarray(d); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(a.boot)]; return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
for name, sel in (('ALL episodes', lambda e: True), ('EMERGENCY episodes', lambda e: e['emergency']), ('no-emergency episodes', lambda e: not e['emergency'])):
    ids = [i for i in common if sel(epd[i])]
    if not ids: continue
    print(f'\n##### {name} (n={len(ids)}) #####'); print(f"{'metric':20s}{'dataset fb':>12s}{'enforcer fb':>13s}{'diff (en-ds)':>14s}{'95% CI (paired bootstrap)':>30s}")
    for k, lab, sc in KEYS:
        x = np.array([agg(epd[i], k) for i in ids]) * sc; y = np.array([agg(epe[i], k) for i in ids]) * sc; m, lo, hi = ci(y - x)
        print(f'{lab:20s}{x.mean():12.3f}{y.mean():13.3f}{m:+14.3f}{f"[{lo:+.3f}, {hi:+.3f}]":>30s}')
td = {r['obs_id']: r for r in jl(a.ds)}; te = {r['obs_id']: r for r in jl(a.en)}; ids = sorted(set(td) & set(te))
ok = [i for i in ids if 'policy' in td[i] and 'policy' in te[i]]
print(f'\n##### chain behaviour (windows: {len(ids)}, both produced a policy: {len(ok)}) #####')
print('Orient tag changed by closed loop :', f"{sum(td[i]['context']['scenario_tag'] != te[i]['context']['scenario_tag'] for i in ok)}/{len(ok)}")
print('PRB split changed by closed loop  :', f"{sum(td[i]['policy']['prb_priority'] != te[i]['policy']['prb_priority'] for i in ok)}/{len(ok)}")
c = collections.Counter((td[i]['context']['scenario_tag'], te[i]['context']['scenario_tag']) for i in ok if td[i]['context']['scenario_tag'] != te[i]['context']['scenario_tag'])
print('tag changes (dataset -> enforcer) :', dict(c.most_common(8)) or 'none')
ov = [te[i]['cl_qos_override'] for i in te if 'cl_qos_override' in te[i]]
print('QoS severity seen by Orient       : dataset', dict(collections.Counter(x['old_severity'] for x in ov)), '| measured', dict(collections.Counter(x['new_severity'] for x in ov)))
for nm, t in (('dataset fb', td), ('enforcer fb', te)):
    esc = collections.Counter(r['escalation']['reason'] for r in t.values() if 'escalation' in r)
    m = [r['cl_act_teacher_match'] for r in t.values() if 'cl_act_teacher_match' in r]
    print(f'Act escalations [{nm:11s}]   :', dict(esc.most_common()), f"| Act == teacher on its own input {np.mean(m):.1%}" if m else '')
print('\nNote: chain_ok / act_decision vs gold in the trace compare with the DATASET-feedback teacher, so they are not meaningful in enforcer mode;')
print('use the cell-level table above and "Act == teacher" instead. Escalations are logged but not acted on by the loop.')
