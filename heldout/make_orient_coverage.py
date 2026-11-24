#!/usr/bin/env python3
"""Coverage-complete Orient training data.  The Orient label depends only on eight small prompt fields (n, load, qos, standstill, event, mode, oem, reg),
and the existing teacher code (build_orient.py) computes it exactly.  The real data only ever has emergencies (event=1) with 3-12 vehicles, calm load and
mild QoS, so the model learned shortcuts.  This script adds synthetic prompts covering the WHOLE space, labelled by the SAME teacher functions, and
holds out whole groups of conditions so the test measures extrapolation, not lookup.
  python make_orient_coverage.py --ooda ..\\OODA --out ..\\OODA_cov [--n_synth 4000]
Writes <out>\\data_compact2_cov\\ (copy of data_compact2 + orient_S9.jsonl) and <out>\\codebook2c.json (the old codebook with NEW codes APPENDED, so old codes keep their meaning).
Nothing in --ooda is modified."""
import argparse, collections, glob, itertools, json, os, random, shutil, sys
ap = argparse.ArgumentParser(); ap.add_argument('--ooda', required=True); ap.add_argument('--out', required=True); ap.add_argument('--n_synth', type=int, default=4000)
ap.add_argument('--seed', type=int, default=7); a = ap.parse_args()
ooda, out = os.path.abspath(a.ooda), os.path.abspath(a.out)
if os.path.exists(os.path.join(out, 'data_compact2_cov')): sys.exit(f'{out}\\data_compact2_cov already exists - choose a new --out folder')
sys.path.insert(0, ooda); os.chdir(ooda)
import compact as K, compact2 as C2, build_orient as BO, config as C
SEV = BO.SEV; cb = json.load(open(os.path.join(ooda, 'codebook2.json')))

def gold_codes(n, load, qos, standstill, event, mode, oem, reg):
    o = {'special_events': [{'vehicle_id': 'AMBULANCE_01', 'in_scope': True, 'n_vehicles_within_300m': 5}] if event else [], 'n_active_vehicles': n, 'load_level': load,
         'qos_status': {'severity': qos, 'mean_loss_pct': 0.0}, 'anomalies': ['standstill_cluster'] if standstill else []}
    tag = BO.classify(o); active, suppressed, conflict = BO.resolve(tag, qos, reg, mode, {'name': 'OEM-A', 'min_embb_mbps': 5} if oem else None)
    g = {'scenario_tag': tag, 'qos_weights': BO.weights(tag, qos, active, reg), 'constraint_overrides': BO.overrides(tag, qos, reg, o['special_events']),
         'active_intents': active, 'intent_resolution': {'conflict_type': conflict, 'suppressed_intents': suppressed}}
    return tag, C2._orient_keys(g)

def parse(s): return dict(x.split('=', 1) for x in s.split())
# ---- 1. faithfulness check: the generator must reproduce the code of EVERY existing training/val/test record
rec = [json.loads(l) for f in sorted(glob.glob('data_compact2/orient_S?.jsonl')) for l in open(f, encoding='utf-8')]
bad = 0
for r in rec:
    d = parse(r['messages'][1]['content']); tag, (x, y) = gold_codes(int(d['n']), d['load'], d['qos'], int(d['standstill']), int(d['event']), d['mode'], int(d['oem']), d['reg'])
    code = r['messages'][2]['content']
    if code.startswith('000000') and r.get('uncodable'): continue
    bad += (cb['orient_a'][int(code[:3])], cb['orient_b'][int(code[3:6])]) != (x, y)
print(f'faithfulness: generator reproduces the teacher code of {len(rec) - bad}/{len(rec)} existing records' + ('' if bad == 0 else '   <-- MISMATCHES, DO NOT TRUST'))
if bad: sys.exit(1)
# ---- 2. the grid
NS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 13, 15, 18, 20, 25, 30, 40, 50, 60, 80]
nb = lambda n: 0 if n <= 4 else 1 if n <= 8 else 2 if n <= 12 else 3 if n <= 20 else 4 if n <= 40 else 5
LOADS, QOS = ['low', 'moderate', 'high', 'very_high'], ['nominal', 'elevated', 'degraded', 'critical']; REG = ['none', 'regional_power_cap', 'safety_zone_mandate', 'tmc_emergency_corridor']
grid = list(itertools.product(NS, LOADS, QOS, [0, 1], [0, 1], ['normal', 'energy_saving'], [0, 1], REG))
rng = random.Random(a.seed); strata = collections.defaultdict(list)
for c in grid: strata[(c[3], c[4], c[1], c[2])].append(c)                      # standstill, event, load, qos
per = max(1, a.n_synth // len(strata)); sample = []
for k in sorted(strata): rng.shuffle(strata[k]); sample += strata[k][:per]
print(f'grid {len(grid)} cells -> {len(sample)} synthetic prompts ({per} per standstill x event x load x qos stratum)')
# ---- 3. grouped hold-out: whole groups of conditions never appear in training
gk = lambda c: (nb(c[0]), c[1], c[2], c[3], c[4]); keys = sorted({gk(c) for c in sample}); random.Random(a.seed + 1).shuffle(keys)
test_k, val_k = set(keys[:int(.14 * len(keys))]), set(keys[int(.14 * len(keys)):int(.20 * len(keys))])
split = lambda c: 'test' if gk(c) in test_k else 'val' if gk(c) in val_k else 'train'
tr = [c for c in sample if split(c) == 'train']
for name, idx in (('n bucket', 0), ('load', 1), ('qos', 2), ('standstill', 3), ('event', 4)):
    miss = {gk(c)[idx if idx == 0 else idx] if idx == 0 else c[idx] for c in sample} - {nb(c[0]) if idx == 0 else c[idx] for c in tr}
    assert not miss, f'a {name} level never appears in training: {miss}'
print(f'groups (n bucket, load, qos, standstill, event): {len(keys)}; held out for test {len(test_k)}, val {len(val_k)}; every single factor level still appears in training')
# ---- 4. codebook extension (append only) and records
newa, newb = [], []
rows = []
for c in sample:
    n, load, qos, ss, ev, mode, oem, reg = c; tag, (x, y) = gold_codes(*c)
    if x not in cb['orient_a'] and x not in newa: newa.append(x)
    if y not in cb['orient_b'] and y not in newb: newb.append(y)
    rows.append((c, tag, x, y))
cbc = dict(cb); cbc['orient_a'] = cb['orient_a'] + sorted(newa); cbc['orient_b'] = cb['orient_b'] + sorted(newb)
assert len(cbc['orient_a']) < 1000 and len(cbc['orient_b']) < 1000
print(f'codebook: orient_a {len(cb["orient_a"])} -> {len(cbc["orient_a"])}, orient_b {len(cb["orient_b"])} -> {len(cbc["orient_b"])} (new codes appended)')
os.makedirs(out, exist_ok=True); shutil.copytree(os.path.join(ooda, 'data_compact2'), os.path.join(out, 'data_compact2_cov'))
json.dump(cbc, open(os.path.join(out, 'codebook2c.json'), 'w'), indent=1)
trp = {r['messages'][1]['content'] for f in glob.glob('data_compact2/orient_S?.jsonl') for r in map(json.loads, open(f, encoding='utf-8')) if r['split'] != 'test'}
with open(os.path.join(out, 'data_compact2_cov', 'orient_S9.jsonl'), 'w', encoding='utf-8') as fo:
    for i, (c, tag, x, y) in enumerate(rows):
        n, load, qos, ss, ev, mode, oem, reg = c
        p = f'n={n} load={load} qos={qos} standstill={ss} event={ev} mode={mode} oem={oem} reg={reg}'
        fo.write(json.dumps({'id': f'S9-syn-{i}', 'agent': 'orient', 'situation': 'S9', 'split': split(c), 'obs_id': f'S9-syn-{i}', 'scenario_tag': tag, 'input_seen_in_train': p in trp,
                             'messages': [{'role': 'system', 'content': C2.ORIENT_SYSTEM2}, {'role': 'user', 'content': p},
                                          {'role': 'assistant', 'content': f'{(cbc["orient_a"].index(x)):03d}{(cbc["orient_b"].index(y)):03d}'}]}) + '\n')
sp = collections.Counter(split(c) for c in sample); print('synthetic records by split:', dict(sp))
print('event=1 synthetic training prompts by qos:', dict(collections.Counter(c[2] for c in tr if c[4] == 1)), ' by load:', dict(collections.Counter(c[1] for c in tr if c[4] == 1)))
print(f'\nwritten: {out}\\data_compact2_cov  and  {out}\\codebook2c.json')
