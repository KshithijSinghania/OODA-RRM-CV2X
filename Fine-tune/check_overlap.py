import argparse, collections, glob, json, os
ap = argparse.ArgumentParser(); ap.add_argument('--trace', required=True); ap.add_argument('--compact_dir', required=True); ap.add_argument('--train_dir', required=True); a = ap.parse_args()
train = set()
for f in glob.glob(os.path.join(a.train_dir, 'decide_S?.jsonl')):
    for l in open(f, encoding='utf-8'):
        r = json.loads(l)
        if r['split'] != 'test': train.add(r['messages'][1]['content'])
rec = {}
for f in glob.glob(os.path.join(a.compact_dir, 'decide_S?.jsonl')):
    for l in open(f, encoding='utf-8'):
        r = json.loads(l); rec.setdefault((r['obs_id'], r['scenario_tag']), r)
print(f'new Decide training prompts: {len(train)}; compact Decide records of this dataset: {len(rec)}')
G = collections.defaultdict(lambda: [0, 0, 0, 0])
miss = 0
for l in open(a.trace, encoding='utf-8'):
    t = json.loads(l)
    if 'chain_ok' not in t: continue
    r = rec.get((t['obs_id'], t['scenario_tag']))
    if not r: miss += 1; continue
    code = r['messages'][2]['content']; gu, ge = int(code[:3]), int(code[3:6]); p = t['prb']
    seen = 'in new training set' if r['messages'][1]['content'] in train else 'NOT in training set'
    for key in ((t['scenario_tag'], seen), ('ALL', seen)):
        g = G[key]; g[0] += 1; g[1] += (p['urllc_pct'], p['embb_pct']) == (gu, ge); g[2] += abs(p['urllc_pct'] - gu) <= 1 and abs(p['embb_pct'] - ge) <= 1; g[3] += bool(t['chain_ok'])
print(f'chain windows with no matching compact record: {miss}\n\n{"scenario":22s}{"prompt":24s}{"windows":>8s}{"PRB exact":>11s}{"PRB +-1":>9s}{"chain OK":>10s}')
for k in sorted(G):
    n, e, e1, c = G[k]; print(f'{k[0]:22s}{k[1]:24s}{n:8d}{100.0 * e / n:10.1f}%{100.0 * e1 / n:8.1f}%{100.0 * c / n:9.1f}%')
