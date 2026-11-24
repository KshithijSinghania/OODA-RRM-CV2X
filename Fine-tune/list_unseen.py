import glob, json, os, sys
tr, cd, td = sys.argv[1:4]
train = {json.loads(l)['messages'][1]['content'] for f in glob.glob(os.path.join(td, 'decide_S?.jsonl')) for l in open(f, encoding='utf-8') if json.loads(l)['split'] != 'test'}
rec = {}
for f in glob.glob(os.path.join(cd, 'decide_S?.jsonl')):
    for l in open(f, encoding='utf-8'):
        r = json.loads(l); rec.setdefault((r['obs_id'], r['scenario_tag']), r)
seen = set(); out = []
for l in open(tr, encoding='utf-8'):
    t = json.loads(l)
    if 'chain_ok' not in t: continue
    r = rec.get((t['obs_id'], t['scenario_tag']))
    if r and r['messages'][1]['content'] not in train: out.append(r['messages'][1]['content'])
for p in sorted(set(out), key=lambda s: (s.split()[0], int(s.split(' n=')[1].split()[0]))): print(p)
print(len(out), 'unseen windows,', len(set(out)), 'distinct prompts')
