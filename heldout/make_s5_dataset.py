#!/usr/bin/env python3
"""Build the held-out S5 test records (Observe/Orient/Decide/Act, full + token-diet v2) in a SEPARATE work folder.
Nothing in your OODA folder is modified: scripts are copied, S5 is added to the COPY of config.py, every S5 window is split='test',
and the token-diet records are encoded with your EXISTING codebook2.json (so S5 gold values that the training codebook cannot express
show up as 'uncodable' instead of silently extending the codebook).
usage (from anywhere):
  python make_s5_dataset.py --ooda C:\\path\\OODA --csv indore_S5_combined_5s_v3.csv --work C:\\path\\OODA_S5 [--tag S5] [--n_vehicles 313]
result: <work>\\data (full JSON records)  and  <work>\\data_compact2_S5 (token-diet records, same layout as data_compact2)"""
import argparse, glob, json, os, shutil, subprocess, sys, collections
ap = argparse.ArgumentParser()
ap.add_argument('--ooda', required=True); ap.add_argument('--csv', required=True); ap.add_argument('--work', required=True)
ap.add_argument('--tag', default='S5'); ap.add_argument('--n_vehicles', type=int, default=313)
ap.add_argument('--codebook', default=None, help='default: <ooda>/codebook2.json')
a = ap.parse_args(); T = a.tag
ooda, work = os.path.abspath(a.ooda), os.path.abspath(a.work); csv = os.path.abspath(a.csv)
cbfile = os.path.abspath(a.codebook or os.path.join(ooda, 'codebook2.json'))
assert os.path.isfile(cbfile), f'codebook not found: {cbfile}'
if os.path.exists(os.path.join(work, 'data')): sys.exit(f'{work}\\data already exists - choose a new --work folder (never reuse one)')
os.makedirs(work, exist_ok=True)
for f in glob.glob(os.path.join(ooda, '*.py')): shutil.copy(f, work)
shutil.copytree(os.path.join(ooda, 'schemas'), os.path.join(work, 'schemas'), dirs_exist_ok=True)
os.makedirs(os.path.join(work, 'data'), exist_ok=True)

def patch(fn, old, new):
    p = os.path.join(work, fn); s = open(p, encoding='utf-8').read()
    assert s.count(old) == 1, f'{fn}: expected text not found once: {old[:60]!r}'
    open(p, 'w', encoding='utf-8').write(s.replace(old, new))
patch('config.py', 'SITUATIONS = {', f'SITUATIONS = {{\n    "{T}": {{"n_vehicles": {a.n_vehicles}, "ambulance": False, "expected_tag": "baseline"}},')
patch('build_observe.py', '    win_split = assign_splits(df, windows)\n',
      f'    win_split = assign_splits(df, windows)\n    win_split = {{w: "test" for w in windows}}      # HELD-OUT: every window is test\n')

def run(*cmd):
    print('>', ' '.join(cmd), flush=True)
    r = subprocess.run([sys.executable, *cmd], cwd=work, capture_output=True, text=True)
    print((r.stdout + r.stderr).strip()[-1500:]); 
    if r.returncode: sys.exit(f'FAILED: {" ".join(cmd)}')
run('build_observe.py', '--csv', csv, '--situation', T)
run('build_orient.py'); run('build_decide.py', '--teacher', 'cplex'); run('build_act.py')
for ag, sch in (('observe', 'observe_summary'), ('orient', 'orient_context'), ('decide', 'decide_policy'), ('act', 'act_command')):
    run('validate.py', f'data/{ag}_{T}.jsonl', sch)

# ---- token-diet v2 records with the EXISTING codebook
sys.path.insert(0, work)
import compact as K, compact2 as C2
cb = json.load(open(cbfile)); out = os.path.join(work, f'data_compact2_{T}'); os.makedirs(out, exist_ok=True)
def load(ag): return [json.loads(l) for l in open(os.path.join(work, 'data', f'{ag}_{T}.jsonl'), encoding='utf-8')]
def train_inputs(ag):
    s = set()
    for f in glob.glob(os.path.join(ooda, 'data_compact2', f'{ag}_S?.jsonl')):
        for l in open(f, encoding='utf-8'):
            r = json.loads(l)
            if r['split'] != 'test': s.add(r['messages'][1]['content'])
    return s
rep = {}
for ag in ('observe', 'orient', 'decide', 'act'):
    R = load(ag); seen = train_inputs(ag); rows = []; unc = bad = 0
    for r in R:
        inp = json.loads(r['messages'][1]['content']); gold = json.loads(r['messages'][2]['content']); codable = True
        if ag == 'orient':
            user = C2.orient_input2(inp)
            try: code = C2.encode_orient2(gold, cb); bad += C2.decode_orient2(code, inp, cb) != gold
            except ValueError: code, codable = '0' * 6, False
            system = C2.ORIENT_SYSTEM2
        elif ag == 'decide':
            user = C2.decide_input2(inp['context'], inp['cell_state'])
            try: code = C2.encode_decide2(gold, cb); bad += C2.decode_decide2(code, inp['context'], cb, gold['policy_id']) != gold
            except ValueError: code, codable = '0' * 12, False
            system = C2.DECIDE_SYSTEM2
        elif ag == 'observe': user = C2.observe_input2(inp); code = C2.encode_observe2(gold); system = C2.OBSERVE_SYSTEM2
        else: user = C2.act_input2(inp); code = C2.encode_act2(gold); system = C2.ACT_SYSTEM2
        unc += not codable
        rec = {k: v for k, v in r.items() if k != 'messages'}; rec['uncodable'] = not codable; rec['input_seen_in_train'] = user in seen
        rec['messages'] = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}, {'role': 'assistant', 'content': code}]
        rows.append(rec)
    with open(os.path.join(out, f'{ag}_{T}.jsonl'), 'w', encoding='utf-8') as fo:
        for rec in rows: fo.write(json.dumps(rec) + '\n')
    rep[ag] = (len(rows), unc, sum(r['input_seen_in_train'] for r in rows) / max(1, len(rows)), bad)
print('\n=== SUMMARY ' + T + ' ===')
print(f'{"agent":8s} {"records":>8s} {"uncodable gold":>15s} {"prompt also in train/val":>26s}  roundtrip-mismatch')
for ag, (n, u, s, b) in rep.items(): print(f'{ag:8s} {n:8d} {u:8d} ({u/max(n,1):5.1%}) {s:20.1%}  {b}')
print('\nfull records :', os.path.join(work, 'data')); print('compact2     :', out)
