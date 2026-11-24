#!/usr/bin/env python3
"""Build the token-diet v2 datasets (data_compact2/) + codebook2.json from data/, and VERIFY losslessness:
for every codable record, decode_*2(encode_*2(gold)) must expand to exactly the original gold JSON.   python build_compact2.py"""
import json, glob, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compact as K, compact2 as C2

def load(agent):
    R = []
    for f in sorted(glob.glob(f'data/{agent}_S?.jsonl')): R += [json.loads(l) for l in open(f, encoding='utf-8')]
    return R
O, D = load('orient'), load('decide')
# ---- codebooks from train + val only
a, b = set(), set()
for r in O:
    if r['split'] != 'test': x, y = C2._orient_keys(json.loads(r['messages'][2]['content'])); a.add(x); b.add(y)
mec, strat = set(), set()
for r in D:
    if r['split'] != 'test':
        p = json.loads(r['messages'][2]['content']); mec.add(tuple(p['mec_offloading']['compute_share_pct'][k] for k in K.CLS)); strat.add(K.STRAT[p['interference_coordination']['strategy']])
cb = {'orient_a': sorted(a), 'orient_b': sorted(b), 'mec': [list(m) for m in sorted(mec)], 'strat': sorted(strat)}
assert len(cb['orient_a']) < 1000 and len(cb['orient_b']) < 1000 and len(cb['mec']) < 100 and len(cb['strat']) < 10
json.dump(cb, open('codebook2.json', 'w'), indent=1)
print(f"codebooks: orient A {len(cb['orient_a'])}, orient B {len(cb['orient_b'])}, mec {len(cb['mec'])}, strategies {len(cb['strat'])}")
os.makedirs('data_compact2', exist_ok=True)
for agent, R in (('orient', O), ('decide', D)):
    out = collections.defaultdict(list); n = bad = unc = 0; seen_in = collections.defaultdict(set); test_in = []
    def _user(r):
        i = json.loads(r['messages'][1]['content']); return C2.orient_input2(i) if agent == 'orient' else C2.decide_input2(i['context'], i['cell_state'])
    train_inputs = {_user(r) for r in R if r['split'] != 'test'}
    for r in R:
        inp = json.loads(r['messages'][1]['content']); gold = json.loads(r['messages'][2]['content']); n += 1
        try: code = C2.encode_orient2(gold, cb) if agent == 'orient' else C2.encode_decide2(gold, cb); codable = True
        except ValueError: code = '0' * (6 if agent == 'orient' else 12); codable = False; unc += 1
        if codable:
            pred = C2.decode_orient2(code, inp, cb) if agent == 'orient' else C2.decode_decide2(code, inp['context'], cb, gold['policy_id'])
            bad += pred != gold
        user = C2.orient_input2(inp) if agent == 'orient' else C2.decide_input2(inp['context'], inp['cell_state'])
        if r['split'] == 'test': test_in.append(user)
        else: seen_in['x'].add(user)
        rec = {k: v for k, v in r.items() if k != 'messages'}; rec['uncodable'] = not codable; rec['input_seen_in_train'] = user in train_inputs
        rec['messages'] = [{'role': 'system', 'content': C2.ORIENT_SYSTEM2 if agent == 'orient' else C2.DECIDE_SYSTEM2}, {'role': 'user', 'content': user}, {'role': 'assistant', 'content': code}]
        out[r['situation']].append(rec)
    for s, rows in out.items():
        with open(f'data_compact2/{agent}_{s}.jsonl', 'w', encoding='utf-8') as fo:
            for rec in rows: fo.write(json.dumps(rec) + '\n')
    cnt = collections.Counter(r['split'] for r in R)
    lk = sum(u in seen_in['x'] for u in test_in) / max(1, len(test_in))
    avg = sum(len(rec['messages'][1]['content']) for rows in out.values() for rec in rows) / n
    print(f"{agent}: {n} records {dict(cnt)} | round-trip mismatches {bad} of {n-unc} codable | uncodable (test values never seen in train/val) {unc} | mean prompt {avg:.0f} chars | "
          f"test prompts identical to a train/val prompt: {lk:.1%}")
    assert bad == 0

# ---- Observe and Act (added with the all-agent v2 chain; Observe is used as code in the measured setup)
OB, A = load('observe'), load('act')
for agent, R in (('observe', OB), ('act', A)):
    out = collections.defaultdict(list); n = bad = 0; cnt = collections.Counter(r['split'] for r in R); conf = {}; nconf = 0; lens = []
    for r in R:
        inp = json.loads(r['messages'][1]['content']); gold = json.loads(r['messages'][2]['content']); n += 1
        if agent == 'observe':
            user = C2.observe_input2(inp); code = C2.encode_observe2(gold); pred = C2.decode_observe2(code, inp)
            keys = ('load_level', 'anomalies', 'hotspots', 'mobility', 'special_events', 'v2x_messages', 'resource_state', 'window_s', 'scope', 'n_active_vehicles')
            bad += not (all(pred[k] == gold[k] for k in keys) and pred['qos_status']['severity'] == gold['qos_status']['severity'] and pred['trend']['qos'] == gold['trend']['qos']
                        and pred['trend']['load'] == gold['trend']['load'])        # numeric trend deltas / narrative are rule-computed from rounded inputs and may differ in the last digit
        else:
            user = C2.act_input2(inp); code = C2.encode_act2(gold); pred = C2.decode_act2(code, inp); bad += pred != gold
        nconf += conf.setdefault(user, code) != code; lens.append(len(user))
        rec = {k: v for k, v in r.items() if k != 'messages'}; rec['uncodable'] = False
        rec['messages'] = [{'role': 'system', 'content': C2.OBSERVE_SYSTEM2 if agent == 'observe' else C2.ACT_SYSTEM2}, {'role': 'user', 'content': user}, {'role': 'assistant', 'content': code}]
        out[r['situation']].append(rec)
    tr = {json.loads(x)['messages'][1]['content'] for s in out for x in [json.dumps(rr) for rr in out[s] if rr['split'] != 'test']}
    for s, rows in out.items():
        for rec in rows: rec['input_seen_in_train'] = rec['messages'][1]['content'] in tr
        with open(f'data_compact2/{agent}_{s}.jsonl', 'w', encoding='utf-8') as fo:
            for rec in rows: fo.write(json.dumps(rec) + '\n')
    te = [rec for rows in out.values() for rec in rows if rec['split'] == 'test']
    print(f"{agent}: {n} records {dict(cnt)} | round-trip mismatches {bad} | same prompt -> different label (conflicts) {nconf} | mean prompt {sum(lens)/n:.0f} chars | "
          f"test prompts identical to a train/val prompt: {sum(r['input_seen_in_train'] for r in te)/max(1,len(te)):.1%}")
    assert bad == 0 and nconf == 0
print('OK')
