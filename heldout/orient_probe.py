#!/usr/bin/env python3
"""Robustness probe for the Orient model, independent of any simulation: does an emergency (event=1) ALWAYS give emergency_preempt, whatever
the traffic around it?  The gold rule is: event=1 -> emergency_preempt, event=0 -> never.  Every probe is a real prompt (or a twin of one)
sent to the Orient llama-server; nothing is scored against a simulation label.
  python orient_probe.py --ooda ..\\OODA --url http://127.0.0.1:8081 --sources ..\\OODA_S5\\data_compact2_S5,..\\OODA_S7\\data_compact2_S7,..\\OODA\\data_compact2
  (--selftest runs the whole script against a rule-based stand-in instead of the server)"""
import argparse, collections, glob, json, os, sys
ap = argparse.ArgumentParser(); ap.add_argument('--ooda', required=True); ap.add_argument('--url', default='http://127.0.0.1:8081'); ap.add_argument('--sources', required=True)
ap.add_argument('--model', default='meta-llama/Llama-3.2-3B-Instruct'); ap.add_argument('--selftest', action='store_true'); ap.add_argument('--max_prompts', type=int, default=0)
ap.add_argument('--sweep', default='1,2,3,4,6,8,10,12,13,15,20,25,30,40,60'); ap.add_argument('--codebook', default=None, help='default: <ooda>/codebook2.json; use codebook2c.json for the coverage-retrained model')
ap.add_argument('--eval_file', default=None, help='jsonl with Orient records (e.g. orient_S9.jsonl): exact-code accuracy on its test split, by factor'); a = ap.parse_args()
sys.path.insert(0, os.path.abspath(a.ooda)); import compact as K, compact2 as C2
cb = json.load(open(a.codebook or os.path.join(a.ooda, 'codebook2.json'))); TAGS = {v: k for k, v in K.TAG.items()}; TAGS_INV = K.TAG
parse = lambda s: dict(x.split('=', 1) for x in s.split()); fmt = lambda d: ' '.join(f'{k}={v}' for k, v in d.items())
P, train = set(), set()
for d in a.sources.split(','):
    for f in sorted(glob.glob(os.path.join(d.strip(), 'orient_S?.jsonl'))):
        for l in open(f, encoding='utf-8'):
            r = json.loads(l); P.add(r['messages'][1]['content'])
            if r['split'] != 'test' and os.path.basename(d.strip().rstrip('\\/')) == 'data_compact2': train.add(r['messages'][1]['content'])
P = sorted(P); print(f'{len(P)} distinct real Orient prompts collected; {len(train)} of them come from the training/validation files')
if a.max_prompts: P = P[:a.max_prompts]
RAW = {}
if a.selftest:
    def ask_text(prompt):
        if prompt in RAW: return RAW[prompt]
        i = ask(prompt); return f'{i:03d}000'
    def ask(prompt):                                    # stand-in: correct for small n, blind to emergencies in busy cells (to exercise the report)
        d = parse(prompt); tag = 'emerg' if d['event'] == '1' and int(d['n']) <= 12 else 'base'
        return next(i for i, s in enumerate(cb['orient_a']) if s.split('|')[0] == tag)
else:
    from transformers import AutoTokenizer; import llama_engine as LE
    tok = AutoTokenizer.from_pretrained(a.model); eng = LE.LlamaEngine({'orient': a.url.rstrip('/')}, tok)
    if not eng.health('orient'): sys.exit(f'no llama-server answering at {a.url}')
    def ask_text(prompt): return eng.gen('orient', [{'role': 'system', 'content': C2.ORIENT_SYSTEM2}, {'role': 'user', 'content': prompt}], max_new=16)[0].strip()
    def ask(prompt):
        try: return int(ask_text(prompt)[:3])
        except Exception: return None
def tag_of(prompt):
    i = ask(prompt); return 'invalid' if i is None or not 0 <= i < len(cb['orient_a']) else cb['orient_a'][i].split('|')[0]
nb = lambda n: '1-2' if n <= 2 else '3-5' if n <= 5 else '6-12' if n <= 12 else '13-20' if n <= 20 else '21-40' if n <= 40 else '41+'
def table(title, rows, keyf):
    G = collections.defaultdict(lambda: [0, 0])
    for p, ok in rows: g = G[keyf(parse(p))]; g[0] += 1; g[1] += ok
    print(f'  {title}: ' + '   '.join(f'{k}: {v[1]}/{v[0]} ({100.0 * v[1] / v[0]:.0f}%)' for k, v in sorted(G.items(), key=lambda x: x[0])))
print('\n=== probe A: twins of every real prompt ===')
ev1, ev0 = [], []
for k, p in enumerate(P):
    d = parse(p); d1 = dict(d, event='1'); d0 = dict(d, event='0')
    ev1.append((fmt(d1), tag_of(fmt(d1)) == 'emerg')); ev0.append((fmt(d0), tag_of(fmt(d0)) != 'emerg'))
    if (k + 1) % 500 == 0: print(f'   ... {k + 1}/{len(P)}', flush=True)
good = sum(ok for _, ok in ev1); print(f'event=1 -> emergency_preempt: {good}/{len(ev1)} = {100.0 * good / len(ev1):.1f}%')
table('by n (vehicles in cell)', ev1, lambda d: nb(int(d['n']))); table('by load', ev1, lambda d: d['load']); table('by QoS severity', ev1, lambda d: d['qos'])
table('by standstill', ev1, lambda d: d['standstill']); table('by regulatory ctx', ev1, lambda d: d['reg'])
seen = [(p, ok) for p, ok in ev1 if p in train]; new = [(p, ok) for p, ok in ev1 if p not in train]
print(f'  prompt in training: {sum(o for _, o in seen)}/{len(seen)}   prompt NOT in training: {sum(o for _, o in new)}/{len(new)}')
good0 = sum(ok for _, ok in ev0); print(f'event=0 -> NOT emergency (false-alarm check): {good0}/{len(ev0)} = {100.0 * good0 / len(ev0):.1f}%')
print('\n=== probe B: emergency in every traffic condition (sweep n over each observed combination of the other fields) ===')
combos = sorted({tuple((k, v) for k, v in parse(p).items() if k not in ('n', 'event')) for p in P}); sw = [int(x) for x in a.sweep.split(',')]
B = []
for c in combos:
    for n in sw: d = dict(c); d = dict(n=str(n), **{k: v for k, v in c if k in ('load', 'qos', 'standstill')}, event='1', **{k: v for k, v in c if k in ('mode', 'oem', 'reg')}); B.append((fmt(d), tag_of(fmt(d)) == 'emerg'))
print(f'{len(combos)} combinations x {len(sw)} n-values = {len(B)} probes; emergency detected in {sum(o for _, o in B)}/{len(B)} = {100.0 * sum(o for _, o in B) / len(B):.1f}%')
table('by n', B, lambda d: f'{int(d["n"]):02d}')

if a.eval_file:
    print('\n=== held-out synthetic test split: exact code match (tag AND weights, constraints, intents) ===')
    R = [json.loads(l) for l in open(a.eval_file, encoding='utf-8')]; R = [r for r in R if r['split'] == 'test']
    if a.selftest:
        for k, r in enumerate(R): d = parse(r['messages'][1]['content']); RAW[r['messages'][1]['content']] = r['messages'][2]['content'] if int(d['n']) <= 30 else '000000'
    rows = []
    for r in R:
        p = r['messages'][1]['content']; got = ask_text(p); gold = r['messages'][2]['content']
        try: gt = cb['orient_a'][int(got[:3])].split('|')[0]
        except Exception: gt = 'invalid'
        rows.append((p, got == gold, gt == TAGS_INV[r['scenario_tag']]))
    print(f'{len(rows)} test prompts (whole groups of conditions held out of training): exact code {sum(o for _, o, _ in rows)}/{len(rows)} = {100.0 * sum(o for _, o, _ in rows) / len(rows):.1f}%   tag only {sum(t for _, _, t in rows)}/{len(rows)} = {100.0 * sum(t for _, _, t in rows) / len(rows):.1f}%')
    for title, kf in (('by event', lambda d: 'event=' + d['event']), ('by qos', lambda d: d['qos']), ('by load', lambda d: d['load']), ('by n', lambda d: nb(int(d['n']))), ('by standstill', lambda d: d['standstill'])):
        G = collections.defaultdict(lambda: [0, 0])
        for p, o, t in rows: g = G[kf(parse(p))]; g[0] += 1; g[1] += o
        print(f'  {title:14s}' + '   '.join(f'{k}: {v[1]}/{v[0]} ({100.0 * v[1] / v[0]:.0f}%)' for k, v in sorted(G.items(), key=lambda x: str(x[0]))))
