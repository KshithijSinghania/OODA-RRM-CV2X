import argparse, collections, glob, json, os, sys, random
ap = argparse.ArgumentParser(); ap.add_argument('--ooda', required=True); ap.add_argument('--data', required=True); ap.add_argument('--url', default='http://127.0.0.1:8082')
ap.add_argument('--model', default='meta-llama/Llama-3.2-3B-Instruct'); ap.add_argument('--all', action='store_true', help='also score train and val records (default: test split only)')
ap.add_argument('--max_prompts', type=int, default=0); ap.add_argument('--selftest', action='store_true'); a = ap.parse_args()
R = [json.loads(l) for l in open(os.path.join(a.data, 'decide_S9.jsonl'), encoding='utf-8')]
if not a.all: R = [r for r in R if r['split'] == 'test']
if a.max_prompts: random.Random(0).shuffle(R); R = R[:a.max_prompts]
nb = lambda n: '01-04' if n <= 4 else '05-08' if n <= 8 else '09-12' if n <= 12 else '13-20' if n <= 20 else '21-40' if n <= 40 else '41-60' if n <= 60 else '61+'
nof = lambda p: int(dict(x.split('=', 1) for x in p.split())['n'])
if a.selftest:
    rng = random.Random(1)
    def ask(p, gold): return gold if rng.random() < .8 else str(int(gold[:3]) - 4).zfill(3) + gold[3:]
else:
    sys.path.insert(0, os.path.abspath(a.ooda)); import compact2 as C2; from transformers import AutoTokenizer; import llama_engine as LE
    eng = LE.LlamaEngine({'decide': a.url.rstrip('/')}, AutoTokenizer.from_pretrained(a.model))
    if not eng.health('decide'): sys.exit(f'no llama-server answering at {a.url}')
    def ask(p, gold): return eng.gen('decide', [{'role': 'system', 'content': C2.DECIDE_SYSTEM2}, {'role': 'user', 'content': p}], max_new=16)[0].strip()
def score(p, gold, out):
    out = ''.join(ch for ch in out if ch.isdigit())[:12]
    if len(out) < 12: return None
    gu, ge, ou, oe = int(gold[:3]), int(gold[3:6]), int(out[:3]), int(out[3:6])
    return {'exact': out == gold, 'prb': (ou, oe) == (gu, ge), 'prb1': abs(ou - gu) <= 1 and abs(oe - ge) <= 1, 'prb5': abs(ou - gu) + abs(oe - ge) <= 5,
            'power': out[6:9] == gold[6:9], 'strat': out[9] == gold[9], 'mec': out[10:12] == gold[10:12], 'unsafe': ou < gu - 5}
rows = []
for i, r in enumerate(R):
    p, gold = r['messages'][1]['content'], r['messages'][2]['content']
    try: s = score(p, gold, ask(p, gold))
    except Exception as e: s = None
    rows.append((r, p, s))
    if (i + 1) % 200 == 0: print(f'  {i + 1}/{len(R)}', flush=True)
COLS = [('exact', 'whole code'), ('prb', 'PRB exact'), ('prb1', 'PRB +-1pt'), ('prb5', 'PRB L1<=5'), ('power', 'power'), ('strat', 'strategy'), ('mec', 'MEC'), ('unsafe', 'URLLC<gold-5')]
def table(title, keyf):
    G = collections.defaultdict(list)
    for r, p, s in rows:
        k = keyf(r, p)
        if k is not None: G[k].append(s)
    print(f'\n== {title}\n{"":26s}{"n":>6s}{"invalid":>9s}' + ''.join(f'{c[1]:>13s}' for c in COLS))
    for k in sorted(G):
        v = [x for x in G[k] if x]; n = len(G[k]); line = f'{str(k):26s}{n:6d}{100.0 * (n - len(v)) / n:8.1f}%'
        for c, _ in COLS: line += f'{100.0 * sum(x[c] for x in v) / max(len(v), 1):12.1f}%'
        print(line)
table('ALL scored records', lambda r, p: 'all')
table('by split / extrapolation', lambda r, p: ('EXTRAP n>=80' if r.get('extrap') else r['split']))
table('by scenario tag', lambda r, p: r['scenario_tag'])
table('by vehicle count n', lambda r, p: nb(nof(p)))
table('emergency windows by n', lambda r, p: nb(nof(p)) if r['scenario_tag'] == 'emergency_preempt' else None)
table('prompt seen in ORIGINAL training data?', lambda r, p: 'seen' if r.get('input_seen_in_train') else 'new')
