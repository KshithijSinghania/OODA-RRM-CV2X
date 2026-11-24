#!/usr/bin/env python3
"""Why did Orient miss some emergency windows?  Is it lookup (unseen prompt) or a real gap in training coverage?
  python emergency_robustness.py --trace loop_S7_3llm_full.jsonl --data_dir ..\\OODA_S7\\data --compact_dir ..\\OODA_S7\\data_compact2_S7 --train_dir ..\\OODA\\data_compact2"""
import argparse, collections, glob, json, os, re
ap = argparse.ArgumentParser(); [ap.add_argument('--' + k, required=True) for k in ('trace', 'data_dir', 'compact_dir', 'train_dir')]; a = ap.parse_args()
E = 'emergency_preempt'
rd = lambda d, ag: [json.loads(l) for f in sorted(glob.glob(os.path.join(d, ag + '_S?.jsonl'))) for l in open(f, encoding='utf-8')]
gold = {}
for r in rd(a.data_dir, 'orient'): gold.setdefault(r['obs_id'], r['scenario_tag'])
prompt, seen = {}, {}
for r in rd(a.compact_dir, 'orient'): prompt.setdefault(r['obs_id'], r['messages'][1]['content']); seen.setdefault(r['obs_id'], r['input_seen_in_train'])
parse = lambda s: dict(x.split('=', 1) for x in s.split())
train = [r for r in rd(a.train_dir, 'orient') if r['split'] != 'test']; tp = [r['messages'][1]['content'] for r in train]
ev1 = [p for p in tp if parse(p).get('event') == '1']
print(f'training Orient prompts: {len(tp)} ({len(set(tp))} distinct); with event=1: {len(ev1)} ({len(set(ev1))} distinct)')
print('  n_active_vehicles values seen with event=1 in training:', sorted({int(parse(p)['n']) for p in ev1}))
print('  regulatory contexts with event=1:', dict(collections.Counter(parse(p)['reg'] for p in ev1)))
T = [json.loads(l) for l in open(a.trace, encoding='utf-8')]; T = [t for t in T if 'chain_ok' in t and gold.get(t['obs_id']) == E]
hit = [t for t in T if t['scenario_tag'] == E]; miss = [t for t in T if t['scenario_tag'] != E]
print(f'\nemergency windows: {len(T)}; caught {len(hit)}, missed {len(miss)}')
print('\nseen-in-training x outcome (does the model only succeed on prompts it has memorised?):')
for lab, S in (('prompt seen in training', [t for t in T if seen.get(t['obs_id'])]), ('prompt NEW', [t for t in T if not seen.get(t['obs_id'])])):
    print(f'  {lab:26s} {len(S):4d} windows, caught {sum(t["scenario_tag"] == E for t in S):4d} ({100.0 * sum(t["scenario_tag"] == E for t in S) / max(1, len(S)):5.1f}%)')
print('\nfeature profile (caught vs missed):')
for k in ('n', 'load', 'qos', 'standstill', 'mode', 'oem', 'reg'):
    print(f'  {k:10s} caught {dict(collections.Counter(parse(prompt[t["obs_id"]])[k] for t in hit if t["obs_id"] in prompt))}\n  {"":10s} missed {dict(collections.Counter(parse(prompt[t["obs_id"]])[k] for t in miss if t["obs_id"] in prompt))}' if k != 'n' else
          f'  n          caught: min {min(int(parse(prompt[t["obs_id"]])["n"]) for t in hit)} max {max(int(parse(prompt[t["obs_id"]])["n"]) for t in hit)}  | missed: ' + str(sorted(int(parse(prompt[t["obs_id"]])["n"]) for t in miss)))
print('\nmissed prompts:'); [print('  ', t['obs_id'], '|', prompt[t['obs_id']], '| seen' if seen.get(t['obs_id']) else '| NEW') for t in miss]
