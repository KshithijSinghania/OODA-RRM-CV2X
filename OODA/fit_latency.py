#!/usr/bin/env python3
"""Where does the time of an Orient/Decide hop go, and what would another engine give?   (CPU only, reads your trace files)
  1) fits   seconds = overhead + prefill_s_per_token * prompt_tokens + decode_s_per_token * generated_tokens   on the traces written by ooda_loop.py
  2) projects the per-window time of other engines from rates YOU measured with llama-bench (pp = prompt tokens/s, tg = generated tokens/s)
  python fit_latency.py loop_compact.jsonl llm_policies.jsonl --cfg llama3b:1500:60 --cfg llama1b:4000:150
Projection = overhead + prompt/pp + generated/tg per hop; 'prefix cache' removes the static system prompt (about 200 tokens: estimate, --sys_tokens) from the prompt;
'compact input' replaces the user part by --compact_in tokens (needs Orient/Decide retrained on a compact input: NOT done, an assumption)."""
import argparse, json, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('traces', nargs='+'); ap.add_argument('--cfg', action='append', default=[], help='name:pp_tok_per_s:tg_tok_per_s')
ap.add_argument('--overhead', type=float, default=0.03, help='per-hop HTTP/client overhead in s (assumption)'); ap.add_argument('--sys_tokens', type=int, default=210)
ap.add_argument('--v2', action='store_true', help='project with ESTIMATED v2 token counts (compact2.py): ~110 prompt tokens and 3 / 5 generated tokens for Orient / Decide; replace with your trace after running --compact2')
ap.add_argument('--compact_in', type=int, default=250, help='assumed prompt tokens with a compact input encoding')
a = ap.parse_args()
rows = [json.loads(l) for f in a.traces for l in open(f)]
hops = ('orient', 'decide'); X = []; y = []; mean = {}
for h in hops:
    d = [(r['tok'][h][0], r['tok'][h][1], r['sec'][h]) for r in rows if r.get('tok', {}).get(h, [0])[0] and r['tok'][h][1] < 100]     # compact-format hops only
    if not d: continue
    d = np.array(d, float); mean[h] = d.mean(0); X += [[1, p, o] for p, o, _ in d]; y += list(d[:, 2])
X = np.array(X); y = np.array(y); co = np.linalg.lstsq(X, y, rcond=None)[0]; pred = X @ co; r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f'{len(rows)} windows, {len(y)} hop measurements (compact outputs).')
for h, m in mean.items(): print(f'  {h:7s} mean prompt {m[0]:.0f} tokens, generated {m[1]:.1f} tokens, {m[2]:.2f} s')
print(f'fit: seconds = {co[0]:.3f} + {co[1]:.5f} x prompt_tokens + {co[2]:.4f} x generated_tokens   (R2 {r2:.2f})')
print(f'  => decode {1/co[2]:.1f} tokens/s, prefill {1/co[1] if co[1] > 0 else float("inf"):.0f} tokens/s')
tot = sum(m[2] for m in mean.values()); print(f'  share of a typical hop: overhead {co[0]/np.mean(y):.0%}, prefill {co[1]*np.mean(X[:,1])/np.mean(y):.0%}, decode {co[2]*np.mean(X[:,2])/np.mean(y):.0%}')
print(f'\nmeasured now (HF + 4-bit + LoRA): {tot:.2f} s per window for Orient + Decide')
if a.v2: mean = {'orient': np.array([110., 3., 0.]), 'decide': np.array([110., 5., 0.])}; print('projection basis: v2 token counts (ESTIMATE): orient 110 prompt / 3 generated, decide 110 / 5')
print(f'HF path with these token counts: {sum(co[0] + co[1]*m[0] + co[2]*m[1] for m in mean.values()):.2f} s per window')
if a.cfg:
    print(f"\n{'engine (your llama-bench rates)':34s}{'as is':>9s}" + ('' if a.v2 else f"{'prefix cache':>14s}{'+ compact input':>17s}"))
    for c in a.cfg:
        name, pp, tg = c.split(':'); pp = float(pp); tg = float(tg); out = []
        for mode in (('asis',) if a.v2 else ('asis', 'cache', 'compact')):
            t = 0.0
            for h, m in mean.items():
                P = m[0] if mode == 'asis' else (m[0] - a.sys_tokens if mode == 'cache' else min(m[0], a.compact_in) - (0))
                t += a.overhead + P / pp + m[1] / tg
            out.append(t)
        print(f'{name+" pp "+str(int(pp))+" tg "+str(int(tg)):34s}{out[0]:8.2f}s' + ('' if a.v2 else f'{out[1]:13.2f}s{out[2]:16.2f}s'))
    print('\n(pp, tg come from your own llama-bench run; nothing here is measured on a real llama.cpp server yet.)')
