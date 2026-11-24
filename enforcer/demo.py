"""One-command demo (replay mode, no GPU needed).

Takes the policies the REAL fine-tuned chain produced (llm_policies.jsonl, written by ooda_loop.py) and shows, episode by episode,
what the cell experiences under four controllers at a chosen OODA refresh delay:
    static | follow the LLM policy only | DRL alone | LLM policy + DRL enforcer (soft)
Usage (from the enforcer folder):
    python demo.py --trace runs/llm_policies.jsonl --delay 4.4 --n 8
Add --delay 67 to see the original slow chain, --delay 0.2 for an (as yet unachieved) 200 ms chain."""
import argparse, json, numpy as np
import cell_env as E, traces as T, baselines as Bl
from ppo_numpy import Agent
from train_ppo import drl_controller

ap = argparse.ArgumentParser()
ap.add_argument('--trace', default='runs/llm_policies.jsonl'); ap.add_argument('--runs', default='runs')
ap.add_argument('--delay', type=float, default=4.4); ap.add_argument('--n', type=int, default=8); ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.trace)]
llm = {r['obs_id']: (r['policy'], r['context']) for r in rows if r.get('policy') and r.get('context')}
eps = sorted(T.build('test', llm), key=lambda e: e['ids'][0])
def load(p):
    g = Agent(E.OBS_DIM, E.ACT_DIM); g.load(p); return g
b3 = drl_controller(load(f"{a.runs}/b3_cplex_s{a.seed}_best.npz")); b2s = drl_controller(load(f"{a.runs}/b2s_cplex_s{a.seed}_best.npz"))
CTL = (('static', Bl.ctl_static, False, False), ('policy-only', Bl.ctl_policy, False, False), ('DRL alone', b3, True, False), ('LLM+DRL soft', b2s, False, True))

print(f'\nOODA refresh delay {a.delay:g} s | {len(eps)} episodes from the real model chain (each = 4 windows x 5 s, 100 ms ticks)')
print('Metric per controller: URLLC traffic left unserved (%) / ticks over the safety budget (%)\n')
hdr = f"{'episode':18s}{'scenario (LLM Orient)':24s}" + ''.join(f'{n:>17s}' for n, *_ in CTL); print(hdr); print('-' * len(hdr))
tot = {n: [] for n, *_ in CTL}
show = set(np.linspace(0, len(eps) - 1, min(a.n, len(eps))).round().astype(int).tolist())
for i, e in enumerate(eps):
    tag = llm[e['ids'][1]][1].get('scenario_tag', '?') if len(e['ids']) > 1 else '?'
    cells = []
    for n, c, vd, sf in CTL:
        r = Bl.run([e], c, delay=a.delay, seed=4000, view_default=vd, soft=sf)
        cells.append(f"{100*r['unserved'].mean():7.2f} /{100*r['over'].mean():5.1f}"); tot[n].append((r['unserved'].mean(), r['over'].mean()))
    if i in show: print(f"{e['ids'][0][:17]:18s}{tag:24s}" + ''.join(f'{c:>17s}' for c in cells))
print('-' * len(hdr))
print(f"{'ALL ' + str(len(eps)) + ' episodes':18s}{'':24s}" + ''.join(f"{100*np.mean([x[0] for x in tot[n]]):7.2f} /{100*np.mean([x[1] for x in tot[n]]):5.1f}".rjust(17) for n, *_ in CTL))
print('\nRead: lower is better in both numbers. Policy-only degrades as the delay grows (the policy is stale); the DRL enforcer keeps the cell safe.')
print('Caveat: placeholder radio model; relative comparison only. This demo uses one trained enforcer seed and one noise seed, so totals differ slightly from the 3x3-seed table in eval_llm_policies.py.')
