r"""Enforcer-in-the-loop replay on S5/S7: live-chain policies vs CPLEX gold policies, through the trained enforcers.
Run from the enforcer folder:
  python eval_s7_replay.py --data_dir ..\OODA_S7\data --sit S7 --trace ..\Fine-tune\loop_S7_dcov2.jsonl --runs runs --delays 0,0.3,1 --nseeds 3
Same 4-window x 5 s episodes, same radio model, same seeds and safety layer for every controller (paired)."""
import argparse, json, os, re, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell_env as E, baselines as Bl, traces as T
from ppo_numpy import Agent
from train_ppo import drl_controller
import system_model as M

ap = argparse.ArgumentParser()
ap.add_argument('--data_dir', required=True); ap.add_argument('--sit', default='S7'); ap.add_argument('--trace', required=True)
ap.add_argument('--runs', default='runs'); ap.add_argument('--delays', default='0,0.3,1'); ap.add_argument('--nseeds', type=int, default=3)
ap.add_argument('--agent_seeds', default='0,1,2'); ap.add_argument('--variant', default='v0'); ap.add_argument('--W', type=int, default=4)
a = ap.parse_args()
jl = lambda p: [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]

# ---- policy sources ----
gold = {}; rows = jl(os.path.join(a.data_dir, f'decide_{a.sit}.jsonl'))
use_var = any(f'-{a.variant}-' in r['id'] for r in rows)
for r in rows:
    if use_var and f'-{a.variant}-' not in r['id']: continue
    gold[r['obs_id']] = (json.loads(r['messages'][2]['content']), json.loads(r['messages'][1]['content'])['context'])
chain = {}; trows = jl(a.trace); keys = collections.Counter(k for r in trows[:50] for k in r)
for r in trows:
    oid = r.get('obs_id') or r.get('id')
    if oid and r.get('policy') and r.get('context'): chain[oid] = (r['policy'], r['context'])
print(f'gold policies {len(gold)} | chain policies {len(chain)} (trace records {len(trows)}; keys seen: {sorted(keys)[:14]})')
common = set(gold) & set(chain)
print(f'windows with both sources: {len(common)}')
if not common: sys.exit('no overlap between trace obs_id and decide obs_id - check --trace / --sit')

# ---- episodes (consecutive 5 s windows per cell) ----
obs = [o for o in jl(os.path.join(a.data_dir, f'observe_{a.sit}.jsonl')) if o.get('layer', 'rsu_edge') == 'rsu_edge' and o['id'] in common]
seqs = collections.defaultdict(list)
for o in obs:
    cell = o.get('cell') or (re.search(r'(C\d+)', o['id']) or [None, 'cell'])[1]
    ob = json.loads(o['messages'][2]['content']); seqs[cell].append((ob['window_s'][0], o['id'], ob))
def mk(src):
    eps = []
    for cell, L in seqs.items():
        L.sort(); i = 0
        while i + a.W <= len(L):
            ch = L[i:i + a.W]
            if all(abs(ch[k + 1][0] - ch[k][0] - 5.0) < 1e-6 for k in range(a.W - 1)):
                N = np.array([c[2]['n_active_vehicles'] for c in ch], float)
                d0 = np.array([[M.cell_load(c[2])[0][k] for k in M.CL] for c in ch])
                ev = np.array([1.0 if c[2]['special_events'] else 0.0 for c in ch])
                pv = [E.default_policy(N[0], d0[0])] + [T.pol_vec(*src[c[1]]) for c in ch]
                pol = {k: np.array([p[k] for p in pv]) for k in ('x', 'lvl', 'alpha', 'umin', 'emax', 's')}
                eps.append(dict(N=N, d0=d0, ev=ev, pol=pol, sit=a.sit, cell=cell, ids=[c[1] for c in ch])); i += a.W
            else: i += 1
    return eps
EPS = {'chain': mk(chain), 'gold': mk(gold)}
print({k: len(v) for k, v in EPS.items()}, 'episodes;', 'with emergency window:', sum(bool(e['ev'].any()) for e in EPS['gold']))
if not EPS['gold']: sys.exit('no complete 4-window episodes')

# ---- controllers ----
sd = [int(x) for x in a.agent_seeds.split(',')]
def load(mode, s):
    p = f'{a.runs}/{mode}_cplex_s{s}_best.npz'
    if not os.path.exists(p): return None
    g = Agent(E.OBS_DIM, E.ACT_DIM); g.load(p); return g
AG = {m: [g for g in (load(m, s) for s in sd) if g] for m in ('b2', 'b2s', 'b3')}
print({m: len(v) for m, v in AG.items()}, 'trained enforcers loaded')
CTL = [('B0 static', 'b0'), ('B1 policy-follow', 'b1'), ('B3 DRL alone', 'b3'), ('B2 OODA+DRL', 'b2'), ('B2s OODA+DRL soft', 'b2s'), ('B4 oracle bound', 'b4')]
def run(kind, eps, D, k):
    sdn = 4000 + k
    if kind == 'b0': return [Bl.run(eps, Bl.ctl_static, delay=D, seed=sdn, view_default=True)]
    if kind == 'b1': return [Bl.run(eps, Bl.ctl_policy, delay=D, seed=sdn)]
    if kind == 'b4': return [Bl.run(eps, Bl.ctl_oracle, delay=D, seed=sdn)]
    return [Bl.run(eps, drl_controller(g), delay=D, seed=sdn, view_default=(kind == 'b3'), soft=(kind == 'b2s')) for g in AG[kind]]

def sub(eps, emg): return [e for e in eps if bool(e['ev'].any()) == emg] if emg is not None else eps
def stat(rs, f): return float(np.mean([f(r) for r in rs])) if rs else float('nan')
for D in [float(x) for x in a.delays.split(',')]:
    for subset, emg in (('ALL episodes', None), ('EMERGENCY episodes', True), ('no-emergency episodes', False)):
        n = len(sub(EPS['gold'], emg))
        if n == 0: continue
        print(f'\n##### {a.sit} | refresh delay {D:g} s | {subset} (n={n}) #####')
        print(f"{'controller':22s}{'policy src':>10s}{'URLLC uns%':>12s}{'ticks>bud%':>12s}{'eMBB sat':>10s}{'power':>8s}{'reward':>10s}")
        for nm, kind in CTL:
            for src in (('chain', 'gold') if kind in ('b1', 'b2', 'b2s') else ('gold',)):
                eps = sub(EPS[src], emg)
                if not eps: continue
                rs = [r for k in range(a.nseeds) for r in run(kind, eps, D, k)]
                if not rs: continue
                lab = src if kind in ('b1', 'b2', 'b2s') else '-'
                print(f"{nm:22s}{lab:>10s}{100*stat(rs, lambda r: r['unserved'].mean()):12.3f}{100*stat(rs, lambda r: r['over'].mean()):12.1f}"
                      f"{stat(rs, lambda r: r['sat'][:, 1].mean()):10.3f}{stat(rs, lambda r: r['power'].mean()):8.2f}{stat(rs, lambda r: r['rtrue'].mean()):+10.3f}")
print('\nRead: chain vs gold on the same row = cost of using the live LLM chain instead of the CPLEX teacher. B0 and B3 ignore the policy; B4 is an unrealisable bound on gold constraints.')
