"""Hidden-context benchmark evaluation. Scenarios x refresh delays x controllers; trained agents are averaged over their seeds (mean +- sd across seeds).
python eval_hidden.py --runs runs_ctx --seeds 0,1,2 --delays 0,4.4,17"""
import argparse, json, os, numpy as np
import cell_env as E, traces as T, baselines as Bl, events as V
from ppo_numpy import Agent
from train_ctx import CFG, controller, env_kw
ap = argparse.ArgumentParser(); ap.add_argument('--runs', default='runs_ctx'); ap.add_argument('--seeds', default='0,1,2'); ap.add_argument('--delays', default='0,4.4,17')
ap.add_argument('--nseeds', type=int, default=3); ap.add_argument('--out', default='runs_ctx/hidden_eval.json'); ap.add_argument('--scen', default='none,cap,emerg,energy,all')
a = ap.parse_args(); sd = [int(x) for x in a.seeds.split(',')]
def load(mode, s):
    p = f'{a.runs}/{mode}_s{s}_best.npz'
    if not os.path.exists(p): return None
    g = Agent(CFG[mode]['od'], E.ACT_DIM); g.load(p); return g
agents = {m: [g for g in (load(m, s) for s in sd) if g is not None] for m in ('blind', 'pol', 'ctx')}
print({m: len(v) for m, v in agents.items()}, 'trained agents found')
base = T.build('test'); print(len(base), 'test episodes (gold CPLEX policies + injected hidden events)')
NAMES = {'static': 'static, no OODA', 'policy': 'follow policy only', 'blind': 'DRL alone (blind, trained)', 'pol': 'policy+DRL (no ctx features)', 'ctx': 'context-aware DRL'}
res = {}
for scen in a.scen.split(','):
    eps = V.attach(base, np.random.default_rng(2024), *V.SCEN[scen]); res[scen] = {}
    for D in [float(x) for x in a.delays.split(',')]:
        res[scen][D] = {}
        rows = {'static': [V.run_ev(eps, Bl.ctl_static, delay=D, seed=4000 + k, view_default=True) for k in range(a.nseeds)],
                'policy': [V.run_ev(eps, Bl.ctl_policy, delay=D, seed=4000 + k) for k in range(a.nseeds)]}
        for m, ags in agents.items():
            rows[m] = [V.run_ev(eps, controller(g), delay=D, seed=4000 + k, view_default=CFG[m]['view_default'], **env_kw(m)) for g in ags for k in range(a.nseeds)]
        print(f'\n##### scenario {scen} | refresh delay {D:g} s #####')
        print(f"{'controller':30s}{'URLLC uns%':>11s}{'emg miss%':>10s}{'cap viol%':>10s}{'power':>7s}{'eMBB':>7s}{'reward':>16s}")
        for m, rs in rows.items():
            if not rs: continue
            f = lambda k: np.mean([r[k] for r in rs])
            per_agent = [np.mean([r['reward'] for r in rs[i * a.nseeds:(i + 1) * a.nseeds]]) for i in range(len(rs) // a.nseeds)] if m in agents else [f('reward')]
            res[scen][D][m] = {k: float(f(k)) for k in rs[0]}; res[scen][D][m]['reward_sd_seeds'] = float(np.std(per_agent))
            print(f"{NAMES[m]:30s}{100*f('unserved'):11.3f}{100*f('emg_miss'):10.2f}{100*f('cap_viol'):10.1f}{f('power'):7.2f}{f('embb'):7.3f}{f('reward'):+10.3f} +-{np.std(per_agent):.3f}")
json.dump({s: {str(d): v for d, v in x.items()} for s, x in res.items()}, open(a.out, 'w'), indent=1)
