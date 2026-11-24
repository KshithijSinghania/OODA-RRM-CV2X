"""Stress tests on TEST episodes (never trained on):
 (A) bounded OODA policy error e: policies corrupted (proposal Sec. 5.1: safety must hold for ||P_OODA - P*|| <= e)
 (B) unannounced URLLC surge (x3) + correlated 6 dB fade for 2 s that no policy anticipated.
Usage: python eval_stress.py --runs runs --seeds 0,1,2"""
import argparse, json, numpy as np
import cell_env as E, traces as T, baselines as Bl
from ppo_numpy import Agent
from train_ppo import drl_controller
ap = argparse.ArgumentParser(); ap.add_argument('--runs', default='runs'); ap.add_argument('--seeds', default='0'); ap.add_argument('--nseeds', type=int, default=3)
a = ap.parse_args(); sd = [int(x) for x in a.seeds.split(',')]
def load(p):
    g = Agent(E.OBS_DIM, E.ACT_DIM); g.load(p); return g
b2 = [load(f'{a.runs}/b2_cplex_s{s}_best.npz') for s in sd]
import os
b2s = [load(f'{a.runs}/b2s_cplex_s{s}_best.npz') for s in sd if os.path.exists(f'{a.runs}/b2s_cplex_s{s}_best.npz')]; b3 = [load(f'{a.runs}/b3_cplex_s{s}_best.npz') for s in sd]
base = T.build('test', 'cplex'); res = {}
def evaluate(eps, name, stress=None):
    row = {}
    for nm, ctls, vd, sf in (('B0 static', [Bl.ctl_static], False, False), ('B1 policy-follow', [Bl.ctl_policy], False, False),
                         ('B3 DRL alone', [drl_controller(g) for g in b3], True, False), ('B2 OODA+DRL (hard)', [drl_controller(g) for g in b2], False, False),
                         ('B2s OODA+DRL (soft)', [drl_controller(g) for g in b2s], False, True)):
        if not ctls: continue
        vals = []
        for c in ctls:
            for k in range(a.nseeds):
                r = Bl.run(eps, c, seed=2000 + k, view_default=vd, stress=stress, soft=sf); vals.append((r['unserved'].mean(), r['over'].mean(), r['rtrue'].mean(), r['sat'].mean(0)[1], r['power'].mean()))
        v = np.array(vals); row[nm] = dict(unserved=v[:, 0].mean(), over=v[:, 1].mean(), reward=v[:, 2].mean(), embb=v[:, 3].mean(), power=v[:, 4].mean(), reward_sd=v[:, 2].std(), unserved_sd=v[:, 0].std())
        print(f"{name:14s} {nm:20s} URLLC unserved {100*v[:,0].mean():7.3f}% (sd {100*v[:,0].std():.3f}) | ticks>budget {100*v[:,1].mean():5.1f}% | eMBB {v[:,3].mean():.3f} | power {v[:,4].mean():.2f} | reward {v[:,2].mean():+.3f} (sd {v[:,2].std():.3f})", flush=True)
    return row
print(f'(models: {len(sd)} training seed(s) each for B2 and B3)')
print('\n=== A. corrupted OODA policy ===')
for e in (0.0, 0.25, 0.5, 0.75, 1.0):
    res[f'policy_err_{e}'] = evaluate(T.corrupt_policies(base, e, seed=7), f'err={e}')
print('\n=== B. unannounced surge + fade ===')
res['surge'] = evaluate(base, 'surge', stress=dict(start=60, dur=20, urllc_mult=3.0, fade_db=6.0))
res['surge_long'] = evaluate(base, 'surge(5s)', stress=dict(start=60, dur=50, urllc_mult=3.0, fade_db=6.0))
json.dump(res, open(f'{a.runs}/stress_results.json', 'w'), indent=1, default=float)
