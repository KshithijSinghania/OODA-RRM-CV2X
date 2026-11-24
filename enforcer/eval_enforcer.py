"""Evaluate B0..B4 on the TEST episodes: policy sources x policy delay, several seeds. Writes results.json + latency_sweep.png.
Usage: python eval_enforcer.py --runs runs --seed 0"""
import argparse, json, numpy as np
import cell_env as E, traces as T, baselines as Bl
from ppo_numpy import Agent
from train_ppo import drl_controller

ap = argparse.ArgumentParser(); ap.add_argument('--runs', default='runs'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--nseeds', type=int, default=5)
ap.add_argument('--sources', default='cplex,rule'); ap.add_argument('--b2', default=None); ap.add_argument('--b3', default=None)
a = ap.parse_args()
def load(p):
    g = Agent(E.OBS_DIM, E.ACT_DIM); g.load(p); return g
b2 = load(a.b2 or f'{a.runs}/b2_cplex_s{a.seed}_best.npz'); b3 = load(a.b3 or f'{a.runs}/b3_cplex_s{a.seed}_best.npz')
DELAYS = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]
methods = {'B0 static': dict(c=Bl.ctl_static, vd=False), 'B1 policy-follow': dict(c=Bl.ctl_policy, vd=False),
           'B3 DRL alone': dict(c=drl_controller(b3), vd=True), 'B2 OODA+DRL': dict(c=drl_controller(b2), vd=False),
           'B4 oracle (bound)': dict(c=Bl.ctl_oracle, vd=False)}
res = {}
for src in a.sources.split(','):
    eps = T.build('test', src); res[src] = {}
    print(f'\n##### policy source: {src}  ({len(eps)} test episodes, {a.nseeds} seeds) #####')
    for L in DELAYS:
        print(f'--- policy delay {L:.1f} s ---'); res[src][L] = {}
        for nm, m in methods.items():
            if nm == 'B3 DRL alone' and False: pass
            runs = [Bl.run(eps, m['c'], delay=L, seed=1000 + s, view_default=m['vd']) for s in range(a.nseeds)]
            agg = {k: float(np.mean([r[k].mean() if k != 'sat' else r['sat'].mean(0)[0] for r in runs])) for k in ('sat', 'over', 'unserved', 'power', 'rtrue')}
            agg['embb'] = float(np.mean([r['sat'].mean(0)[1] for r in runs])); agg['mmtc'] = float(np.mean([r['sat'].mean(0)[2] for r in runs]))
            agg['reward']=agg['rtrue']; agg['reward_sd_over_seeds'] = float(np.std([r['rtrue'].mean() for r in runs]))
            res[src][L][nm] = agg
            print(f"{nm:20s} URLLC sat {agg['sat']:.4f} unserved {100*agg['unserved']:.3f}% ticks>budget {100*agg['over']:5.1f}% | eMBB {agg['embb']:.3f} mMTC {agg['mmtc']:.3f} | power {agg['power']:.2f} | reward {agg['reward']:+.3f} (sd {agg['reward_sd_over_seeds']:.3f})")
json.dump({s: {str(L): v for L, v in d.items()} for s, d in res.items()}, open(f'{a.runs}/results.json', 'w'), indent=1)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8)); col = {'B0 static': '#888', 'B1 policy-follow': '#d98c2b', 'B3 DRL alone': '#7b5ea7', 'B2 OODA+DRL': '#1f7a8c', 'B4 oracle (bound)': '#2e8b57'}
src0 = a.sources.split(',')[0]
for j, (key, ttl) in enumerate((('unserved', 'URLLC unserved traffic (%)'), ('reward', 'Reward per tick'), ('embb', 'eMBB satisfaction'))):
    for nm in methods:
        y = [res[src0][L][nm][key] * (100 if key == 'unserved' else 1) for L in DELAYS]; ax[j].plot(DELAYS, y, marker='o', label=nm, color=col[nm])
    ax[j].set_title(ttl); ax[j].set_xlabel('policy delay (s)'); ax[j].grid(alpha=.3)
ax[0].legend(fontsize=7); plt.suptitle(f'Effect of OODA policy delay (test episodes, policy source = {src0})'); plt.tight_layout(); plt.savefig(f'{a.runs}/latency_sweep.png', dpi=130)
