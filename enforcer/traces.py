"""Episodes for the enforcer from real observe windows (RSU-cell layer) + the Decide policy of each window.
policy source: 'cplex' (decide_S*.jsonl gold), 'rule' (decide_rule_S*.jsonl), or a dict {obs_id: policy_json} (e.g. LLM chain output)."""
import os, sys, json, collections
import numpy as np
_here = os.path.dirname(os.path.abspath(__file__))
PIPE = os.environ.get('OODA_PIPELINE') or next((p for p in (os.path.join(_here, '..', 'ooda_pipeline'), os.path.join(_here, '..', 'OODA')) if os.path.exists(os.path.join(p, 'config.py'))), os.path.join(_here, '..', 'ooda_pipeline'))
sys.path.insert(0, PIPE)
import system_model as M, config as C
from cell_env import default_policy, STRATS, DEFAULT_ALPHA, DEFAULT_CONS
DATA = os.path.join(PIPE, 'data')
def jl(p): return [json.loads(l) for l in open(p)]

def pol_vec(rec_out, ctx):
    p = rec_out['prb_priority']; pw = rec_out['power_budget_dbm']
    q = ctx['qos_weights']; co = ctx['constraint_overrides']
    return dict(x=np.array([p['urllc_pct'], p['embb_pct'], p['mmtc_pct']]) / 100.0, lvl=np.array([pw['urllc'], pw['embb'], pw['mmtc']], float),
                alpha=np.array([q['urllc'], q['embb'], q['mmtc']], float), umin=float(co['urllc_min_prb_share_pct']), emax=float(co['embb_max_prb_share_pct']),
                s=STRATS.index(rec_out['interference_coordination']['strategy']))

def build(split, source='cplex', W=4, situations=('S1', 'S2', 'S3', 'S4'), variant=None):
    """variant='v0': keep only the v0 Orient/Decide variant of each window (what ooda_loop.py runs), for a paired comparison with LLM policies.
    Default None: last variant in the file wins (used for training/the earlier evaluations)."""
    eps = []
    for S in situations:
        obs = [o for o in jl(f'{DATA}/observe_{S}.jsonl') if o['split'] == split and o['layer'] == 'rsu_edge']
        fn = 'decide_' if source == 'cplex' else 'decide_rule_'
        dec = {}
        if isinstance(source, dict): dec = source
        else:
            for r in jl(f'{DATA}/{fn}{S}.jsonl'):
                if variant and f'-{variant}-' not in r['id']: continue
                dec[r['obs_id']] = (json.loads(r['messages'][2]['content']), json.loads(r['messages'][1]['content'])['context'])
        seqs = collections.defaultdict(list)
        for o in obs:
            if o['id'] in dec:
                ob = json.loads(o['messages'][2]['content']); seqs[o['cell']].append((ob['window_s'][0], o['id'], ob))
        for cell, L in seqs.items():
            L.sort(); i = 0
            while i + W <= len(L):
                ch = L[i:i + W]
                if all(abs(ch[k + 1][0] - ch[k][0] - 5.0) < 1e-6 for k in range(W - 1)):
                    N = np.array([c[2]['n_active_vehicles'] for c in ch], float)
                    d0 = np.array([[M.cell_load(c[2])[0][k] for k in M.CL] for c in ch])
                    ev = np.array([1.0 if c[2]['special_events'] else 0.0 for c in ch])
                    pv = [default_policy(N[0], d0[0])] + [pol_vec(*dec[c[1]]) for c in ch]
                    pol = {k: np.array([p[k] for p in pv]) for k in ('x', 'lvl', 'alpha', 'umin', 'emax', 's')}
                    eps.append(dict(N=N, d0=d0, ev=ev, pol=pol, sit=S, cell=cell, ids=[c[1] for c in ch]))
                    i += W
                else: i += 1
    return eps
if __name__ == '__main__':
    for sp in ('train', 'val', 'test'):
        e = build(sp); print(sp, len(e), collections.Counter(x['sit'] for x in e))


def corrupt_policies(eps, e, seed=0):
    """Bounded OODA policy error of size e in [0,1]: shares blended with a random split, power budgets lowered by up to 10*e dB,
    strategy replaced at random with prob e. Orient constraints (urllc_min, embb_max, alpha) are left as given (trusted)."""
    import copy
    r = np.random.default_rng(seed); out = []
    for ep in eps:
        ep = copy.deepcopy(ep); p = ep['pol']; W1 = len(p['umin'])
        for j in range(1, W1):
            x = (1 - e) * p['x'][j] + e * r.dirichlet(np.ones(3)); p['x'][j] = x / x.sum()
            p['lvl'][j] = np.maximum(p['lvl'][j] - e * r.uniform(0, 10, 3), 0)
            if r.random() < e: p['s'][j] = r.integers(0, 4)
        out.append(ep)
    return out


def corrupt_mixed(eps, rng, p=0.5):
    """Training augmentation: with prob p an episode gets a corrupted policy of random size e~U(0,1)."""
    out = []
    for ep in eps:
        out.append(corrupt_policies([ep], float(rng.uniform(0, 1)), seed=int(rng.integers(1 << 30)))[0] if rng.random() < p else ep)
    return out
