import numpy as np
def _fit(X, Y, depth, rng, min_leaf=2):
    n = len(X)
    if depth == 0 or n < 2 * min_leaf or (Y.sum(0) == n).any(): return ('L', int(Y.sum(0).argmax()))
    best = None
    for j in range(X.shape[1]):
        o = np.argsort(X[:, j], kind='stable'); xs = X[o, j]; cum = np.cumsum(Y[o], 0); tot = cum[-1]
        i = np.nonzero(xs[:-1] < xs[1:])[0]
        i = i[(i + 1 >= min_leaf) & (n - i - 1 >= min_leaf)]
        if not len(i): continue
        nl = (i + 1)[:, None]; nr = n - nl; L = cum[i]; Rr = tot - L
        g = (nl[:, 0] * (1 - ((L / nl) ** 2).sum(1)) + nr[:, 0] * (1 - ((Rr / nr) ** 2).sum(1)))
        k = g.argmin()
        if best is None or g[k] < best[0]: best = (g[k], j, (xs[i[k]] + xs[i[k] + 1]) / 2)
    if best is None: return ('L', int(Y.sum(0).argmax()))
    _, j, t = best; m = X[:, j] <= t
    return ('N', j, t, _fit(X[m], Y[m], depth - 1, rng, min_leaf), _fit(X[~m], Y[~m], depth - 1, rng, min_leaf))
def _pred(t, x):
    while t[0] == 'N': t = t[3] if x[t[1]] <= t[2] else t[4]
    return t[1]
def forest_predict(Xtr, ytr, Xte, n_trees=9, depth=14, seed=0):
    rng = np.random.default_rng(seed); K = int(max(ytr.max(), 0)) + 1; Y = np.eye(K)[ytr]; votes = np.zeros((len(Xte), K))
    for _ in range(n_trees):
        b = rng.integers(0, len(Xtr), len(Xtr)); t = _fit(Xtr[b], Y[b], depth, rng)
        for r, x in enumerate(Xte): votes[r, _pred(t, x)] += 1
    return votes.argmax(1)

import json, glob, os, sys
sys.path.insert(0, r'..\OODA'); import compact2 as C2
cb = C2.load_codebook(); R = []
for dd in sys.argv[1:]:
    for f in glob.glob(os.path.join(dd, 'decide_S?.jsonl')):
        for l in open(f, encoding='utf-8'):
            r = json.loads(l)
            try:
                u = json.loads(r['messages'][1]['content']); p = json.loads(r['messages'][2]['content']); code = C2.encode_decide2(p, cb)
            except Exception: continue
            c = u['context']; k = c['constraint_overrides']; w = c['qos_weights']; cs = u['cell_state']; ev = cs['special_events']; rs = cs['resource_state']
            A = [w['urllc'], w['embb'], w['mmtc'], k['urllc_min_prb_share_pct'], k['embb_max_prb_share_pct'], k['max_policy_latency_ms'], k.get('tx_power_cap_dbm', -1),
                 int('preempt_vehicle_ids' in k), cs['n_active_vehicles'], int(bool(ev) and bool(ev[0]['in_scope'])), ev[0]['n_vehicles_within_300m'] if ev else -1]
            prb = list(u['current_allocation']['prb_pct'].values()); pw = list(u['current_allocation']['power_dbm'].values())
            res = [rs['offered_load_ratio'], rs['interference_iot_db'], rs['mec_load_ratio']]
            o = r['obs_id'].split('-')
            R.append((r['scenario_tag'], o[0] + o[2], code, A, prb, pw, res))
print('records', len(R))
SETS = {'compact prompt': lambda x: x[3], '+ current prb_pct': lambda x: x[3] + x[4], '+ current power': lambda x: x[3] + x[5],
        '+ resource_state': lambda x: x[3] + x[6], '+ everything': lambda x: x[3] + x[4] + x[5] + x[6]}
FIELDS = {'PRB split': slice(0, 6), 'power': slice(6, 9), 'strategy': slice(9, 10), 'MEC': slice(10, 12)}
rng0 = np.random.default_rng(0)
for tag in ('emergency_preempt', 'intersection_dense', 'baseline'):
    rs_ = [x for x in R if x[0] == tag]
    if len(rs_) < 50: continue
    if len(rs_) > 1500: rs_ = [rs_[i] for i in rng0.choice(len(rs_), 1500, replace=False)]
    cells = sorted({x[1] for x in rs_}); fold = {c: i % 5 for i, c in enumerate(cells)}; F = np.array([fold[x[1]] for x in rs_])
    print('\n=== %s (%d windows, %d cells) accuracy on HELD-OUT cells' % (tag, len(rs_), len(cells)))
    print('%-20s' % '' + ''.join('%12s' % f for f in FIELDS))
    for name, fx in SETS.items():
        X = np.array([fx(x) for x in rs_], float); line = '%-20s' % name
        for sl in FIELDS.values():
            _, y = np.unique([x[2][sl] for x in rs_], return_inverse=True); hit = 0
            for k in range(5):
                te = F == k; tr = ~te
                if te.sum() == 0 or tr.sum() == 0: continue
                hit += (forest_predict(X[tr], y[tr], X[te]) == y[te]).sum()
            line += '%11.1f%%' % (100 * hit / len(y))
        print(line)
