"""Step 1: sanity + safety tests of cell_env. Run: python test_env.py"""
import numpy as np, sys
import cell_env as E, traces as T
rng = np.random.default_rng(0)
tr = T.build('train'); te = T.build('test')
print('episodes', len(tr), len(te))
ok = True
def check(name, cond, extra=''):
    global ok; ok &= bool(cond); print(('PASS ' if cond else 'FAIL ') + name, extra)

# 1. safety layer fuzz: random + extreme raw actions, random policies from real episodes
viol = dict(sum1=0, umin=0, emax=0, minshare=0, pcap=0, p23=0, intf=0); n = 0
for trial in range(200):
    eps = [tr[i] for i in rng.integers(0, len(tr), 64)]
    env = E.CellEnv(eps, seed=trial); env.reset()
    for t in range(20):
        a = rng.normal(0, 3, (64, E.ACT_DIM)) * (rng.random() < 0.5 or 1)
        if trial % 5 == 0: a = np.sign(a) * 50                      # extreme
        pol, w = env._pol(); N = env.N[:, w]
        xr, lr = E.map_action(a, pol)
        x, lvl = E.safety_layer(xr, lr, pol, N)
        tol = 1e-6; n += 64
        viol['sum1'] += (abs(x.sum(1) - 1) > 1e-6).sum()
        viol['umin'] += (x[:, 0] < pol['umin'] / 100 - tol).sum()
        viol['emax'] += (x[:, 1] > pol['emax'] / 100 + tol).sum()
        viol['minshare'] += (x.min(1) < E.MIN_SHARE - tol).sum()
        viol['pcap'] += (lvl > pol['lvl'] + tol).sum()
        viol['p23'] += (lvl > 23 + tol).sum()
        cap = E.cap_dbm_np(N, pol['s'].astype(int)); pav = (x * 10 ** (lvl / 10)).sum(1)
        viol['intf'] += (10 * np.log10(pav) > cap + 1e-6).sum()
        env.step(a)
        if env.t >= env.T: break
print('violations over', n, 'checks:', viol)
check('safety layer: zero violations', sum(int(v) for v in viol.values()) == 0)

def rollout(eps, mode, delay=0, seed=1):
    env = E.CellEnv(eps, delay_ticks=delay, seed=seed); env.reset(); B = len(eps)
    S = np.zeros((B, 3)); U = np.zeros(B); P = np.zeros(B); R = np.zeros(B); k = 0
    while True:
        pol, w = env._pol(); d0 = env.d0[:, min(w, env.W - 1)]
        if mode == 'B0': direct = (d0 / d0.sum(1, keepdims=True), np.full((B, 3), 23.0))
        elif mode == 'B1': direct = (pol['x'], pol['lvl'])
        elif mode == 'lowpow': direct = (pol['x'], np.full((B, 3), 5.0))
        obs, r, done, info = env.step(np.zeros((B, E.ACT_DIM)), direct=direct)
        S += info['sat']; U += (info['u'] > E.BUDGET); P += info['power']; R += r; k += 1
        if done: break
    return dict(sat=S / k, viol_ticks=U / k, power=P / k, reward=R / k, v=info)

def summary(eps, name, **kw):
    r = rollout(eps, **kw); s = r['sat'].mean(0)
    print(f"{name:22s} sat urllc/embb/mmtc = {s[0]:.3f} {s[1]:.3f} {s[2]:.3f} | urllc ticks over budget {r['viol_ticks'].mean():.3f} | power {r['power'].mean():.2f} | reward {r['reward'].mean():+.3f}")
    return r
print('\n--- baselines on TRAIN (all) ---')
r0 = summary(tr, 'B0 static', mode='B0'); r1 = summary(tr, 'B1 policy-follow', mode='B1'); rl = summary(tr, 'low power (5 dBm)', mode='lowpow')
for S_ in ('S2', 'S3', 'S4'):
    sub = [e for e in tr if e['sit'] == S_]
    summary(sub, f'B0 {S_} (n={len(sub)})', mode='B0'); summary(sub, f'B1 {S_}', mode='B1')

# 2. physics: more vehicles -> lower URLLC/eMBB satisfaction under B0
Nm = np.array([e['N'].mean() for e in tr]); sat0 = r0['sat']
c = np.corrcoef(Nm, sat0[:, 1])[0, 1]; check('more vehicles -> lower eMBB satisfaction (B0)', c < -0.05, f'corr={c:+.2f}')
check('full power >= low power on URLLC satisfaction', r0['sat'][:, 0].mean() >= rl['sat'][:, 0].mean() - 1e-9)
check('all rewards finite', np.isfinite(r0['reward']).all() and np.isfinite(r1['reward']).all())
check('baselines not degenerate (B0 eMBB sat in 0.05..0.98)', 0.05 < r0['sat'][:, 1].mean() < 0.98, f"{r0['sat'][:,1].mean():.3f}")

# 3. delay logic: which policy index is active at each tick
eps = tr[:2]
for L in (0, 10, 50):
    env = E.CellEnv(eps, delay_ticks=L, seed=0); env.reset(); seen = []
    for t in range(2 * E.TICKS_PER_WIN + 1):
        seen.append(env._active_idx()); env.t += 1
    exp = [ (t // 50) + (1 if (t % 50) >= L else 0) for t in range(len(seen))]
    check(f'policy delay L={L} ticks: active index correct', seen == exp)
print('\nOK' if ok else '\nSOME CHECKS FAILED')
