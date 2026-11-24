"""Step 2: baselines + evaluation harness. All run on the same episodes/seeds/safety layer (paired comparison).
B0 static | B1 policy-follow | B3 DRL-alone (see train_ppo) | B4 per-tick search oracle | B2 OODA+DRL (see train_ppo)."""
import numpy as np
import cell_env as E

def run(eps, controller, delay=0.0, seed=1, view_default=False, stress=None, soft=False):
    """controller(env, obs) -> dict(a=(B,6)) or dict(direct=(x,lvl)).  delay in seconds."""
    env = E.CellEnv(eps, delay_ticks=int(round(delay / E.TICK_S)), seed=seed, stress=stress, soft=soft); obs = env.reset(); B = len(eps)
    acc = dict(sat=np.zeros((B, 3)), over=np.zeros(B), unserved=np.zeros(B), power=np.zeros(B), reward=np.zeros(B), util=np.zeros(B), rtrue=np.zeros(B)); k = 0
    while True:
        c = controller(env, obs)
        obs, r, done, info = env.step(c.get('a', np.zeros((B, E.ACT_DIM))), view_default=view_default, direct=c.get('direct'))
        acc['sat'] += info['sat']; acc['over'] += info['u'] > E.BUDGET; acc['unserved'] += info['u']
        acc['power'] += info['power']; acc['reward'] += r; acc['rtrue'] += info['reward_true']; acc['util'] += info['util']; k += 1
        if done: break
    return {kk: v / k for kk, v in acc.items()}

def ctl_static(env, obs):
    pol, w = env._pol(); d0 = env.d0[:, min(w, env.W - 1)]
    return dict(direct=(d0 / d0.sum(1, keepdims=True), np.full((env.B, 3), 23.0)))
def ctl_policy(env, obs):
    pol, w = env._pol(); return dict(direct=(pol['x'], pol['lvl']))

def ctl_oracle(env, obs, n_cand=64, seed=0):
    """Per-tick random search over (x, lvl) with perfect knowledge of the CURRENT demand, queue and fading; same safety layer.
    Maximises the immediate reward-like score (alpha-weighted satisfaction, heavy URLLC penalty). Upper bound, not realisable."""
    rng = np.random.default_rng(seed + env.t)
    pol, w = env._pol(); w = min(w, env.W - 1); B = env.B
    N = env.N[:, w]; d0 = env.d0[:, w]; d = d0 * env.b; s_idx = pol['s'].astype(int)
    best = None; best_sc = np.full(B, -1e9)
    for i in range(n_cand):
        if i == 0: xr = pol['x']; lv = pol['lvl']
        elif i == 1: xr = d0 / d0.sum(1, keepdims=True); lv = np.full((B, 3), 23.0)
        else:
            xr = rng.dirichlet(np.ones(3) * 2, B) ; lv = rng.uniform(8, 23, (B, 3))
        x, lvl = E.safety_layer(xr, lv, pol, N)
        cap = E.rho_np(lvl + env.g, s_idx, N) * 100.0 * x
        arr = d * E.TICK_S; avail = env.q + arr; served = np.minimum(avail, cap * E.TICK_S); served[:, 0] = np.minimum(arr[:, 0], cap[:, 0] * E.TICK_S)
        denom = np.where(np.arange(3) == 0, arr, avail); sat = np.clip(served / np.maximum(denom, 1e-9), 0, 1)
        a = pol['alpha']; u = 1 - sat[:, 0]
        sc = (a * sat).sum(1) - 0.1 * a[:, 2] * (x * lvl).sum(1) / 23 - E.LAMBDA * (env.Z * u + u) / 10
        better = sc > best_sc; best_sc = np.where(better, sc, best_sc)
        best = (x.copy(), lvl.copy()) if best is None else (np.where(better[:, None], x, best[0]), np.where(better[:, None], lvl, best[1]))
    return dict(direct=best)

def fmt(name, r):
    s = r['sat'].mean(0)
    return f"{name:26s} urllc {s[0]:.3f} embb {s[1]:.3f} mmtc {s[2]:.3f} | urllc unserved {100*r['unserved'].mean():.3f}% | ticks>budget {100*r['over'].mean():5.1f}% | power {r['power'].mean():.2f} | reward {r['rtrue'].mean():+.3f}"

if __name__ == '__main__':
    import traces as T
    for split in ('val', 'test'):
        eps = T.build(split); print(f'\n=== {split}: {len(eps)} episodes ===')
        for nm, c in (('B0 static', ctl_static), ('B1 policy-follow', ctl_policy), ('B4 oracle (bound)', ctl_oracle)):
            print(fmt(nm, run(eps, c)))
