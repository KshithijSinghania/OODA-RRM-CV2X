"""Hidden-context benchmark: events that are TRUE in the world but reach the enforcer only through the slow loop.

  cap     regulatory per-class transmit-power cap (14/17/20 dBm) from some window on. Orient's context carries it (real S3 windows contain
          `tx_power_cap_dbm`); Decide's power budget respects it. A context-blind controller cannot know it.
  emerg   emergency window(s): bursts of extra URLLC demand with a deadline (real S4 windows: preempt_vehicle_ids, urllc_min 45 %, embb_max 30 %,
          max_policy_latency 50 ms). The extra demand is NOT in the enforcer's tick-level observation.
  energy  operator intent switches to energy saving: the true objective adds ETA_HI * (normalised power). Orient's `active_intents` carries it.

The slow loop learns about an event with the same lag as any policy (window of observation + OODA latency), so refresh speed matters here.
SYNTHETIC INJECTION: the real data has these fields only in some S3/S4 windows; here they are injected with controlled frequency so the benchmark has
enough events. The interface (which context field carries what) is the real one. Event timing, caps, burst sizes and ETA_HI are my choices."""
import copy, numpy as np
import cell_env as E

NO_CAP = 99.0
SCEN = {'none': (0, 0, 0), 'cap': (1, 0, 0), 'emerg': (0, 1, 0), 'energy': (0, 0, 1), 'all': (1, 1, 1), 'mixed': (0.5, 0.5, 0.5)}   # (p_cap, p_emerg, p_energy)

def attach(eps, rng, p_cap=0.5, p_emg=0.5, p_eta=0.5, caps=(14.0, 17.0, 20.0)):
    out = []
    for e in eps:
        e = copy.deepcopy(e); W = len(e['N'])
        cap = np.full(W, NO_CAP); emg = np.zeros(W); eta = np.zeros(W)
        if rng.random() < p_cap: w0 = int(rng.integers(1, W)); cap[w0:] = float(rng.choice(caps))
        if rng.random() < p_emg: w0 = int(rng.integers(1, W)); emg[w0:w0 + 2] = 1.0
        if rng.random() < p_eta: w0 = int(rng.integers(1, W)); eta[w0:] = E.ETA_HI
        e['truth'] = dict(cap=cap, emg=emg, eta=eta)
        p = e['pol']                                              # entry 0 = default (no context); entry k+1 = policy computed from window k's context
        p['cap'] = np.concatenate([[NO_CAP], cap]); p['emg'] = np.concatenate([[0.0], emg]); p['eta'] = np.concatenate([[0.0], eta])
        for j in range(1, W + 1):                                 # the policy is consistent with its own context (as the real Decide is)
            if p['cap'][j] < 90: p['lvl'][j] = np.minimum(p['lvl'][j], p['cap'][j])
            if p['emg'][j] > 0:
                p['umin'][j] = max(p['umin'][j], 45.0); p['emax'][j] = min(p['emax'][j], 30.0)
                a = max(p['x'][j][0], 0.45); b = min(p['x'][j][1], 0.30); c = 1 - a - b
                if c < 0.05: a -= 0.05 - c; c = 0.05
                p['x'][j] = np.array([a, b, c])
        out.append(e)
    return out

def run_ev(eps, controller, delay=0.0, seed=1, view_default=False, soft=False, ctx_obs=False, wide=False):
    """Like baselines.run but returns the hidden-context metrics. delay in seconds. Means over ticks and episodes (event metrics over the relevant ticks)."""
    env = E.CellEnv(eps, delay_ticks=int(round(delay / E.TICK_S)), seed=seed, soft=soft, ctx_obs=ctx_obs, wide=wide); obs = env.reset(); B = len(eps)
    s = dict(u=0.0, emg_u=0.0, emg_n=0.0, vio=0.0, vdb=0.0, cap_n=0.0, power=0.0, embb=0.0, mmtc=0.0, rew=0.0, over=0.0); k = 0
    while True:
        c = controller(env, obs)
        obs, r, done, info = env.step(c.get('a', np.zeros((B, E.ACT_DIM))), view_default=view_default, direct=c.get('direct'))
        em = info['emg_on']; cp = info['capped'].astype(float)
        s['u'] += info['u'].sum(); s['emg_u'] += (info['u'] * em).sum(); s['emg_n'] += em.sum()
        s['vio'] += ((info['viol'] > 0.1) * cp).sum(); s['vdb'] += (info['viol'] * cp).sum(); s['cap_n'] += cp.sum()
        s['power'] += info['power'].sum(); s['embb'] += info['sat'][:, 1].sum(); s['mmtc'] += info['sat'][:, 2].sum(); s['rew'] += info['reward_true'].sum()
        s['over'] += (info['u'] > E.BUDGET).sum(); k += 1
        if done: break
    n = B * k
    return dict(unserved=s['u'] / n, over=s['over'] / n, emg_miss=s['emg_u'] / max(s['emg_n'], 1), emg_ticks=s['emg_n'] / n, cap_viol=s['vio'] / max(s['cap_n'], 1),
                cap_db=s['vdb'] / max(s['cap_n'], 1), cap_ticks=s['cap_n'] / n, power=s['power'] / n, embb=s['embb'] / n, mmtc=s['mmtc'] / n, reward=s['rew'] / n)
