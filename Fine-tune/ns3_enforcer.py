r"""DRL enforcer for the live ns-3 loop (used by ns3_bridge.NS3Loop with --ns3_enforcer; put this file next to ooda_loop.py).

What it does (and what it does not):
  The trained numpy-PPO enforcer (enforcer/runs/b2s_cplex_s<seed>_best.npz, the same weights cl_env.py uses) runs for every cell that the OODA chain
  decides in a window. For that cell it steps the enforcer's analytic cell environment (enforcer/cell_env.py) through the 50 ticks (100 ms) of the
  window the new OODA policy will govern. The enforcer acts every tick on its own observation (queues, sat, noisy CSI, ...), exactly as in training.
  The average power it applied, converted to ONE per-cell TX offset with the same mapping as policy_to_schedule.py
      p_avg = sum_k x_k * 10^(level_k/10),  delta_dB = 10*log10(p_avg / 10^(23/10)),
  is compared with what the OODA policy would have applied through the same safety layer (policy-follow). The difference, `residual_db`, is added
  to the offset the live loop already computes (RSRP rule + protect, or the OODA delta), and ns-3 receives the sum (clipped to [default-18, default]).
  - The enforcer's state is the ANALYTIC model fed with the dataset telemetry of the window (vehicle count, special events -> per-class demand),
    not ns-3 measurements: ns-3 has no traffic classes, so its URLLC/eMBB/mMTC observations and PRB shares have no ns-3 counterpart.
    Only the enforcer's POWER behaviour is transferred to ns-3; its PRB shares are computed (they enter the p_avg weights) but nothing else uses them.
  - ns-3 feedback reaches the enforcer only indirectly, through the rule's offset and through the OODA Observe input (measured loss).
  Per-cell environments persist across windows (queues, virtual queue, fading state), as in the enforcer's training episodes.
Log: DIR/enforcer_log.csv, one line per (window, cell) decision."""
import os, sys, math
import numpy as np

class NS3Enforcer:
    def __init__(s, enf_dir, runs, kind='b2s', agent_seed=0, W=26, seed=0, log_path=None):
        enf_dir = os.path.abspath(enf_dir)
        if enf_dir not in sys.path: sys.path.insert(0, enf_dir)
        import cell_env as E, traces as T, system_model as M
        from ppo_numpy import Agent
        s.E, s.T, s.M = E, T, M
        if kind not in ('b2', 'b2s'): raise SystemExit("--cl_enforcer must be b2 or b2s for --ns3_enforcer")
        p = os.path.join(runs, f'{kind}_cplex_s{agent_seed}_best.npz')
        if not os.path.exists(p): raise SystemExit(f'enforcer weights not found: {p}')
        s.agent = Agent(E.OBS_DIM, E.ACT_DIM); s.agent.load(p)
        s.kind, s.W, s.seed, s.soft = kind, W, seed, (kind == 'b2s')
        s.envs = {}; s.n = 0; s.n_fail = 0; s.sum_res = 0.0; s.log = None
        if log_path:
            s.log = open(log_path, 'a'); s.log.write('k,site,N,policy_delta_db,enf_delta_db,residual_db,urllc_unserved_mean,x_urllc,x_embb,x_mmtc,lvl_urllc,lvl_embb,lvl_mmtc\n'); s.log.flush()

    def _env(s, site):
        if site not in s.envs:
            E, W = s.E, s.W
            dp = E.default_policy(1.0, np.ones(3))
            pol = {k: np.array([np.asarray(dp[k], float)] * (W + 1)) for k in ('x', 'lvl', 'alpha', 'umin', 'emax', 's')}
            ep = dict(N=np.ones(W), d0=np.ones((W, 3)), ev=np.zeros(W), pol=pol)
            env = E.CellEnv([ep], delay_ticks=50, seed=s.seed + 1000 * (site + 1), soft=s.soft)    # delay 50 ticks = the policy of window k governs window k+1 (as cl_env.py)
            env.reset(); s.envs[site] = env
        return s.envs[site]

    def run(s, site, k, g_next, pol, ctx):
        """policy `pol` computed at the end of ns-3 window k (for `site`) governs window k+1. g_next = dataset telemetry of that window
        (n_active_vehicles, special_events): the world's demand. Returns (residual_db, info)."""
        E, M, T = s.E, s.M, s.T
        env = s._env(site); wi = k + 1
        if wi >= env.W: return 0.0, None
        N = float(g_next['n_active_vehicles']); d0 = np.array([M.cell_load(g_next)[0][c] for c in M.CL], float)
        env.N[0, wi] = N; env.d0[0, wi] = d0; env.ev[0, wi] = 1.0 if g_next.get('special_events') else 0.0
        pv = T.pol_vec(pol, ctx)
        for key in ('x', 'lvl', 'alpha', 'umin', 'emax', 's'): env.P[key][0, wi] = pv[key]       # entry wi = policy active in window wi under delay 50
        if env.t < wi * E.TICKS_PER_WIN: env.t = wi * E.TICKS_PER_WIN                              # cells skip windows: jump the clock, keep queue/fading state
        obs = env._obs()
        pol_arr = {key: env.P[key][:, wi] for key in env.P}
        x0, l0 = E.safety_layer(np.asarray(pv['x'], float)[None], np.asarray(pv['lvl'], float)[None], pol_arr, env.N[:, wi], s.soft)
        pavg0 = float((x0 * 10 ** (l0 / 10)).sum())                                                 # policy-follow through the same safety layer
        pavgs, us, xs, ls = [], [], [], []
        for _ in range(E.TICKS_PER_WIN):
            if obs is None or env.t >= env.T: break
            a = s.agent.act(obs, deterministic=True)[0]
            obs, _, _, info = env.step(a)
            x, l = info['x'][0], info['lvl'][0]
            pavgs.append(float((x * 10 ** (l / 10)).sum())); us.append(float(info['u'][0])); xs.append(x.copy()); ls.append(l.copy())
        if not pavgs: return 0.0, None
        pe = float(np.mean(pavgs))
        enf_d = 10 * math.log10(max(pe, 1e-9) / 10 ** (23.0 / 10)); pol_d = 10 * math.log10(max(pavg0, 1e-9) / 10 ** (23.0 / 10))
        res = enf_d - pol_d
        xm, lm = np.mean(xs, 0), np.mean(ls, 0)
        s.n += 1; s.sum_res += res
        if s.log:
            s.log.write(f'{k},{site},{N:.0f},{pol_d:.3f},{enf_d:.3f},{res:.3f},{np.mean(us):.5f},' + ','.join(f'{v:.3f}' for v in xm) + ',' + ','.join(f'{v:.2f}' for v in lm) + '\n'); s.log.flush()
        return res, dict(policy_delta_db=round(pol_d, 3), enforcer_delta_db=round(enf_d, 3), residual_db=round(res, 3))

    def summary(s): return f'enforcer: {s.n} cell-window decisions, mean residual {s.sum_res / max(1, s.n):+.2f} dB, {s.n_fail} failures'
