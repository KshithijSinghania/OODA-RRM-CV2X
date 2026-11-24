r"""Closed-loop hook for ooda_loop.py (analytic cell environment + trained enforcer in the loop).

Per 4-window episode (one cell, consecutive 5 s windows), causal timing (policy delay = 5 s = 50 ticks):
  window v is executed under the policy computed from window v-1's telemetry (window 0: the default policy).
  For window w:  Observe(w) -> Orient -> Decide -> policy(w) -> executed in window w+1 -> measured -> Act(w) feedback.
Modes (--cl_mode):
  dataset : feedback and QoS exactly as in the dataset (random synthetic faults, real next-window telemetry). Control run.
  enforcer: (1) Observe's qos_status of window w is replaced by what the enforcer measured in window w,
            (2) Act's feedback is built from the enforcer's measurements of window w+1.
In both modes the environment is stepped, so cell-level outcomes are recorded for the same episodes and noise seeds.
Timing assumption: the chain's total latency is <= 5 s (it is ~0.3 s), so a policy computed at the end of window w is active from window w+1."""
import os, sys, json, collections
import numpy as np

class ClosedLoop:
    def __init__(self, enforcer_dir, runs, kind='b2s', agent_seed=0, mode='enforcer', W=4, loss_thr=0.10, prr_target=1e-3,
                 viol_frac=0.5, rb_tol=0.10, seed=0, ep_out=None):
        sys.path.insert(0, os.path.abspath(enforcer_dir))
        import cell_env as E, baselines as Bl, traces as T
        import config as C, system_model as M
        from ppo_numpy import Agent
        self.E, self.Bl, self.T, self.C, self.M = E, Bl, T, C, M
        self.kind, self.mode, self.W = kind, mode, W
        self.loss_thr, self.prr_target, self.viol_frac, self.rb_tol, self.seed = loss_thr, prr_target, viol_frac, rb_tol, seed
        self.agent = None
        if kind in ('b2', 'b2s'):
            p = os.path.join(runs, f'{kind}_cplex_s{agent_seed}_best.npz')
            if not os.path.exists(p): sys.exit(f'enforcer weights not found: {p}')
            self.agent = Agent(E.OBS_DIM, E.ACT_DIM); self.agent.load(p)
        elif kind != 'b1': sys.exit("--cl_enforcer must be b2s, b2 or b1")
        self.ep_out = open(ep_out, 'w', encoding='utf-8') if ep_out else None
        self.cur = None; self.eps = []; self.ep_of = {}
        self.sev = collections.Counter(); self.viol = collections.Counter(); self.n_win = 0; self.n_set = 0; self.n_set_fail = 0

    # ---------------------------------------------------------------- episodes
    def group(self, chains):
        """Reorder chains into per-cell episodes of W consecutive 5 s windows; windows outside a full episode are dropped."""
        by = collections.defaultdict(list)
        for ch in chains:
            gobs = json.loads(ch[0]['messages'][2]['content'])
            by[ch[0]['id'].rsplit('-', 1)[0]].append((float(gobs['window_s'][0]), ch))
        out = []
        for cell in sorted(by):
            L = sorted(by[cell], key=lambda z: z[0]); i = 0
            while i + self.W <= len(L):
                seg = L[i:i + self.W]
                if all(abs(seg[k + 1][0] - seg[k][0] - 5.0) < 1e-6 for k in range(self.W - 1)):
                    ei = len(self.eps); self.eps.append([s[1] for s in seg])
                    for p, s in enumerate(seg): self.ep_of[s[1][0]['id']] = (ei, p); out.append(s[1])
                    i += self.W
                else: i += 1
        return out

    def start_episode(self, ei):
        E, M, T = self.E, self.M, self.T
        ch = self.eps[ei]; self.cur = ei
        self.gobs = [json.loads(c[0]['messages'][2]['content']) for c in ch]
        gctx = [json.loads(c[1]['messages'][2]['content']) for c in ch]
        self.alpha_true = np.array([[x['qos_weights'][k] for k in ('urllc', 'embb', 'mmtc')] for x in gctx], float)
        N = np.array([g['n_active_vehicles'] for g in self.gobs], float)
        d0 = np.array([[M.cell_load(g)[0][k] for k in M.CL] for g in self.gobs])
        ev = np.array([1.0 if g['special_events'] else 0.0 for g in self.gobs])
        pv = [E.default_policy(N[0], d0[0])] * (self.W + 1)
        pol = {k: np.array([p[k] for p in pv]) for k in ('x', 'lvl', 'alpha', 'umin', 'emax', 's')}
        self.env = E.CellEnv([dict(N=N, d0=d0, ev=ev, pol=pol)], delay_ticks=50, seed=self.seed + ei, soft=(self.kind == 'b2s'))
        self.obs = self.env.reset(); self.meas = []; self.ev = ev; self.pset = [False] * self.W
        self.ids = [c[0]['id'] for c in ch]

    def begin(self, go):
        ei, pos = self.ep_of[go['id']]
        if ei != self.cur: self.finish_episode(); self.start_episode(ei)
        self.pos = pos; self.ensure(pos)

    def ensure(self, v):
        while v < self.W and len(self.meas) <= v: self.meas.append(self.run_window(len(self.meas)))

    # ---------------------------------------------------------------- one 5 s window of the environment
    def run_window(self, v):
        E, Bl, env = self.E, self.Bl, self.env
        u, sat, pw, rt, fp, fi, fr = [], [], [], [], [], [], []
        at = self.alpha_true[v]
        for _ in range(E.TICKS_PER_WIN):
            if self.obs is None: break
            if self.kind == 'b1': a = np.zeros((1, E.ACT_DIM)); direct = Bl.ctl_policy(env, self.obs)['direct']
            else: a = self.agent.act(self.obs, deterministic=True)[0]; direct = None
            pol, _ = env._pol(); Zb = env.Z.copy()
            x_raw, lvl_raw = direct if direct is not None else E.map_action(a, pol, env.soft, env.wide)
            self.obs, _, _, info = env.step(a, direct=direct)
            lim = 23.0 if env.soft else np.minimum(pol['lvl'], 23.0)
            lvl1 = np.maximum(np.minimum(lvl_raw, lim), 0.0)
            fp.append(float((lvl_raw - lvl1).max() > 0.5)); fi.append(float((lvl1 - info['lvl']).max() > 0.5))
            fr.append(float(np.abs(info['x'] - x_raw).sum() / 2 > self.rb_tol))
            s = info['sat'][0]; uu = float(info['u'][0]); pp = float(info['power'][0])
            u.append(uu); sat.append(s); pw.append(pp)
            rt.append(at[0] * s[0] + at[1] * s[1] + at[2] * (s[2] - 0.1 * pp) - E.LAMBDA * (float(Zb[0]) * uu + uu) / 10.0)
        u = np.array(u); sat = np.array(sat)
        m = dict(window=v, u_mean=float(u.mean()), u_p95=float(np.percentile(u, 95)), f_gt=float((u > self.loss_thr).mean()), over=float((u > E.BUDGET).mean()),
                 embb_sat=float(sat[:, 1].mean()), urllc_sat=float(sat[:, 0].mean()), power=float(np.mean(pw)), rtrue=float(np.mean(rt)),
                 f_power=float(np.mean(fp)), f_intf=float(np.mean(fi)), f_rb=float(np.mean(fr)))
        m['severity'] = self.qos_fields(m)[0]; self.sev[m['severity']] += 1; self.n_win += 1
        return m

    def qos_fields(self, m):
        C = self.C; f = m['f_gt']; sev = C.SEVERITY_BINS[-1][1]
        for hi, name in C.SEVERITY_BINS:
            if f < hi: sev = name; break
        loss = 100.0 * m['u_mean']; delay = 4.5 + 4.0 * loss; p95 = 100.0 * m['u_p95']     # delay is a PROXY (the environment has no delay model)
        if sev == 'nominal': loss = min(loss, C.ACT_NOMINAL_LOSS_MAX - 0.1); delay = min(delay, C.ACT_NOMINAL_DELAY_MAX - 0.1)
        return sev, round(min(loss, 100.0), 2), round(min(p95, 100.0), 2), round(f, 3), round(min(delay, 150.0), 2)

    # ---------------------------------------------------------------- hooks called from ooda_loop.py
    def override_obs(self, obs):
        q = obs['qos_status']; m = self.meas[self.pos]; sev, loss, p95, f, delay = self.qos_fields(m)
        old = q.get('severity'); q.update(severity=sev, mean_loss_pct=loss, p95_loss_pct=p95, frac_loss_gt_10pct=f, frac_delay_gt_thr=f, mean_delay_ms=delay)
        return {'old_severity': old, 'new_severity': sev, 'mean_loss_pct': loss}

    def set_policy(self, pol, ctx):
        idx = self.pos + 1; ok = True
        try:
            pv = self.T.pol_vec(pol, ctx)
            for k in ('x', 'lvl', 'alpha', 'umin', 'emax', 's'): self.env.P[k][0, idx] = pv[k]
            self.pset[self.pos] = True; self.n_set += 1
        except Exception as e: ok = False; self.n_set_fail += 1; print('CL set_policy failed', repr(e))
        return {'ep': self.cur, 'pos': self.pos, 'policy_entry_set': ok, 'meas_this_window': self.meas[self.pos]}

    def feedback(self, obs_out):
        w = self.pos
        if w + 1 >= self.W: return None
        self.ensure(w + 1); m = self.meas[w + 1]; sev, loss, p95, f, delay = self.qos_fields(m)
        viol = []
        if m['u_mean'] > self.prr_target: viol.append('urllc_prr_below_target')
        if m['f_power'] >= self.viol_frac: viol.append('power_cap_exceeded')
        if m['f_intf'] >= self.viol_frac: viol.append('interference_limit_exceeded')
        if m['f_rb'] >= self.viol_frac: viol.append('rb_allocation_rejected')
        for v in viol: self.viol[v] += 1
        try: util = round(100.0 * min(1.0, float(self.gobs[w + 1]['resource_state']['offered_load_ratio'])), 1)
        except Exception: util = 50.0
        ws = float(obs_out['window_s'][1])
        return {'policy_id': None, 'window_s': [ws, ws + 5.0], 'auth_ok': True,
                'enforcer': {'applied': True, 'violations': sorted(viol), 'prb_util_pct': util},
                'observed': {'severity': sev, 'mean_loss_pct': loss, 'mean_delay_ms': delay, 'frac_loss_gt_10pct': f},
                'emergency_event_observed': bool(self.gobs[w + 1]['special_events'])}

    # ---------------------------------------------------------------- episode summaries
    def finish_episode(self):
        if self.cur is None: return
        self.ensure(self.W - 1)
        rec = {'ep': self.cur, 'ids': self.ids, 'emergency': bool(self.ev.any()), 'policy_set': self.pset, 'windows': self.meas}
        if self.ep_out: self.ep_out.write(json.dumps(rec) + '\n'); self.ep_out.flush()
        self.cur = None

    def finish_all(self):
        self.finish_episode()
        print(f"\n=== closed loop ({self.mode}, enforcer={self.kind}) ===")
        print(f"  episodes {len(self.eps)}   environment windows {self.n_win}   chain policies applied {self.n_set} (failed to set {self.n_set_fail})")
        print('  measured severity distribution (all windows):', dict(self.sev))
        if self.mode == 'enforcer': print('  violations reported to Act:', dict(self.viol))
        print('  NOTE: mean_delay_ms is a proxy; mec_overload cannot be produced by this environment.')
        if self.ep_out: self.ep_out.close()
