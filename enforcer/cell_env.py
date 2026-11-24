"""Vectorised RSU-cell environment for the DRL enforcer (proposal Sec. 3.2).

Radio / traffic model = the SAME analytic placeholder model the Decide labels were optimised on (system_model.py + config.OPT),
so the enforcer and the OODA policies share one definition of capacity, interference cap and class mix.
Real data enters through the demand: every 5 s window carries the vehicle count and special event of a real held-out RSU-cell
window (SUMO/ns-3-derived telemetry). Inside a window demand bursts and the channel fades stochastically.

One env tick = 100 ms. (The proposal's 1 ms TTI is abstracted: a tick aggregates 100 TTIs; per-PRB slot assignment is NOT simulated.)

Enforcer action (6 numbers, tanh-squashed): residual PRB shares around the OODA policy and per-class power back-off below the
policy budget. A hard SAFETY LAYER then enforces the Orient constraints (URLLC minimum share, eMBB maximum share, min share per class,
power budget, 23 dBm, interference cap), so the enforcer cannot violate them whatever it outputs.
"""
import os, sys, math
import numpy as np
_here = os.path.dirname(os.path.abspath(__file__))
PIPE = os.environ.get('OODA_PIPELINE') or next((p for p in (os.path.join(_here, '..', 'ooda_pipeline'), os.path.join(_here, '..', 'OODA')) if os.path.exists(os.path.join(p, 'config.py'))), os.path.join(_here, '..', 'ooda_pipeline'))
sys.path.insert(0, PIPE)
import config as C
import system_model as M

CL = M.CL; O = C.OPT; MIN_SHARE = M.MIN_SHARE / 100.0
STRATS = M.STRATS
TICK_S = 0.1; WIN_S = 5.0; TICKS_PER_WIN = int(round(WIN_S / TICK_S))
SIGMA_BURST = np.array([0.10, 0.35, 0.15]); RHO_BURST = 0.90          # demand burstiness per class (urllc periodic, embb bursty)
SIGMA_FADE_COMMON, SIGMA_FADE_CLASS, RHO_FADE = 3.0, 1.5, 0.95        # dB
BUDGET = 1e-3          # tolerated URLLC unserved fraction per tick (Lyapunov virtual-queue drift budget)
LAMBDA = 50.0          # weight of the Lyapunov drift-plus-penalty term
OBS_DIM = 33; ACT_DIM = 6
OBS_DIM_CTX = 37       # + 4 slow-loop context features (power cap, cap flag, emergency flag, energy-saving intent); only for ctx_obs=True
# --- hidden-context benchmark (events.py): facts that exist in the world and reach the enforcer ONLY through the slow loop (Orient/Decide context) ---
EMG_MBPS = 1.0         # extra URLLC demand of an emergency burst (deadline traffic, NOT in the enforcer's tick-level observation)
EMG_P_START = 0.12     # per-tick start probability of a burst inside an emergency window; burst lasts 3-6 ticks
ETA_HI = 0.5           # extra power-penalty weight when the operator's intent is energy saving
REG_PEN = 1.0          # reward penalty at >= 6 dB above a regulatory power cap

S_FAC = np.array([[O['strategy'][s][i] for i in range(4)] for s in STRATS])      # (4, 4): iot factor, prb factor, cap relief dB, reuse

def eff_np(level, s_idx, N):
    iot = np.minimum(O['iot_db_max'], O['iot_db_per_vehicle'] * N) * S_FAC[s_idx, 0]
    return np.log2(1 + 10 ** ((level - M.PL_DB - M.NOISE_DBM - iot[..., None]) / 10))

def rho_np(level, s_idx, N):
    """Mbps per 1 % of the PRBs. level (B,3) dBm, s_idx (B,), N (B,)."""
    return O['cal_capacity_mbps'] / 100 * eff_np(level, s_idx, N) / M.REF * S_FAC[s_idx, 1][:, None]

def cap_dbm_np(N, s_idx):
    return O['cap_dbm_base'] - np.minimum(O['cap_db_max_drop'], O['cap_db_per_vehicle'] * N) + S_FAC[s_idx, 2]

DEFAULT_CONS = dict(umin=20.0, emax=70.0)
DEFAULT_ALPHA = np.array([0.45, 0.35, 0.20])

def default_policy(N, d0):
    """What runs when no OODA policy exists yet (and the policy of the 'DRL alone' baseline): demand-proportional PRBs, full power."""
    x = d0 / d0.sum()
    return dict(x=x, lvl=np.full(3, 23.0), alpha=DEFAULT_ALPHA.copy(), umin=DEFAULT_CONS['umin'], emax=DEFAULT_CONS['emax'], s=0)

def safety_layer(x_raw, lvl_raw, pol, N, soft=False):
    """Project (x, lvl) onto the feasible set. x_raw (B,3) >0 ; lvl_raw (B,3) dBm. pol: dict of per-env arrays.
    PRB shares: exact Euclidean projection onto {sum=1, lo<=x<=hi}, lo=[max(URLLC min,5%),5%,5%], hi=[1,eMBB max,1],
    found by bisection on the shift (x = clip(x_raw + shift, lo, hi))."""
    B = x_raw.shape[0]
    lo = np.stack([np.maximum(pol['umin'] / 100.0, MIN_SHARE), np.full(B, MIN_SHARE), np.full(B, MIN_SHARE)], 1)
    hi = np.stack([np.ones(B), np.maximum(pol['emax'] / 100.0, MIN_SHARE), np.ones(B)], 1)
    a = np.full(B, -2.0); b = np.full(B, 2.0)
    for _ in range(60):
        m = 0.5 * (a + b); sm = np.clip(x_raw + m[:, None], lo, hi).sum(1)
        a = np.where(sm < 1, m, a); b = np.where(sm >= 1, m, b)
    x = np.clip(x_raw + b[:, None], lo, hi); x = x / x.sum(1, keepdims=True)
    lvl = np.minimum(lvl_raw, 23.0 if soft else np.minimum(pol['lvl'], 23.0))   # soft: the policy's power budget is guidance, only 23 dBm + interference cap are hard
    lvl = np.maximum(lvl, 0.0)
    if 'cap' in pol: lvl = np.minimum(lvl, pol['cap'][:, None])   # regulatory per-class cap, only when the controller's context contains it
    capdbm = cap_dbm_np(N, pol['s'].astype(int))
    p_avg = (x * 10 ** (lvl / 10)).sum(1)
    ratio = np.minimum(1.0, 10 ** (capdbm / 10) / np.maximum(p_avg, 1e-9))
    lvl = lvl + 10 * np.log10(ratio)[:, None]
    return x, lvl

def map_action(a, pol, soft=False, wide=False):
    """tanh-squashed residual action -> raw (x, lvl) before the safety layer. wide=True (context-blind DRL in the hidden-context benchmark):
    soft share logits around the (default) policy and a free power range 9..23 dBm (start 20), so it can learn any context-free compromise."""
    t = np.tanh(a)
    if wide:
        logits = np.log(np.maximum(pol['x'], 1e-3)) + 3.0 * t[:, :3]
        x = np.exp(logits - logits.max(1, keepdims=True)); x /= x.sum(1, keepdims=True)
        return x, 20.0 + 11.0 * t[:, 3:]
    logits = np.log(np.maximum(pol['x'], 1e-3)) + (3.0 if soft else 1.5) * t[:, :3]
    x = np.exp(logits - logits.max(1, keepdims=True)); x /= x.sum(1, keepdims=True)
    if soft: return x, pol['lvl'] + 8.0 * t[:, 3:]                   # +/-8 dB around the policy's budget (clipped by the hard limits)
    lvl = pol['lvl'] - 10.0 / (1 + np.exp(-(4.0 * t[:, 3:] - 4.0)))   # back-off below the policy budget: ~0.2 dB at a=0 (starts equal to policy-follow), up to 5 dB
    return x, lvl


class CellEnv:
    """B parallel episodes, each W windows long. episode dict (all numpy):
        N (W,) vehicles, d0 (W,3) mean demand Mbps, ev (W,) event flag,
        pol: dict with arrays of length W+1: entry 0 = default policy, entry k+1 = OODA policy computed from window k-1's observation
             keys x(W+1,3) lvl(W+1,3) alpha(W+1,3) umin(W+1) emax(W+1) s(W+1)
    policy_delay_ticks L: a new policy takes effect L ticks into its window (models OODA compute + network latency)."""
    def __init__(self, episodes, delay_ticks=0, seed=0, stress=None, soft=False, ctx_obs=False, wide=False):
        self.soft = soft; self.ctx_obs = ctx_obs; self.wide = wide; self.ev_on = 'truth' in episodes[0]
        self.stress = stress          # None | dict(start=tick, dur=ticks, urllc_mult=float, fade_db=float): unannounced surge + correlated fade, unknown to the policy
        self.B = len(episodes); self.W = len(episodes[0]['N']); assert all(len(e['N']) == self.W for e in episodes)
        self.eps = episodes; self.L = delay_ticks; self.rng = np.random.default_rng(seed)
        st = lambda k: np.stack([e[k] for e in episodes])
        self.N = st('N').astype(float); self.d0 = st('d0'); self.ev = st('ev').astype(float)
        self.P = {k: np.stack([e['pol'][k] for e in episodes]) for k in ('x', 'lvl', 'alpha', 'umin', 'emax', 's') + (('cap', 'emg', 'eta') if self.ev_on else ())}
        if self.ev_on: self.tr = {k: np.stack([e['truth'][k] for e in episodes]) for k in ('cap', 'emg', 'eta')}   # TRUE per-window events
        self.T = self.W * TICKS_PER_WIN

    def reset(self):
        B = self.B; self.t = 0
        self.logb = self.rng.normal(0, 1, (B, 3)) * SIGMA_BURST
        self.gc = self.rng.normal(0, SIGMA_FADE_COMMON, B); self.gk = self.rng.normal(0, SIGMA_FADE_CLASS, (B, 3))
        self.q = np.zeros((B, 3)); self.Z = np.zeros(B); self.sat_prev = np.ones((B, 3)); self.emg_left = np.zeros(B, int); self.emg_on = np.zeros(B)
        self._advance_stochastic(first=True)
        return self._obs()

    def _active_idx(self):
        """Index of the policy in force. Entry j of the policy arrays = policy computed from window j-1's observation.
        Total OODA latency L ticks = m whole windows + r ticks: the policy of window w arrives at window w+m, r ticks in."""
        w = self.t // TICKS_PER_WIN; within = self.t % TICKS_PER_WIN
        m, r = divmod(self.L, TICKS_PER_WIN)
        return max(0, w + (1 if within >= r else 0) - m)

    def _pol(self):
        k = self._active_idx(); w = self.t // TICKS_PER_WIN
        g = lambda key: self.P[key][:, k]
        return {key: g(key) for key in self.P}, w

    def _advance_stochastic(self, first=False):
        if not first:
            r = self.rng
            self.logb = RHO_BURST * self.logb + SIGMA_BURST * math.sqrt(1 - RHO_BURST ** 2) * r.normal(0, 1, (self.B, 3))
            self.gc = RHO_FADE * self.gc + SIGMA_FADE_COMMON * math.sqrt(1 - RHO_FADE ** 2) * r.normal(0, 1, self.B)
            self.gk = RHO_FADE * self.gk + SIGMA_FADE_CLASS * math.sqrt(1 - RHO_FADE ** 2) * r.normal(0, 1, (self.B, 3))
        self.b = np.exp(self.logb - SIGMA_BURST ** 2 / 2)
        self.g = self.gc[:, None] + self.gk
        st = self.stress
        if st is not None and st['start'] <= self.t < st['start'] + st['dur']:
            self.b = self.b.copy(); self.b[:, 0] *= st['urllc_mult']; self.g = self.g - st['fade_db']
        if self.ev_on and not first:                      # emergency bursts (hidden): Markov on/off inside emergency windows
            w = min(self.t // TICKS_PER_WIN, self.W - 1); act = self.tr['emg'][:, w] > 0
            left = np.maximum(self.emg_left - 1, 0); start = (left == 0) & act & (self.rng.random(self.B) < EMG_P_START)
            self.emg_left = np.where(start, self.rng.integers(3, 7, self.B), left); self.emg_on = ((self.emg_left > 0) & act).astype(float)

    def _obs(self, view_default=False):
        pol, w = self._pol(); w = min(w, self.W - 1)
        N = self.N[:, w]; d0 = self.d0[:, w]; d = d0 * self.b
        if view_default:
            x_pol = d0 / d0.sum(1, keepdims=True); lvl_pol = np.full((self.B, 3), 23.0); alpha = np.tile(DEFAULT_ALPHA, (self.B, 1))
            umin = np.full(self.B, DEFAULT_CONS['umin']); emax = np.full(self.B, DEFAULT_CONS['emax']); s = np.zeros(self.B, int)
        else:
            x_pol, lvl_pol, alpha, umin, emax, s = pol['x'], pol['lvl'], pol['alpha'], pol['umin'], pol['emax'], pol['s'].astype(int)
        g_obs = self.g + self.rng.normal(0, 1.0, self.g.shape)                 # noisy CSI estimate
        qn = np.clip(self.q / np.maximum(d0, 1e-3), 0, 3)
        ctx = []
        if self.ctx_obs:
            if view_default: ctx = [np.zeros((self.B, 4))]
            else: ctx = [np.stack([np.minimum(pol['cap'], 23.0) / 23.0, (pol['cap'] < 90).astype(float), pol['emg'], pol['eta'] / ETA_HI], 1)]
        return np.concatenate([
            (N / 60.0)[:, None], self.ev[:, w][:, None], d0 / O['cal_capacity_mbps'], self.b, g_obs / 5.0, self.sat_prev, qn,
            (self.Z / 5.0)[:, None], x_pol, lvl_pol / 23.0, alpha, (umin / 100.0)[:, None], (emax / 100.0)[:, None],
            np.eye(4)[s]] + ctx, 1).astype(np.float32)

    def step(self, a, view_default=False, direct=None):
        """a: (B,6) raw action. direct=(x,lvl) bypasses the enforcer (policy-follow / static / oracle baselines); safety layer still applies."""
        pol, w = self._pol(); w = min(w, self.W - 1)
        N = self.N[:, w]; d0 = self.d0[:, w]; d = d0 * self.b
        emg_now = self.emg_on * (self.tr['emg'][:, w] > 0) if self.ev_on else np.zeros(self.B)
        if self.ev_on: d = d.copy(); d[:, 0] += EMG_MBPS * emg_now                                          # hidden emergency demand
        if view_default:        # DRL-alone baseline: the policy it sees (and is bounded by) is the generic default
            pol = dict(x=d0 / d0.sum(1, keepdims=True), lvl=np.full((self.B, 3), 23.0), alpha=np.tile(DEFAULT_ALPHA, (self.B, 1)),
                       umin=np.full(self.B, DEFAULT_CONS['umin']), emax=np.full(self.B, DEFAULT_CONS['emax']), s=np.zeros(self.B))
        x_raw, lvl_raw = direct if direct is not None else map_action(a, pol, self.soft, self.wide)
        x, lvl = safety_layer(x_raw, lvl_raw, pol, N, self.soft)
        s_idx = pol['s'].astype(int)
        cap = rho_np(lvl + self.g, s_idx, N) * 100.0 * x                       # Mbps per class
        arr = d * TICK_S; qmax = 2.0 * d0 + 1e-3
        avail = self.q + arr; served = np.minimum(avail, cap * TICK_S)
        served[:, 0] = np.minimum(arr[:, 0], cap[:, 0] * TICK_S)                 # URLLC: deadline traffic, no queue
        newq = np.minimum(avail - served, qmax); newq[:, 0] = 0
        denom = np.where(np.arange(3) == 0, arr, avail)
        sat = np.clip(served / np.maximum(denom, 1e-9), 0, 1)
        u = 1 - sat[:, 0]                                                        # URLLC unserved fraction
        alpha = self.P['alpha'][:, w + 1] if not view_default else None          # metric alpha = true OODA alpha of this window
        a_train = pol['alpha']
        p_pen = ((x * lvl).sum(1) / 23.0)
        util = a_train[:, 0] * sat[:, 0] + a_train[:, 1] * sat[:, 1] + a_train[:, 2] * (sat[:, 2] - 0.1 * p_pen)
        reward = util - LAMBDA * (self.Z * u + u) / 10.0
        at = self.P['alpha'][:, w + 1]                                         # TRUE OODA weights of this window (common yardstick for every method)
        util_t = at[:, 0] * sat[:, 0] + at[:, 1] * sat[:, 1] + at[:, 2] * (sat[:, 2] - 0.1 * p_pen)
        reward_true = util_t - LAMBDA * (self.Z * u + u) / 10.0
        viol = np.zeros(self.B)
        if self.ev_on:                                                          # hidden-context objective: true power-cap violation and energy intent
            viol = np.maximum(lvl.max(1) - self.tr['cap'][:, w], 0.0)
            reward_true = reward_true - self.tr['eta'][:, w] * p_pen - REG_PEN * np.minimum(viol / 6.0, 1.0)
            reward = reward_true                                                 # every controller is trained on the TRUE objective (fair)
        self.Z = np.maximum(self.Z + u - BUDGET, 0)
        self.q = newq; self.sat_prev = sat
        self.t += 1
        done = self.t >= self.T
        if not done: self._advance_stochastic()
        obs = self._obs(view_default) if not done else None
        info = dict(reward_true=reward_true, sat=sat, u=u, x=x, lvl=lvl, util=util, power=p_pen, alpha_true=self.P['alpha'][:, min(self.t // TICKS_PER_WIN, self.W - 1) + 1 if not done else self.W],
                    umin=pol['umin'], emax=pol['emax'], viol=viol, emg_on=emg_now,
                    capped=(self.tr['cap'][:, w] < 90) if self.ev_on else np.zeros(self.B, bool))
        return obs, reward.astype(np.float32), done, info
