"""Minimal actor-critic PPO in numpy (no torch in this environment). Gaussian policy, tanh MLPs, GAE, Adam, manual backprop."""
import numpy as np, json

class MLP:
    def __init__(self, sizes, rng, last_scale=1.0):
        self.W = []; self.b = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            s = np.sqrt(2.0 / a) * (last_scale if i == len(sizes) - 2 else 1.0)
            self.W.append(rng.normal(0, s, (a, b)).astype(np.float64)); self.b.append(np.zeros(b))
    def forward(self, x):
        hs = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = hs[-1] @ W + b
            hs.append(np.tanh(z) if i < len(self.W) - 1 else z)
        return hs
    def backward(self, hs, dout):
        gW = [None] * len(self.W); gb = [None] * len(self.W); d = dout
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = hs[i].T @ d; gb[i] = d.sum(0)
            if i > 0: d = (d @ self.W[i].T) * (1 - hs[i] ** 2)
        return gW, gb
    def n_params(self): return sum(w.size + b.size for w, b in zip(self.W, self.b))

class Adam:
    def __init__(self, params, lr): self.p = params; self.lr = lr; self.m = [np.zeros_like(p) for p in params]; self.v = [np.zeros_like(p) for p in params]; self.t = 0
    def step(self, grads, max_norm=0.5):
        n = np.sqrt(sum((g ** 2).sum() for g in grads)); sc = min(1.0, max_norm / (n + 1e-8)); self.t += 1
        for i, (p, g) in enumerate(zip(self.p, grads)):
            g = g * sc; self.m[i] = 0.9 * self.m[i] + 0.1 * g; self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            p -= self.lr * (self.m[i] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[i] / (1 - 0.999 ** self.t)) + 1e-8)

class RunNorm:
    def __init__(self, d): self.n = 1e-4; self.mean = np.zeros(d); self.var = np.ones(d)
    def update(self, x):
        bm, bv, bn = x.mean(0), x.var(0), len(x); tot = self.n + bn; delta = bm - self.mean
        self.mean = self.mean + delta * bn / tot
        self.var = (self.var * self.n + bv * bn + delta ** 2 * self.n * bn / tot) / tot; self.n = tot
    def __call__(self, x): return np.clip((x - self.mean) / np.sqrt(self.var + 1e-6), -8, 8)

class Agent:
    def __init__(self, obs_dim, act_dim, hidden=128, seed=0, init_logstd=-0.7):
        r = np.random.default_rng(seed); self.rng = r
        self.actor = MLP([obs_dim, hidden, hidden, act_dim], r, last_scale=0.01)
        self.critic = MLP([obs_dim, hidden, hidden, 1], r, last_scale=1.0)
        self.logstd = np.full(act_dim, init_logstd); self.norm = RunNorm(obs_dim); self.ret_scale = 1.0
    def n_params(self): return self.actor.n_params() + self.critic.n_params() + self.logstd.size
    def act(self, obs, deterministic=False):
        o = self.norm(obs); mu = self.actor.forward(o)[-1]; v = self.critic.forward(o)[-1][:, 0] * self.ret_scale
        if deterministic: return mu, None, v
        std = np.exp(self.logstd); a = mu + std * self.rng.normal(size=mu.shape)
        lp = (-0.5 * ((a - mu) / std) ** 2 - self.logstd - 0.5 * np.log(2 * np.pi)).sum(1)
        return a, lp, v
    def save(self, path):
        np.savez(path, **{f'aW{i}': w for i, w in enumerate(self.actor.W)}, **{f'ab{i}': b for i, b in enumerate(self.actor.b)},
                 **{f'cW{i}': w for i, w in enumerate(self.critic.W)}, **{f'cb{i}': b for i, b in enumerate(self.critic.b)},
                 logstd=self.logstd, nmean=self.norm.mean, nvar=self.norm.var, ret_scale=self.ret_scale)
    def load(self, path):
        d = np.load(path)
        for i in range(len(self.actor.W)): self.actor.W[i][:] = d[f'aW{i}']; self.actor.b[i][:] = d[f'ab{i}']
        for i in range(len(self.critic.W)): self.critic.W[i][:] = d[f'cW{i}']; self.critic.b[i][:] = d[f'cb{i}']
        self.logstd[:] = d['logstd']; self.norm.mean = d['nmean']; self.norm.var = d['nvar']; self.ret_scale = float(d['ret_scale'])

def gae(rew, val, done_last, gamma=0.99, lam=0.95):
    """rew,val: (T,B). Terminal at T. Returns adv, ret (T,B)."""
    T = rew.shape[0]; adv = np.zeros_like(rew); last = 0
    for t in range(T - 1, -1, -1):
        nv = val[t + 1] if t + 1 < T else 0.0
        delta = rew[t] + gamma * nv - val[t]; last = delta + gamma * lam * last; adv[t] = last
    return adv, adv + val

def ppo_update(ag, opt_a, opt_c, obs, act, logp_old, adv, ret, rng, epochs=4, mb=4096, clip=0.2, ent=0.003):
    n = len(obs); o = ag.norm(obs); stats = []
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, mb):
            j = idx[s:s + mb]; m = len(j)
            ha = ag.actor.forward(o[j]); mu = ha[-1]; std = np.exp(ag.logstd)
            lp = (-0.5 * ((act[j] - mu) / std) ** 2 - ag.logstd - 0.5 * np.log(2 * np.pi)).sum(1)
            r = np.exp(lp - logp_old[j]); A = adv[j]
            active = ((A >= 0) & (r < 1 + clip)) | ((A < 0) & (r > 1 - clip))
            coef = np.where(active, -A * r, 0.0)[:, None] / m
            dmu = coef * (act[j] - mu) / std ** 2
            dls = (coef * (((act[j] - mu) / std) ** 2 - 1)).sum(0) - ent
            gW, gb = ag.actor.backward(ha, dmu)
            opt_a.step([*gW, *gb, dls])
            hc = ag.critic.forward(o[j]); v = hc[-1][:, 0]
            dv = (v - ret[j] / ag.ret_scale)[:, None] / m
            gW2, gb2 = ag.critic.backward(hc, dv); opt_c.step([*gW2, *gb2])
            stats.append((np.mean(np.abs(r - 1)), np.mean(~active)))
    return np.mean(stats, 0)
