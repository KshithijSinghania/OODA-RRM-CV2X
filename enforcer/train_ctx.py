"""Train enforcers for the hidden-context benchmark (events.py). All modes are trained on the SAME true objective and the same random events.
  blind : sees only the generic default policy and its own tick-level state; free power range (what a DRL with no slow loop can learn)
  pol   : policy-aware (x, lvl, alpha, constraints in its observation, safety layer bounds incl. the policy's cap) but NO explicit context features
  ctx   : pol + 4 explicit context features (power cap, cap flag, emergency flag, energy intent)
Usage: python train_ctx.py --mode ctx --seed 0 --iters 150"""
import argparse, time, json, os, numpy as np
import cell_env as E, traces as T, events as V
from ppo_numpy import Agent, Adam, gae, ppo_update

CFG = {'blind': dict(view_default=True, soft=True, ctx_obs=False, wide=True, od=E.OBS_DIM, aug=False),
       'pol':   dict(view_default=False, soft=True, ctx_obs=False, wide=False, od=E.OBS_DIM, aug=True),
       'ctx':   dict(view_default=False, soft=True, ctx_obs=True, wide=False, od=E.OBS_DIM_CTX, aug=True)}

def controller(agent):
    def f(env, obs):
        mu, _, _ = agent.act(obs, deterministic=True); return dict(a=mu)
    return f
def env_kw(mode): c = CFG[mode]; return dict(soft=c['soft'], ctx_obs=c['ctx_obs'], wide=c['wide'])

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', default='ctx'); ap.add_argument('--iters', type=int, default=150)
    ap.add_argument('--batch', type=int, default=128); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--out', default='runs_ctx'); ap.add_argument('--eval_every', type=int, default=10)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); tag = f'{a.mode}_s{a.seed}'; C = CFG[a.mode]
    tr = T.build('train'); va = V.attach(T.build('val'), np.random.default_rng(999), *V.SCEN['mixed']); rng = np.random.default_rng(a.seed)
    ag = Agent(C['od'], E.ACT_DIM, seed=a.seed); print('params', ag.n_params(), 'train eps', len(tr), 'val eps', len(va), flush=True)
    oa = Adam([*ag.actor.W, *ag.actor.b, ag.logstd], a.lr); oc = Adam([*ag.critic.W, *ag.critic.b], a.lr)
    best = -1e9; log = []
    for it in range(1, a.iters + 1):
        t0 = time.time(); eps = [tr[i] for i in rng.integers(0, len(tr), a.batch)]
        if C['aug']: eps = T.corrupt_mixed(eps, rng)
        eps = V.attach(eps, rng, *V.SCEN['mixed'])
        delay = int(rng.choice([0, 5, 20, 50])); env = E.CellEnv(eps, delay_ticks=delay, seed=int(rng.integers(1 << 30)), **env_kw(a.mode))
        obs = env.reset(); O, A, LP, Vv, R = [], [], [], [], []
        while True:
            act, lp, v = ag.act(obs); O.append(obs); A.append(act); LP.append(lp); Vv.append(v)
            obs, r, done, info = env.step(act, view_default=C['view_default']); R.append(r * 0.2)
            if done: break
        O = np.array(O); A = np.array(A); LP = np.array(LP); Vv = np.array(Vv); R = np.array(R)
        adv, ret = gae(R, Vv, True)
        ag.ret_scale = 0.9 * ag.ret_scale + 0.1 * max(ret.std(), 1e-3) if it > 1 else max(ret.std(), 1e-3)
        ag.norm.update(O.reshape(-1, O.shape[-1])); advn = (adv - adv.mean()) / (adv.std() + 1e-8)
        st = ppo_update(ag, oa, oc, O.reshape(-1, O.shape[-1]), A.reshape(-1, A.shape[-1]), LP.reshape(-1), advn.reshape(-1), ret.reshape(-1), rng)
        msg = f'it {it:3d} delay {delay:2d} train reward/tick {R.mean()/0.2:+.3f} std {np.exp(ag.logstd).mean():.2f} dr {st[0]:.3f} clipfrac {st[1]:.2f} [{time.time()-t0:.1f}s]'
        if it % a.eval_every == 0 or it == a.iters:
            rv = np.mean([V.run_ev(va, controller(ag), delay=d, seed=123, view_default=C['view_default'], **env_kw(a.mode))['reward'] for d in (0.0, 2.0, 4.4)])
            msg += f' | VAL reward {rv:+.3f}'
            if rv > best: best = rv; ag.save(f'{a.out}/{tag}_best.npz'); msg += ' *'
            log.append(dict(it=it, val_reward=float(rv)))
        print(msg, flush=True)
    ag.save(f'{a.out}/{tag}_last.npz'); json.dump(log, open(f'{a.out}/{tag}_log.json', 'w'))
