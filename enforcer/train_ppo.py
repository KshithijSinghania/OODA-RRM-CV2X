"""Step 3: train the DRL enforcer. mode 'b2' = OODA policy + DRL (policy-conditioned, trained under random policy delay);
mode 'b3' = DRL alone (sees only the generic default policy).  Usage: python train_ppo.py --mode b2 --iters 150"""
import argparse, time, json, os, numpy as np
import cell_env as E, traces as T, baselines as Bl
from ppo_numpy import Agent, Adam, gae, ppo_update

def drl_controller(agent):
    # evaluation controller; pass soft=True to Bl.run for 'b2s' models
    def f(env, obs):
        mu, _, _ = agent.act(obs, deterministic=True); return dict(a=mu)
    return f

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', default='b2'); ap.add_argument('--iters', type=int, default=150)
    ap.add_argument('--batch', type=int, default=128); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--source', default='cplex'); ap.add_argument('--out', default='runs'); ap.add_argument('--eval_every', type=int, default=10)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); tag = f'{a.mode}_{a.source}_s{a.seed}'
    vd = a.mode == 'b3'; soft = a.mode == 'b2s'; aug = a.mode == 'b2s'
    tr = T.build('train', a.source); va = T.build('val', a.source); rng = np.random.default_rng(a.seed)
    ag = Agent(E.OBS_DIM, E.ACT_DIM, seed=a.seed); print('params', ag.n_params(), 'train eps', len(tr), 'val eps', len(va), flush=True)
    oa = Adam([*ag.actor.W, *ag.actor.b, ag.logstd], a.lr); oc = Adam([*ag.critic.W, *ag.critic.b], a.lr)
    ref = Bl.run(va, Bl.ctl_policy, view_default=vd, seed=123) if not vd else Bl.run(va, Bl.ctl_static, view_default=True, seed=123)
    print('val ref (B1 policy-follow)' if not vd else 'val ref (B0 static)', f"reward {ref['reward'].mean():+.3f}", flush=True)
    best = -1e9; log = []
    for it in range(1, a.iters + 1):
        t0 = time.time(); eps = [tr[i] for i in rng.integers(0, len(tr), a.batch)]
        if aug: eps = T.corrupt_mixed(eps, rng)
        delay = int(rng.choice([0, 5, 20, 50])); env = E.CellEnv(eps, delay_ticks=delay, seed=int(rng.integers(1 << 30)), soft=soft)
        obs = env.reset(); O, A, LP, V, R = [], [], [], [], []
        while True:
            act, lp, v = ag.act(obs); O.append(obs); A.append(act); LP.append(lp); V.append(v)
            obs, r, done, info = env.step(act, view_default=vd); R.append(r * 0.2)
            if done: break
        O = np.array(O); A = np.array(A); LP = np.array(LP); V = np.array(V); R = np.array(R)
        adv, ret = gae(R, V, True)
        ag.ret_scale = 0.9 * ag.ret_scale + 0.1 * max(ret.std(), 1e-3) if it > 1 else max(ret.std(), 1e-3)
        ag.norm.update(O.reshape(-1, O.shape[-1])); advn = (adv - adv.mean()) / (adv.std() + 1e-8)
        st = ppo_update(ag, oa, oc, O.reshape(-1, O.shape[-1]), A.reshape(-1, A.shape[-1]), LP.reshape(-1), advn.reshape(-1), ret.reshape(-1), rng)
        msg = f'it {it:3d} delay {delay:2d} train reward/tick {R.mean()/0.2:+.3f} std {np.exp(ag.logstd).mean():.2f} dr {st[0]:.3f} clipfrac {st[1]:.2f} [{time.time()-t0:.1f}s]'
        if it % a.eval_every == 0 or it == a.iters:
            rv = Bl.run(va, drl_controller(ag), view_default=vd, seed=123, soft=soft); msg += f" | VAL reward {rv['reward'].mean():+.3f} urllc {rv['sat'][:,0].mean():.3f} embb {rv['sat'][:,1].mean():.3f}"
            if rv['reward'].mean() > best: best = rv['reward'].mean(); ag.save(f'{a.out}/{tag}_best.npz'); msg += ' *'
            log.append(dict(it=it, val_reward=float(rv['reward'].mean())))
        print(msg, flush=True)
    ag.save(f'{a.out}/{tag}_last.npz'); json.dump(log, open(f'{a.out}/{tag}_log.json', 'w'))
