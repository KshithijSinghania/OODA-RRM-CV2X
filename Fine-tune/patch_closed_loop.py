r"""Adds the closed-loop mode to ooda_loop.py (backup: ooda_loop.py.bak_cl).  Usage:  python patch_closed_loop.py ..\OODA\ooda_loop.py"""
import sys, shutil
p = sys.argv[1]; src = open(p, encoding='utf-8').read()
if "--cl_mode" in src: sys.exit('already patched')
shutil.copy(p, p + '.bak_cl')
def sub(old, new):
    global src
    assert src.count(old) == 1, f'anchor not unique / not found ({src.count(old)}x): {old[:70]!r}'
    src = src.replace(old, new)

sub("ap.add_argument('--oracle', action='store_true', help='use gold outputs instead of models (glue self-test, no GPU needed)')",
"""ap.add_argument('--oracle', action='store_true', help='use gold outputs instead of models (glue self-test, no GPU needed)')
ap.add_argument('--cl_mode', default='off', choices=['off', 'dataset', 'enforcer'], help="closed loop with the cell environment + trained DRL enforcer (cl_env.py). dataset = control (dataset feedback), enforcer = measured feedback")
ap.add_argument('--cl_enforcer_dir', default=r'..\\enforcer'); ap.add_argument('--cl_runs', default=None, help='folder with <kind>_cplex_s<seed>_best.npz (default <cl_enforcer_dir>/runs)')
ap.add_argument('--cl_enforcer', default='b2s', choices=['b2s', 'b2', 'b1']); ap.add_argument('--cl_agent_seed', type=int, default=0); ap.add_argument('--cl_seed', type=int, default=4000)
ap.add_argument('--cl_loss_thr', type=float, default=0.10, help='a tick counts as a QoS violation when URLLC unserved > this'); ap.add_argument('--cl_prr_target', type=float, default=1e-3)
ap.add_argument('--cl_viol_frac', type=float, default=0.5, help='window violation flags need this fraction of ticks'); ap.add_argument('--cl_rb_tol', type=float, default=0.10)""")

sub("print(f'{len(chains)} test windows with a complete gold chain', flush=True)",
"""print(f'{len(chains)} test windows with a complete gold chain', flush=True)
CL = None
if a.cl_mode != 'off':
    import cl_env
    CL = cl_env.ClosedLoop(a.cl_enforcer_dir, a.cl_runs or os.path.join(a.cl_enforcer_dir, 'runs'), kind=a.cl_enforcer, agent_seed=a.cl_agent_seed, mode=a.cl_mode,
                           loss_thr=a.cl_loss_thr, prr_target=a.cl_prr_target, viol_frac=a.cl_viol_frac, rb_tol=a.cl_rb_tol, seed=a.cl_seed, ep_out=a.out + '.episodes.jsonl')
    n0 = len(chains); chains = CL.group(chains)
    print(f'closed loop ({a.cl_mode}): {len(CL.eps)} episodes x {CL.W} windows = {len(chains)} windows ({n0 - len(chains)} windows outside a full episode dropped)', flush=True)""")

sub("def act_prompt(gold_act, obs_out, ctx, policy):", "def act_prompt(gold_act, obs_out, ctx, policy, fb_override='keep'):")
sub("fb = g['feedback']", "fb = g['feedback'] if fb_override == 'keep' else fb_override")
sub("gold_pol, gold_act = json.loads(g2['messages'][2]['content']), json.loads(g3['messages'][2]['content'])",
    "gold_pol, gold_act = json.loads(g2['messages'][2]['content']), json.loads(g3['messages'][2]['content'])\n        if CL: CL.begin(go)")
sub("if obs is None or not ok1: tr['failed_at'] = 'observe'; tr['raw'] = raw; fo.write(json.dumps(tr) + '\\n'); continue",
    "if obs is None or not ok1: tr['failed_at'] = 'observe'; tr['raw'] = raw; fo.write(json.dumps(tr) + '\\n'); continue\n        if CL and a.cl_mode == 'enforcer': tr['cl_qos_override'] = CL.override_obs(obs)")
sub("tr['context'] = ctx; tr['policy'] = pol; tr['prb_matches_cplex'] = pol['prb_priority'] == gold_pol['prb_priority']",
    "tr['context'] = ctx; tr['policy'] = pol; tr['prb_matches_cplex'] = pol['prb_priority'] == gold_pol['prb_priority']\n        if CL: tr['cl'] = CL.set_policy(pol, ctx)")
sub("up = act_prompt(g3, obs, ctx, pol)",
    "if CL and a.cl_mode == 'enforcer':\n            fb_cl = CL.feedback(obs); tr['cl_fb'] = fb_cl; up = act_prompt(g3, obs, ctx, pol, fb_cl)\n        else: up = act_prompt(g3, obs, ctx, pol)")
sub("stats['act_exact'] += cmd == gold_act",
    "stats['act_exact'] += cmd == gold_act\n        if CL: _t = BA.act_teacher(ainp)[0]; tr['cl_teacher_escalation'] = _t['escalation']; tr['cl_act_teacher_match'] = cmd == _t")
sub("n = max(1, stats['windows'])", "if CL: CL.finish_all()\nn = max(1, stats['windows'])")
open(p, 'w', encoding='utf-8').write(src); print('patched', p, '(backup', p + '.bak_cl)')
