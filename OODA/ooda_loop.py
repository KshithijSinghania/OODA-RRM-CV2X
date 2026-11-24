#!/usr/bin/env python3
"""End-to-end OODA loop: ONE frozen 4-bit Llama-3.2-3B base + four LoRA adapters (observe, orient, decide, act).

For each held-out (test) RSU-cell window the loop runs
    telemetry --Observe--> summary --Orient--> context --Decide--> policy --Act--> A1/E2 command
where every agent's input is built from the previous agent's *generated* output (nothing is copied from the gold chain
except what comes from the environment: intent registry / retrieved snippets, enforcer feedback, system model).
Glue code (not an LLM) does what it would do in deployment: assigns the policy_id, builds constraints, and builds the
next agent's prompt. Every hop is JSON- and schema-validated.

  python ooda_loop.py --data_dir data --schema_dir schemas --adapters adapters --n 30 --out loop_trace.jsonl
  python ooda_loop.py --data_dir data --schema_dir schemas --oracle        # no GPU: gold outputs in place of the models,
                                                                           # checks the glue reproduces the dataset prompts exactly
"""
import argparse, glob, json, os, sys, time

SCHEMA = {'observe': 'observe_summary', 'orient': 'orient_context', 'decide': 'decide_policy', 'act': 'act_command'}
ap = argparse.ArgumentParser()
ap.add_argument('--data_dir', default='data'); ap.add_argument('--schema_dir', default='schemas')
ap.add_argument('--adapters', default='adapters', help='folder holding <agent>_FINAL (or <agent>) adapter folders')
ap.add_argument('--model', default='meta-llama/Llama-3.2-3B-Instruct')
ap.add_argument('--n', type=int, default=30, help='test windows to run (0 = all)')
ap.add_argument('--max_new', type=int, default=700)
ap.add_argument('--out', default='loop_trace.jsonl')
ap.add_argument('--obs_ids', default=None, help='file with one observe id per line: run exactly these windows, in this order (overrides --n)')
ap.add_argument('--skip_act', action='store_true', help='stop after Decide (saves ~20 s/window); used to collect LLM policies for the DRL enforcer')
ap.add_argument('--upstream', default='llm', choices=['llm', 'gold'], help="'gold': feed gold Observe/Orient outputs and run only the later agents")
ap.add_argument('--act_mode', default='llm', choices=['llm', 'rules'], help="'rules': Act = the deterministic rule function (microseconds, no LLM) - removes ~20 s per window")
ap.add_argument('--prompt_lookup', type=int, default=0, help='prompt-lookup (assisted) decoding with N draft tokens, e.g. 10: same output as greedy decoding, usually 1.5-3x faster on JSON that repeats the prompt')
ap.add_argument('--observe_mode', default='llm', choices=['llm', 'rules'], help="'rules': Observe = deterministic observe_rules.py (telemetry copy + thresholds, microseconds) - removes ~25 s per window")
ap.add_argument('--compact', action='store_true', help="use the compact-format Orient/Decide adapters ('orient_compact', 'decide_compact' under --adapters; ~5x fewer generated tokens); output is expanded to the full JSON by compact.py")
ap.add_argument('--compact2', action='store_true', help="token-diet v2 (compact2.py): ~100-token prompts and 2-4 digit-code output tokens per hop; adapters 'orient_compact2' / 'decide_compact2' (or GGUF servers trained on data_compact2)")
ap.add_argument('--adapter_override', default='', help="use a specific adapter folder name for an agent, e.g. act=act_compact2b (comma separated)")
ap.add_argument('--no_guard', action='store_true', help='disable the deterministic Act guard (act_guard.py) - default: guard ON')
ap.add_argument('--engine', default='hf', choices=['hf', 'llama'], help="'llama': generate through llama-server (merged + quantised GGUF per agent), see README_INFER.md; default: Hugging Face 4-bit + LoRA adapters")
ap.add_argument('--llama_urls', default='', help='agent=url pairs for --engine llama, e.g. orient=http://127.0.0.1:8081,decide=http://127.0.0.1:8082')
ap.add_argument('--oracle', action='store_true', help='use gold outputs instead of models (glue self-test, no GPU needed)')
ap.add_argument('--cl_mode', default='off', choices=['off', 'dataset', 'enforcer', 'ns3'], help="closed loop with the cell environment + trained DRL enforcer (cl_env.py). dataset = control (dataset feedback), enforcer = measured feedback")
ap.add_argument('--ns3_dir', default=None, help='ns-3 --ctrlDir folder as seen from this machine (e.g. C:\\ns3ctrl\\live_s7_s1)')
ap.add_argument('--ns3_tstart', type=float, default=90.0); ap.add_argument('--ns3_t1', type=float, default=200.0)
ap.add_argument('--ns3_txMacro', type=float, default=37.0); ap.add_argument('--ns3_txRsu', type=float, default=30.0); ap.add_argument('--ns3_full_dbm', type=float, default=23.0)
ap.add_argument('--ns3_rule', default=None, help='rsrp_teacher.json: the RSRP rule sets every cell\'s power depth (see ns3_bridge.py)')
ap.add_argument('--ns3_rule_mode', default='protect', choices=['protect', 'rule'], help="protect = rule, but non-baseline cells never cut deeper than the OODA policy; rule = rule only")
ap.add_argument('--ns3_persist', type=int, default=2, help='windows a new rule target must persist before it is applied')
ap.add_argument('--ns3_enforcer', action='store_true', help='DRL enforcer in the ns-3 loop: its power residual vs the OODA policy is added to each cell\'s offset (see ns3_enforcer.py)')
ap.add_argument('--ns3_correct', action='store_true', help='add the feedback correction (hill-climbing on measured loss) on top of the OODA powers')
ap.add_argument('--ns3_timeout', type=float, default=7200.0); ap.add_argument('--ns3_delay_thr', type=float, default=None, help='ms threshold behind frac_delay_gt_thr (default config value or 20)')
ap.add_argument('--cl_enforcer_dir', default=r'..\enforcer'); ap.add_argument('--cl_runs', default=None, help='folder with <kind>_cplex_s<seed>_best.npz (default <cl_enforcer_dir>/runs)')
ap.add_argument('--cl_enforcer', default='b2s', choices=['b2s', 'b2', 'b1']); ap.add_argument('--cl_agent_seed', type=int, default=0); ap.add_argument('--cl_seed', type=int, default=4000)
ap.add_argument('--cl_loss_thr', type=float, default=0.10, help='a tick counts as a QoS violation when URLLC unserved > this'); ap.add_argument('--cl_prr_target', type=float, default=1e-3)
ap.add_argument('--cl_viol_frac', type=float, default=0.5, help='window violation flags need this fraction of ticks'); ap.add_argument('--cl_rb_tol', type=float, default=0.10)
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # act_guard.py / config.py / schemas/ live next to this script
import act_guard as G
import build_act as BA
import observe_rules as OR
import compact as K
import compact2 as C2
V2 = a.compact2; a.compact = a.compact or V2
CB = C2.load_codebook() if V2 else None
V2_LLM = {'observe': a.observe_mode == 'llm', 'act': a.act_mode == 'llm' and not a.skip_act}       # observe/act get v2 codes only when they run as LLMs
def MAXNEW(agent): return (16 if V2 else 96) if (a.compact and agent in ('orient', 'decide')) or (V2 and agent in ('observe', 'act')) else a.max_new
TOK = {'n': 0}

# ---------------------------------------------------------------- data
def rows(agent):
    out = []
    for f in sorted(glob.glob(os.path.join(a.data_dir, f'{agent}_S?.jsonl'))):
        out += [json.loads(l) for l in open(f, encoding='utf-8')]
    return out

obs_rows = [r for r in rows('observe') if r['split'] == 'test' and json.loads(r['messages'][1]['content'])['scope']['type'] == 'cell']
orient_by_obs = {}
for r in rows('orient'):
    if r['id'].endswith('-v0'): orient_by_obs[r['obs_id']] = r                 # one Orient variant (v0) per window
decide_by_orient = {r['id'][:-2]: r for r in rows('decide')}                    # '<orient id>-d'
act_by_decide = {r['decide_id']: r for r in rows('act') if r['kind'] == 'clean'}
chains = []
for o in obs_rows:
    r1 = orient_by_obs.get(o['id'])
    r2 = decide_by_orient.get(r1['id']) if r1 else None
    r3 = act_by_decide.get(r2['id']) if r2 else None
    if r3: chains.append((o, r1, r2, r3))
if a.obs_ids:
    want = [l.strip() for l in open(a.obs_ids) if l.strip()]; by = {c[0]['id']: c for c in chains}
    missing = [w for w in want if w not in by]
    if missing: print(f'WARNING: {len(missing)} requested ids have no complete gold chain (skipped), e.g. {missing[:3]}')
    chains = [by[w] for w in want if w in by]
elif a.n: chains = chains[::max(1, len(chains) // a.n)][:a.n]
print(f'{len(chains)} test windows with a complete gold chain', flush=True)
CL = None
if a.cl_mode != 'off':
    import ns3_bridge
    if a.cl_mode != 'ns3': import cl_env
    CL = ns3_bridge.NS3Loop(a) if a.cl_mode == 'ns3' else cl_env.ClosedLoop(a.cl_enforcer_dir, a.cl_runs or os.path.join(a.cl_enforcer_dir, 'runs'), kind=a.cl_enforcer, agent_seed=a.cl_agent_seed, mode=a.cl_mode,
                           loss_thr=a.cl_loss_thr, prr_target=a.cl_prr_target, viol_frac=a.cl_viol_frac, rb_tol=a.cl_rb_tol, seed=a.cl_seed, ep_out=a.out + '.episodes.jsonl')
    n0 = len(chains); chains = CL.group(chains)
    print(f'closed loop ({a.cl_mode}): {len(CL.eps)} episodes x {CL.W} windows = {len(chains)} windows ({n0 - len(chains)} windows outside a full episode dropped)', flush=True)

def J(x): return json.dumps(x, separators=(',', ':'))

# ---------------------------------------------------------------- glue: build each agent's prompt from the previous output
def orient_prompt(gold_orient, obs_out):
    inp = json.loads(gold_orient['messages'][1]['content'])                      # intent registry + retrieved snippets = environment
    inp['observe_summary'] = obs_out
    return J(inp)

def decide_prompt(gold_decide, obs_out, ctx):
    g = json.loads(gold_decide['messages'][1]['content'])
    state = {k: obs_out[k] for k in ('window_s', 'n_active_vehicles', 'load_level', 'trend', 'special_events', 'anomalies')}
    state['qos'] = {k: obs_out['qos_status'][k] for k in ('severity', 'mean_loss_pct', 'p95_loss_pct', 'mean_delay_ms', 'frac_loss_gt_10pct')}
    state['cell_id'] = obs_out['scope']['cell_id']
    state['v2x_messages'] = obs_out['v2x_messages']
    state['resource_state'] = {k: obs_out['resource_state'][k] for k in ('estimated', 'offered_load_ratio', 'interference_iot_db', 'mec_load_ratio')}
    return J({'context': ctx, 'cell_state': state, 'current_allocation': obs_out['resource_state']['current_allocation'],
              'system_model': g['system_model']})

def act_prompt(gold_act, obs_out, ctx, policy, fb_override='keep'):
    g = json.loads(gold_act['messages'][1]['content']); cons = ctx['constraint_overrides']
    cs_qos = obs_out['qos_status']
    c = {k: cons[k] for k in ('urllc_min_prb_share_pct', 'embb_max_prb_share_pct')}
    if 'tx_power_cap_dbm' in cons: c['tx_power_cap_dbm'] = cons['tx_power_cap_dbm']
    c['preempt_vehicle_ids'] = cons.get('preempt_vehicle_ids', [])
    c['max_vehicle_power_dbm'] = g['constraints']['max_vehicle_power_dbm']
    fb = g['feedback'] if fb_override == 'keep' else fb_override                                                           # enforcer feedback = environment
    if fb is not None: fb = {**fb, 'policy_id': policy['policy_id']}
    return J({'cell_id': obs_out['scope']['cell_id'], 'policy': policy, 'constraints': c,
              'baseline_qos': {'severity': cs_qos['severity'], 'mean_loss_pct': cs_qos['mean_loss_pct']},
              'expected_window_start_s': float(obs_out['window_s'][1]), 'feedback': fb})

def feasible(policy, ctx):
    """Plain-python feasibility of a Decide policy against the Orient constraints (no LLM)."""
    p = policy['prb_priority']; cons = ctx['constraint_overrides']; g = policy['mec_offloading']['compute_share_pct']
    return (p['urllc_pct'] + p['embb_pct'] + p['mmtc_pct'] == 100 and sum(g.values()) == 100
            and p['urllc_pct'] >= cons['urllc_min_prb_share_pct'] and p['embb_pct'] <= cons['embb_max_prb_share_pct'])

# ---------------------------------------------------------------- models
if a.oracle:
    def gen(agent, messages, gold):
        TOK['n'] = 0
        if V2 and agent in ('orient', 'decide'): return (C2.encode_orient2(gold, CB) if agent == 'orient' else C2.encode_decide2(gold, CB)), 0.0
        if V2 and agent in ('observe', 'act'): return (C2.encode_observe2(gold) if agent == 'observe' else C2.encode_act2(gold)), 0.0
        if a.compact and agent in ('orient', 'decide'): return (K.encode_orient(gold) if agent == 'orient' else K.encode_decide(gold)), 0.0   # exercises the decoder offline
        return gold, 0.0
elif a.engine == 'llama':
    from transformers import AutoTokenizer
    import llama_engine as LE
    tok = AutoTokenizer.from_pretrained(a.model); eng = LE.LlamaEngine(LE.parse_urls(a.llama_urls), tok)
    need = [ag for ag in SCHEMA if not (ag == 'observe' and a.observe_mode == 'rules') and not (ag == 'act' and (a.act_mode == 'rules' or a.skip_act)) and not (a.upstream == 'gold' and ag in ('observe', 'orient') and False)]
    miss = [ag for ag in need if ag not in eng.urls or not eng.health(ag)]
    if miss: sys.exit(f'llama-server not reachable for: {miss}. Start one server per agent and pass --llama_urls (README_INFER.md, step 5).')
    def gen(agent, messages, gold):
        txt, dt, n, p = eng.gen(agent, messages, max_new=MAXNEW(agent)); TOK['n'] = n; TOK['prompt'] = p
        return txt, dt
else:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.pad_token or '<|reserved_special_token_0|>'
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    import transformers
    _v = tuple(int(x) for x in transformers.__version__.split('.')[:2])
    base = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, device_map={'': 0},
                                                **{'dtype' if _v >= (4, 56) else 'torch_dtype': torch.bfloat16})
    model = None
    for ag in SCHEMA:
        ov = dict(x.split('=') for x in a.adapter_override.split(',') if x)
        names = ([ov[ag]] if ag in ov else []) + ([f'{ag}_compact2', f'{ag}_compact2_FINAL'] if V2 and (ag in ('orient', 'decide') or V2_LLM[ag]) else []) + ([f'{ag}_compact', f'{ag}_compact_FINAL'] if a.compact and not V2 and ag in ('orient', 'decide') else []) + [f'{ag}_FINAL', ag]
        d = next((os.path.join(a.adapters, n) for n in names if os.path.isdir(os.path.join(a.adapters, n))), None)
        if d is None: sys.exit(f'no adapter folder for {ag} under {a.adapters}')
        if V2 and ag in V2_LLM and V2_LLM[ag] and 'compact2' not in d: sys.exit(f'--compact2 with {ag} as an LLM needs the adapter {ag}_compact2 (train it, README_INFER.md step 1b); found only {d}')
        print(f'adapter {ag}: {d}', flush=True)
        if model is None: model = PeftModel.from_pretrained(base, d, adapter_name=ag)
        else: model.load_adapter(d, adapter_name=ag)          # returns a load report in newer PEFT versions: do not reassign model
    model.eval()
    stop = [tok.eos_token_id, tok.convert_tokens_to_ids('<|eot_id|>')]

    def gen(agent, messages, gold):
        model.set_adapter(agent)
        enc = tok(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), return_tensors='pt', add_special_tokens=False).to('cuda')
        t = time.time()
        with torch.no_grad():
            kw = {'prompt_lookup_num_tokens': a.prompt_lookup} if a.prompt_lookup else {}
            out = model.generate(**enc, max_new_tokens=MAXNEW(agent), do_sample=False, eos_token_id=stop, pad_token_id=tok.pad_token_id, **kw)
        TOK['n'] = int(out.shape[1] - enc['input_ids'].shape[1]); TOK['prompt'] = int(enc['input_ids'].shape[1])
        txt = tok.decode(out[0][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        torch.cuda.synchronize(); return txt, time.time() - t

from jsonschema import validate, ValidationError
schemas = {ag: json.load(open(os.path.join(a.schema_dir, s + '.schema.json'))) for ag, s in SCHEMA.items()}

def hop(agent, system, user, gold_out_text, decoder=None):
    """run one agent; returns (parsed dict or None, schema_ok, seconds, raw text on failure). decoder: compact text -> full dict (compact mode)"""
    gold = gold_out_text if not a.oracle else json.loads(gold_out_text)
    if decoder is not None and not a.oracle: system = ({'observe': C2.OBSERVE_SYSTEM2, 'orient': C2.ORIENT_SYSTEM2, 'decide': C2.DECIDE_SYSTEM2, 'act': C2.ACT_SYSTEM2} if V2 else {'orient': K.ORIENT_COMPACT_SYSTEM, 'decide': K.DECIDE_COMPACT_SYSTEM})[agent]
    res, dt = gen(agent, [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], gold)
    try: parsed = decoder(res) if decoder is not None else (res if a.oracle else json.loads(res))
    except Exception as e: print('DECODE_FAIL', agent, repr(e), repr(res)); return None, False, dt, res
    try: validate(parsed, schemas[agent]); ok = True
    except ValidationError: ok = False
    return parsed, ok, dt, None

# ---------------------------------------------------------------- run
stats = {k: 0 for k in ('windows', 'observe_ok', 'orient_ok', 'decide_ok', 'act_ok', 'chain_ok', 'decide_feasible', 'act_no_escalation',
                        'scenario_match', 'load_match', 'act_decision_match', 'act_exact', 'prb_exact', 'prompt_mismatch_decide', 'prompt_mismatch_act', 'guard_fires', 'guard_changed', 'prompt_mismatch_orient')}
lat = {ag: 0.0 for ag in SCHEMA}
with open(a.out, 'w', encoding='utf-8') as fo:
    for i, (go, g1, g2, g3) in enumerate(chains):
        stats['windows'] += 1; tr = {'obs_id': go['id'], 'situation': go['situation']}
        gold_obs, gold_ctx = json.loads(go['messages'][2]['content']), json.loads(g1['messages'][2]['content'])
        gold_pol, gold_act = json.loads(g2['messages'][2]['content']), json.loads(g3['messages'][2]['content'])
        if CL: CL.begin(go)
        # 1 OBSERVE
        if a.upstream == 'gold': obs, ok1, dt, raw = gold_obs, True, 0.0, None
        elif a.observe_mode == 'rules':
            t0 = time.time(); obs = OR.observe_rules(json.loads(go['messages'][1]['content'])); dt = time.time() - t0; raw = None
            try: validate(obs, schemas['observe']); ok1 = True
            except ValidationError: ok1 = False
        elif V2:
            oinp = json.loads(go['messages'][1]['content'])
            obs, ok1, dt, raw = hop('observe', go['messages'][0]['content'], C2.observe_input2(oinp), go['messages'][2]['content'], decoder=lambda t, oinp=oinp: C2.decode_observe2(t, oinp))
        else: obs, ok1, dt, raw = hop('observe', go['messages'][0]['content'], go['messages'][1]['content'], go['messages'][2]['content'])
        lat['observe'] += dt; tr.setdefault('sec', {})['observe'] = dt; tr.setdefault('tok', {})['observe'] = [TOK.get('prompt', 0), TOK['n']]; stats['observe_ok'] += ok1; tr['observe_ok'] = ok1
        if obs is None or not ok1: tr['failed_at'] = 'observe'; tr['raw'] = raw; fo.write(json.dumps(tr) + '\n'); continue
        if CL and a.cl_mode == 'enforcer': tr['cl_qos_override'] = CL.override_obs(obs)
        # 2 ORIENT
        up = orient_prompt(g1, obs)
        if a.oracle and up != g1['messages'][1]['content']: stats['prompt_mismatch_orient'] += 1
        if a.upstream == 'gold': ctx, ok2, dt, raw = gold_ctx, True, 0.0, None
        else: ctx, ok2, dt, raw = hop('orient', g1['messages'][0]['content'], C2.orient_input2(json.loads(up)) if V2 else up, g1['messages'][2]['content'],
                                     decoder=((lambda t, up=up: C2.decode_orient2(t, json.loads(up), CB)) if V2 else (lambda t, up=up: K.decode_orient(t, json.loads(up)))) if a.compact else None)
        lat['orient'] += dt; tr['sec']['orient'] = dt; tr.setdefault('tok', {})['orient'] = [TOK.get('prompt', 0), TOK['n']]; stats['orient_ok'] += ok2; tr['orient_ok'] = ok2
        if ctx is None or not ok2: tr['failed_at'] = 'orient'; tr['raw'] = raw; fo.write(json.dumps(tr) + '\n'); continue
        stats['scenario_match'] += ctx['scenario_tag'] == gold_ctx['scenario_tag']; stats['load_match'] += obs['load_level'] == gold_obs['load_level']
        # 3 DECIDE
        up = decide_prompt(g2, obs, ctx)
        if a.oracle and up != g2['messages'][1]['content']: stats['prompt_mismatch_decide'] += 1
        pol, ok3, dt, raw = hop('decide', g2['messages'][0]['content'], C2.decide_input2(ctx, obs) if V2 else up, g2['messages'][2]['content'],
                                decoder=((lambda t, ctx=ctx: C2.decode_decide2(t, ctx, CB, gold_pol['policy_id'])) if V2 else (lambda t, ctx=ctx: K.decode_decide(t, ctx, gold_pol['policy_id']))) if a.compact else None)
        lat['decide'] += dt; tr['sec']['decide'] = dt; tr.setdefault('tok', {})['decide'] = [TOK.get('prompt', 0), TOK['n']]; stats['decide_ok'] += ok3; tr['decide_ok'] = ok3
        if pol is None or not ok3: tr['failed_at'] = 'decide'; tr['raw'] = raw; fo.write(json.dumps(tr) + '\n'); continue
        pol['policy_id'] = gold_pol['policy_id']             # glue assigns the id (the LLM cannot know a random hash)
        feas = feasible(pol, ctx); stats['decide_feasible'] += feas; tr['decide_feasible'] = feas
        stats['prb_exact'] += pol['prb_priority'] == gold_pol['prb_priority']
        tr['context'] = ctx; tr['policy'] = pol; tr['prb_matches_cplex'] = pol['prb_priority'] == gold_pol['prb_priority']
        if CL: tr['cl'] = CL.set_policy(pol, ctx)
        if a.skip_act:
            tr['chain_ok'] = bool(ok1 and ok2 and ok3 and feas); stats['chain_ok'] += tr['chain_ok']; fo.write(json.dumps(tr) + '\n'); fo.flush()
            if (i + 1) % 10 == 0: print(f'  window {i + 1}/{len(chains)}', flush=True)
            continue
        # 4 ACT
        if CL and a.cl_mode in ('enforcer', 'ns3'):
            fb_cl = CL.feedback(obs); tr['cl_fb'] = fb_cl; up = act_prompt(g3, obs, ctx, pol, fb_cl)
        else: up = act_prompt(g3, obs, ctx, pol)
        if a.oracle and up != g3['messages'][1]['content']: stats['prompt_mismatch_act'] += 1
        if a.act_mode == 'rules':
            t0 = time.time(); cmd = BA.act_teacher(json.loads(up))[0]; ok4 = True; dt = time.time() - t0; raw = None
        elif V2:
            ainp0 = json.loads(up)
            cmd, ok4, dt, raw = hop('act', g3['messages'][0]['content'], C2.act_input2(ainp0), g3['messages'][2]['content'], decoder=lambda t, ainp0=ainp0: C2.decode_act2(t, ainp0))
        else: cmd, ok4, dt, raw = hop('act', g3['messages'][0]['content'], up, g3['messages'][2]['content'])
        lat['act'] += dt; tr['sec']['act'] = dt; tr.setdefault('tok', {})['act'] = [TOK.get('prompt', 0), TOK['n']] if a.act_mode == 'llm' else [0, 0]; stats['act_ok'] += ok4; tr['act_ok'] = ok4
        if cmd is None or not ok4: tr['failed_at'] = 'act'; tr['raw'] = raw; fo.write(json.dumps(tr) + '\n'); continue
        ainp = json.loads(up); gv = G.verdict(ainp)                                         # deterministic hard-case check
        if gv[0] is not None: stats['guard_fires'] += 1; tr['guard'] = {'reason': gv[0], 'target': gv[1]}
        if not a.no_guard:
            fixed = G.apply(cmd, ainp); stats['guard_changed'] += fixed != cmd; cmd = fixed
        no_esc = not cmd['escalation']['escalate']; stats['act_no_escalation'] += no_esc
        dec_match = cmd['escalation']['escalate'] == gold_act['escalation']['escalate']      # gold Act itself escalates when QoS persists
        stats['act_decision_match'] += dec_match
        chain_ok = ok1 and ok2 and ok3 and ok4 and feas and dec_match and (bool(cmd['e2_control']) == bool(gold_act['e2_control']))
        stats['chain_ok'] += chain_ok
        tr.update({'scenario_tag': ctx['scenario_tag'], 'prb': pol['prb_priority'], 'escalation': cmd['escalation'], 'chain_ok': chain_ok,
                   'act_matches_gold': cmd == gold_act, 'cmd': cmd})
        stats['act_exact'] += cmd == gold_act
        if CL: _t = BA.act_teacher(ainp)[0]; tr['cl_teacher_escalation'] = _t['escalation']; tr['cl_act_teacher_match'] = cmd == _t
        fo.write(json.dumps(tr) + '\n'); fo.flush()
        if (i + 1) % 10 == 0: print(f'  window {i + 1}/{len(chains)}  chain_ok={stats["chain_ok"]}', flush=True)

if CL: CL.finish_all()
n = max(1, stats['windows'])
print(f"\n=== end-to-end ({'ORACLE glue test' if a.oracle else 'models'}) on {stats['windows']} test windows ===")
for k in ('observe_ok', 'orient_ok', 'decide_ok', 'act_ok'): print(f'  {k.replace("_ok", ""):<9} schema-valid     {stats[k] / n:.1%}')
print(f'  orient scenario_tag matches gold   {stats["scenario_match"] / n:.1%}')
print(f'  observe load_level matches gold    {stats["load_match"] / n:.1%}')
print(f'  decide policy feasible (rules)     {stats["decide_feasible"] / n:.1%}')
print(f'  decide PRB split == CPLEX          {stats["prb_exact"] / n:.1%}')
print(f'  act escalate/accept decision == gold {stats["act_decision_match"] / n:.1%}   (act escalates in {1 - stats["act_no_escalation"] / n:.1%} of windows)')
print(f'  act guard ({"OFF" if a.no_guard else "ON"}): fired on {stats["guard_fires"]} windows (a Decide policy that is malformed/infeasible or feedback that is spoofed), changed the Act output on {stats["guard_changed"]}')
print(f'  act output identical to gold Act     {stats["act_exact"] / n:.1%}')
print(f'  FULL CHAIN OK (all hops valid, policy feasible, Act decision == gold)  {stats["chain_ok"] / n:.1%}')
if not a.oracle: print(f'  settings: engine={a.engine} compact={a.compact} compact2={V2} observe_mode={a.observe_mode} act_mode={a.act_mode} prompt_lookup={a.prompt_lookup} upstream={a.upstream}')
if not a.oracle: print('  mean seconds/hop: ' + '  '.join(f'{k} {v / n:.1f}' for k, v in lat.items()) + f'   (total {sum(lat.values()) / n:.1f} s per cell-window)')
if a.oracle: print(f'  orient prompts differing from the dataset prompt (observe-rules deviations): {stats["prompt_mismatch_orient"]}')
if a.oracle: print(f'  glue prompt mismatches: decide {stats["prompt_mismatch_decide"]}, act {stats["prompt_mismatch_act"]}  (must be 0)')
print('trace written to', a.out)

