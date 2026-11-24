"""Token-diet v2 for Orient and Decide: SHORT PROMPTS (only the fields the teachers read) and DIGIT-CODE OUTPUTS (each 3-digit chunk is one Llama-3 token).

Why (measured on your traces, fit_latency.py): a hop costs ~0.19 s + 0.00045 s per prompt token + 0.073 s per generated token on HF 4-bit + LoRA.
v1 (compact.py) generates 20-27 tokens from ~700-850 prompt tokens. v2 generates 2-4 content tokens from ~100-150 prompt tokens.

Orient  in : n=<vehicles> load=<level> qos=<severity> standstill=<0|1> event=<0|1> mode=<operator mode> oem=<0|1> reg=<regulatory context>
        out: AAABBB   AAA = codebook index of (tag, QoS weights, constraints); BBB = index of (active intents, conflict, suppressed intents, extras flags)
Decide in : tag=<tag> w=<wu,we,wm> c=<urllc_min,embb_max,max_latency> cap=<dBm|-> pre=<0|1> n=<vehicles> ev=<in_scope,n_within_300m|->
        out: UUUEEEPPPSMM  U,E = URLLC and eMBB PRB share in % (mMTC = 100-U-E); PPP = three power-level indices (0..4 = 10,14,17,20,23 dBm) for URLLC, eMBB, mMTC;
             S = interference strategy index; MM = index of the MEC compute-share triple.
Both are lossless on the dataset (verified in build_compact2.py: decode(encode(x)) expands to exactly the original gold JSON) EXCEPT values that never occur in
training/validation (codebook misses; counted as errors in eval). Codebooks are built from train+val only and shipped as codebook2.json.
The expanders reuse compact.py (v1) so everything downstream is unchanged. The model never sees the retrieved RAG snippets, the narrative or unused telemetry;
rag_refs, rationale, confidence, valid_for_ms, preemption ids etc. are produced by code from the full inputs, as in v1."""
import json, os
import compact as K

HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = [10.0, 14.0, 17.0, 20.0, 23.0]
ORIENT_SYSTEM2 = "ORIENT agent (C-V2X radio resource manager). Output only the 6-digit class code for the scenario, QoS weights, constraints and intent resolution."
DECIDE_SYSTEM2 = "DECIDE agent (C-V2X radio resource manager). Output only the 12-digit policy code: URLLC share, eMBB share, power levels, strategy, MEC shares."

def load_codebook(path=None):
    return json.load(open(path or os.environ.get('OODA_CODEBOOK') or os.path.join(HERE, 'codebook2.json')))

# ----------------------------------------------------------------------------- inputs
def orient_input2(inp):
    o = inp['observe_summary']; r = inp['intent_registry']
    return (f"n={o['n_active_vehicles']} load={o['load_level']} qos={o['qos_status']['severity']} standstill={int('standstill_cluster' in o['anomalies'])} "
            f"event={int(bool(o['special_events']))} mode={r['operator_mode']} oem={int(bool(r['oem_contract']))} reg={r['regulatory_context']}")

def decide_input2(ctx, obs):
    c = ctx['constraint_overrides']; w = ctx['qos_weights']; ev = obs['special_events']
    e = f"{int(bool(ev[0]['in_scope']))},{ev[0]['n_vehicles_within_300m']}" if ev else '-'
    cap = K._num(c['tx_power_cap_dbm']) if 'tx_power_cap_dbm' in c else '-'
    return (f"tag={ctx['scenario_tag']} w={K._num(w['urllc'])},{K._num(w['embb'])},{K._num(w['mmtc'])} "
            f"c={c['urllc_min_prb_share_pct']},{c['embb_max_prb_share_pct']},{c['max_policy_latency_ms']} cap={cap} pre={int('preempt_vehicle_ids' in c)} n={obs['n_active_vehicles']} ev={e}")

# ----------------------------------------------------------------------------- Orient codes
def _orient_keys(gold):
    f = K.encode_orient(gold).split('|'); extras = []
    for x in f[6:]: extras.append('pre' if x.startswith('pre:') else x)
    return '|'.join(f[:3]), '|'.join(f[3:6] + extras)

def encode_orient2(gold, cb):
    a, b = _orient_keys(gold); return f"{cb['orient_a'].index(a):03d}{cb['orient_b'].index(b):03d}"      # ValueError if not in codebook

def decode_orient2(code, inp, cb):
    code = code.strip(); a = cb['orient_a'][int(code[:3])]; b = cb['orient_b'][int(code[3:6])].split('|')
    f = b[:3]; vid = inp['observe_summary']['special_events'][0]['vehicle_id'] if inp['observe_summary']['special_events'] else None
    ex = [('pre:' + vid if x == 'pre' else x) for x in b[3:]]
    return K.decode_orient('|'.join([a] + f + ex), inp)

# ----------------------------------------------------------------------------- Decide codes
def encode_decide2(p, cb):
    pr, pw, m = p['prb_priority'], p['power_budget_dbm'], p['mec_offloading']['compute_share_pct']
    assert pr['urllc_pct'] + pr['embb_pct'] + pr['mmtc_pct'] == 100
    pi = ''.join(str(LEVELS.index(float(pw[k]))) for k in K.CLS)
    st = K.STRAT[p['interference_coordination']['strategy']]; ms = cb['mec'].index([m[k] for k in K.CLS])
    return f"{pr['urllc_pct']:03d}{pr['embb_pct']:03d}{pi}{cb['strat'].index(st)}{ms:02d}"

def decode_decide2(code, ctx, cb, policy_id='P-00000000'):
    c = code.strip(); u, e = int(c[:3]), int(c[3:6]); pw = [LEVELS[int(x)] for x in c[6:9]]; st = cb['strat'][int(c[9])]; mec = cb['mec'][int(c[10:12])]
    v1 = f"{u},{e},{100-u-e}|{','.join(K._num(x) for x in pw)}|{st}|{','.join(str(x) for x in mec)}"
    return K.decode_decide(v1, ctx, policy_id)

# ============================================================================= Observe and Act (token diet v2, added for the four-LLM chain)
# Observe: the telemetry window is ~790 chars/~330 tokens in, an ObserveSummary of ~800 chars out. Everything in the summary except five categorical decisions is a COPY of
#   the input (or a text template). The LLM reads the fields the rules read and emits 9 digits = LOAD SEV TQOS TLOAD + 5 anomaly bits; code fills in the rest (observe_rules.py).
# Act:     translation of the policy into A1/E2 is a deterministic template (build_act.translate). The LLM does the JUDGEMENT: reject the policy? spoofed feedback? which escalation?
#   It emits 2 digits = REASON REJECTED; code builds the full command (target/applied/trusted are functions of REASON; a1/e2/watch come from the policy itself).
import observe_rules as OR
import build_act as BA
OBSERVE_SYSTEM2 = "OBSERVE agent (C-V2X radio resource manager). Output only the 9-digit class code: load level, QoS severity, QoS trend, load trend, five anomaly flags."
ACT_SYSTEM2 = "ACT agent (C-V2X radio resource manager). Output only the 2-digit code: escalation reason, policy-rejected flag."
LOADS = ['low', 'moderate', 'high', 'very_high']; SEVS = ['nominal', 'elevated', 'degraded', 'critical']
TQ = ['unknown', 'worsening', 'improving', 'stable']; TLD = ['unknown', 'ramping_up', 'draining', 'steady']
ANOMS = ['loss_spike', 'high_delay_tail', 'standstill_cluster', 'high_jitter', 'throughput_drop']
REASONS = ['none', 'malformed_policy', 'constraint_violation', 'spoofed_feedback', 'policy_execution_failure', 'qos_violation_persists', 'qos_violation_onset', 'scenario_change']

def observe_input2(inp):
    prev = inp['prev_windows_mean']; n = inp['n_active_vehicles']
    if prev is None: dl = dv = '-'
    else: dl = f"{inp['loss_pct']['mean'] - prev['loss_pct']:.2f}"; dv = f"{100.0 * (n - prev['n_vehicles']) / max(prev['n_vehicles'], 1):.1f}"
    return (f"n={n} dens={inp['density_300m']['mean']} fl10={inp['frac_loss_gt_10pct']} fd={inp['frac_delay_gt_thr']} fd100={inp['frac_delay_gt_100ms']} g50={inp['n_loss_gt_50pct']} "
            f"stat={inp['frac_speed_lt_0.5']} wait={inp['waiting_s_mean']} jit={inp['jitter_ms_mean']} dr={inp['delivery_ratio']} dl={dl} dv={dv}")

def encode_observe2(g):
    an = ''.join('1' if x in g['anomalies'] else '0' for x in ANOMS)
    return f"{LOADS.index(g['load_level'])}{SEVS.index(g['qos_status']['severity'])}{TQ.index(g['trend']['qos'])}{TLD.index(g['trend']['load'])}{an}"

def decode_observe2(code, inp):
    c = code.strip()
    ov = (LOADS[int(c[0])], SEVS[int(c[1])], TQ[int(c[2])], TLD[int(c[3])], [x for x, b in zip(ANOMS, c[4:9]) if b == '1'])
    return OR.observe_rules(inp, ov=ov)

_ORD = {'urllc': 'U', 'embb': 'E', 'mmtc': 'M'}
def _p(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d: return '-'
        d = d[k]
    return d
def act_input2(inp):
    pol, cons, fb = inp['policy'], inp['constraints'], inp['feedback']
    prb = _p(pol, 'prb_priority'); pw = _p(pol, 'power_budget_dbm'); mec = _p(pol, 'mec_offloading'); ic = _p(pol, 'interference_coordination'); pre = _p(pol, 'preemption')
    j = lambda d, ks: ','.join(str(_p(d, k)) if isinstance(d, dict) else '-' for k in ks)
    order = ''.join(_ORD.get(x, '?') for x in prb['priority_order']) if isinstance(prb, dict) and isinstance(prb.get('priority_order'), list) else '-'
    cs = _p(mec, 'compute_share_pct')
    p = (f"id={_p(pol, 'policy_id')} vf={_p(pol, 'valid_for_ms')} prb={j(prb, ['urllc_pct', 'embb_pct', 'mmtc_pct'])} ord={order} pw={j(pw, ['urllc', 'embb', 'mmtc'])} "
         f"mec={j(mec, ['mode', 'offload_ratio', 'priority_class'])},{j(cs, ['urllc', 'embb', 'mmtc'])} ic={j(ic, ['strategy', 'reuse_factor'])} "
         f"pre={int(bool(pre['enabled'])) if isinstance(pre, dict) and 'enabled' in pre else '-'}:{';'.join(pre['vehicle_ids']) if isinstance(pre, dict) and 'vehicle_ids' in pre else '-'}")
    c = (f"cons umin={cons['urllc_min_prb_share_pct']} emax={cons['embb_max_prb_share_pct']} cap={cons.get('tx_power_cap_dbm', '-')} mx={cons['max_vehicle_power_dbm']} "
         f"pre={';'.join(cons.get('preempt_vehicle_ids', [])) or '-'}")
    b = f"base={inp['baseline_qos']['severity']},{inp['baseline_qos']['mean_loss_pct']} start={inp['expected_window_start_s']}"
    if fb is None: f = 'fb=none'
    else:
        o = fb['observed']; e = fb['enforcer']
        f = (f"fb id={fb['policy_id']} auth={int(bool(fb['auth_ok']))} win={fb['window_s'][0]} app={int(bool(e['applied']))} viol={len(e['violations'])} util={e['prb_util_pct']} "
             f"sev={o['severity']} loss={o['mean_loss_pct']} dly={o['mean_delay_ms']} f10={o['frac_loss_gt_10pct']} emg={int(bool(fb['emergency_event_observed']))}")
    return f"{p} | {c} | {b} | {f}"

def encode_act2(g): return f"{REASONS.index(g['escalation']['reason'])}{int(g['a1_policy'] is None)}"

def decode_act2(code, inp):
    c = code.strip(); reason = REASONS[int(c[0])]; rej = c[1] == '1'; pol = inp['policy']
    esc = lambda r, t: {'escalate': r != 'none', 'reason': r, 'target': t}
    if not rej:
        try: a1, e2 = BA.translate(pol, inp['cell_id']); wm = BA.watch(pol)
        except Exception: rej = True; reason = 'malformed_policy'          # the model said "ok" but the policy cannot even be translated: reject it
    if rej:
        return {'a1_policy': None, 'e2_control': [], 'escalation': esc(reason, 'decide'),
                'feedback_to_observe': {'policy_applied': False, 'feedback_trusted': True, 'watch_metrics': ['packet_loss', 'delay']}}
    tgt = 'none' if reason == 'none' else 'orient' if reason == 'scenario_change' else 'decide'
    return {'a1_policy': a1, 'e2_control': e2, 'escalation': esc(reason, tgt),
            'feedback_to_observe': {'policy_applied': reason != 'policy_execution_failure', 'feedback_trusted': reason != 'spoofed_feedback', 'watch_metrics': wm}}

