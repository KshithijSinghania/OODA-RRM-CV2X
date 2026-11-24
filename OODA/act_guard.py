"""Deterministic guard placed IN FRONT OF / ALONGSIDE the Act agent (no LLM, microseconds).
It re-implements the Act safety rules from the Act system prompt:   reject malformed policy > constraint violation > spoofed feedback.
The LLM Act agent still does the O-RAN translation and the soft decisions (QoS escalation); the guard has the final say on the three
HARD cases, so a mistake by the 3B model can never let a malformed / infeasible policy or spoofed feedback through.
Usage:  import act_guard as G;  v = G.verdict(act_input_dict);  out = G.apply(model_output_or_None, act_input_dict)"""
import json, os
import jsonschema
import config as C
HERE = os.path.dirname(os.path.abspath(__file__))
_DSL = jsonschema.Draft202012Validator(json.load(open(os.path.join(HERE, 'schemas', 'decide_policy.schema.json'), encoding='utf-8')))

def policy_problem(pol, cons):
    """None if the policy is well-formed and respects the constraints, else 'malformed_policy' / 'constraint_violation'."""
    if list(_DSL.iter_errors(pol)): return 'malformed_policy'
    p = pol['prb_priority']
    if p['urllc_pct'] + p['embb_pct'] + p['mmtc_pct'] != 100 or sum(pol['mec_offloading']['compute_share_pct'].values()) != 100: return 'malformed_policy'
    cap = min(cons['max_vehicle_power_dbm'], cons.get('tx_power_cap_dbm', 1e9))
    if (p['urllc_pct'] < cons['urllc_min_prb_share_pct'] or p['embb_pct'] > cons['embb_max_prb_share_pct'] or max(pol['power_budget_dbm'].values()) > cap
            or pol['preemption']['vehicle_ids'] != cons['preempt_vehicle_ids'] or pol['preemption']['enabled'] != bool(cons['preempt_vehicle_ids'])):
        return 'constraint_violation'
    return None

def feedback_spoofed(fb, pol, expected_window_start_s):
    if fb is None: return False
    o, e = fb['observed'], fb['enforcer']
    return bool(fb['policy_id'] != pol['policy_id'] or not fb['auth_ok'] or fb['window_s'][0] != expected_window_start_s
                or not 0 <= o['mean_loss_pct'] <= 100 or not 0 <= o['frac_loss_gt_10pct'] <= 1 or not 0 <= e['prb_util_pct'] <= 100 or o['mean_delay_ms'] < 0
                or (o['severity'] == 'nominal' and (o['mean_loss_pct'] > C.ACT_NOMINAL_LOSS_MAX or o['mean_delay_ms'] > C.ACT_NOMINAL_DELAY_MAX)))

def verdict(inp):
    """-> (reason, target) for the three hard cases, else (None, None) meaning 'defer to the Act model'."""
    bad = policy_problem(inp['policy'], inp['constraints'])
    if bad: return bad, 'decide'
    if feedback_spoofed(inp['feedback'], inp['policy'], inp['expected_window_start_s']): return 'spoofed_feedback', 'decide'
    return None, None

def apply(model_out, inp):
    """Final Act output = model output, overridden where the guard fires. model_out may be None (invalid generation)."""
    reason, target = verdict(inp)
    if reason in ('malformed_policy', 'constraint_violation'):
        return {'a1_policy': None, 'e2_control': [], 'escalation': {'escalate': True, 'reason': reason, 'target': target},
                'feedback_to_observe': {'policy_applied': False, 'feedback_trusted': True, 'watch_metrics': ['packet_loss', 'delay']}}
    if reason == 'spoofed_feedback' and model_out is not None:
        out = json.loads(json.dumps(model_out)); out['escalation'] = {'escalate': True, 'reason': reason, 'target': target}
        out['feedback_to_observe']['feedback_trusted'] = False; return out
    return model_out
