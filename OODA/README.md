# OODA-RRM-CV2X dataset pipeline

Order of work: schemas -> Observe -> Orient -> Decide (rule placeholder, then optimizer) -> Act -> fine-tune.

## Done
- `schemas/` JSON Schema (Draft 2020-12) for the four inter-agent messages:
  ObserveSummary, ContextObject (Orient), Command-DSL policy (Decide), Act output.
- `config.py` situations S1-S4, QoS thresholds, zone size, split settings.
- `build_observe.py` telemetry -> ObserveSummary samples (1 network + 1 per RSU cell per 5 s window; cells = nearest of the 12 ns-3 sites).
- `validate.py` schema check + token-size stats.

## Run
    python build_observe.py --csv <situation.csv> --situation S2
    python validate.py data/observe_S2.jsonl observe_summary
    python build_orient.py && python validate.py data/orient_S4.jsonl orient_context

## Status
- Observe-DS built for all four situations with one script (data/observe_S1..S4.jsonl): 79 + 2843 + 2264 + 2253 = 7,439 samples (network + per-RSU-cell scope).
- Inactive/placeholder rows (no vehicle_type, 0 packets) are dropped; S1/S3 CSVs contain many of them.
- Thresholds are calibrated to the simulation (delay 30 ms, loss 10 %, jitter 10 ms), NOT to the proposal's 1 ms / 99.999 % URLLC target (ns-3 delay floor is ~3-5 ms).
- Orient-DS built (data/orient_S1..S4.jsonl): 65 + 2872 + 1758 + 2336 = 7,031 (cell scope only) samples, splits inherited from Observe.
  Teacher = deterministic rules in build_orient.py (scenario, intent resolution, weights, overrides). Only 3 of 6 scenario
  classes occur in the data (baseline, intersection_dense, emergency_preempt); platoon_* / handover_imminent need new traces.
  DRAFTS TO APPROVE: the 5 intent-conflict types, 4 regulatory contexts, QoS-weight values and PRB-share overrides.
  RAG snippets in config.py are paraphrased placeholders: replace with verbatim chunks from the real corpus.
- Decide-DS (data/decide_S*.jsonl, 7,031 per-RSU-cell samples): IBM CPLEX MILPs on the PROPOSAL's system model (placeholder parameters); rule baseline in data/decide_rule_S*.jsonl. `python build_decide.py [--teacher cplex|rule]`, `python check_decide_optimality.py`. See ASSUMPTIONS.md 6b.
- Act-DS (data/act_S*.jsonl, 9,029 samples incl. ~2K adversarial): rule teacher, `python build_act.py`, audited by `python check_act.py`. See ASSUMPTIONS.md 6d.
- Full list of placeholders, assumptions and TODOs: ASSUMPTIONS.md
- Quiet-tail / tiny-scope samples are kept; downsample them when assembling the training mix (filter on n_active_vehicles).

## Pipeline order
    build_observe.py (x4 situations) -> build_orient.py -> build_decide.py -> build_act.py -> check_consistency.py / check_act.py
    system_model.py = shared placeholder system model (Observe's modelled resource_state and Decide's optimizer both use it)

## Notes
- Records carry metadata (`cell`, `situation`, `split`, `obs_id`, `scenario_tag`): use ONLY `messages` as training text. Zone ids inside `messages` are per-sample pseudonyms.
- Splits are contiguous 20-window blocks, stratified by congestion; split by block, never by row.
- Emergency vehicles are detected if `vehicle_type` contains "emerg" or "ambul" (needed for S4).
- Thresholds in `config.py` are first guesses; tune after inspecting all four situations.
