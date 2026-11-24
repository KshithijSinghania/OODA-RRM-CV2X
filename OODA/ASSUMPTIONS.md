# OODA-RRM-CV2X pipeline: placeholders, assumptions and TODOs

Status as of this document: Observe-DS, Orient-DS, Decide-DS and Act-DS are built for S1-S4 (all with placeholder or rule-based parts, listed below). Fine-tuning has not started.
Everything below is something **I decided, not something you or the proposal specified**, unless marked "from data" or "from proposal".

Legend for "Risk": **High** = could invalidate a paper claim if left as is; **Med** = affects dataset quality; **Low** = cosmetic or easy to change.

---

## Decisions log

| # | Decision (from you) | Consequence |
|---|---|---|
| 1 | "rsu fork": use the RSU option. | Per-RSU cells are decision domains (zones = nearest of the 12 cell sites, M8). |
| 2 | "keep the system as proposal only". | Decide follows the proposal's system model (M RSUs with MEC + macro gNB; V2V PC5, V2I UL and DL share K PRBs; per-vehicle power budget, MEC capacity, interference limits, URLLC reliability). The ns-3 traces are used only as a source of state (vehicle counts, load, ambulance events); the simulator's downlink-only structure no longer shapes the policy schema. My earlier "align with the simulator" working assumption is withdrawn. |

| 3 | "make observe and orient consistent". | Observe, Orient and Decide now share one vocabulary and one system model (section 6c). I interpreted this as aligning them with the proposal and with each other; correct me if you meant something else. |

I interpreted "rsu fork" as choosing the RSU option; correct me if you meant otherwise.

---

## 0. What the ns-3 sources tell us

**Two versions of the script have been seen.** (i) The uploaded `indore-nr-sumo-s1-corrected.cc`: 1 gNB at the centre of the bounding box. (ii) A later pasted "fixed network" version: 12 cells, namely 1 macro gNB (25 m) plus 11 RSUs (6 m) at fixed coordinates, all on the same 20 MHz carrier, with NR A3 handover (3 dB hysteresis, 256 ms time-to-trigger) over an X2 mesh, and per-vehicle UDP start/stop bounded to the vehicle's presence in the SUMO trace. I could not verify which version produced which CSV, so I tested the data.

**Evidence that S2 came from the 12-cell network.** In S2's congested period (t = 450-700 s), the nearest of the 12 sites explains 67 % of the variance in per-vehicle loss, versus 39 % for distance to a single central gNB. The busy-corridor cells (sites 0, 2, 4) lose 52-71 % of packets while far cells (6-11) lose about 0 %. S4's worst cells are the same three (0, 2, 4), which is consistent. S1 and S3 are inconclusive because loss is low everywhere. **My earlier conclusion that the data came from a single cell with about 74 Mbps capacity was wrong and is withdrawn.**

**Facts from script (ii):**

| Item | Value |
|---|---|
| Cells | 12 (macro C00 + RSUs C01-C11), layout fixed for all situations, sized for the p90 concurrent load (about 174 vehicles) of S1+S2; the macro site is a known recurring chokepoint |
| Radio | 5.9 GHz, 20 MHz, numerology 0 (15 kHz), gNB 8x8, UE 2x2, 3GPP UMi, shadowing off; every cell uses the same band, so reuse-1 with inter-cell interference |
| Power | 38.5 dBm for every cell, RSUs included (the accompanying note says 37 dBm; the code says 38.5) |
| Traffic | UDP downlink only, 512 B payload (540 B on the wire), 100 packets/s per UE; no uplink, no sidelink |
| QoS | No dedicated bearers; default QoS flow and default scheduler for everyone |
| Core | PGW to remote host 1 ms delay |
| Output | Per vehicle x 5 s window: packets, loss, throughput, delay, jitter, bytes. **It does not log the serving cell, handover events, RSRP/SINR or PRB use.** |

**Mismatches and open issues:**

| # | Issue | Risk | What to do |
|---|---|---|---|
| M1 | The simulation is NR Uu downlink. The proposal's V2V sidelink (PC5), V2I uplink and MEC are not simulated; the "RSUs" are NR gNBs. **Decision: follow the proposal; ns-3 is a source of vehicle and downlink-QoS state only.** | Med | Add uplink/sidelink/MEC to the simulator or testbed if you want trace-derived (not model-based) labels. |
| M2 | Nothing is controlled (default QoS flow, default scheduler, identical power and carrier in all cells). The traces are an uncontrolled baseline; policy effects are never simulated. | **High** | Closing the loop needs ns-3 changes (per-class bearers/5QI, scheduler or PRB quotas, per-cell power) and new runs. |
| M3 | The traces cover only V2I downlink: no uplink, sidelink, MEC or per-class traces. The Decide labels for power, MEC and interference are therefore model-based (CPLEX on a placeholder model), not derived from traces. | **High** | See 6b. |
| M4 | Delay floor of 3-5 ms: plausible contributors are the 1 ms core link, 1 ms slots and scheduling. The proposal's 1 ms URLLC target is unreachable here. | Med | Set realistic targets for this simulator. |
| M5 | 5 s windows versus the proposal's 100-500 ms OODA cycle. | Med | Set `WINDOW_SIZE` to 0.5 and rerun (about 10x more rows). |
| M6 | UMi channel, not a V2X channel model. If the paper claims V2X channel modelling, check TR 37.885 (NR-V2X evaluation methodology; TR 36.885 is LTE-V2X). Verify. | Low | |
| M7 | The S2-S4 scripts and the SUMO context-layer script have not been seen, and I cannot tell which script version made S1, S3 and S4. | Med | Send them; confirm each CSV was run on the 12-cell script. |
| M8 | **Zones are now the nearest of the 12 sites** (C00 = macro, C01-C11 = RSUs), replacing my 600 m grid. This is a proxy for the serving cell: it ignores handover hysteresis. | Med | Log the real serving cell per UE per window in ns-3 (M9). |
| M9 | No serving-cell, handover, RSRP/SINR or per-cell PRB logging. Without it, `handover_imminent` cannot be labelled, per-RSU PRB occupancy is unavailable, and the proposal's CSI/PRB telemetry is missing. | **High** | Add logging (serving cell per UE, handover start/end events, per-cell PRB use) and rerun. |
| M10 | Script inconsistencies: the header still says 1 gNB and 25 m; 37 dBm in the note versus 38.5 in code; RSUs transmit at the macro's power; the script's own comment says the A3 attribute names were not verified. | Med | Confirm it compiled, and count the handovers that actually occurred. |
| M11 | The layout was derived from S1+S2 telemetry, so S2's hot cells are partly by design. Models must use load features, not cell identity. **Done:** the `zone_id` in model-visible text is now a per-sample pseudonym (R100-R999, unrelated across samples); the true cell is in the record's `cell` field, which is metadata only and must not be part of training text. | Med | Train on `messages` only. |
| M12 | The same memorisation risk applies to SUMO edge IDs in `hotspots.edges` / `worst_edges` (a model could learn "edge X is always congested"). **Not yet anonymised.** | Med | Anonymise per sample, or drop them from the Observe input, unless Act needs real edge IDs. |

---

## 1. Decide system-model parameters (proposal-based; all values are placeholders)

Check specification numbers against current versions; I wrote the identifiers from memory.

| Parameter | Placeholder now | Where to get the real value |
|---|---|---|
| PRBs per RSU (K) and PRB bandwidth (B) | 106 PRBs x 180 kHz (20 MHz at 15 kHz) | your OAI NR-V2X / testbed configuration; 3GPP TS 38.101-1; sidelink resource pool in TS 38.331 |
| Max transmit power per vehicle | 23 dBm (UE power class 3; verify); levels 10/14/17/20/23 dBm | TS 38.101-1; ETSI EN 302 571; India DoT/WPC allocation (status unknown to me) |
| Link budget (reference distance 150 m, path-loss exponent 2.8, noise figure 9 dB) | as stated | your channel emulator / a V2X channel model (check TR 37.885) |
| Interference model (IoT 0.3 dB per vehicle, power cap shrinking 0.25 dB per vehicle, strategy factors) | invented | ns-3 or testbed measurements of inter-RSU interference |
| Capacity anchor | 9 Mbps for an interference-free cell at 23 dBm | a controlled saturation run |
| Per-vehicle QoS profile (class, minimum rate, maximum latency, reliability) | class mix 20/60/20 URLLC/eMBB/mMTC; alert cars and the ambulance all URLLC | TS 22.186, TS 23.287 (PQI); your intent registry; define in the simulator |
| URLLC reliability | 99 % of demand served | the proposal's target is PRR 99.999 % at 1 ms, which this class-level model cannot express |
| MEC capacity and task profile | 150 tasks/s; 1 / 2 / 0.5 tasks/s per vehicle per class; 0.05 Mbps uplink per task/s | measure on your Jetson Orin; ETSI GS MEC 003 / MEC 030 |
| O-RAN message formats for Act | open | O-RAN Alliance A1 and E2 specs (E2SM-RC) |
| Optimizer | IBM CPLEX 22.2 Community Edition (1,000 variables/constraints) | academic license if you need larger models |

---

## 2. Data and preprocessing assumptions

| # | Assumption | Where | Risk | What to do |
|---|---|---|---|---|
| D1 | Rows with no `vehicle_type` or `tx_packets == 0` are placeholders and are dropped (S1: 359 of 748 rows, S3: 175,286 of 187,800, S4: 2 rows). I assumed they are vehicles not yet in the simulation. | `build_observe.py` main | Med | Confirm with whoever generated the CSVs. The 2 dropped S4 rows are moving cars (about 18 m/s) with zero packets, which may be real. |
| D2 | S2 has 626 vehicles (brief said 627). I set the expected value to 626. | `config.SITUATIONS` | Low | Confirm. |
| D3 | "313 vehicles" is the total spawned over the run, not simultaneous. Peak concurrency: S1 39, S2 237, S3 86, S4 91. | from data | Med | Make sure the paper describes load as peak concurrency, not total. |
| D4 | Cells are the nearest of the 12 sites in the ns-3 fixed-network script (C00 = macro, C01-C11 = RSUs), a proxy for the serving cell; assigned from the window's end position; minimum 3 vehicles per cell. Observe also has one `network` sample per window (Non-RT RIC view, 24 % of Observe samples); Orient and Decide use cell samples only. | `config.CELL_SITES`, `MIN_VEHICLES_PER_ZONE` | Med | Replace with the logged serving cell (M8, M9). |
| D5 | `local_density_300m` = number of other vehicles within 300 m of the end position in the same window (matches S2's column exactly). I applied the same definition to S1, S3 and S4 and included the ambulance in the count. | `add_density` | Low | Verified on S2 only. |
| D6 | `congestion_ratio` (like `local_density_300m`) comes from the SUMO context layer merged by window, per the script header. It is dropped from the Observe input because S1/S3/S4 lack it and its definition is unknown. | `build_observe.py` | Low | Ask what it measures. |
| D7 | Ambulance = `vehicle_type` containing "emerg" or "ambul". Alert radius 300 m (from data: every car with `alert_zone_samples > 0` has `min_dist_to_ambulance_m` at most 299.9). An "emergency event" exists for a scope if the ambulance is in it or any car in it is within 300 m. | `config.ALERT_RADIUS_M` | Low | Confirm the 300 m comes from your simulation settings. |
| D8 | Observe input is aggregated statistics (a "telemetry DB query result"), not raw per-vehicle rows. | design | Med | The proposal describes raw multi-source telemetry plus RAG over a telemetry database. Decide how to describe this in the paper. |
| D9 | The data lacks CSI, PRB occupancy, interference maps, MEC load and BSM/CAM rates, which the proposal lists as Observe inputs. **Partly addressed:** Observe now carries (a) `v2x_messages` (mean message rate and fraction of vehicles at alert rate), a real trace-derived proxy for BSM/CAM rates, and (b) `resource_state` (offered-load ratio, interference-over-thermal, MEC load ratio, current allocation), which is **modelled by the shared placeholder system model, not measured**, and flagged `estimated: true`. CSI is still absent. | **High** | Log the real quantities in ns-3 / the testbed and replace the modelled values. |
| D10 | 5 s windows (`WINDOW_SIZE` in the ns-3 script); the final window per run is shorter (e.g. 3.6 s) and is not treated specially. The proposal expects OODA cycles of 100-500 ms. | data | Med | Rerun ns-3 with `WINDOW_SIZE = 0.5` (see M5). |

## 3. Thresholds (all in `config.py`)

All are **calibrated to the simulation, not taken from a standard**. The proposal's URLLC target (1 ms, 99.999 %) cannot be used: even free-flow S1 has a median delay of about 5 ms.

| Setting | Value | Basis | Risk |
|---|---|---|---|
| `LOSS_VIOLATION_PCT` | 10 % | round number | Med |
| `DELAY_VIOLATION_MS` | 30 ms | about p96 of free-flow delay in S1/S3/S4 (20 ms sat inside baseline noise) | Med |
| `DELAY_TAIL_MS` / `LOSS_SPIKE_PCT` | 100 ms / 50 % | round numbers | Low |
| `HIGH_JITTER_MS` | 10 ms | S3 baseline jitter p90 is about 15 ms | Low |
| `DELIVERY_RATIO_LOW` | 0.80 | round number (rx bytes / tx bytes); equivalent to roughly 20 % loss, so partly redundant with loss | Low |
| `STATIONARY_SPEED_MPS` | 0.5 | round number | Low |
| `SEVERITY_BINS` | 5 / 20 / 50 % of vehicles violating | round numbers | Med |
| `LOAD_BINS` | 10 / 30 / 60 neighbours within 300 m | quantiles of S2 only | Med |
| Trend: `TREND_LOOKBACK`, `TREND_LOSS_PP`, `TREND_LOAD_PCT` | 3 windows, 3 pp, 10 % | round numbers | Low |

**TODO:** once the standards-based targets and the ns-3 delay floor are understood, either set per-class thresholds from TS 22.186 / TS 23.287 or define them relative to a measured free-flow baseline, and document the choice. Changing any value means rebuilding Observe, then Orient.

## 4. Observe-DS assumptions

| # | Assumption | Risk | What to do |
|---|---|---|---|
| O1 | Summaries (severity, anomalies, trend, hotspots) are produced by deterministic rules; narratives are templates. The proposal expects 1K human-written summaries and LLM-generated ones with human review. | **High** | Write about 1K human summaries; use an LLM to rewrite the rest; keep the rule output as the structured part. |
| O2 | Samples are one global sample plus one per zone per window. Zone and global samples from the same window are correlated, so the effective sample size is below the raw count (7,439). | Med | Report the number of windows as well as samples. |
| O3 | Splits: blocks of 20 consecutive windows (3 for S1, which has 17 windows), stratified by (congested, ambulance present), 80/10/10, seed 42. S1 test has 12 samples, S3 val has 60. | Med | Add a held-out-situation split (see section 8). |
| O4 | Class balance is skewed: 75-82 % of samples are nominal in every situation (S1 79 %, S2 76 %, S3 82 %, S4 75 %). Tiny scopes (5 or fewer vehicles) are 20-41 % of samples (S2 20 %, S1 31 %, S3 41 %, S4 40 %). | Med | Downsample or reweight when assembling the training mix. |
| O5 | Token counts are estimated as characters / 4. | Low | Recount with the Llama-3 tokenizer. |
| O6 | Observe total is 7,439 samples (1,775 network + 5,664 cell) against a 15K target. | Med | More variety is needed, not more rows from the same four runs (more topologies, seeds, densities). The proposal specifies 6 road topologies. |

## 5. Orient-DS assumptions

| # | Assumption | Where | Risk | What to do |
|---|---|---|---|---|
| R1 | Scenario rules: `emergency_preempt` if any emergency event; else `intersection_dense` if at least 5 vehicles and severity degraded/critical or a standstill cluster, or load high/very_high; else `baseline`. | `classify` | **High** | The labels are rules, so the model learns the rules (see section 8). |
| R2 | Only 3 of 6 scenario classes occur in the data. | from data | **High** | New SUMO runs (platoons, merges, handover) or synthetic augmentation. |
| R3 | The 5 intent-conflict types and 4 regulatory contexts are my invention; the proposal gives only the counts. | `build_orient.py`, schema | **High** | Replace with your definitions. |
| R4 | Intent priority: regulatory > safety > emergency > OEM contract > throughput > energy. | `config.INTENT_PRIORITY` | **High** | Confirm with stakeholders (or the cited intent-management spec). |
| R5 | Context sampling: regulatory context probabilities (emergency: none .45, TMC corridor .30, power cap .10, safety zone .15; otherwise none .70, power cap .15, safety zone .15); energy-saving mode 25 %; OEM contract 30 % with min eMBB 2/5/10 Mbps ("OEM-A"). | `sample_context` | Med | Arbitrary; the model's behaviour on conflict cases depends on them. |
| R6 | QoS weight base values (baseline .40/.40/.20, dense .55/.30/.15, emergency .75/.15/.10), plus shifts for severity (+.03/.06/.10 to URLLC), energy-saving, OEM contract, safety zone. | `weights` | **High** | Fix after Decide's optimizer shows which weights give good outcomes. |
| R7 | Constraint overrides: URLLC min PRB share 20/30(35 if critical)/40(50 under TMC corridor); eMBB max 70/50(40)/30(25); policy latency 500/200/50 ms; power cap 20 dBm; safety-zone +5. The 200 and 50 ms values loosely follow the proposal's OODA-cycle (under 200 ms) and emergency-preemption (under 50 ms) targets. | `overrides` | **High** | Replace with limits derived from the system model. |
| R8 | Confidence is a hand-set heuristic (e.g. 0.97 / 0.88 / 0.92 / 0.80 / 0.95 / 0.72, minus 0.05 with a conflict). Not calibrated. | `confidence` | Med | Drop it, or calibrate on real classifier error. |
| R9 | RAG snippets are my paraphrases, not verbatim 3GPP text; no real DPR/FAISS retrieval; no clause numbers (the proposal wants chains annotated with supporting clauses). "NRV2X-MODE1/2", "ROAD-EVENT-ONTOLOGY" and "TMC-REG" are placeholder IDs. | `config.RAG_KB` | **High** | Build the real knowledge base and retrieve with it. Note 36.885 is a Technical Report (TR), not a TS as the proposal writes. |
| R10 | Retrieval noise: each relevant snippet is dropped with p = 0.2, plus 2 random distractors. Snippet sets still predict the tag at about 82 % (majority baseline 65 %). | `build_one` | Med | Replace with real retrieval from the Observe summary. |
| R11 | Variants per Observe sample: emergency x3, dense x2, baseline x1. Variants share the same summary with different intent contexts, so they are correlated. | `ORIENT_VARIANTS` | Med | Report unique summaries. |
| R12 | Orient is trained on the gold Observe summaries, not the Observe model's noisy output. | design | **High** | Add noise-injection / Observe-model-output training and evaluation. |
| R13 | Orient total is 7,031 samples against a 20K target (cell scope only; network-scope samples were dropped when Orient was aligned with the proposal's per-RSU OODA loop). | | Med | See O6. |

## 6. Schemas (`schemas/`)

All four schemas are my design. Only Observe and Orient have data behind them.

| # | Assumption | Risk | What to do |
|---|---|---|---|
| S1 | The Decide policy ("Command DSL") is my design, following the proposal's four outputs: PRB priority mapping, per-class power budget (dBm), MEC offloading policy (mode, ratio, priority class, compute shares) and interference coordination strategy (`none`, `frequency_reuse`, `sidelink_mode1_scheduling`, `power_backoff_edge`), plus preemption. The proposal mentions a C-V2X Command DSL with a formal type checker but I have not seen it. | **High** | Send me the real DSL (or its grammar) and I will conform to it. |
| S2 | Act output uses made-up names: A1 policy type `CV2X-RRM-POLICY-1`; E2 functions `RC-PRB-QUOTA`, `RC-TX-POWER`, `RC-MEC-OFFLOAD`, `RC-INTERFERENCE`, `RC-PREEMPT` and action strings such as `SET_CLASS_PRB_QUOTA`. They are **not** real O-RAN E2SM-RC identifiers. | **High** | Map to the real O-RAN A1/E2 specs. |
| S3 | PRB allocation is class-level percentages, not per-PRB assignment (the proposal says Decide outputs class-level parameters). | Low | Confirm. |
| S4 | The escalation reasons and targets in the Act schema (`malformed_policy`, `constraint_violation`, `spoofed_feedback`, `policy_execution_failure`, `qos_violation_persists/onset`, `scenario_change`; targets `decide`/`orient`) are my guess; the proposal only says Act escalates failures to Decide. | Med | Define with your adversarial cases (RQ6). |

## 6b. Decide-DS (CPLEX on the proposal's system model, built)

7,031 samples (S1 65, S2 2,872, S3 1,758, S4 2,336), one policy per RSU cell and window (cell = nearest of the 12 sites; the macro gNB cell is treated like an RSU). The input also carries a `current_allocation` snapshot (the proposal's "current PRB/power state"), which in this data is my assumption of the uncontrolled baseline.

**Teacher.** For each of the four interference strategies, one MILP solved with IBM CPLEX 22.2; the best strategy is kept (ties go to `none`). Variables: integer PRB share per class, power level per class, MEC offload level and integer compute share per class. Objective: Orient-weighted demand satisfaction on the radio side plus a weighted MEC term, minus a URLLC shortfall penalty, a small pull toward the Orient weights and a small power penalty. Constraints: PRB shares sum to 100 with the Orient URLLC minimum and eMBB maximum and a 5 % floor per class; a PRB-averaged transmit-power (interference) limit that shrinks with cell load; per-vehicle power levels up to 23 dBm or the regional cap; MEC compute capacity; 99 % URLLC service. The constraint families are the proposal's; their functional forms and numbers are mine.

**Checks.** All 864 distinct problems (3,456 MILPs) returned OPTIMAL; schema and constraint checks found no violations; an independent plain-Python recomputation with random and local search on 120 samples never beat CPLEX. The rule baseline is kept in `data/decide_rule_S*.jsonl`.

**Degenerate outputs (placeholder artefacts, not findings).** MEC mode is `full` in 100 % of samples, because the placeholder MEC capacity (150 tasks/s) far exceeds the cells' task load. `frequency_reuse` is never selected, because it halves the usable PRBs. A model trained on this data would learn those as constants.

| # | Assumption | Risk | What to do |
|---|---|---|---|
| DC1 | System model follows the proposal (your decision). The traces give state only; uplink, sidelink and MEC effects are model-based. | Med | Add them to the simulator/testbed for trace-derived labels. |
| DC2 | MILP functional forms (satisfaction objective, interference cap, MEC term, uplink cost of offloading) are my construction. | **High** | Replace with the real utility and constraints (and the Lyapunov-based URLLC constraint if desired). |
| DC3 | Link budget and capacity anchor (150 m, exponent 2.8, 9 Mbps) are placeholders; the anchor comes from ns-3's hot cells. | **High** | See section 1. |
| DC4 | Interference model and strategy factors (IoT, power-cap relief, usable PRBs, reuse) are invented. | **High** | Measure or derive. |
| DC5 | Traffic class mix is invented (the simulator has one traffic type). | **High** | Define classes per vehicle. |
| DC6 | MEC parameters are invented; the result is degenerate (always full offload). | **High** | Measure on the Orin; then re-run. |
| DC7 | Power levels {10, 14, 17, 20, 23} dBm and the 20 dBm regional cap (from Orient) are placeholders. | Med | Use real limits. |
| DC8 | `current_allocation` is an assumed uncontrolled baseline: demand-proportional PRBs, maximum power, no offload, no coordination. | Med | Use real state once the simulator logs it. |
| DC9 | Policies are per cell with the nearest-site proxy; there is no inter-RSU coordination (the proposal's Non-RT RIC layer) and no DRL enforcer. | Med | Add the coordination layer and the enforcer. |
| DC10 | `valid_for_ms` = 5 x the Orient latency bound; reuse factor follows the strategy (2 for `frequency_reuse`). | Low | Replace. |
| DC11 | 7,031 samples against the 25K target; variants of the same Observe sample are correlated. | Med | More runs and topologies. |
| DC12 | Policies cannot be applied or evaluated in the current ns-3 (M2), so there is no ground truth for policy quality. | **High** | Closed-loop simulator changes. |
| DC13 | "Optimal" is relative to my model and tie-breaks (small power penalty, pull toward Orient weights, `none` first). CPLEX Community Edition caps models at 1,000 variables/constraints (ours have about 60); check license terms. | Med | |

---

## 6c. Consistency across Observe, Orient and Decide (checked by `check_consistency.py`)

What was aligned, following the proposal (OODA cycles run at the RSU edge; the Non-RT RIC sees the network):

| Aspect | Before | Now |
|---|---|---|
| Vocabulary | Observe said "zone", Decide said "cell"/"RSU", scope values `global`/`zone` | One vocabulary: scope `network` or `cell`, field `cell_id`; anonymised labels `R100-R999`. ("alert zone" and "safety_zone_mandate" remain as domain terms.) |
| Scope | Orient covered `global` and `zone` samples; Decide only zones | Orient and Decide use `cell` samples only (one per RSU cell and window). Observe keeps `network` samples as the Non-RT RIC view. |
| Proposal's Observe inputs | no message rates or resource state | `v2x_messages` (trace-derived) and `resource_state` (modelled, flagged) added |
| State passed to Decide | Decide made up its own `current_allocation` | Decide receives Observe's `resource_state` verbatim; both compute it from one shared module, `system_model.py` |
| Orient input | Observe summary only | the same summary, now richer |

**Automated checks (all pass):** Orient embeds exactly the Observe output; Decide's context equals Orient's output; Decide's cell state and current allocation equal Observe's; splits and situations propagate; the emergency tag appears iff the Observe event exists; preemption ids come from the Observe event; `resource_state` is reproducible from the shared model; no real cell id or situation id in any text.

**New observations:**
- In S4, 274 of the 275 emergency-event samples also show vehicles at alert message rate (150 Hz), so the event is visible in two independent trace-derived signals. S2 and S3 never show it.
- The modelled MEC load never exceeds 0.49, which is why Decide's MEC mode is always `full` (DC6).

| # | Assumption | Risk | What to do |
|---|---|---|---|
| CO1 | `resource_state` is deterministic in `n_active_vehicles` and the event, so it adds no information beyond them and a model can compute it exactly. Its value is consistency, not signal. | Med | Replace with logged values from ns-3. |
| CO2 | The alert-rate threshold (1.25 x 100 Hz) and the message-rate definition (tx packets / window length) are my choices, inferred from the S4 data. | Low | Confirm against the S4 script. |
| CO3 | Observe output is now about 250 tokens on average (max 343, budget 512). | Low | |
| CO4 | CPLEX is now run with the MIP gap tolerance set to 0 (the default left a 1e-5 gap that an independent search could beat). | Low | |

---

## 6d. Act-DS (rule teacher, built)

9,029 samples (S1 83, S2 3,688, S3 2,258, S4 3,000): 7,031 clean (one per Decide sample), 999 malformed-policy and 999 spoofed-feedback adversarial samples (target: 10K total with 2K adversarial). Mean output about 210 tokens. Built by `build_act.py`; audited by `check_act.py`, which re-implements the Act rules from the system prompt independently of the builder: all 9,029 labels match, every adversarial case is detected as intended, no clean sample is flagged malformed or spoofed, translations to E2 are faithful to the policy, clean policies equal Decide's output, and feedback equals the cell's real next-window Observe summary.

**Provenance.** `build_act.py`, the Act schema and the Act config were already in the project folder when I started this step; I did not write them in this conversation. I audited rather than replaced them (the build reproduces byte-for-byte), and made two changes: malformed-policy samples now keep their feedback, and two dead lines were removed (see AC9).

**What an Act sample is.** Input: the Decide policy for one RSU cell, the Orient constraints, the QoS Observe saw before the policy, and the enforcer's execution feedback (or null). Output: either a rejection (no A1/E2 commands, escalate to Decide) or one A1 policy plus E2 control messages, a trust flag for the feedback, an escalation decision, and metrics for Observe to watch. Priority: reject policy > spoofed feedback > execution failure > enforcer violations > new emergency under a non-preempting policy > QoS degradation.

| # | Assumption | Risk | What to do |
|---|---|---|---|
| AC1 | **Feedback is not causal.** The traces are an uncontrolled baseline, so the "enforcer report" is the same cell's real next-window Observe summary repackaged, plus synthetic enforcer faults (4 % not applied, 4 % violation flags). A policy never changes it. `qos_violation_*` labels therefore mean "the cell is degraded", not "this policy failed". | **High** | Closed-loop ns-3 (M2) so feedback reflects the applied policy. |
| AC2 | The enforcer report fields are invented (`prb_util_pct` is modelled from the shared system model, not measured; violation names, `auth_ok`). There is no DRL enforcer. | **High** | Replace with real enforcer output. |
| AC3 | O-RAN names are placeholders (S2 above); translation is a template. | **High** | Real A1/E2 formats. |
| AC4 | Malformed-policy taxonomy (9 kinds: DSL error, PRB or compute sum, unknown enum, missing field, bad id, below URLLC minimum, above eMBB maximum, over power cap, preemption mismatch) and spoofed-feedback taxonomy (5 kinds: wrong policy id, failed authentication, stale window, impossible value, self-contradictory) are my design. Attacks are single-field perturbations; coordinated or subtle attacks (GPS denial, spoofed BSMs from RQ6) are not covered. | **High** | Define threat models with your supervisor. |
| AC5 | Thresholds for "self-contradictory" feedback: severity `nominal` with mean loss above 8 % or delay above 40 ms. Checked against the data: nominal-severity cell windows reach at most 3.58 % loss and 20.28 ms. | Low | Tune if thresholds in section 3 change. |
| AC6 | Spoofed feedback escalates to `decide`. The proposal only says Act escalates policy execution failures to Decide; where spoofing should go is undefined. | Med | Decide, or add an Observe re-verification target. |
| AC7 | **Class imbalance.** Clean samples: no escalation 5,145; QoS persists 1,070; onset 289; execution failure 265; constraint violation 261; scenario change **1**. Val and test contain only 2 onset samples each and no scenario-change sample, so the Orient-escalation path cannot be learned or evaluated. | **High** | Add traces with emergencies appearing mid-run, or synthesise them and label them as synthetic. |
| AC8 | Act receives Orient's constraints and the baseline QoS, which are not in the proposal's list of Act inputs (policy plus execution feedback); they are needed to check a policy. | Low | Confirm. |
| AC9 | Fixed: malformed-policy samples originally had no feedback, while only 2.8 % of clean samples lacked it, a shortcut a model could learn. They now keep the real feedback; the teacher rejects the policy before using it. Two dead lines in the builder were removed. | Low | |
| AC10 | 9,029 samples against the 10K target; 1,998 adversarial against 2K (rounding per situation). Adversarial parents are drawn from the same situation and split as their clean parent. | Med | More runs and topologies. |
| AC11 | Labels come from one deterministic rule set, so a fine-tuned Act model learns the rules (section 8). For malformed and constraint-violating policies, the rule needs only the policy and constraints, which a few lines of code already do correctly. | **High** | Evaluate on hard and noisy cases; compare with the rule baseline. |

---

## 7. Not built yet (and what each needs)

| Stage | Needs | Current plan |
|---|---|---|
| **Decide-DS** (25K target) | The system model from section 1; the DSL (S1); a utility and URLLC-constraint formulation | Built with CPLEX on the proposal's system model (section 6b); next, replace the placeholder model parameters with measured values and re-run. |
| **Act-DS** (10K target) | Real A1/E2 formats (S2); a real enforcer and closed-loop feedback (AC1, AC2); threat models (AC4) | Built with a rule teacher (section 6d); replace the placeholders and rebalance (AC7). |
| **Fine-tuning** | Base model choice (proposal: Llama-3-8B), GPU, framework | One LoRA adapter per agent (rank 32, alpha 64, QLoRA) as in the proposal; start with Observe and Orient. |

## 8. Evaluation caveats (what the paper can and cannot claim)

1. **Rule-labelled data.** Observe and Orient labels come from deterministic rules. A high score on held-out data labelled by the same rules shows the model learned the rules, not that an LLM is needed.
2. **Needed for a credible claim:** evaluate on the Observe model's actual (noisy) output; hold out a whole situation or topology; add hard cases (conflicting intents, mixed signals) where a simple rule would fail; compare to a rule-based baseline and a DRL baseline.
3. **Built-in control:** S3 (313 cars, no ambulance) vs S4 (313 cars + ambulance). In S4, 1,032 of 1,587 emergency samples have nominal QoS, so a model must read the ambulance event, not network trouble.
4. **Single city, few runs.** All four traces are one Indore map. Generalisation claims need more topologies and random seeds.

## 9. Prioritised TODO

**Before training anything**
- [ ] Close the loop in ns-3 (M2) and add the logging in M9; add uplink/sidelink/MEC if you want trace-derived Decide labels (M1/M3).
- [ ] Send the S2-S4 ns-3 scripts and the SUMO context-layer script (M7); confirm placeholder rows (D1).
- [ ] Get or confirm the Command DSL (S1) and O-RAN formats (S2).
- [ ] Confirm or replace the conflict types, regulatory contexts, intent priority (R3, R4).
- [ ] Decide threshold policy (section 3).

**Before the final dataset**
- [ ] Real RAG knowledge base with verbatim clauses (R9, R10).
- [ ] Human-written summaries for about 1K Observe samples (O1).
- [ ] Add platoon and handover scenarios (R2) and more topologies/seeds (O6).
- [ ] Replace the Decide placeholders: capacity and link budget (DC3), interference (DC4), traffic classes (DC5), MEC (DC6); fix the degenerate outputs; replace the Act placeholders (AC1-AC4) and fix its class imbalance (AC7).
- [ ] Rebalance the training mix and recount tokens (O4, O5).

**Before writing results**
- [ ] Noise-injection and held-out-situation evaluation (section 8).
- [ ] Rule-based and DRL baselines.
- [ ] Re-check every number in this document against your final configuration; any change to `config.py` requires rebuilding Observe, then Orient.
