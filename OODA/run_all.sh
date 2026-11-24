#!/usr/bin/env bash
# Usage: ./run_all.sh /path/to/folder/with/indore_0{1..4}_combined_5s.csv
set -e
D="${1:?folder with indore_0N_combined_5s.csv}"
for n in 1 2 3 4; do python build_observe.py --csv "$D/indore_0${n}_combined_5s.csv" --situation S$n; done
python build_orient.py
python build_decide.py --teacher rule | tail -1
python build_decide.py --teacher cplex
for n in 1 2 3 4; do python validate.py data/observe_S$n.jsonl observe_summary | head -1; python validate.py data/orient_S$n.jsonl orient_context | head -1; python validate.py data/decide_S$n.jsonl decide_policy | head -1; done
python build_act.py
for n in 1 2 3 4; do python validate.py data/act_S$n.jsonl act_command | head -1; done
python check_decide_optimality.py
python check_consistency.py
python check_act.py
