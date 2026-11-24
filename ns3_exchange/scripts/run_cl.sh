#!/bin/bash
# Closed-loop batch runner. Run from the ns-3 root:  ./run_cl.sh JOBS_FILE [PARALLEL]
# JOBS_FILE: one job per line   name|ns-3 arguments|controller arguments     (lines starting with # are ignored)
# Each job gets its own folder ctrl/<name>, its own controller process (log: chunk_logs/<name>_ctrl.log) and ns-3 log chunk_logs/<name>.log
JOBS=$1; PAR=${2:-4}; mkdir -p chunk_logs ctrl
grep -v '^[[:space:]]*#' "$JOBS" | grep -v '^[[:space:]]*$' | xargs -P "$PAR" -d '\n' -I{} bash -c '
  IFS="|" read -r n x c <<< "$1"
  d="$PWD/ctrl/$n"; rm -rf "$d"; mkdir -p "$d"
  python3 ~/exchange/scripts/ctrl_server.py --dir "$d" $c > chunk_logs/${n}_ctrl.log 2>&1 &
  cpid=$!
  ./ns3 run --no-build "indore-nr-12cell $x --ctrlDir=$d --outPrefix=$n" > chunk_logs/$n.log 2>&1
  sleep 2; kill $cpid 2>/dev/null
' _ {}
cp cl_*_5s.csv ~/exchange/results/ 2>/dev/null
