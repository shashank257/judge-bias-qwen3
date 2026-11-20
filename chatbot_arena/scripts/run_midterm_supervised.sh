#!/usr/bin/env bash
# Keeps the batched midterm run going until it finishes: restarts after any crash
# (resume is automatic from the log) and steps the KV token budget down after a GPU
# out-of-memory error. Usage: bash scripts/run_midterm_supervised.sh
cd "$(dirname "$0")/.."
OUT=${MIDTERM_OUT:-api/run_logs/midterm_batched_stdout.txt}
budgets=(24000 18000 13000 9000 5000)
i=0
tries=0
while true; do
  b=${budgets[$i]}
  echo "=== $(date +%H:%M:%S) start TOKEN_BUDGET=$b (try $tries)" >> "$OUT"
  PY=$( [ -x .venv/bin/python ] && echo .venv/bin/python || { [ -x ../.venv/bin/python ] && echo ../.venv/bin/python || echo python3; } )
  TOKEN_BUDGET=$b "$PY" scripts/run_midterm_batched.py >> "$OUT" 2>&1
  if tail -5 "$OUT" | grep -q "midterm done"; then
    echo "=== finished" >> "$OUT"; break
  fi
  tries=$((tries + 1))
  if tail -40 "$OUT" | grep -q "OutOfMemory\|Insufficient Memory"; then
    [ $i -lt $((${#budgets[@]} - 1)) ] && i=$((i + 1))
    echo "=== OOM -> lowering token budget" >> "$OUT"
  fi
  [ $tries -ge 30 ] && { echo "=== giving up after 30 restarts" >> "$OUT"; exit 1; }
  sleep 10
done
