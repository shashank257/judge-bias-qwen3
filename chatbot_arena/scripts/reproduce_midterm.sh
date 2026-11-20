#!/usr/bin/env bash
# Reproduce every midterm number, figure and the results-pack PDF from the committed
# judge logs (api/pilot_local/, api/midterm_remote/). No GPU or model download needed.
#
#   cd chatbot_arena && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
#   bash scripts/reproduce_midterm.sh            # uses ./.venv or ../.venv if present
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || { [ -x ../.venv/bin/python ] && echo ../.venv/bin/python || echo python3; } )}
export PYTHONPATH=src

echo "[1/3] main results + figures  -> report/results.csv, runs.csv, stats.json, fig_*.pdf"
"$PY" scripts/report_stats.py > /dev/null
echo "[2/3] robustness analyses     -> report/extra.json"
"$PY" scripts/report_extra.py > /dev/null

echo "[3/3] results pack PDF        -> share/DSC261_G3_midterm_results_pack.pdf"
if command -v pdflatex > /dev/null; then
  mkdir -p report/_build
  (cd report && pdflatex -interaction=nonstopmode -output-directory=_build results_pack.tex > /dev/null \
     && pdflatex -interaction=nonstopmode -output-directory=_build results_pack.tex > /dev/null)
  mkdir -p share && cp report/_build/results_pack.pdf share/DSC261_G3_midterm_results_pack.pdf
else
  echo "      pdflatex not found: skipped (tables/figures above are still regenerated)"
fi

"$PY" - <<'EOF'
import pandas as pd
r = pd.read_csv("report/results.csv", index_col=0)
cols = {"pos_excess": "position excess", "fmt_excess": "format excess", "md_pref": "markdown pref",
        "human_acc": "human agreement"}
print("\nMidterm results (100 Chatbot Arena pairs; percent / points):")
print((r[list(cols)] * 100).round(1).rename(columns=cols).to_string())
EOF
echo -e "\nDone. Compare with the tables in share/DSC261_G3_midterm_results_pack.pdf."
