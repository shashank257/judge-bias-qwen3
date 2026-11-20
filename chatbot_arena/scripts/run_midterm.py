"""Midterm run: the proposal's full position + format design on the 100 frozen
pilot pairs, Qwen3 0.6B and 1.7B, thinking off and on (local MLX backend).

Per pair and configuration (manifest.format_requests): (A, B) in {orig, plain}^2
in both orders, plus orig/orig repeated in both orders (noise floor) = 10 calls.
The orig/orig repeat-0 calls are the same request ids as the earlier pilot, so
they are reused from the pilot log, not re-run.

  python scripts/run_midterm.py            # resumable; finished request ids are skipped
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dsc261 import common as C, judge as J, manifest as MF  # noqa: E402

sizes = os.environ.get("MIDTERM_SIZES", "0.6B,1.7B").split(",")
pairs = pd.read_parquet(C.path("data", "processed", "pilot_pairs.parquet"))
log_path = C.path("api", "pilot_local", "pilot_log.jsonl")  # one log for pilot + midterm
backend = J.MLXBackend()

prior = {}
if os.path.exists(log_path):
    for line in open(log_path):
        x = json.loads(line)
        prior.setdefault(x["request_id"], []).append(x)
done = J.done_request_ids(log_path)
by_pair = pairs.set_index("pair_id")


def text(p, ans, var):
    return p[f"answer_{ans}"] if var == "orig" else p[f"plain_{ans}"]


# cheap configs first so a partial run is already usable
configs = [(s, False) for s in sizes] + [(s, True) for s in sizes]
for size, thinking in configs:
    reqs = MF.format_requests(pairs, size, thinking, exp="midterm")
    todo = [r for r in reqs if r["request_id"] not in done]
    print(f"{MF.config_name(size, thinking)}: {len(todo)}/{len(reqs)} to run", flush=True)
    for i, r in enumerate(todo):
        p = by_pair.loc[r["pair_id"]]
        msgs = J.build_messages(p.question, text(p, r["slot1_answer"], r["slot1_variant"]),
                                text(p, r["slot2_answer"], r["slot2_variant"]), r["prompt_version"])
        res = J.judge_with_retry(backend, msgs, size, thinking, r["sampling_profile"], r["request_id"],
                                 log_path, prior_attempts=prior.get(r["request_id"], []))
        if i % 50 == 0:
            print(f"  {i}/{len(todo)} verdict={res.get('verdict')} lat={res.get('latency_s', 0):.1f}s", flush=True)
print("midterm done")
