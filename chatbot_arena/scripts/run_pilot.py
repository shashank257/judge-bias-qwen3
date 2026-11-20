"""Pilot: 100 frozen pilot pairs x both orders x {sizes} x {thinking off, on}.

Backend is chosen on the command line:
  python scripts/run_pilot.py local                 # Apple-silicon MLX (what we ran in week 1)
  python scripts/run_pilot.py featherless           # hosted API, needs FEATHERLESS_API_KEY
  python scripts/run_pilot.py openai-compatible URL # any vLLM / OpenAI-compatible server (OPENAI_API_KEY)
Resumable: finished request ids in the log are skipped.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dsc261 import common as C, judge as J, manifest as MF  # noqa: E402

backend_name = sys.argv[1] if len(sys.argv) > 1 else "local"
sizes = os.environ.get("PILOT_SIZES", "0.6B,1.7B").split(",")
n_pairs = int(os.environ.get("PILOT_N", "100"))
n_pairs_think = int(os.environ.get("PILOT_N_THINK", n_pairs))  # local hardware: thinking runs on fewer pairs

if backend_name == "local":
    backend = J.MLXBackend()
elif backend_name == "featherless":
    backend = J.OpenAICompletionsBackend("https://api.featherless.ai/v1", os.environ["FEATHERLESS_API_KEY"],
                                         {s: J.MODEL_REVISIONS[s][0] for s in J.MODEL_REVISIONS})
else:
    backend = J.OpenAICompletionsBackend(sys.argv[2], os.environ.get("OPENAI_API_KEY", "EMPTY"),
                                         {s: J.MODEL_REVISIONS[s][0] for s in J.MODEL_REVISIONS})

pairs = pd.read_parquet(C.path("data", "processed", "pilot_pairs.parquet")).head(n_pairs)
log_path = C.path("api", f"pilot_{backend_name}", "pilot_log.jsonl")
done = J.done_request_ids(log_path)
prior = {}
if os.path.exists(log_path):
    import json
    for line in open(log_path):
        x = json.loads(line); prior.setdefault(x["request_id"], []).append(x)
# cheap configs first so partial runs are already useful
configs = [(s, False) for s in sizes] + [(s, True) for s in sizes]
for size, thinking in configs:
    sub = pairs.head(n_pairs_think if thinking else n_pairs)
    reqs = MF.position_requests(sub, size, thinking, repeats=(0,), exp="pilot")
    todo = [r for r in reqs if r["request_id"] not in done]
    print(f"{MF.config_name(size, thinking)}: {len(todo)}/{len(reqs)} to run", flush=True)
    by_pair = pairs.set_index("pair_id")
    for i, r in enumerate(todo):
        p = by_pair.loc[r["pair_id"]]
        first, second = p[f"answer_{r['slot1_answer']}"], p[f"answer_{r['slot2_answer']}"]
        msgs = J.build_messages(p.question, first, second, r["prompt_version"])
        res = J.judge_with_retry(backend, msgs, size, thinking, r["sampling_profile"], r["request_id"], log_path,
                                 prior_attempts=prior.get(r["request_id"], []))
        if i % 20 == 0:
            print(f"  {i}/{len(todo)} verdict={res.get('verdict')} lat={res.get('latency_s', 0):.1f}s "
                  f"out_tok={res.get('n_completion_tokens')}", flush=True)
print("pilot done")
