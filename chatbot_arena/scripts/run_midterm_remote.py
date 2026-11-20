"""Midterm run against an OpenAI-compatible server (vLLM on a GPU box, Featherless, ...).

Same design as run_midterm.py (100 pilot pairs x 10 calls x thinking off/on), same
prompt/template/thinking-budget rules (judge.judge_once), resumable, many requests
in flight at once.

  # on the GPU machine, terminal 1 (BF16 = contract-compliant; needs >= 24 GB GPU):
  vllm serve Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
       --dtype bfloat16 --max-model-len 12288 --seed 261 --port 8000
  # terminal 2:
  python scripts/run_midterm_remote.py --url http://localhost:8000/v1 --size 8B

Writes api/midterm_remote/<size>_log.jsonl. Copy that file back into the same path in
the main project; notebook 06 reads it automatically.
"""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dsc261 import common as C, judge as J, manifest as MF  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True, help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1")
ap.add_argument("--size", default="8B", help="0.6B / 1.7B / 4B / 8B (must match the served model)")
ap.add_argument("--model", default=None, help="served model name (default: the HF repo id)")
ap.add_argument("--workers", type=int, default=32, help="requests in flight (vLLM batches them)")
ap.add_argument("--timeout", type=float, default=1800.0, help="per-call seconds, incl. time queued in the server")
ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
a = ap.parse_args()

repo = J.MODEL_REVISIONS[J.QUANT_MODELS.get(a.size, (a.size,))[0]][0]
backend = J.OpenAICompletionsBackend(a.url, a.api_key, {a.size: a.model or repo}, timeout=a.timeout)
backend.name = f"openai-compatible:{a.url}"
J.get_tokenizer(a.size)  # load once up front: transformers' lazy import is not thread-safe
pairs = pd.read_parquet(C.path("data", "processed", "pilot_pairs.parquet")).set_index("pair_id")
log_path = C.path("api", "midterm_remote", f"{a.size}_log.jsonl")
lock = threading.Lock()

prior = {}
if os.path.exists(log_path):
    for line in open(log_path):
        x = json.loads(line)
        if x.get("error") is None:
            prior.setdefault(x["request_id"], []).append(x)


def write(rec):
    with lock, open(log_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def one(r):
    """Verdict with one retry; provider errors retried (not counted as attempts)."""
    p = pairs.loc[r["pair_id"]]
    t = lambda ans, v: p[f"answer_{ans}"] if v == "orig" else p[f"plain_{ans}"]  # noqa: E731
    msgs = J.build_messages(p.question, t(r["slot1_answer"], r["slot1_variant"]),
                            t(r["slot2_answer"], r["slot2_variant"]), r["prompt_version"])
    attempt = len(prior.get(r["request_id"], []))
    errors = 0
    while attempt < 2 and errors < 8:
        try:
            rec = J.judge_once(backend, msgs, a.size, r["thinking"], r["sampling_profile"], r["request_id"], attempt)
            rec["error"] = None
        except Exception as e:  # network / server error: logged, retried, not an attempt
            write({"request_id": r["request_id"], "attempt": attempt, "error": repr(e)[:300], "verdict": None})
            errors += 1
            continue
        write(rec)
        attempt += 1
        if rec["verdict"]:
            return True
    return False


for thinking in (False, True):
    reqs = MF.format_requests(pairs.reset_index(), a.size, thinking, exp="midterm")
    todo = [r for r in reqs if not any(x.get("verdict") for x in prior.get(r["request_id"], []))
            and len(prior.get(r["request_id"], [])) < 2]
    print(f"{MF.config_name(a.size, thinking)}: {len(todo)}/{len(reqs)} to run", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, r) for r in todo]
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 100 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}", flush=True)
print("midterm done")
