"""One-line progress for the midterm run: done/total, elapsed, estimated time left."""
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dsc261 import common as C, judge as J, manifest as MF  # noqa: E402
import json  # noqa: E402

log = C.path("api", "pilot_local", "pilot_log.jsonl")
pairs = pd.read_parquet(C.path("data", "processed", "pilot_pairs.parquet"))
done = J.done_request_ids(log)
lat = {}
for line in open(log):
    r = json.loads(line)
    if r.get("latency_s") and r.get("size"):
        lat.setdefault(MF.config_name(r["size"], r["thinking"]), []).append(r["latency_s"])
total = left = 0
eta = 0.0
parts = []
for s in os.environ.get("MIDTERM_SIZES", "0.6B,1.7B").split(","):
    for th in (False, True):
        cfg = MF.config_name(s, th)
        ids = [r["request_id"] for r in MF.format_requests(pairs, s, th)]
        d = sum(i in done for i in ids)
        total += len(ids); left += len(ids) - d
        eta += (len(ids) - d) * (sum(lat.get(cfg, [10])) / len(lat.get(cfg, [10])))
        parts.append(f"{cfg} {d}/{len(ids)}")
start = os.stat(os.environ.get("MIDTERM_OUT", C.path("api", "run_logs", "midterm_stdout.txt"))).st_birthtime
el = time.time() - start
fmt = lambda s: f"{int(s // 3600)}h{int(s % 3600 // 60):02d}m"  # noqa: E731
print(f"{time.strftime('%H:%M')} | {total - left}/{total} done | elapsed {fmt(el)} | ~{fmt(eta)} left | " + ", ".join(parts))
